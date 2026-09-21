
"""
Orchestrator Agent

Coordinates the Airport Operations AI Copilot workflow.

Day 3 flow:

    User Query
        ↓
    Conversation Memory
        ↓
    Context Resolution
        ↓
    Investigator
        ↓
    ReAct Controller
        ↓
    Policy & Compliance
        ↓
    Resolution
        ↓
    Final Response
        ↓
    Conversation Memory


Day 4 safety-enhanced flow:

    User Query
        ↓
    Conversation Memory
        ↓
    Context Resolution
        ↓
    Investigator
        ↓
    ReAct Controller
        ↓
    Policy & Compliance
        ↓
    Resolution
        ↓
    AI Recommendation
        ↓
    Safety / Guardrail Layer
        ↓
    Risk Classification
        ↓
    Human Approval (when required)
        ↓
    Safe Executor
        ↓
    Audit Trail + Distillation
        ↓
    Final Response
        ↓
    Conversation Memory


Important architecture rule:

The Resolution Agent recommends actions.

The Resolution Agent does NOT directly execute sensitive actions.

All executable recommendations must pass through safe_executor.py.
"""

import re

from typing import Any, Dict, Optional

from src.agents.state import (
    add_error,
    increment_iteration,
    create_initial_state,
    update_state,
    can_continue,
    summarize_state,
)

from src.agents.investigator import investigate
from src.agents.policy_agent import policy_agent
from src.agents.resolution_agent import resolution_agent

from src.agents.react_controller import (
    run_react_iteration,
    should_continue_react_loop,
)

from src.memory.conversation_memory import ConversationMemory
from src.safe_executor import execute_safely


MAX_ITERATIONS = 5

SUPPORTED_AIRPORTS = [
    "SFO",
    "LAX",
    "JFK",
]


# =============================================================
# GENERAL HELPERS
# =============================================================

def format_for_response(value: Any) -> str:
    """
    Convert a state value into safe display text.

    Handles strings, lists, dictionaries, numbers, and None.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "\n".join(
            f"- {item}"
            for item in value
        )

    if isinstance(value, dict):
        return "\n".join(
            f"- {key}: {val}"
            for key, val in value.items()
        )

    return str(value)


def detect_airport_from_text(
    text: str,
) -> Optional[str]:
    """
    Detect an airport code from text.

    Returns:
        SFO, LAX, JFK, or None.
    """

    if not text:
        return None

    text_upper = text.upper()

    for airport in SUPPORTED_AIRPORTS:
        if airport in text_upper:
            return airport

    return None


# =============================================================
# CONVERSATION CONTEXT
# =============================================================

def resolve_airport_from_conversation(
    user_query: str,
    memory: Optional[ConversationMemory],
) -> Optional[str]:
    """
    Resolve the airport for the current query.

    Priority:

    1. Airport explicitly mentioned in current query.
    2. Airport mentioned in recent conversation.
    3. None if no airport can be determined.
    """

    # ---------------------------------------------------------
    # FIRST: CURRENT QUERY
    # ---------------------------------------------------------

    airport = detect_airport_from_text(user_query)

    if airport:
        return airport

    # ---------------------------------------------------------
    # SECOND: CONVERSATION MEMORY
    # ---------------------------------------------------------

    if memory is None:
        return None

    recent_messages = memory.get_recent_messages(
        count=10
    )

    for message in reversed(recent_messages):
        content = message.get(
            "content",
            "",
        )

        airport = detect_airport_from_text(content)

        if airport:
            return airport

    return None


# =============================================================
# MEMORY RECALL
# =============================================================

def is_memory_recall_query(
    user_query: str,
) -> bool:
    """
    Detect questions that explicitly ask about previous
    conversation context.

    These queries should be answered from short-term memory
    rather than starting a new operational investigation.
    """

    if not user_query:
        return False

    query = user_query.lower().strip()

    recall_phrases = (
        "previous investigation",
        "previous recommendation",
        "previous decision",
        "previous conversation",
        "earlier investigation",
        "earlier recommendation",
        "earlier decision",
        "last investigation",
        "last recommendation",
        "last decision",
        "what did we recommend",
        "what was the recommendation",
        "what did you recommend",
        "what was recommended",
        "what did we decide",
        "what was the decision",
        "remind me what",
        "from our previous",
        "from the previous",
        "from our earlier",
        "from the earlier",
    )

    return any(
        phrase in query
        for phrase in recall_phrases
    )


def _extract_memory_section(
    response: str,
    heading: str,
    next_headings: tuple,
) -> str:
    """
    Extract a section from a previous orchestrator response.

    The search is performed from the supplied heading and stops
    at the next known heading.

    This helper is intentionally used only after we have located
    the correct part of the response.
    """

    if not response:
        return ""

    heading_pattern = re.escape(heading)

    next_patterns = [
        re.escape(item)
        for item in next_headings
    ]

    if next_patterns:
        stop_pattern = "|".join(next_patterns)

        pattern = (
            heading_pattern
            + r"\s*(.*?)"
            + r"(?=\n\s*(?:"
            + stop_pattern
            + r")|\Z)"
        )
    else:
        pattern = (
            heading_pattern
            + r"\s*(.*)"
        )

    match = re.search(
        pattern,
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return ""

    value = match.group(1).strip()

    # Remove accidental markdown bullet from the beginning.
    value = re.sub(
        r"^\s*-\s*",
        "",
        value,
    ).strip()

    return value


def _extract_memory_line_value(
    response: str,
    label: str,
) -> str:
    """
    Extract a single-line value.

    Example:

        Policy approval required: False

    returns:

        False
    """

    if not response:
        return ""

    pattern = (
        r"^\s*-\s*"
        + re.escape(label)
        + r"\s*(.+?)\s*$"
    )

    match = re.search(
        pattern,
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not match:
        return ""

    return match.group(1).strip()


def _extract_historical_recommendation(
    response: str,
) -> Dict[str, str]:
    """
    Extract only the historical fields needed for memory recall.

    Important:

    The final orchestrator response contains multiple occurrences
    of words such as "Reasoning:".

    The first "Reasoning:" belongs to the ReAct section.

    Therefore we MUST locate the recommendation first and only
    search for the recommendation reasoning after that point.

    This prevents ReAct reasoning from leaking into memory recall.
    """

    if not response:
        return {
            "recommendation": "",
            "reasoning": "",
            "approval_required": "",
            "execution_status": "",
        }

    # ---------------------------------------------------------
    # FIND THE RECOMMENDATION SECTION FIRST
    # ---------------------------------------------------------

    recommendation_match = re.search(
        r"Recommended action:\s*(.*?)(?=\n\s*Reasoning:|\n\s*Policy & Compliance:|\n\s*Safety & Execution:|\n\s*Agent notes:|\Z)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    recommendation = ""

    if recommendation_match:
        recommendation = (
            recommendation_match.group(1)
            .strip()
        )

    # ---------------------------------------------------------
    # FIND REASONING ONLY AFTER RECOMMENDATION
    # ---------------------------------------------------------

    reasoning = ""

    if recommendation_match:
        remaining_response = response[
            recommendation_match.end():
        ]

        reasoning_match = re.search(
            r"^\s*Reasoning:\s*(.*?)(?=\n\s*Policy & Compliance:|\n\s*Possible actions:|\n\s*Safety & Execution:|\n\s*Agent notes:|\Z)",
            remaining_response,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if reasoning_match:
            reasoning = (
                reasoning_match.group(1)
                .strip()
            )

    # ---------------------------------------------------------
    # POLICY APPROVAL
    # ---------------------------------------------------------

    approval_required = ""

    policy_section_match = re.search(
        r"Policy\s*&\s*Compliance:\s*(.*?)(?=\n\s*Possible actions:|\n\s*Recommended action:|\n\s*Safety & Execution:|\n\s*Agent notes:|\Z)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if policy_section_match:
        policy_section = (
            policy_section_match.group(1)
        )

        approval_match = re.search(
            r"^\s*-\s*Policy approval required:\s*(.+?)\s*$",
            policy_section,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if approval_match:
            approval_required = (
                approval_match.group(1)
                .strip()
            )

    # ---------------------------------------------------------
    # EXECUTION STATUS
    # ---------------------------------------------------------

    execution_status = ""

    safety_section_match = re.search(
        r"Safety\s*&\s*Execution:\s*(.*?)(?=\n\s*Agent notes:|\Z)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if safety_section_match:
        safety_section = (
            safety_section_match.group(1)
        )

        status_match = re.search(
            r"^\s*-\s*Execution status:\s*(.+?)\s*$",
            safety_section,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if status_match:
            execution_status = (
                status_match.group(1)
                .strip()
            )

    return {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "approval_required": approval_required,
        "execution_status": execution_status,
    }


def build_memory_recall_response(
    memory: Optional[ConversationMemory],
    airport: Optional[str],
) -> Optional[str]:
    """
    Answer a historical recommendation question directly
    from short-term conversation memory.

    This function does NOT start a new investigation.

    It extracts only:

        Recommended action
        Reasoning
        Policy approval requirement
        Execution status
    """

    if memory is None:
        return None

    if memory.is_empty():
        return None

    recent_messages = memory.get_recent_messages(
        count=10
    )

    # Search newest-to-oldest.
    for message in reversed(recent_messages):

        role = str(
            message.get(
                "role",
                "",
            )
        ).lower()

        if role != "assistant":
            continue

        response = str(
            message.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if not response:
            continue

        response_airport = detect_airport_from_text(
            response
        )

        # If an airport is known, make sure the historical
        # response belongs to that airport.
        if (
            airport
            and response_airport != airport
        ):
            continue

        extracted = _extract_historical_recommendation(
            response
        )

        recommendation = extracted.get(
            "recommendation",
            "",
        )

        if not recommendation:
            continue

        airport_label = (
            airport
            or response_airport
            or "the previous airport"
        )

        lines = [
            (
                f"From our previous "
                f"{airport_label} investigation:"
            ),
            "",
            f"Recommended action: {recommendation}",
        ]

        reasoning = extracted.get(
            "reasoning",
            "",
        )

        if reasoning:
            lines.extend(
                [
                    "",
                    f"Reasoning: {reasoning}",
                ]
            )

        approval_required = extracted.get(
            "approval_required",
            "",
        )

        if approval_required:
            lines.extend(
                [
                    "",
                    (
                        "Policy approval required "
                        f"in that investigation: "
                        f"{approval_required}"
                    ),
                ]
            )

        execution_status = extracted.get(
            "execution_status",
            "",
        )

        if execution_status:
            lines.extend(
                [
                    "",
                    (
                        "Execution status from "
                        f"that investigation: "
                        f"{execution_status}"
                    ),
                ]
            )

        return "\n".join(lines)

    return None


# =============================================================
# STRUCTURED EXECUTION RECOMMENDATION
# =============================================================

def get_structured_execution_recommendation(
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Return a structured executable recommendation when one
    is explicitly present in shared state.

    Day 4 safety rule:

    Do NOT attempt to convert arbitrary natural-language
    recommendations into executable actions by guessing
    parameters.
    """

    recommendation = state.get(
        "execution_recommendation"
    )

    if recommendation is None:
        return None

    if not isinstance(
        recommendation,
        dict,
    ):
        return None

    if not recommendation.get(
        "action"
    ):
        return None

    return dict(
        recommendation
    )


# =============================================================
# SAFETY LAYER
# =============================================================

def _normalize_pending_approval_result(
    execution_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize the Safe Executor result.

    Important distinction:

        awaiting_approval
            No human decision has happened yet.

        rejected
            A human explicitly rejected the action.

        executed
            The action was actually executed.

    The Safe Executor can return approval information inside the
    nested ``risk`` and ``human_approval`` dictionaries.  The
    orchestrator therefore normalizes both the top-level and nested
    representations here.

    A high-risk recommendation must NEVER be labelled ``rejected``
    merely because interactive approval was not supplied.
    """

    if not isinstance(execution_result, dict):
        return execution_result

    # ---------------------------------------------------------
    # READ APPROVAL INFORMATION FROM ALL SUPPORTED LOCATIONS
    # ---------------------------------------------------------

    risk = execution_result.get("risk")
    if not isinstance(risk, dict):
        risk = {}

    human_approval = execution_result.get("human_approval")
    if not isinstance(human_approval, dict):
        human_approval = {}

    # Safe Executor versions may expose approval_required either at
    # the top level or inside the risk/human_approval objects.
    approval_required = execution_result.get("approval_required")

    if approval_required is None:
        approval_required = risk.get("approval_required")

    if approval_required is None:
        approval_required = human_approval.get("approval_required")

    approval_required = bool(approval_required)

    # Safe Executor versions may expose approval_decision at the
    # top level or inside human_approval.
    approval_decision = execution_result.get(
        "approval_decision"
    )

    if approval_decision is None:
        approval_decision = human_approval.get(
            "approval_decision"
        )

    approval_decision = str(
        approval_decision or ""
    ).lower().strip()

    status = str(
        execution_result.get(
            "status",
            "",
        )
        or ""
    ).lower().strip()

    # ---------------------------------------------------------
    # EXPLICIT HUMAN REJECTION
    # ---------------------------------------------------------

    explicit_rejection = approval_decision in {
        "rejected",
        "reject",
        "denied",
    }

    if explicit_rejection:
        execution_result["status"] = "rejected"
        execution_result["execution_status"] = "rejected"
        execution_result["execution_allowed"] = False
        execution_result["approval_required"] = approval_required
        execution_result["approval_decision"] = "rejected"

        human_approval["approval_required"] = approval_required
        human_approval["approval_decision"] = "rejected"
        human_approval["approved"] = False
        execution_result["human_approval"] = human_approval

        return execution_result

    # ---------------------------------------------------------
    # PENDING HUMAN APPROVAL
    # ---------------------------------------------------------

    if approval_required:
        # Only an explicit approval may move a high-risk action
        # beyond the pending state.
        if approval_decision not in {
            "approved",
            "approve",
        }:
            execution_result["status"] = (
                "awaiting_approval"
            )

            execution_result["execution_status"] = (
                "awaiting_approval"
            )

            execution_result["execution_allowed"] = False
            execution_result["approval_required"] = True
            execution_result["approval_decision"] = "pending"

            execution_result["message"] = (
                "High-risk action requires human "
                "approval before execution."
            )

            human_approval["approval_required"] = True
            human_approval["approval_decision"] = "pending"
            human_approval["approved"] = False
            execution_result["human_approval"] = human_approval

            return execution_result

    # ---------------------------------------------------------
    # EXECUTED
    # ---------------------------------------------------------

    if status == "executed":
        execution_result["execution_status"] = "executed"

    # Preserve the normalized approval requirement even when the
    # action did not require human approval.
    execution_result["approval_required"] = approval_required

    return execution_result


def run_safety_layer(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pass an explicitly structured recommendation through
    the Day 4 Safe Executor.
    """

    state["safety_check_performed"] = True

    recommendation = (
        get_structured_execution_recommendation(
            state
        )
    )

    # ---------------------------------------------------------
    # NO EXECUTABLE RECOMMENDATION
    # ---------------------------------------------------------

    if recommendation is None:

        state["execution_result"] = {
            "status": "advisory_only",
            "execution_allowed": False,
            "message": (
                "Resolution produced an advisory recommendation "
                "without a structured executable action. "
                "No operational action was executed."
            ),
        }

        state["execution_status"] = (
            "advisory_only"
        )

        state["human_approval_required"] = False

        return state

    # ---------------------------------------------------------
    # SAFE EXECUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "DAY 4 SAFETY LAYER"
    )
    print("=" * 70)

    print(
        "\nStructured AI recommendation:"
    )

    print(
        format_for_response(
            recommendation
        )
    )

    try:

        execution_result = execute_safely(
            recommendation=recommendation,
            user_request=state.get(
                "user_query",
                "",
            ),
            policy_results=state.get(
                "policy_results",
                [],
            ),
            human_approval=None,
            interactive_approval=False,
            state=state,
        )

        # -----------------------------------------------------
        # NORMALIZE APPROVAL STATE
        # -----------------------------------------------------

        execution_result = (
            _normalize_pending_approval_result(
                execution_result
            )
        )

        state["execution_result"] = (
            execution_result
        )

        state["execution_status"] = (
            execution_result.get(
                "execution_status",
                execution_result.get(
                    "status",
                    "unknown",
                ),
            )
            if isinstance(
                execution_result,
                dict,
            )
            else "unknown"
        )

        if isinstance(execution_result, dict):
            risk = execution_result.get("risk")
            if not isinstance(risk, dict):
                risk = {}

            state["human_approval_required"] = bool(
                execution_result.get(
                    "approval_required",
                    risk.get(
                        "approval_required",
                        False,
                    ),
                )
            )
        else:
            state["human_approval_required"] = False

        print("\nSafety decision:")
        print("-" * 70)

        print(
            format_for_response(
                execution_result
            )
        )

        return state

    except Exception as error:

        error_message = str(
            error
        )

        add_error(
            state,
            "orchestrator_safety_layer",
            error_message,
        )

        state["execution_result"] = {
            "status": "blocked",
            "execution_status": "blocked",
            "execution_allowed": False,
            "message": (
                "Safety layer blocked execution "
                f"because: {error_message}"
            ),
        }

        state["execution_status"] = (
            "blocked"
        )

        state["human_approval_required"] = False

        return state


# =============================================================
# FINAL RESPONSE
# =============================================================

def build_final_response(
    state: Dict[str, Any],
) -> str:
    """
    Build the final user-facing response from shared state.
    """

    airport = state.get(
        "detected_airport",
        "unknown",
    )

    severity = state.get(
        "severity_level",
        "unknown",
    )

    operational_data = state.get(
        "operational_data",
        {},
    )

    policy_decision = state.get(
        "policy_decision",
        "Insufficient Information",
    )

    # This is POLICY approval, not human safety approval.
    approval_required = state.get(
        "approval_required",
        "unknown",
    )

    possible_actions = state.get(
        "possible_actions",
        [],
    )

    recommended_action = state.get(
        "recommended_action",
        "",
    )

    recommendation_reasoning = state.get(
        "recommendation_reasoning",
        "",
    )

    investigator_findings = state.get(
        "investigator_findings",
        "",
    )

    react_trace = state.get(
        "react_trace",
        [],
    )

    errors = state.get(
        "errors",
        [],
    )

    execution_result = state.get(
        "execution_result"
    )

    execution_status = state.get(
        "execution_status"
    )

    response_lines = []

    response_lines.append(
        f"Airport Operations Summary: {airport}"
    )

    response_lines.append("")

    response_lines.append(
        f"Severity: {str(severity).upper()}"
    )

    # ---------------------------------------------------------
    # OPERATIONAL METRICS
    # ---------------------------------------------------------

    if operational_data:

        timestamp = operational_data.get(
            "timestamp"
        )

        completion_rate = operational_data.get(
            "completion_rate"
        )

        average_eta = operational_data.get(
            "average_eta_minutes"
        )

        active_drivers = operational_data.get(
            "active_drivers"
        )

        queue_size = operational_data.get(
            "queue_size"
        )

        surge_multiplier = operational_data.get(
            "surge_multiplier"
        )

        cancellation_rate = operational_data.get(
            "driver_cancellation_rate"
        )

        request_volume = operational_data.get(
            "request_volume"
        )

        response_lines.append("")

        response_lines.append(
            "Current operational metrics:"
        )

        if timestamp is not None:
            response_lines.append(
                f"- Timestamp: {timestamp}"
            )

        if request_volume is not None:
            response_lines.append(
                f"- Request volume: {request_volume}"
            )

        if active_drivers is not None:
            response_lines.append(
                f"- Active drivers: {active_drivers}"
            )

        if completion_rate is not None:
            response_lines.append(
                f"- Completion rate: "
                f"{completion_rate:.2%}"
            )

        if average_eta is not None:
            response_lines.append(
                f"- Average ETA: "
                f"{average_eta:.2f} minutes"
            )

        if queue_size is not None:
            response_lines.append(
                f"- Queue size: {queue_size}"
            )

        if surge_multiplier is not None:
            response_lines.append(
                f"- Surge multiplier: "
                f"{surge_multiplier:.2f}x"
            )

        if cancellation_rate is not None:
            response_lines.append(
                f"- Driver cancellation rate: "
                f"{cancellation_rate:.2%}"
            )

    # ---------------------------------------------------------
    # INVESTIGATION FINDINGS
    # ---------------------------------------------------------

    if investigator_findings:

        response_lines.append("")

        response_lines.append(
            "Investigation findings:"
        )

        findings_text = format_for_response(
            investigator_findings
        )

        response_lines.append(
            findings_text
        )

    # ---------------------------------------------------------
    # REACT REASONING
    # ---------------------------------------------------------

    if react_trace:

        latest_trace = react_trace[-1]

        response_lines.append("")

        response_lines.append(
            "ReAct reasoning:"
        )

        response_lines.append(
            f"- Observation: "
            f"{format_for_response(latest_trace.get('observation'))}"
        )

        response_lines.append(
            f"- Reasoning: "
            f"{format_for_response(latest_trace.get('reasoning'))}"
        )

        response_lines.append(
            f"- Action: "
            f"{format_for_response(latest_trace.get('action'))}"
        )

        response_lines.append(
            f"- More information required: "
            f"{latest_trace.get('needs_more_information')}"
        )

    # ---------------------------------------------------------
    # POLICY
    # ---------------------------------------------------------

    response_lines.append("")

    response_lines.append(
        "Policy & Compliance:"
    )

    response_lines.append(
        f"- Decision: {policy_decision}"
    )

    response_lines.append(
        f"- Policy approval required: "
        f"{approval_required}"
    )

    # ---------------------------------------------------------
    # POSSIBLE ACTIONS
    # ---------------------------------------------------------

    if possible_actions:

        response_lines.append("")

        response_lines.append(
            "Possible actions:"
        )

        if isinstance(
            possible_actions,
            list,
        ):
            for action in possible_actions:

                response_lines.append(
                    f"- "
                    f"{format_for_response(action)}"
                )

        else:

            response_lines.append(
                format_for_response(
                    possible_actions
                )
            )

    # ---------------------------------------------------------
    # RECOMMENDED ACTION
    # ---------------------------------------------------------

    if recommended_action:

        response_lines.append("")

        response_lines.append(
            "Recommended action:"
        )

        response_lines.append(
            format_for_response(
                recommended_action
            )
        )

    # ---------------------------------------------------------
    # REASONING
    # ---------------------------------------------------------

    if recommendation_reasoning:

        response_lines.append("")

        response_lines.append(
            "Reasoning:"
        )

        response_lines.append(
            format_for_response(
                recommendation_reasoning
            )
        )

    # ---------------------------------------------------------
    # DAY 4 SAFETY / EXECUTION
    # ---------------------------------------------------------

    if state.get(
        "safety_check_performed"
    ):

        response_lines.append("")

        response_lines.append(
            "Safety & Execution:"
        )

        if execution_status:

            response_lines.append(
                f"- Execution status: "
                f"{execution_status}"
            )

        if execution_result:

            if isinstance(
                execution_result,
                dict,
            ):

                status = execution_result.get(
                    "status"
                )

                if status:

                    response_lines.append(
                        f"- Safety decision: "
                        f"{status}"
                    )

                risk_level = execution_result.get(
                    "risk_level"
                )

                if risk_level:

                    response_lines.append(
                        f"- Risk level: "
                        f"{risk_level}"
                    )

                human_approval_required = (
                    execution_result.get(
                        "approval_required"
                    )
                )

                if human_approval_required is not None:

                    response_lines.append(
                        f"- Human approval required: "
                        f"{human_approval_required}"
                    )

                approval_decision = (
                    execution_result.get(
                        "approval_decision"
                    )
                )

                if approval_decision:

                    response_lines.append(
                        f"- Approval decision: "
                        f"{approval_decision}"
                    )

                message = execution_result.get(
                    "message"
                )

                if message:

                    response_lines.append(
                        f"- Result: "
                        f"{message}"
                    )

            else:

                response_lines.append(
                    format_for_response(
                        execution_result
                    )
                )

    # ---------------------------------------------------------
    # AGENT ERRORS / NOTES
    # ---------------------------------------------------------

    if errors:

        response_lines.append("")

        response_lines.append(
            "Agent notes:"
        )

        for error in errors:

            agent = error.get(
                "agent",
                "unknown",
            )

            error_message = error.get(
                "error",
                "unknown error",
            )

            response_lines.append(
                f"- {agent}: "
                f"{format_for_response(error_message)}"
            )

    response_lines = [
        str(item)
        for item in response_lines
    ]

    return "\n".join(
        response_lines
    )


# =============================================================
# MAIN ORCHESTRATOR
# =============================================================

def orchestrate(
    user_query: str,
    memory: Optional[ConversationMemory] = None,
) -> Dict[str, Any]:
    """
    Run the complete multi-agent workflow.
    """

    print("\n" + "=" * 70)
    print(
        "AIRPORT OPERATIONS AI COPILOT"
    )
    print("ORCHESTRATOR")
    print("=" * 70)

    print(
        f"\nUser query: {user_query}"
    )

    # ---------------------------------------------------------
    # CREATE INITIAL STATE
    # ---------------------------------------------------------

    state = create_initial_state(
        user_query
    )

    state["max_iterations"] = (
        MAX_ITERATIONS
    )

    # ---------------------------------------------------------
    # LOAD CONVERSATION MEMORY
    # ---------------------------------------------------------

    if memory is not None:

        previous_context = (
            memory.get_context()
        )

        state["conversation_history"] = (
            memory.get_messages()
        )

        print("\n" + "-" * 70)
        print(
            "SHORT-TERM MEMORY"
        )
        print("-" * 70)

        if memory.is_empty():

            print(
                "No previous conversation context."
            )

        else:

            print(
                "Previous conversation context:"
            )

            print(
                previous_context
            )

    # ---------------------------------------------------------
    # RESOLVE CONTEXTUAL AIRPORT
    # ---------------------------------------------------------

    resolved_airport = (
        resolve_airport_from_conversation(
            user_query,
            memory,
        )
    )

    if resolved_airport:

        state["detected_airport"] = (
            resolved_airport
        )

        print("\n" + "-" * 70)
        print(
            "CONTEXT RESOLUTION"
        )
        print("-" * 70)

        print(
            f"Resolved airport: "
            f"{resolved_airport}"
        )

    # ---------------------------------------------------------
    # MEMORY RECALL / HISTORICAL FOLLOW-UP
    # ---------------------------------------------------------

    if (
        memory is not None
        and not memory.is_empty()
        and is_memory_recall_query(
            user_query
        )
    ):

        memory_response = (
            build_memory_recall_response(
                memory=memory,
                airport=resolved_airport,
            )
        )

        if memory_response:

            state["detected_airport"] = (
                resolved_airport
            )

            # Mark this state explicitly as a memory-only
            # response. The Streamlit layer can use this later
            # to preserve the previous operational dashboard.
            state["memory_recall"] = True

            state["recommended_action"] = (
                memory_response
            )

            state["execution_status"] = (
                "memory_recall"
            )

            state["completed"] = True

            update_state(
                state,
                current_agent="orchestrator",
                final_response=memory_response,
            )

            memory.add_exchange(
                user_query,
                memory_response,
            )

            print("\n" + "=" * 70)
            print(
                "MEMORY RECALL"
            )
            print("=" * 70)

            print(
                memory_response
            )

            return state

    # ---------------------------------------------------------
    # NORMAL WORKFLOW
    # ---------------------------------------------------------

    try:

        # =====================================================
        # STEP 1 — OPERATIONS INVESTIGATOR
        # =====================================================

        if not can_continue(state):

            raise RuntimeError(
                "Maximum iteration limit reached "
                "before investigation."
            )

        print("\n" + "=" * 70)
        print(
            "HANDOFF → OPERATIONS INVESTIGATOR"
        )
        print("=" * 70)

        increment_iteration(
            state
        )

        # -----------------------------------------------------
        # CONTEXTUAL FOLLOW-UP HANDLING
        # -----------------------------------------------------

        if (
            resolved_airport
            and not detect_airport_from_text(
                user_query
            )
        ):

            original_query = (
                state["user_query"]
            )

            state["user_query"] = (
                f"{original_query} "
                f"(Context: current conversation "
                f"airport is {resolved_airport}.)"
            )

        state = investigate(
            state
        )

        # -----------------------------------------------------
        # AIRPORT VALIDATION
        # -----------------------------------------------------

        if not state.get(
            "detected_airport"
        ):

            add_error(
                state,
                "orchestrator",
                "Unable to determine airport "
                "from user query or conversation context.",
            )

            state["completed"] = True

            final_response = (
                "I could not determine which airport "
                "you are asking about. "
                "Please specify SFO, LAX, or JFK."
            )

            update_state(
                state,
                current_agent="orchestrator",
                final_response=final_response,
            )

            if memory is not None:

                memory.add_exchange(
                    user_query,
                    final_response,
                )

            return state

        # =====================================================
        # STEP 2 — CONTROLLED REACT CONTROLLER
        # =====================================================

        if not can_continue(state):

            raise RuntimeError(
                "Maximum iteration limit reached "
                "before ReAct reasoning."
            )

        print("\n" + "=" * 70)
        print(
            "REACT CONTROLLER"
        )
        print("=" * 70)

        state = run_react_iteration(
            state
        )

        react_trace = state.get(
            "react_trace",
            [],
        )

        if react_trace:

            latest_trace = react_trace[-1]

            print("\nObservation:")
            print("-" * 70)

            print(
                latest_trace.get(
                    "observation",
                    "",
                )
            )

            print("\nReasoning:")
            print("-" * 70)

            print(
                latest_trace.get(
                    "reasoning",
                    "",
                )
            )

            print("\nAction:")
            print("-" * 70)

            print(
                latest_trace.get(
                    "action",
                    "",
                )
            )

            print("\nNeed more information:")

            print(
                latest_trace.get(
                    "needs_more_information",
                    False,
                )
            )

        # -----------------------------------------------------
        # CONTROLLED REACT LOOP
        # -----------------------------------------------------

        react_iterations = 1

        while should_continue_react_loop(
            state
        ):

            if react_iterations >= MAX_ITERATIONS:
                break

            increment_iteration(
                state
            )

            state = run_react_iteration(
                state
            )

            react_iterations += 1

        print("\nReAct iterations:")

        print(
            len(
                state.get(
                    "react_trace",
                    [],
                )
            )
        )

        print(
            "ReAct loop continue:",
            should_continue_react_loop(
                state
            ),
        )

        # =====================================================
        # STEP 3 — POLICY AGENT
        # =====================================================

        if not can_continue(state):

            raise RuntimeError(
                "Maximum iteration limit reached "
                "before policy analysis."
            )

        print("\n" + "=" * 70)
        print(
            "HANDOFF → POLICY & COMPLIANCE AGENT"
        )
        print("=" * 70)

        increment_iteration(
            state
        )

        state = policy_agent(
            state
        )

        # =====================================================
        # STEP 4 — RESOLUTION AGENT
        # =====================================================

        if not can_continue(state):

            raise RuntimeError(
                "Maximum iteration limit reached "
                "before resolution."
            )

        print("\n" + "=" * 70)
        print(
            "HANDOFF → RESOLUTION AGENT"
        )
        print("=" * 70)

        increment_iteration(
            state
        )

        state = resolution_agent(
            state
        )

        # =====================================================
        # STEP 5 — DAY 4 SAFETY LAYER
        # =====================================================

        print("\n" + "=" * 70)
        print(
            "HANDOFF → SAFETY & SAFE EXECUTOR"
        )
        print("=" * 70)

        state = run_safety_layer(
            state
        )

        # =====================================================
        # STEP 6 — FINAL RESPONSE
        # =====================================================

        print("\n" + "=" * 70)
        print(
            "ORCHESTRATOR → FINAL RESPONSE"
        )
        print("=" * 70)

        final_response = (
            build_final_response(
                state
            )
        )

        update_state(
            state,
            current_agent="orchestrator",
            final_response=final_response,
        )

        state["completed"] = True

        print("\n" + "-" * 70)
        print(final_response)
        print("-" * 70)

        # -----------------------------------------------------
        # SAVE INTERACTION TO MEMORY
        # -----------------------------------------------------

        if memory is not None:

            memory.add_exchange(
                user_query,
                final_response,
            )

            print("\n" + "-" * 70)
            print(
                "Conversation exchange saved "
                "to short-term memory."
            )

            print(
                f"Memory size: "
                f"{memory.size()}"
            )

        return state

    except Exception as error:

        error_message = str(
            error
        )

        print(
            f"\nOrchestrator failed: "
            f"{error_message}"
        )

        add_error(
            state,
            "orchestrator",
            error_message,
        )

        final_response = (
            build_final_response(
                state
            )
        )

        update_state(
            state,
            current_agent="orchestrator",
            final_response=final_response,
        )

        state["completed"] = True

        if memory is not None:

            memory.add_exchange(
                user_query,
                final_response,
            )

        return state


# =============================================================
# WORKFLOW SUMMARY
# =============================================================

def print_workflow_summary(
    state: Dict[str, Any],
) -> None:
    """
    Print a compact workflow summary.
    """

    summary = summarize_state(
        state
    )

    react_trace = state.get(
        "react_trace",
        [],
    )

    print("\n" + "=" * 70)
    print(
        "WORKFLOW SUMMARY"
    )
    print("=" * 70)

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"ReAct trace records: "
        f"{len(react_trace)}"
    )

    if react_trace:

        latest_trace = react_trace[-1]

        print(
            "ReAct final action: "
            f"{latest_trace.get('action')}"
        )

        print(
            "ReAct needs more information: "
            f"{latest_trace.get('needs_more_information')}"
        )

    # ---------------------------------------------------------
    # DAY 4 SAFETY SUMMARY
    # ---------------------------------------------------------

    print(
        f"Safety check performed: "
        f"{state.get('safety_check_performed', False)}"
    )

    print(
        f"Human approval required: "
        f"{state.get('human_approval_required', False)}"
    )

    print(
        f"Execution status: "
        f"{state.get('execution_status', 'not_run')}"
    )


# =============================================================
# MEMORY INTEGRATION TEST
# =============================================================

def run_memory_integration_test() -> None:
    """
    Test:

        Memory
        +
        ReAct
        +
        Multi-Agent workflow
        +
        Day 4 Safety Layer
        +
        Historical recommendation recall
    """

    print("\n" + "=" * 70)
    print(
        "ORCHESTRATOR + MEMORY + REACT + SAFETY TEST"
    )
    print("=" * 70)

    memory = ConversationMemory(
        max_messages=10
    )

    # ---------------------------------------------------------
    # FIRST QUERY
    # ---------------------------------------------------------

    first_query = (
        "What's happening at SFO right now?"
    )

    first_state = orchestrate(
        first_query,
        memory=memory,
    )

    print_workflow_summary(
        first_state
    )

    # ---------------------------------------------------------
    # SECOND QUERY
    # ---------------------------------------------------------

    second_query = (
        "Does the current surge require approval?"
    )

    second_state = orchestrate(
        second_query,
        memory=memory,
    )

    print_workflow_summary(
        second_state
    )

    # ---------------------------------------------------------
    # THIRD QUERY — MEMORY RECALL
    # ---------------------------------------------------------

    third_query = (
        "What was the recommendation from "
        "our previous investigation?"
    )

    third_state = orchestrate(
        third_query,
        memory=memory,
    )

    print_workflow_summary(
        third_state
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "INTEGRATION VALIDATION"
    )
    print("=" * 70)

    print(
        f"\nFirst query: "
        f"{first_query}"
    )

    print(
        f"First airport: "
        f"{first_state.get('detected_airport')}"
    )

    print(
        f"First ReAct trace records: "
        f"{len(first_state.get('react_trace', []))}"
    )

    print(
        f"First safety status: "
        f"{first_state.get('execution_status')}"
    )

    print(
        f"\nSecond query: "
        f"{second_query}"
    )

    print(
        f"Second airport: "
        f"{second_state.get('detected_airport')}"
    )

    print(
        f"Second policy decision: "
        f"{second_state.get('policy_decision')}"
    )

    print(
        f"Second policy approval required: "
        f"{second_state.get('approval_required')}"
    )

    print(
        f"Second human approval required: "
        f"{second_state.get('human_approval_required')}"
    )

    print(
        f"Second ReAct trace records: "
        f"{len(second_state.get('react_trace', []))}"
    )

    print(
        f"Second safety status: "
        f"{second_state.get('execution_status')}"
    )

    print(
        f"\nThird query: "
        f"{third_query}"
    )

    print(
        f"Third airport: "
        f"{third_state.get('detected_airport')}"
    )

    print(
        f"Third execution status: "
        f"{third_state.get('execution_status')}"
    )

    print(
        f"Third recommendation: "
        f"{third_state.get('recommended_action')}"
    )

    print(
        f"\nTotal messages stored: "
        f"{memory.size()}"
    )

    print("\nConversation memory:")
    print("-" * 70)

    print(
        memory.get_context()
    )

    print("\n" + "=" * 70)
    print(
        "ORCHESTRATOR + MEMORY + REACT + SAFETY TEST COMPLETE"
    )
    print("=" * 70)


# =============================================================
# STRUCTURED SAFETY INTEGRATION TEST
# =============================================================

def run_structured_safety_integration_test() -> None:
    """
    Directly test the Orchestrator's Day 4 safety integration.

    This test injects a structured high-risk recommendation
    into a state and verifies that it is sent through the
    Safe Executor.

    interactive_approval=False is used so the automated test
    never waits for terminal input.
    """

    print("\n" + "=" * 70)
    print(
        "STRUCTURED SAFETY INTEGRATION TEST"
    )
    print("=" * 70)

    state = create_initial_state(
        "Increase SFO surge to 1.4x because of airport supply pressure."
    )

    state["detected_airport"] = "SFO"

    state["policy_results"] = [
        {
            "source": "sfo_pricing.md",
            "policy_id": "SFO-PRC-001",
            "section": "3",
            "relevant_rule": (
                "Maximum surge multiplier allowed "
                "at SFO is 1.5x."
            ),
        }
    ]

    state["severity_level"] = "high"

    state["execution_recommendation"] = {
        "action": "increase surge",
        "airport_code": "SFO",
        "new_multiplier": 1.4,
        "reason": (
            "Increase SFO surge to address "
            "temporary supply pressure."
        ),
    }

    state = run_safety_layer(
        state
    )

    execution_result = state.get(
        "execution_result",
        {},
    )

    print("\nSafety integration result:")
    print("-" * 70)

    print(
        format_for_response(
            execution_result
        )
    )

    status = execution_result.get(
        "status"
    )

    # 1.4x is high risk.
    # No human approval is supplied.
    # Therefore the action must remain pending.

    assert status == "awaiting_approval", (
        "Expected high-risk action to remain "
        "awaiting human approval."
    )

    assert (
        execution_result.get(
            "execution_allowed",
            False,
        )
        is False
    ), (
        "High-risk action must not execute "
        "without human approval."
    )

    assert (
        state.get(
            "execution_status"
        )
        == "awaiting_approval"
    ), (
        "Execution status should be "
        "'awaiting_approval' until a human "
        "actually approves or rejects the action."
    )

    print(
        "\nPASS - High-risk recommendation reached "
        "the Safe Executor."
    )

    print(
        "PASS - Execution is awaiting human approval."
    )

    print(
        "PASS - Pending approval is not incorrectly "
        "reported as rejected."
    )

    print("\n" + "=" * 70)
    print(
        "STRUCTURED SAFETY INTEGRATION TEST PASSED"
    )
    print("=" * 70)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    run_memory_integration_test()

    run_structured_safety_integration_test()
