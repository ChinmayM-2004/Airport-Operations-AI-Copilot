
"""
Uber Global Airport Operations AI Copilot
=========================================

Day 5 - Streamlit AI Copilot & End-to-End Integration

This application integrates:

    RAG
    Vector Search
    Operational Tools
    Function Calling
    Multi-Agent Workflow
    ReAct Loop
    Conversational Memory
    Guardrails
    Human-in-the-Loop Approval
    Safe Execution
    Audit Trail
    Distilled Training Data

Run with:

    streamlit run app.py
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from src.agents.orchestrator import (
    orchestrate,
    detect_airport_from_text,
)

from src.memory.conversation_memory import (
    ConversationMemory,
)

from src.safe_executor import (
    execute_safely,
)

from src.guardrails import (
    classify_action_risk,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airport Operations AI Copilot",
    page_icon="✈️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

AIRPORTS = [
    "SFO",
    "LAX",
    "JFK",
]


AIRPORT_NAMES = {
    "SFO": "San Francisco International Airport",
    "LAX": "Los Angeles International Airport",
    "JFK": "John F. Kennedy International Airport",
}


# ============================================================
# SESSION STATE
# ============================================================


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state.

    Streamlit reruns this file whenever the user interacts
    with the application, so persistent application state
    must live inside st.session_state.
    """

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(
            max_messages=20
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_airport" not in st.session_state:
        st.session_state.selected_airport = "SFO"

    if "last_state" not in st.session_state:
        st.session_state.last_state = None

    if "pending_recommendation" not in st.session_state:
        st.session_state.pending_recommendation = None

    if "pending_policy_results" not in st.session_state:
        st.session_state.pending_policy_results = []

    if "pending_user_request" not in st.session_state:
        st.session_state.pending_user_request = ""

    if "execution_result" not in st.session_state:
        st.session_state.execution_result = None

    if "workflow_running" not in st.session_state:
        st.session_state.workflow_running = False


initialize_session_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def get_value(
    dictionary: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve a dictionary value.
    """

    if not isinstance(
        dictionary,
        dict,
    ):
        return default

    return dictionary.get(
        key,
        default,
    )


def get_latest_react_trace(
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Return the latest ReAct trace record.
    """

    trace = state.get(
        "react_trace",
        [],
    )

    if not trace:
        return None

    if isinstance(
        trace,
        list,
    ):
        latest = trace[-1]

        if isinstance(
            latest,
            dict,
        ):
            return latest

    return None


def normalize_policy_result(
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize a retrieved policy result so that both the
    Day 5 UI and the Day 4 safety layer can access the
    policy text consistently.

    Different stages of the project use slightly different
    field names:

        RAG Retriever:
            text
            file_name

        Policy Agent:
            relevant_rule
            source

    We preserve the original fields and add compatible
    aliases where necessary.
    """

    if not isinstance(
        policy,
        dict,
    ):
        return {}

    normalized = dict(
        policy
    )

    # --------------------------------------------------------
    # Normalize document/source name
    # --------------------------------------------------------

    source = normalized.get(
        "source"
    )

    if not source:
        source = normalized.get(
            "file_name"
        )

    if source:
        normalized["source"] = source

        normalized.setdefault(
            "file_name",
            source,
        )

    # --------------------------------------------------------
    # Normalize policy text
    # --------------------------------------------------------

    relevant_rule = normalized.get(
        "relevant_rule"
    )

    text = normalized.get(
        "text"
    )

    if not relevant_rule and text:
        relevant_rule = text

    if relevant_rule:
        normalized["relevant_rule"] = (
            relevant_rule
        )

        if not text:
            normalized["text"] = (
                relevant_rule
            )

    # --------------------------------------------------------
    # Normalize section
    # --------------------------------------------------------

    if not normalized.get(
        "section"
    ):
        normalized["section"] = (
            normalized.get(
                "section_name"
            )
            or
            "Unknown section"
        )

    # --------------------------------------------------------
    # Normalize policy ID
    # --------------------------------------------------------

    if not normalized.get(
        "policy_id"
    ):
        normalized["policy_id"] = (
            normalized.get(
                "document_id"
            )
            or
            "Unknown policy"
        )

    return normalized


def normalize_policy_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normalize all policy results consistently.
    """

    if not isinstance(
        results,
        list,
    ):
        return []

    normalized_results = []

    for policy in results:

        normalized = normalize_policy_result(
            policy
        )

        if normalized:
            normalized_results.append(
                normalized
            )

    return normalized_results


def get_policy_results(
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Safely return retrieved policy results.

    The policy results are normalized before being returned
    so that the UI, policy validation, and safe executor use
    the same policy representation.
    """

    results = state.get(
        "policy_results",
        [],
    )

    return normalize_policy_results(
        results
    )


def format_policy_source(
    policy: Dict[str, Any],
) -> str:
    """
    Format a policy source for display.
    """

    normalized = normalize_policy_result(
        policy
    )

    source = normalized.get(
        "source",
        "Unknown document",
    )

    policy_id = normalized.get(
        "policy_id",
        "Unknown policy",
    )

    section = normalized.get(
        "section",
        "Unknown section",
    )

    return (
        f"{source} | "
        f"{policy_id} | "
        f"{section}"
    )


def get_structured_recommendation(
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Get the structured executable recommendation from
    the shared state.

    Natural-language recommendations are not converted
    into executable actions by guessing parameters.
    """

    recommendation = state.get(
        "execution_recommendation"
    )

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


def recommendation_requires_approval(
    recommendation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify recommendation risk using the existing
    Day 4 guardrail implementation.
    """

    action = recommendation.get(
        "action"
    )

    try:
        return classify_action_risk(
            action=action,
            parameters=recommendation,
        )

    except Exception as exc:
        return {
            "action": action,
            "risk_level": "high",
            "approval_required": True,
            "reason": str(exc),
        }


def clear_pending_approval() -> None:
    """
    Clear any pending approval state.
    """

    st.session_state.pending_recommendation = None

    st.session_state.pending_policy_results = []

    st.session_state.pending_user_request = ""

    st.session_state.execution_result = None


def execute_approved_action(
    approval_decision: str,
) -> None:
    """
    Execute or reject the pending structured recommendation.

    This function is called only after the user explicitly
    chooses APPROVE or REJECT in the Streamlit interface.
    """

    recommendation = (
        st.session_state.pending_recommendation
    )

    if not recommendation:
        st.warning(
            "There is no pending action requiring approval."
        )
        return

    # --------------------------------------------------------
    # Normalize policy results before sending them to the
    # safety / policy validation layer.
    # --------------------------------------------------------

    policy_results = normalize_policy_results(
        st.session_state.pending_policy_results
    )

    user_request = (
        st.session_state.pending_user_request
    )

    state = (
        st.session_state.last_state
    )

    approved = (
        approval_decision == "approved"
    )

    human_approval = {
        "approval_required": True,
        "approval_decision": approval_decision,
        "approved": approved,
        "action": recommendation.get(
            "action"
        ),
        "reason": (
            "Approved by the Streamlit user."
            if approved
            else
            "Rejected by the Streamlit user."
        ),
    }

    with st.spinner(
        "Running safety checks and execution..."
    ):

        result = execute_safely(
            recommendation=recommendation,
            user_request=user_request,
            policy_results=policy_results,
            human_approval=human_approval,
            interactive_approval=False,
            state=state,
        )

    st.session_state.execution_result = result

    # Keep the execution result visible while clearing
    # only the pending approval request.

    st.session_state.pending_recommendation = None

    st.session_state.pending_policy_results = []

    st.session_state.pending_user_request = ""


# ============================================================
# HEADER
# ============================================================

st.title(
    "✈️ Airport Operations AI Copilot"
)

st.caption(
    "Uber Global Airport Operations & Supply Disruption Resolver"
)

st.markdown(
    """
This AI Copilot investigates airport operational issues,
retrieves applicable policies, reasons about contributing
factors, generates recommendations, applies safety
guardrails, and requests human approval before sensitive
operational actions.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Airport Selection"
    )

    selected_airport = st.selectbox(
        "Select airport",
        AIRPORTS,
        index=AIRPORTS.index(
            st.session_state.selected_airport
        ),
    )

    st.session_state.selected_airport = (
        selected_airport
    )

    st.divider()

    st.subheader(
        "System Architecture"
    )

    st.markdown(
        """
**User Query**

↓

**Orchestrator**

↓

**Operations Investigator**

↓

**ReAct Controller**

↓

**Policy & Compliance**

↓

**Resolution Agent**

↓

**Safety / Guardrails**

↓

**Human Approval**

↓

**Safe Executor**

↓

**Audit + Distillation**
"""
    )

    st.divider()

    st.subheader(
        "Supported Airports"
    )

    for airport in AIRPORTS:

        st.write(
            f"• {airport} — "
            f"{AIRPORT_NAMES[airport]}"
        )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.memory.clear()

        st.session_state.messages = []

        st.session_state.last_state = None

        clear_pending_approval()

        st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.messages:

    st.subheader(
        "Conversation"
    )

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(
            role
        ):

            st.markdown(
                content
            )


# ============================================================
# CHAT INPUT
# ============================================================

default_prompt = (
    f"Investigate {selected_airport}. "
    "Completion rate appears to be low. "
    "Can we increase surge?"
)

user_prompt = st.chat_input(
    "Ask about airport operations..."
)


# ============================================================
# PROCESS USER QUERY
# ============================================================

if user_prompt:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    # --------------------------------------------------------
    # Add selected airport when the user does not mention
    # an airport explicitly.
    #
    # This keeps the selected airport meaningful while
    # allowing the existing ConversationMemory to handle
    # contextual follow-up questions.
    # --------------------------------------------------------

    airport_in_query = (
        detect_airport_from_text(
            user_prompt
        )
    )

    if airport_in_query is None:

        workflow_query = (
            f"{user_prompt} "
            f"(Selected airport: "
            f"{selected_airport}.)"
        )

    else:

        workflow_query = user_prompt

    # --------------------------------------------------------
    # Start a fresh execution cycle.
    #
    # Any execution result or pending approval from the
    # previous query belongs to that previous workflow and
    # must not appear as the result of the new query.
    # --------------------------------------------------------

    clear_pending_approval()

    # --------------------------------------------------------
    # Run orchestrator
    # --------------------------------------------------------

    st.session_state.workflow_running = True

    with st.spinner(
        "Investigating airport operations..."
    ):

        state = orchestrate(
            workflow_query,
            memory=st.session_state.memory,
        )

    st.session_state.workflow_running = False

    st.session_state.last_state = state

    # --------------------------------------------------------
    # Normalize policy results immediately after the
    # orchestrator completes.
    #
    # This ensures every later Day 5 component sees the same
    # policy representation.
    # --------------------------------------------------------

    normalized_policy_results = (
        normalize_policy_results(
            state.get(
                "policy_results",
                [],
            )
        )
    )

    state["policy_results"] = (
        normalized_policy_results
    )

    st.session_state.last_state = state

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    final_response = state.get(
        "final_response",
        "The workflow completed without a final response.",
    )

    # --------------------------------------------------------
    # Display assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_response,
        }
    )

    # --------------------------------------------------------
    # Detect structured executable recommendation
    # --------------------------------------------------------

    recommendation = (
        get_structured_recommendation(
            state
        )
    )

    if recommendation is not None:

        risk = (
            recommendation_requires_approval(
                recommendation
            )
        )

        # Only create a pending approval request when
        # the recommendation actually requires approval.

        if risk.get(
            "approval_required",
            False,
        ):

            st.session_state.pending_recommendation = (
                recommendation
            )

            st.session_state.pending_policy_results = (
                normalized_policy_results
            )

            st.session_state.pending_user_request = (
                workflow_query
            )

    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

state = st.session_state.last_state


if state is not None:

    airport = state.get(
        "detected_airport",
        selected_airport,
    )

    st.divider()

    st.header(
        f"Airport Operations Dashboard — {airport}"
    )

    # ========================================================
    # OPERATIONAL METRICS
    # ========================================================

    st.subheader(
        "Operational Metrics"
    )

    operational_data = state.get(
        "operational_data",
        {},
    )

    if operational_data:

        completion_rate = safe_float(
            operational_data.get(
                "completion_rate"
            )
        )

        average_eta = safe_float(
            operational_data.get(
                "average_eta_minutes"
            )
        )

        cancellation_rate = safe_float(
            operational_data.get(
                "driver_cancellation_rate"
            )
        )

        queue_size = operational_data.get(
            "queue_size",
            0,
        )

        surge_multiplier = safe_float(
            operational_data.get(
                "surge_multiplier"
            )
        )

        active_drivers = operational_data.get(
            "active_drivers",
            0,
        )

        request_volume = operational_data.get(
            "request_volume",
            0,
        )

        timestamp = operational_data.get(
            "timestamp",
            "Unknown",
        )

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "Completion Rate",
                f"{completion_rate:.1%}",
            )

        with col2:

            st.metric(
                "Average ETA",
                f"{average_eta:.1f} min",
            )

        with col3:

            st.metric(
                "Driver Cancellation",
                f"{cancellation_rate:.1%}",
            )

        col4, col5, col6 = st.columns(
            3
        )

        with col4:

            st.metric(
                "Queue Size",
                str(queue_size),
            )

        with col5:

            st.metric(
                "Surge",
                f"{surge_multiplier:.2f}x",
            )

        with col6:

            st.metric(
                "Active Drivers",
                str(active_drivers),
            )

        st.caption(
            f"Request volume: {request_volume} | "
            f"Telemetry timestamp: {timestamp}"
        )

    else:

        st.info(
            "No operational metrics are available."
        )

    # ========================================================
    # AGENT ACTIVITY
    # ========================================================

    st.subheader(
        "Agent Activity"
    )

    agent_columns = st.columns(
        4
    )

    agents = [
        (
            "Orchestrator",
            "Coordinates workflow",
        ),
        (
            "Operations Investigator",
            "Analyzes telemetry",
        ),
        (
            "Policy Agent",
            "Retrieves policy",
        ),
        (
            "Resolution Agent",
            "Generates recommendation",
        ),
    ]

    for column, (
        agent_name,
        description,
    ) in zip(
        agent_columns,
        agents,
    ):

        with column:

            st.success(
                f"✓ {agent_name}"
            )

            st.caption(
                description
            )

    # ========================================================
    # REACT TRACE
    # ========================================================

    react_trace = state.get(
        "react_trace",
        [],
    )

    if react_trace:

        with st.expander(
            "🔄 ReAct Execution Trace",
            expanded=False,
        ):

            for index, trace in enumerate(
                react_trace,
                start=1,
            ):

                st.markdown(
                    f"**Iteration {index}**"
                )

                st.write(
                    "**Observation:**",
                    trace.get(
                        "observation",
                        "",
                    ),
                )

                st.write(
                    "**Reasoning:**",
                    trace.get(
                        "reasoning",
                        "",
                    ),
                )

                st.write(
                    "**Action:**",
                    trace.get(
                        "action",
                        "",
                    ),
                )

                st.write(
                    "**More information required:**",
                    trace.get(
                        "needs_more_information",
                        False,
                    ),
                )

                if index < len(
                    react_trace
                ):

                    st.divider()

    # ========================================================
    # INVESTIGATION
    # ========================================================

    st.subheader(
        "Investigation"
    )

    severity = state.get(
        "severity_level",
        "unknown",
    )

    findings = state.get(
        "investigator_findings",
        "",
    )

    investigation_col1, investigation_col2 = (
        st.columns(2)
    )

    with investigation_col1:

        st.metric(
            "Severity",
            str(
                severity
            ).upper(),
        )

    with investigation_col2:

        react_latest = (
            get_latest_react_trace(
                state
            )
        )

        if react_latest:

            react_action = react_latest.get(
                "action",
                "Proceed",
            )

            st.metric(
                "ReAct Decision",
                str(
                    react_action
                ),
            )

    if findings:

        st.write(
            "**Contributing Factors**"
        )

        if isinstance(
            findings,
            list,
        ):

            for finding in findings:

                st.write(
                    f"• {finding}"
                )

        else:

            st.write(
                findings
            )

    # ========================================================
    # RAG TRACE
    # ========================================================

    st.subheader(
        "RAG Trace"
    )

    policy_results = get_policy_results(
        state
    )

    if policy_results:

        for index, policy in enumerate(
            policy_results,
            start=1,
        ):

            normalized_policy = (
                normalize_policy_result(
                    policy
                )
            )

            with st.expander(
                f"Retrieved Policy {index}: "
                f"{format_policy_source(normalized_policy)}",
                expanded=(
                    index == 1
                ),
            ):

                st.write(
                    "**Retrieved Document:**"
                )

                st.code(
                    normalized_policy.get(
                        "source",
                        "Unknown document",
                    )
                )

                st.write(
                    "**Policy ID:**"
                )

                st.code(
                    normalized_policy.get(
                        "policy_id",
                        "Unknown policy",
                    )
                )

                st.write(
                    "**Section:**"
                )

                st.write(
                    normalized_policy.get(
                        "section",
                        "Unknown section",
                    )
                )

                st.write(
                    "**Relevant Policy:**"
                )

                policy_text = (
                    normalized_policy.get(
                        "relevant_rule"
                    )
                    or
                    normalized_policy.get(
                        "text"
                    )
                )

                if policy_text:

                    st.info(
                        policy_text
                    )

                else:

                    st.info(
                        "No policy rule available."
                    )

                similarity = (
                    normalized_policy.get(
                        "similarity"
                    )
                )

                if similarity is None:

                    similarity = (
                        normalized_policy.get(
                            "score"
                        )
                    )

                if similarity is not None:

                    st.caption(
                        f"Semantic similarity: "
                        f"{safe_float(similarity):.4f}"
                    )

    else:

        st.info(
            "No policy retrieval results are available."
        )

    # ========================================================
    # POLICY DECISION
    # ========================================================

    st.subheader(
        "Policy & Compliance"
    )

    policy_decision = state.get(
        "policy_decision",
        "Insufficient Information",
    )

    approval_required = state.get(
        "approval_required",
        "unknown",
    )

    policy_col1, policy_col2 = st.columns(
        2
    )

    with policy_col1:

        st.metric(
            "Policy Decision",
            str(
                policy_decision
            ),
        )

    with policy_col2:

        st.metric(
            "Approval Required",
            str(
                approval_required
            ),
        )

    restrictions = state.get(
        "policy_restrictions",
        [],
    )

    if restrictions:

        st.write(
            "**Policy Restrictions**"
        )

        for restriction in restrictions:

            st.warning(
                restriction
            )

    # ========================================================
    # RESOLUTION
    # ========================================================

    st.subheader(
        "Resolution Recommendation"
    )

    recommended_action = state.get(
        "recommended_action",
        "",
    )

    recommendation_reasoning = state.get(
        "recommendation_reasoning",
        "",
    )

    if recommended_action:

        st.info(
            recommended_action
        )

    if recommendation_reasoning:

        st.write(
            "**Reason:**"
        )

        st.write(
            recommendation_reasoning
        )

    possible_actions = state.get(
        "possible_actions",
        [],
    )

    if possible_actions:

        with st.expander(
            "Possible Actions",
            expanded=False,
        ):

            if isinstance(
                possible_actions,
                list,
            ):

                for action in possible_actions:

                    st.write(
                        f"• {action}"
                    )

            else:

                st.write(
                    possible_actions
                )

    # ========================================================
    # PENDING HUMAN APPROVAL
    # ========================================================

    pending_recommendation = (
        st.session_state.pending_recommendation
    )

    if pending_recommendation:

        st.divider()

        st.error(
            "⚠️ HUMAN APPROVAL REQUIRED"
        )

        st.subheader(
            "High-Risk Action"
        )

        risk = (
            recommendation_requires_approval(
                pending_recommendation
            )
        )

        action = pending_recommendation.get(
            "action",
            "Unknown action",
        )

        reason = pending_recommendation.get(
            "reason",
            "No reason provided.",
        )

        st.write(
            f"**Action:** {action}"
        )

        if (
            action == "increase surge"
            and pending_recommendation.get(
                "airport_code"
            )
        ):

            pending_airport = (
                pending_recommendation.get(
                    "airport_code"
                )
            )

            new_multiplier = safe_float(
                pending_recommendation.get(
                    "new_multiplier"
                )
            )

            current_multiplier = safe_float(
                operational_data.get(
                    "surge_multiplier",
                    0.0,
                )
            )

            st.write(
                f"**Surge:** "
                f"{current_multiplier:.2f}x → "
                f"{new_multiplier:.2f}x"
            )

            st.write(
                f"**Airport:** {pending_airport}"
            )

        st.write(
            f"**Reason:** {reason}"
        )

        st.write(
            f"**Risk Level:** "
            f"{str(risk.get('risk_level', 'high')).upper()}"
        )

        st.write(
            f"**Guardrail:** "
            f"{risk.get('reason', 'Human approval required.')}"
        )

        st.warning(
            "This action will not execute until you explicitly "
            "approve it."
        )

        approval_col1, approval_col2 = (
            st.columns(2)
        )

        with approval_col1:

            if st.button(
                "✅ APPROVE",
                type="primary",
                use_container_width=True,
            ):

                execute_approved_action(
                    "approved"
                )

                st.rerun()

        with approval_col2:

            if st.button(
                "❌ REJECT",
                use_container_width=True,
            ):

                execute_approved_action(
                    "rejected"
                )

                st.rerun()

    # ========================================================
    # EXECUTION RESULT
    # ========================================================

    execution_result = (
        st.session_state.execution_result
    )

    if execution_result:

        st.divider()

        st.subheader(
            "Execution Result"
        )

        status = execution_result.get(
            "status",
            "unknown",
        )

        if status == "executed":

            st.success(
                "✅ ACTION EXECUTED"
            )

            tool_result = execution_result.get(
                "tool_result",
                {},
            )

            tool_name = execution_result.get(
                "tool_name",
                "unknown",
            )

            st.write(
                f"**Tool:** {tool_name}"
            )

            st.write(
                "**Execution:** Successful"
            )

            if isinstance(
                tool_result,
                dict,
            ):

                message = tool_result.get(
                    "message"
                )

                if message:

                    st.info(
                        message
                    )

                new_multiplier = tool_result.get(
                    "new_multiplier"
                )

                airport_code = tool_result.get(
                    "airport_code"
                )

                if (
                    new_multiplier is not None
                    and airport_code
                ):

                    st.write(
                        f"**{airport_code} surge updated:** "
                        f"{safe_float(new_multiplier):.2f}x"
                    )

            approval = execution_result.get(
                "human_approval",
                {},
            )

            if isinstance(
                approval,
                dict,
            ):

                st.write(
                    f"**Approval:** "
                    f"{approval.get('approval_decision', 'approved')}"
                )

        elif status == "rejected":

            st.warning(
                "❌ ACTION REJECTED — "
                "No operational action was executed."
            )

            reason = execution_result.get(
                "reason"
            )

            if reason:

                st.write(
                    reason
                )

        elif status == "blocked":

            st.error(
                "🛑 ACTION BLOCKED"
            )

            reason = execution_result.get(
                "reason",
                "The safety layer blocked this action.",
            )

            st.write(
                reason
            )

        else:

            st.info(
                f"Execution status: {status}"
            )

        # ----------------------------------------------------
        # AUDIT INFORMATION
        # ----------------------------------------------------

        audit_record = execution_result.get(
            "audit_record"
        )

        if audit_record:

            with st.expander(
                "📋 Audit Trail Record",
                expanded=False,
            ):

                st.json(
                    audit_record
                )

    # ========================================================
    # WORKFLOW SUMMARY
    # ========================================================

    st.subheader(
        "End-to-End Workflow"
    )

    workflow_steps = [
        (
            "1",
            "Investigate",
            "Operational telemetry analyzed",
        ),
        (
            "2",
            "Identify Factors",
            "Supply and demand signals evaluated",
        ),
        (
            "3",
            "Retrieve Policy",
            "Relevant airport policies retrieved with RAG",
        ),
        (
            "4",
            "Generate Recommendation",
            "Resolution Agent produced recommendation",
        ),
        (
            "5",
            "Apply Guardrail",
            "Risk and policy checks applied",
        ),
        (
            "6",
            "Human Decision",
            "Approval requested for sensitive actions",
        ),
        (
            "7",
            "Execute",
            "Safe Executor controls operational execution",
        ),
        (
            "8",
            "Explain",
            "Audit trail and final response generated",
        ),
    ]

    for number, title, description in workflow_steps:

        st.write(
            f"**Step {number} — {title}:** "
            f"{description}"
        )

    # ========================================================
    # TECHNICAL STATE
    # ========================================================

    with st.expander(
        "🔍 Technical State",
        expanded=False,
    ):

        st.json(
            state
        )


# ============================================================
# EMPTY STATE / WELCOME SCREEN
# ============================================================

else:

    st.divider()

    st.header(
        "Start an Airport Investigation"
    )

    st.markdown(
        f"""
### Try the end-to-end demo

Enter a natural-language request such as:

> **Investigate {selected_airport}. Completion rate appears to be low. Can we increase surge?**

The Copilot will:

1. Investigate operational telemetry
2. Identify contributing factors
3. Retrieve the applicable airport policy
4. Generate a recommendation
5. Apply risk and policy guardrails
6. Request human approval when required
7. Execute only after approval
8. Record the audit trail
9. Distill the successful interaction
"""
    )

    st.info(
        "Select an airport from the sidebar and enter your "
        "question in the chat box below."
    )
