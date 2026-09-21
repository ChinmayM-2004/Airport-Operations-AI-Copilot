"""
Resolution Agent

Uses operational findings and policy/compliance results to determine
possible actions and recommend an appropriate resolution.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from src.agents.state import (
    add_error,
    add_tool_call,
    update_state,
)

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"


def build_resolution_prompt(state: Dict[str, Any]) -> str:
    """Build a grounded prompt for the Resolution Agent."""

    airport = state.get("detected_airport", "unknown")
    operational_data = state.get("operational_data", {})
    investigator_findings = state.get("investigator_findings", "")
    severity = state.get("severity_level", "unknown")
    contributing_factors = state.get("contributing_factors", [])

    policy_decision = state.get(
        "policy_decision",
        "Insufficient Information",
    )

    approval_required = state.get(
        "approval_required",
        "unknown",
    )

    policy_results = state.get("policy_results", [])

    policy_evidence = []

    for result in policy_results:
        policy_evidence.append(
            {
                "source": result.get("source", "unknown"),
                "policy_id": result.get("policy_id", "unknown"),
                "section": result.get("section", "unknown"),
                "relevant_rule": result.get(
                    "relevant_rule",
                    "",
                ),
            }
        )

    return f"""
You are the Resolution Agent in an Airport Operations AI Copilot.

Your responsibility is to recommend operational actions based ONLY
on the investigation findings and retrieved policy information.

Do not invent operational facts.
Do not invent policy rules.
Do not recommend an action that conflicts with a retrieved policy.
If approval is required, clearly state that the action requires approval.
If policy information is insufficient, do not pretend that an action
is compliant.

AIRPORT:
{airport}

OPERATIONAL DATA:
{json.dumps(operational_data, indent=2)}

INVESTIGATOR FINDINGS:
{investigator_findings}

SEVERITY:
{severity}

CONTRIBUTING FACTORS:
{json.dumps(contributing_factors, indent=2)}

POLICY DECISION:
{policy_decision}

APPROVAL REQUIRED:
{approval_required}

RETRIEVED POLICY EVIDENCE:
{json.dumps(policy_evidence, indent=2)}

USER REQUEST:
{state.get("user_query", "")}

Return ONLY valid JSON using exactly this structure:

{{
    "possible_actions": [
        "action 1",
        "action 2",
        "action 3"
    ],
    "recommended_action": "recommended action",
    "recommendation_reasoning": "short explanation",
    "approval_required": true,
    "execution_ready": false,
    "execution_recommendation": null
}}

For an explicitly requested surge change that is supported by policy,
execution_recommendation may be:

{{
    "action": "increase surge",
    "airport_code": "SFO",
    "new_multiplier": 1.5,
    "reason": "short grounded reason"
}}

Rules:

1. If policy decision is "Not Allowed", do not recommend the
   prohibited action.

2. If policy decision is "Insufficient Information", recommend
   investigation or information gathering rather than an
   unsupported operational change.

3. If approval_required is true, execution_ready must be false.

4. If approval_required is false, execution_ready may be true only
   when the action is clearly supported by the retrieved policy.

5. Prefer operationally conservative actions when severity is low
   or when the evidence does not justify a major intervention.

6. Keep the recommended action specific and actionable.

7. Do not invent numerical thresholds that are not present in
   the supplied evidence.

8. If the user explicitly requests a specific surge multiplier,
   preserve that requested multiplier in execution_recommendation
   when the requested action is policy-compliant.

9. Never create an executable recommendation for an action that
   policy does not allow.

10. The safety layer, not this agent, is responsible for final
    human approval and execution authorization.
"""


def analyze_resolution_with_llm(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Ask Gemini to generate resolution options."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = genai.Client(api_key=api_key)

    prompt = build_resolution_prompt(state)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    text = response.text.strip()

    return json.loads(text)


def extract_requested_surge_multiplier(
    user_query: str,
) -> Optional[float]:
    """
    Extract an explicitly requested surge multiplier from the
    user's natural-language request.

    Examples:

        "increase surge to 1.5x" -> 1.5
        "Can we increase surge to 1.4x?" -> 1.4
        "raise surge multiplier to 1.6" -> 1.6

    Returns None when the user did not explicitly request a
    surge multiplier.
    """

    if not user_query:
        return None

    query = user_query.lower()

    surge_requested = any(
        phrase in query
        for phrase in [
            "increase surge",
            "raise surge",
            "increase the surge",
            "raise the surge",
            "increase surge multiplier",
            "raise surge multiplier",
            "surge to",
            "surge multiplier to",
        ]
    )

    if not surge_requested:
        return None

    patterns = [
        r"surge(?:\s+multiplier)?\s*(?:to|at)\s*(\d+(?:\.\d+)?)\s*x?",
        r"(?:increase|raise)\s+(?:the\s+)?surge(?:\s+multiplier)?\s*(?:to|by)?\s*(\d+(?:\.\d+)?)\s*x?",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)

        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None

    return None


def build_structured_surge_recommendation(
    state: Dict[str, Any],
    requested_multiplier: float,
) -> Dict[str, Any]:
    """
    Build a structured executable recommendation for an explicitly
    requested surge change.

    This function does NOT approve or execute the action.
    The Day 4 safety layer performs risk classification,
    policy validation, human approval, and execution authorization.
    """

    airport = state.get(
        "detected_airport",
        "unknown",
    )

    current_multiplier = state.get(
        "operational_data",
        {},
    ).get(
        "surge_multiplier"
    )

    policy_decision = state.get(
        "policy_decision",
        "Insufficient Information",
    )

    policy_approval_required = state.get(
        "approval_required",
        "unknown",
    )

    policy_results = state.get(
        "policy_results",
        []
    )

    requested_text = (
        f"Increase {airport} surge multiplier "
        f"from {current_multiplier:.2f}x to "
        f"{requested_multiplier:.2f}x."
        if isinstance(current_multiplier, (int, float))
        else
        f"Increase {airport} surge multiplier to "
        f"{requested_multiplier:.2f}x."
    )

    # Never create an executable recommendation when policy
    # explicitly rejects the requested change.
    if policy_decision == "Not Allowed":
        return {
            "possible_actions": [
                "Maintain the current operating configuration.",
                "Escalate the request for policy or management review.",
                "Continue monitoring airport operating conditions.",
            ],
            "recommended_action": (
                "Do not execute the requested surge change because "
                "the retrieved policy does not allow it."
            ),
            "recommendation_reasoning": (
                "The requested surge adjustment conflicts with "
                "the retrieved airport policy."
            ),
            "approval_required": policy_approval_required,
            "execution_ready": False,
            "execution_recommendation": None,
        }

    # Never pretend a change is safe when policy evidence is missing.
    if policy_decision == "Insufficient Information":
        return {
            "possible_actions": [
                "Gather additional policy information.",
                "Continue monitoring airport conditions.",
                "Escalate to an Airport Operations Manager for clarification.",
            ],
            "recommended_action": (
                "Do not execute the requested surge change until "
                "policy compliance can be established."
            ),
            "recommendation_reasoning": (
                "The retrieved policy evidence is insufficient to "
                "authorize the requested surge adjustment."
            ),
            "approval_required": policy_approval_required,
            "execution_ready": False,
            "execution_recommendation": None,
        }

    # Policy allows the requested action.
    #
    # The safety layer will independently classify this action.
    # For example, 1.5x is policy-compliant at SFO but is HIGH risk
    # according to our Day 4 project guardrail because it is >= 1.3x.
    return {
        "possible_actions": [
            requested_text,
            "Continue monitoring airport operating conditions.",
            "Review demand and driver supply after the surge adjustment.",
        ],
        "recommended_action": requested_text,
        "recommendation_reasoning": (
            "The user explicitly requested this surge adjustment and "
            "the retrieved policy does not prohibit the requested "
            "multiplier. Final execution remains subject to the "
            "Day 4 safety guardrails and human approval when required."
        ),
        "approval_required": policy_approval_required,
        "execution_ready": False,
        "execution_recommendation": {
            "action": "increase surge",
            "airport_code": airport,
            "new_multiplier": requested_multiplier,
            "reason": (
                "Explicit user request for a surge adjustment; "
                "policy compliance and safety approval must be "
                "checked before execution."
            ),
        },
    }


def deterministic_resolution(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministic fallback used when Gemini is unavailable.

    Explicit operational requests are handled first so that a
    temporary Gemini outage does not remove the requested action
    from the safety and human-approval workflow.
    """

    severity = state.get(
        "severity_level",
        "unknown",
    )

    policy_decision = state.get(
        "policy_decision",
        "Insufficient Information",
    )

    approval_required = state.get(
        "approval_required",
        "unknown",
    )

    airport = state.get(
        "detected_airport",
        "unknown",
    )

    user_query = state.get(
        "user_query",
        "",
    )

    # ------------------------------------------------------------
    # Explicit surge request
    # ------------------------------------------------------------

    requested_multiplier = extract_requested_surge_multiplier(
        user_query
    )

    if requested_multiplier is not None:
        return build_structured_surge_recommendation(
            state,
            requested_multiplier,
        )

    # ------------------------------------------------------------
    # Policy does not allow the requested change.
    # ------------------------------------------------------------

    if policy_decision == "Not Allowed":

        possible_actions = [
            "Maintain the current operating configuration.",
            "Escalate the request for policy or management review.",
            "Continue monitoring airport operating conditions.",
        ]

        return {
            "possible_actions": possible_actions,
            "recommended_action": (
                "Maintain the current configuration and escalate "
                "for policy or management review."
            ),
            "recommendation_reasoning": (
                "The requested operational change is not allowed "
                "under the retrieved policy evidence."
            ),
            "approval_required": approval_required,
            "execution_ready": False,
            "execution_recommendation": None,
        }

    # ------------------------------------------------------------
    # Policy information is insufficient.
    # ------------------------------------------------------------

    if policy_decision == "Insufficient Information":

        possible_actions = [
            "Gather additional policy information.",
            "Continue monitoring the airport conditions.",
            "Escalate to an Airport Operations Manager for clarification.",
        ]

        return {
            "possible_actions": possible_actions,
            "recommended_action": (
                "Gather additional policy information before "
                "making an operational change."
            ),
            "recommendation_reasoning": (
                "The available policy evidence is insufficient "
                "to safely authorize a specific operational change."
            ),
            "approval_required": approval_required,
            "execution_ready": False,
            "execution_recommendation": None,
        }

    # ------------------------------------------------------------
    # Policy allows the change but approval is required.
    # ------------------------------------------------------------

    if approval_required is True:

        possible_actions = [
            "Prepare the proposed operational change for approval.",
            "Continue monitoring current airport conditions.",
            "Escalate the proposed change to the Airport Operations Manager.",
        ]

        return {
            "possible_actions": possible_actions,
            "recommended_action": (
                "Prepare the operational change and obtain the "
                "required Airport Operations Manager approval before execution."
            ),
            "recommendation_reasoning": (
                "The retrieved policy permits the change but requires "
                "approval before implementation."
            ),
            "approval_required": True,
            "execution_ready": False,
            "execution_recommendation": None,
        }

    # ------------------------------------------------------------
    # Low severity.
    # ------------------------------------------------------------

    if severity == "low":

        possible_actions = [
            "Continue monitoring airport operating conditions.",
            "Monitor active driver supply and passenger demand.",
            "Review queue and cancellation trends before taking intervention.",
        ]

        return {
            "possible_actions": possible_actions,
            "recommended_action": (
                f"Continue monitoring {airport} operating conditions "
                "without making a major operational change."
            ),
            "recommendation_reasoning": (
                "Current conditions are classified as low severity. "
                "The available evidence does not justify a major "
                "operational intervention."
            ),
            "approval_required": False,
            "execution_ready": True,
            "execution_recommendation": None,
        }

    # ------------------------------------------------------------
    # Medium severity.
    # ------------------------------------------------------------

    if severity == "medium":

        possible_actions = [
            "Increase operational monitoring.",
            "Review driver supply relative to request demand.",
            "Evaluate a policy-compliant driver incentive.",
            "Escalate if operating conditions continue to deteriorate.",
        ]

        return {
            "possible_actions": possible_actions,
            "recommended_action": (
                "Increase operational monitoring and evaluate a "
                "policy-compliant driver incentive if conditions persist."
            ),
            "recommendation_reasoning": (
                "The airport is experiencing medium-severity operational "
                "pressure. Monitoring and a controlled supply-side response "
                "are appropriate before considering larger interventions."
            ),
            "approval_required": False,
            "execution_ready": True,
            "execution_recommendation": None,
        }

    # ------------------------------------------------------------
    # High or critical severity.
    # ------------------------------------------------------------

    possible_actions = [
        "Escalate the operational issue to the Airport Operations Manager.",
        "Evaluate additional driver supply using approved incentives.",
        "Review whether a policy-compliant surge adjustment is permitted.",
        "Continue monitoring airport conditions.",
    ]

    return {
        "possible_actions": possible_actions,
        "recommended_action": (
            "Escalate the operational issue and evaluate approved "
            "driver-supply interventions."
        ),
        "recommendation_reasoning": (
            f"The airport is experiencing {severity}-severity conditions. "
            "Additional operational intervention may be required, but "
            "actions must remain within the retrieved policy constraints."
        ),
        "approval_required": False,
        "execution_ready": False,
        "execution_recommendation": None,
    }


def resolution_agent(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Main Resolution Agent."""

    print("\n" + "=" * 70)
    print("RESOLUTION AGENT")
    print("=" * 70)

    airport = state.get(
        "detected_airport",
        "unknown",
    )

    print(f"Airport: {airport}")

    try:
        policy_decision = state.get(
            "policy_decision",
            "Insufficient Information",
        )

        print(f"Policy decision: {policy_decision}")

        if not state.get("operational_data"):
            raise ValueError(
                "Operational data is missing."
            )

        print("\nGenerating resolution options...")

        try:
            result = analyze_resolution_with_llm(state)

            print(
                "Gemini resolution generated successfully."
            )

        except Exception as llm_error:

            print(
                f"Gemini resolution unavailable: {llm_error}"
            )

            add_error(
                state,
                "resolution_agent",
                f"Gemini resolution failed: {llm_error}",
            )

            print(
                "Using deterministic resolution fallback..."
            )

            result = deterministic_resolution(state)

        possible_actions = result.get(
            "possible_actions",
            [],
        )

        recommended_action = result.get(
            "recommended_action",
            "",
        )

        recommendation_reasoning = result.get(
            "recommendation_reasoning",
            "",
        )

        approval_required = result.get(
            "approval_required",
            state.get(
                "approval_required",
                "unknown",
            ),
        )

        execution_recommendation = result.get(
            "execution_recommendation"
        )

        update_state(
            state,
            possible_actions=possible_actions,
            recommended_action=recommended_action,
            recommendation_reasoning=recommendation_reasoning,
            approval_required=approval_required,
            execution_recommendation=execution_recommendation,
            current_agent="resolution_agent",
        )

        # add_tool_call() accepts exactly:
        # state, tool_name, arguments, result
        add_tool_call(
            state,
            "resolution_analysis",
            {
                "airport_code": airport,
                "policy_decision": policy_decision,
            },
            {
                "possible_actions_count": len(
                    possible_actions
                ),
                "recommended_action": recommended_action,
                "approval_required": approval_required,
                "execution_recommendation": (
                    execution_recommendation
                ),
            },
        )

        print("\nPossible actions:")
        print("-" * 70)

        for action in possible_actions:
            print(f"- {action}")

        print("\nRecommended action:")
        print("-" * 70)
        print(recommended_action)

        print("\nRecommendation reasoning:")
        print("-" * 70)
        print(recommendation_reasoning)

        print("\nApproval required:")
        print("-" * 70)
        print(approval_required)

        if execution_recommendation:
            print("\nStructured execution recommendation:")
            print("-" * 70)
            print(
                json.dumps(
                    execution_recommendation,
                    indent=2,
                )
            )

        return state

    except Exception as error:

        error_message = str(error)

        print(
            f"Resolution Agent failed: {error_message}"
        )

        add_error(
            state,
            "resolution_agent",
            error_message,
        )

        update_state(
            state,
            current_agent="resolution_agent",
        )

        return state


def run_test() -> None:
    """
    Standalone Resolution Agent test.

    Flow:

        Investigator
            ↓
        Policy Agent
            ↓
        Resolution Agent
    """

    from src.agents.state import create_initial_state
    from src.agents.investigator import investigate
    from src.agents.policy_agent import policy_agent

    print("\n" + "=" * 70)
    print("RESOLUTION AGENT TEST")
    print("=" * 70)

    query = "What's happening at SFO right now?"

    state = create_initial_state(query)

    state = investigate(state)

    state = policy_agent(state)

    state = resolution_agent(state)

    print("\n" + "=" * 70)
    print("RESOLUTION AGENT TEST COMPLETE")
    print("=" * 70)

    print("\nFinal state summary:")

    print(
        f"Airport: {state.get('detected_airport')}"
    )

    print(
        f"Severity: {state.get('severity_level')}"
    )

    print(
        f"Policy decision: {state.get('policy_decision')}"
    )

    print(
        f"Recommended action: "
        f"{state.get('recommended_action')}"
    )

    print(
        f"Execution recommendation: "
        f"{state.get('execution_recommendation')}"
    )

    print(
        f"Errors: {len(state.get('errors', []))}"
    )

    print(
        f"Tool calls: {len(state.get('tool_calls', []))}"
    )


def run_explicit_surge_test() -> None:
    """
    Test the deterministic fallback with an explicit surge request.

    This test does not require Gemini because it directly exercises
    the fallback logic that protects the Day 4 safety workflow from
    temporary model unavailability.
    """

    print("\n" + "=" * 70)
    print("EXPLICIT SURGE FALLBACK TEST")
    print("=" * 70)

    from src.agents.state import create_initial_state

    state = create_initial_state(
        "Investigate SFO. Completion rate appears to be low. "
        "Can we increase surge to 1.5x?"
    )

    state["detected_airport"] = "SFO"

    state["operational_data"] = {
        "airport_code": "SFO",
        "completion_rate": 0.9466,
        "average_eta_minutes": 6.13,
        "active_drivers": 137,
        "driver_cancellation_rate": 0.0379,
        "queue_size": 41,
        "surge_multiplier": 1.26,
        "request_volume": 341,
    }

    state["severity_level"] = "low"

    state["policy_decision"] = "Allowed"

    state["approval_required"] = False

    state["policy_results"] = [
        {
            "source": "sfo_pricing.md",
            "policy_id": "SFO-PRC-001",
            "section": "4. Surge Above 1.5x",
            "relevant_rule": (
                "Surge multiplier increases above 1.5x require "
                "formal request and approval from the Airport "
                "Operations Manager."
            ),
        }
    ]

    result = deterministic_resolution(state)

    execution_recommendation = result.get(
        "execution_recommendation"
    )

    print("\nRecommended action:")
    print(
        result.get("recommended_action")
    )

    print("\nStructured execution recommendation:")
    print(
        json.dumps(
            execution_recommendation,
            indent=2,
        )
    )

    assert execution_recommendation is not None
    assert execution_recommendation["action"] == "increase surge"
    assert execution_recommendation["airport_code"] == "SFO"
    assert execution_recommendation["new_multiplier"] == 1.5

    print(
        "\nPASS - Explicit surge request converted into "
        "structured execution recommendation."
    )

    print(
        "\nPASS - Resolution fallback test completed."
    )


if __name__ == "__main__":
    run_test()
    run_explicit_surge_test()