"""
Safe Executor
=============

Day 4 - AI Safety, Guardrails & Human-in-the-Loop

This module provides a safe execution layer between AI recommendations
and operational tools.

Execution flow:

    AI Recommendation
          |
          v
    Output Validation
          |
          v
    Risk Classification
          |
          v
    Policy Validation
          |
          v
    Human Approval (if required)
          |
          v
    Execute Tool
          |
          v
    Audit Trail
          |
          v
    Distilled Training Data

Important:
- Guardrails always run before execution.
- Policy violations are blocked.
- Medium/high/critical actions require approval.
- Low-risk actions can execute without approval.
- This project uses mock operational tools only.
"""

from typing import Any, Dict, Optional, List

from src.guardrails import (
    GuardrailError,
    classify_action_risk,
    evaluate_execution,
    request_human_approval,
    validate_recommendation,
)

from src.tools import execute_tool

from src.audit_logger import create_audit_from_execution

from src.distillation_logger import DistillationLogger


# ============================================================
# SAFE EXECUTOR
# ============================================================


def execute_safely(
    recommendation: Dict[str, Any],
    user_request: str = "Day 4 safe execution test",
    policy_results: Optional[List[Dict[str, Any]]] = None,
    human_approval: Optional[Dict[str, Any]] = None,
    interactive_approval: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Safely execute an AI recommendation.

    Parameters
    ----------
    recommendation:
        Structured AI recommendation.

        Example:

        {
            "action": "increase surge",
            "airport_code": "SFO",
            "new_multiplier": 1.2,
            "reason": "Temporary supply pressure"
        }

    user_request:
        Original user request used for the audit trail.

    policy_results:
        Retrieved policy evidence used by the policy guardrail.

    human_approval:
        Optional approval result.

        Example:

        {
            "approval_required": True,
            "approval_decision": "approved",
            "approved": True
        }

    interactive_approval:
        If True, the guardrail can ask the user for approval.
        For automated tests this should normally be False.

    state:
        Optional orchestrator state used for distilled training data.

    Returns
    -------
    Dict[str, Any]
        Complete execution and guardrail result.
    """

    agents_invoked = [
        "orchestrator",
        "investigator",
        "policy_agent",
        "resolution_agent",
    ]

    policy_results = policy_results or []

    # --------------------------------------------------------
    # 1. Validate AI recommendation
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # validate_recommendation() returns a validation summary.
    # It does NOT return all original recommendation parameters.
    #
    # Therefore we validate the original recommendation first,
    # then keep the original recommendation for the remaining
    # guardrail steps.
    # --------------------------------------------------------

    try:

        validation_result = validate_recommendation(
            recommendation
        )

        validated_recommendation = dict(recommendation)

        validated_recommendation["action"] = (
            validation_result["action"]
        )

    except GuardrailError as exc:

        execution_result = {
            "execution_allowed": False,
            "status": "blocked",
            "reason": str(exc),
            "final_action": "blocked",
            "execution_result": "blocked",
            "risk": {
                "risk_level": "high",
                "approval_required": True,
            },
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 2. Classify risk
    # --------------------------------------------------------

    action = validated_recommendation["action"]

    try:

        risk = classify_action_risk(
            action=action,
            parameters=validated_recommendation,
        )

    except GuardrailError as exc:

        execution_result = {
            "execution_allowed": False,
            "status": "blocked",
            "reason": str(exc),
            "final_action": "blocked",
            "execution_result": "blocked",
            "risk": {
                "risk_level": "high",
                "approval_required": True,
            },
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 3. First complete guardrail evaluation
    # --------------------------------------------------------
    #
    # evaluate_execution() performs:
    #
    #   1. Output validation
    #   2. Risk classification
    #   3. Policy validation
    #   4. Approval validation
    #
    # It does NOT execute the operational tool.
    # --------------------------------------------------------

    try:

        decision = evaluate_execution(
            recommendation=validated_recommendation,
            policy_results=policy_results,
            human_approval=human_approval,
        )

    except GuardrailError as exc:

        execution_result = {
            "execution_allowed": False,
            "status": "blocked",
            "reason": str(exc),
            "final_action": "blocked",
            "execution_result": "blocked",
            "risk": risk,
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 4. Policy violation / guardrail block
    # --------------------------------------------------------

    if decision.get("status") == "blocked":

        execution_result = {
            **decision,
            "final_action": "blocked",
            "execution_result": "blocked",
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 5. Human approval
    # --------------------------------------------------------
    #
    # Medium/high/critical actions require approval according
    # to the Day 4 project assumptions.
    # --------------------------------------------------------

    approval = decision.get("human_approval")

    if decision.get("status") == "approval_required":

        approval = request_human_approval(
            action=action,
            risk_level=risk["risk_level"],
            reason=risk["reason"],
            interactive=interactive_approval,
        )

        # ----------------------------------------------------
        # Re-run the complete guardrail sequence with the
        # approval decision.
        # ----------------------------------------------------

        try:

            decision = evaluate_execution(
                recommendation=validated_recommendation,
                policy_results=policy_results,
                human_approval=approval,
            )

        except GuardrailError as exc:

            execution_result = {
                "execution_allowed": False,
                "status": "blocked",
                "reason": str(exc),
                "final_action": "blocked",
                "execution_result": "blocked",
                "risk": risk,
                "human_approval": approval,
            }

            create_audit_from_execution(
                user_request=user_request,
                execution_result=execution_result,
                agents_invoked=agents_invoked,
                previous_tool_calls=[],
                retrieved_policies=policy_results,
                recommendation=validated_recommendation,
            )

            return execution_result

    # --------------------------------------------------------
    # 6. Handle human rejection
    # --------------------------------------------------------

    if decision.get("status") == "rejected":

        execution_result = {
            **decision,
            "final_action": "rejected",
            "execution_result": "not_executed",
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 7. Final execution permission check
    # --------------------------------------------------------

    if not decision.get("execution_allowed", False):

        execution_result = {
            **decision,
            "final_action": "blocked",
            "execution_result": "not_executed",
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 8. Determine operational tool
    # --------------------------------------------------------

    tool_name = None
    tool_arguments: Dict[str, Any] = {}

    # --------------------------------------------------------
    # Increase surge
    # --------------------------------------------------------

    if action == "increase surge":

        tool_name = "trigger_surge_override"

        tool_arguments = {
            "airport_code": validated_recommendation[
                "airport_code"
            ],
            "new_multiplier": validated_recommendation[
                "new_multiplier"
            ],
            "reason": validated_recommendation.get(
                "reason",
                "Approved operational surge change.",
            ),
        }

    # --------------------------------------------------------
    # Calculate incentive
    # --------------------------------------------------------

    elif action == "calculate incentive":

        tool_name = "calculate_driver_incentive"

        tool_arguments = {
            "driver_count": validated_recommendation[
                "driver_count"
            ],
            "severity_level": validated_recommendation[
                "severity_level"
            ],
        }

    # --------------------------------------------------------
    # Read airport metrics
    # --------------------------------------------------------

    elif action == "read airport metrics":

        tool_name = "get_airport_metrics"

        tool_arguments = {
            "airport_code": validated_recommendation[
                "airport_code"
            ],
        }

    # --------------------------------------------------------
    # Search policy
    #
    # Policy search is already handled by the RAG layer.
    # It does not execute an operational mutation.
    # --------------------------------------------------------

    elif action == "search policy":

        execution_result = {
            **decision,
            "execution_allowed": True,
            "status": "executed",
            "tool_name": "policy_retrieval",
            "tool_arguments": {
                "airport_code": validated_recommendation.get(
                    "airport_code"
                )
            },
            "tool_result": {
                "status": "success",
                "message": (
                    "Policy search is handled by the "
                    "policy/RAG agent."
                ),
            },
            "final_action": action,
            "execution_result": "success",
        }

        audit_record = create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        execution_result["audit_record"] = audit_record

        if state is not None:

            try:

                distillation_logger = DistillationLogger()

                distillation_logger.create_distilled_record(
                    state=state,
                    execution_result=execution_result,
                )

            except Exception as exc:

                execution_result["distillation_warning"] = str(
                    exc
                )

        return execution_result

    # --------------------------------------------------------
    # Unsupported execution mapping
    # --------------------------------------------------------

    else:

        execution_result = {
            **decision,
            "execution_allowed": False,
            "status": "blocked",
            "reason": (
                f"No executable tool mapping exists for "
                f"action: {action}"
            ),
            "final_action": "blocked",
            "execution_result": "not_executed",
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 9. Execute through the existing tool dispatcher
    # --------------------------------------------------------

    try:

        tool_result = execute_tool(
            tool_name,
            tool_arguments,
        )

    except Exception as exc:

        execution_result = {
            **decision,
            "execution_allowed": True,
            "status": "execution_failed",
            "reason": str(exc),
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "tool_result": None,
            "final_action": action,
            "execution_result": "failed",
        }

        create_audit_from_execution(
            user_request=user_request,
            execution_result=execution_result,
            agents_invoked=agents_invoked,
            previous_tool_calls=[
                {
                    "tool_name": tool_name,
                    "arguments": tool_arguments,
                }
            ],
            retrieved_policies=policy_results,
            recommendation=validated_recommendation,
        )

        return execution_result

    # --------------------------------------------------------
    # 10. Successful execution
    # --------------------------------------------------------

    execution_result = {
        **decision,
        "execution_allowed": True,
        "status": "executed",
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
        "tool_result": tool_result,
        "final_action": action,
        "execution_result": "success",
    }

    # --------------------------------------------------------
    # 11. Persistent audit trail
    # --------------------------------------------------------

    audit_record = create_audit_from_execution(
        user_request=user_request,
        execution_result=execution_result,
        agents_invoked=agents_invoked,
        previous_tool_calls=[
            {
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "result": tool_result,
            }
        ],
        retrieved_policies=policy_results,
        recommendation=validated_recommendation,
    )

    execution_result["audit_record"] = audit_record

    # --------------------------------------------------------
    # 12. Distill successful interaction
    # --------------------------------------------------------

    if state is not None:

        try:

            distillation_logger = DistillationLogger()

            distillation_logger.create_distilled_record(
                state=state,
                execution_result=execution_result,
            )

        except Exception as exc:

            # Distillation failure must not change the result
            # of an already successful operational execution.

            execution_result["distillation_warning"] = str(
                exc
            )

    return execution_result


# ============================================================
# TESTS
# ============================================================


def run_safe_executor_tests() -> None:
    """
    Run Day 4 safe-execution integration tests.

    Tests:

    1. Invalid recommendations are blocked.
    2. Policy violations are blocked.
    3. High-risk actions require approval.
    4. Rejected actions do not execute.
    5. Approved high-risk actions execute.
    6. Low-risk actions execute without approval.
    """

    print()
    print("=" * 60)
    print("DAY 4 - SAFE EXECUTOR INTEGRATION TESTS")
    print("=" * 60)

    # --------------------------------------------------------
    # TEST 1 - Invalid recommendation
    # --------------------------------------------------------

    invalid_recommendation = {
        "action": "increase surge",
        "airport_code": "UNKNOWN",
        "new_multiplier": 100.0,
        "reason": "Invalid test",
    }

    result = execute_safely(
        recommendation=invalid_recommendation,
        user_request="Invalid recommendation test",
        interactive_approval=False,
    )

    assert result["status"] == "blocked"

    print("PASS - Invalid recommendation blocked.")

    # --------------------------------------------------------
    # TEST 2 - Policy violation
    # --------------------------------------------------------

    policy_results = [
        {
            "source": "sfo_pricing.md",
            "policy_id": "SFO-PRC-001",
            "section": "Section 3",
            "relevant_rule": (
                "Maximum surge multiplier allowed without "
                "additional approval is 1.5x."
            ),
        }
    ]

    policy_violation = {
        "action": "increase surge",
        "airport_code": "SFO",
        "new_multiplier": 2.0,
        "reason": "Test policy violation",
    }

    result = execute_safely(
        recommendation=policy_violation,
        user_request="Test SFO policy violation",
        policy_results=policy_results,
        interactive_approval=False,
    )

    assert result["status"] == "blocked"

    print("PASS - Policy violation blocked before execution.")

    # --------------------------------------------------------
    # TEST 3 - High-risk action without approval
    # --------------------------------------------------------
    #
    # 1.3x is high risk according to the Day 4 project
    # threshold.
    #
    # Because interactive_approval=False, the approval
    # function returns "not_provided".
    #
    # Therefore the action must NOT execute.
    # --------------------------------------------------------

    high_risk_recommendation = {
        "action": "increase surge",
        "airport_code": "SFO",
        "new_multiplier": 1.3,
        "reason": "Test high-risk approval requirement",
    }

    compliant_policy = [
        {
            "source": "sfo_pricing.md",
            "policy_id": "SFO-PRC-001",
            "section": "Section 3",
            "relevant_rule": (
                "Maximum surge multiplier allowed without "
                "additional approval is 1.5x."
            ),
        }
    ]

    result = execute_safely(
        recommendation=high_risk_recommendation,
        user_request="Test high-risk action",
        policy_results=compliant_policy,
        interactive_approval=False,
    )

    assert result["status"] in {
        "approval_required",
        "rejected",
    }

    assert result.get("execution_result") != "success"

    print(
        "PASS - High-risk action blocked without human approval."
    )

    # --------------------------------------------------------
    # TEST 4 - Explicit rejection
    # --------------------------------------------------------

    rejected_approval = {
        "approval_required": True,
        "approval_decision": "rejected",
        "approved": False,
        "action": "increase surge",
        "reason": "Human rejected test action.",
    }

    result = execute_safely(
        recommendation=high_risk_recommendation,
        user_request="Test rejected high-risk action",
        policy_results=compliant_policy,
        human_approval=rejected_approval,
        interactive_approval=False,
    )

    assert result["status"] == "rejected"

    assert result["execution_result"] == "not_executed"

    print("PASS - Rejected action was not executed.")

    # --------------------------------------------------------
    # TEST 5 - Approved high-risk action
    # --------------------------------------------------------
    #
    # 1.4x is high risk but within the SFO policy maximum
    # of 1.5x.
    #
    # Explicit human approval allows the mock execution.
    # --------------------------------------------------------

    approved_recommendation = {
        "action": "increase surge",
        "airport_code": "SFO",
        "new_multiplier": 1.4,
        "reason": "Approved test surge increase",
    }

    approved_approval = {
        "approval_required": True,
        "approval_decision": "approved",
        "approved": True,
        "action": "increase surge",
        "reason": "Human approved test action.",
    }

    result = execute_safely(
        recommendation=approved_recommendation,
        user_request="Test approved high-risk action",
        policy_results=compliant_policy,
        human_approval=approved_approval,
        interactive_approval=False,
    )

    assert result["status"] == "executed"

    assert result["execution_result"] == "success"

    assert result["tool_name"] == "trigger_surge_override"

    print("PASS - Approved high-risk action executed safely.")

    # --------------------------------------------------------
    # TEST 6 - Low-risk metrics action
    # --------------------------------------------------------

    metrics_recommendation = {
        "action": "read airport metrics",
        "airport_code": "SFO",
        "reason": "Read current SFO operational metrics",
    }

    result = execute_safely(
        recommendation=metrics_recommendation,
        user_request="Read SFO metrics",
        interactive_approval=False,
    )

    assert result["status"] == "executed"

    assert result["execution_result"] == "success"

    assert result["tool_name"] == "get_airport_metrics"

    print(
        "PASS - Low-risk metrics action executed "
        "without human approval."
    )

    print()
    print("ALL SAFE EXECUTOR INTEGRATION TESTS PASSED")
    print()


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    run_safe_executor_tests()