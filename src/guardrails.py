
"""
Day 4 - AI Safety, Guardrails & Human-in-the-Loop

Central safety layer for:
- Input validation
- Output validation
- Risk classification
- Policy validation
- Human approval
- Execution decisions

Important:
The thresholds in this file are project assumptions for demonstration.
They are not real Uber policies.
"""

from datetime import datetime
import re
from typing import Any, Dict, List, Optional


# ============================================================
# PROJECT SAFETY ASSUMPTIONS
# ============================================================

VALID_AIRPORTS = {"SFO", "LAX", "JFK"}

VALID_SEVERITY_LEVELS = {
    "low",
    "medium",
    "high",
    "critical",
}

# Day 4 risk threshold assumption.
# Surge >= 1.3x is considered high-risk.
HIGH_RISK_SURGE_THRESHOLD = 1.3

# Day 4 project assumption.
# Incentives above this total amount require approval.
INCENTIVE_APPROVAL_THRESHOLD = 1000.0

# Absolute safety boundary for surge values.
# This prevents obviously invalid requests such as 100x.
MAX_ALLOWED_SURGE = 3.0

MIN_ALLOWED_SURGE = 1.0

# Driver count must be non-negative.
MIN_DRIVER_COUNT = 0

# Incentive amount must be non-negative.
MIN_INCENTIVE_AMOUNT = 0.0


# ============================================================
# CUSTOM ERROR
# ============================================================

class GuardrailError(Exception):
    """Raised when a guardrail blocks an unsafe or invalid action."""

    pass


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_airport_code(airport_code: Any) -> str:
    """
    Validate airport code.
    """

    if airport_code is None:
        raise GuardrailError("airport_code is required.")

    if not isinstance(airport_code, str):
        raise GuardrailError("airport_code must be a string.")

    airport_code = airport_code.strip().upper()

    if not airport_code:
        raise GuardrailError("airport_code cannot be empty.")

    if airport_code not in VALID_AIRPORTS:
        raise GuardrailError(
            f"Unsupported airport_code '{airport_code}'. "
            f"Supported airports: {sorted(VALID_AIRPORTS)}"
        )

    return airport_code


def validate_surge_multiplier(surge_multiplier: Any) -> float:
    """
    Validate surge multiplier.

    Project safety assumptions:
        Minimum = 1.0x
        Maximum = 3.0x
    """

    if surge_multiplier is None:
        raise GuardrailError("surge_multiplier is required.")

    try:
        value = float(surge_multiplier)
    except (TypeError, ValueError):
        raise GuardrailError(
            "surge_multiplier must be a numeric value."
        )

    if value < MIN_ALLOWED_SURGE:
        raise GuardrailError(
            f"surge_multiplier cannot be below "
            f"{MIN_ALLOWED_SURGE}x."
        )

    if value > MAX_ALLOWED_SURGE:
        raise GuardrailError(
            f"surge_multiplier cannot exceed "
            f"{MAX_ALLOWED_SURGE}x."
        )

    return value


def validate_driver_count(driver_count: Any) -> int:
    """
    Validate number of drivers.

    Driver count cannot be negative.
    """

    if driver_count is None:
        raise GuardrailError("driver_count is required.")

    try:
        value = int(driver_count)
    except (TypeError, ValueError):
        raise GuardrailError(
            "driver_count must be an integer."
        )

    if value < MIN_DRIVER_COUNT:
        raise GuardrailError(
            "driver_count cannot be negative."
        )

    return value


def validate_incentive_amount(incentive_amount: Any) -> float:
    """
    Validate incentive amount.

    Incentive amount cannot be negative.
    """

    if incentive_amount is None:
        raise GuardrailError("incentive_amount is required.")

    try:
        value = float(incentive_amount)
    except (TypeError, ValueError):
        raise GuardrailError(
            "incentive_amount must be numeric."
        )

    if value < MIN_INCENTIVE_AMOUNT:
        raise GuardrailError(
            "incentive_amount cannot be negative."
        )

    return value


def validate_severity_level(severity_level: Any) -> str:
    """
    Validate severity level.
    """

    if severity_level is None:
        raise GuardrailError("severity_level is required.")

    if not isinstance(severity_level, str):
        raise GuardrailError(
            "severity_level must be a string."
        )

    severity = severity_level.strip().lower()

    if severity not in VALID_SEVERITY_LEVELS:
        raise GuardrailError(
            f"Invalid severity_level '{severity}'. "
            f"Valid levels: {sorted(VALID_SEVERITY_LEVELS)}"
        )

    return severity


def validate_required_parameters(
    action: str,
    parameters: Dict[str, Any],
) -> bool:
    """
    Validate parameters required for a particular action.
    """

    if not action:
        raise GuardrailError("action is required.")

    if not isinstance(parameters, dict):
        raise GuardrailError(
            "parameters must be provided as a dictionary."
        )

    action = action.strip().lower()

    required_parameters = {
        "read airport metrics": ["airport_code"],
        "search policy": ["airport_code"],
        "calculate incentive": [
            "driver_count",
            "severity_level",
        ],
        "increase surge": [
            "airport_code",
            "new_multiplier",
        ],
    }

    required = required_parameters.get(action)

    if required is None:
        raise GuardrailError(
            f"Unsupported action '{action}'."
        )

    missing = [
        parameter
        for parameter in required
        if parameter not in parameters
        or parameters[parameter] is None
    ]

    if missing:
        raise GuardrailError(
            f"Missing required parameters for '{action}': "
            f"{', '.join(missing)}"
        )

    return True


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_action_risk(
    action: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classify an operational action according to Day 4 rules.
    """

    if not action:
        raise GuardrailError("action is required.")

    parameters = parameters or {}

    action_normalized = action.strip().lower()

    # --------------------------------------------------------
    # Low-risk actions
    # --------------------------------------------------------

    if action_normalized == "read airport metrics":
        return {
            "action": action,
            "risk_level": "low",
            "approval_required": False,
            "reason": "Reading operational metrics is non-destructive.",
        }

    if action_normalized == "search policy":
        return {
            "action": action,
            "risk_level": "low",
            "approval_required": False,
            "reason": "Searching policy does not modify operational state.",
        }

    # --------------------------------------------------------
    # Incentive calculation
    # --------------------------------------------------------

    if action_normalized == "calculate incentive":

        incentive_amount = parameters.get(
            "incentive_amount",
            0,
        )

        try:
            incentive_amount = float(incentive_amount)
        except (TypeError, ValueError):
            raise GuardrailError(
                "incentive_amount must be numeric "
                "for risk classification."
            )

        if incentive_amount > INCENTIVE_APPROVAL_THRESHOLD:
            return {
                "action": action,
                "risk_level": "high",
                "approval_required": True,
                "reason": (
                    f"Incentive amount ${incentive_amount:.2f} "
                    f"exceeds the project approval threshold "
                    f"of ${INCENTIVE_APPROVAL_THRESHOLD:.2f}."
                ),
            }

        return {
            "action": action,
            "risk_level": "medium",
            "approval_required": False,
            "reason": (
                f"Incentive amount ${incentive_amount:.2f} "
                f"is within the project approval threshold."
            ),
        }

    # --------------------------------------------------------
    # Surge increase
    # --------------------------------------------------------

    if action_normalized == "increase surge":

        if "new_multiplier" not in parameters:
            raise GuardrailError(
                "new_multiplier is required to classify surge risk."
            )

        new_multiplier = validate_surge_multiplier(
            parameters["new_multiplier"]
        )

        if new_multiplier >= HIGH_RISK_SURGE_THRESHOLD:
            return {
                "action": action,
                "risk_level": "high",
                "approval_required": True,
                "reason": (
                    f"Requested surge {new_multiplier:.2f}x "
                    f"is at or above the high-risk threshold "
                    f"of {HIGH_RISK_SURGE_THRESHOLD:.2f}x."
                ),
            }

        return {
            "action": action,
            "risk_level": "medium",
            "approval_required": True,
            "reason": (
                f"Requested surge {new_multiplier:.2f}x "
                f"is below the high-risk threshold but "
                f"still requires human approval."
            ),
        }

    # --------------------------------------------------------
    # Unknown action
    # --------------------------------------------------------

    return {
        "action": action,
        "risk_level": "high",
        "approval_required": True,
        "reason": (
            "Unknown actions are treated as high-risk "
            "and require human approval."
        ),
    }


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_recommendation(
    recommendation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate an AI-generated recommendation before execution.
    """

    if not isinstance(recommendation, dict):
        raise GuardrailError(
            "AI recommendation must be a dictionary."
        )

    action = recommendation.get("action")

    if not action:
        raise GuardrailError(
            "AI recommendation is missing 'action'."
        )

    if not isinstance(action, str):
        raise GuardrailError(
            "Recommendation action must be a string."
        )

    action = action.strip().lower()

    if action == "increase surge":

        airport_code = recommendation.get("airport_code")
        new_multiplier = recommendation.get("new_multiplier")

        validate_airport_code(airport_code)
        validate_surge_multiplier(new_multiplier)

    elif action == "calculate incentive":

        incentive_amount = recommendation.get(
            "incentive_amount"
        )

        validate_incentive_amount(incentive_amount)

    elif action in {
        "read airport metrics",
        "search policy",
    }:
        pass

    else:
        raise GuardrailError(
            f"AI recommended unsupported action '{action}'."
        )

    return {
        "valid": True,
        "action": action,
        "message": "AI recommendation passed output validation.",
    }


# ============================================================
# POLICY VALIDATION
# ============================================================

def extract_policy_max_surge(
    policy_results: Optional[List[Dict[str, Any]]],
) -> Optional[float]:
    """
    Extract the maximum surge multiplier from retrieved policy
    evidence.

    The function deliberately uses retrieved policy text instead
    of trusting an AI-generated number.

    Supported policy phrasings include:

        "maximum standard surge multiplier ... 1.5x"

        "maximum surge multiplier ... 1.5x"

        "max surge ... 1.5x"

        "surge may increase up to 1.5x"

        "any increase above 1.5x requires approval"

        "increase above 1.5x requires approval"

        "surge above 1.5x ... Airport Operations Manager"

    Returns:
        float
            Explicit maximum surge value.

        None
            No clear policy maximum found.
    """

    if not policy_results:
        return None

    for result in policy_results:

        if not isinstance(result, dict):
            continue

        # ----------------------------------------------------
        # Policy evidence can appear under different keys.
        # ----------------------------------------------------

        text_parts = [
            result.get("text"),
            result.get("content"),
            result.get("relevant_rule"),
            result.get("policy_text"),
        ]

        text = " ".join(
            str(part)
            for part in text_parts
            if part
        )

        if not text:
            continue

        text_lower = text.lower()

        # ----------------------------------------------------
        # Pattern 1:
        #
        # "maximum standard surge multiplier permitted at SFO
        #  without additional approval is 1.5x"
        #
        # "maximum surge multiplier ... 1.5x"
        # ----------------------------------------------------

        patterns = [
            r"maximum\s+(?:standard\s+)?surge"
            r"(?:\s+multiplier)?"
            r"[\s\S]{0,150}?"
            r"(\d+(?:\.\d+)?)\s*x",

            r"max(?:imum)?\s+(?:standard\s+)?surge"
            r"(?:\s+multiplier)?"
            r"[\s\S]{0,150}?"
            r"(\d+(?:\.\d+)?)\s*x",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text_lower,
            )

            if match:
                return float(match.group(1))

        # ----------------------------------------------------
        # Pattern 2:
        #
        # "surge may be increased up to 1.5x"
        #
        # "up to 1.5x"
        # ----------------------------------------------------

        match = re.search(
            r"(?:surge[\s\S]{0,100}?)?"
            r"up\s+to\s+(\d+(?:\.\d+)?)\s*x",
            text_lower,
        )

        if match:
            return float(match.group(1))

        # ----------------------------------------------------
        # Pattern 3:
        #
        # "Any increase above 1.5x requires approval..."
        #
        # This is the exact wording used by the SFO pricing
        # policy retrieved by the application.
        # ----------------------------------------------------

        match = re.search(
            r"(?:increase|surge|multiplier)"
            r"[\s\S]{0,100}?"
            r"above\s+(\d+(?:\.\d+)?)\s*x",
            text_lower,
        )

        if match:
            return float(match.group(1))

        # ----------------------------------------------------
        # Pattern 4:
        #
        # "above 1.5x ... requires approval"
        #
        # This is intentionally more restrictive than simply
        # finding any number followed by x.
        # ----------------------------------------------------

        match = re.search(
            r"above\s+(\d+(?:\.\d+)?)\s*x"
            r"[\s\S]{0,120}?"
            r"requires?\s+(?:additional\s+)?approval",
            text_lower,
        )

        if match:
            return float(match.group(1))

        # ----------------------------------------------------
        # Pattern 5:
        #
        # "increase above 1.5x ... Airport Operations Manager"
        # ----------------------------------------------------

        match = re.search(
            r"(?:increase|surge)"
            r"[\s\S]{0,100}?"
            r"above\s+(\d+(?:\.\d+)?)\s*x"
            r"[\s\S]{0,150}?"
            r"airport\s+operations\s+manager",
            text_lower,
        )

        if match:
            return float(match.group(1))

    return None


def validate_policy(
    airport_code: str,
    new_multiplier: float,
    policy_results: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Compare a proposed surge change against retrieved policy.

    If the retrieved policy explicitly defines a maximum and the
    requested multiplier exceeds it, the action is blocked.

    If the requested multiplier is within the retrieved maximum,
    policy validation passes.

    If policy evidence does not contain a clear maximum, the
    action is blocked rather than being assumed safe.
    """

    airport_code = validate_airport_code(airport_code)
    new_multiplier = validate_surge_multiplier(new_multiplier)

    policy_max = extract_policy_max_surge(
        policy_results
    )

    # --------------------------------------------------------
    # No clear policy maximum
    # --------------------------------------------------------

    if policy_max is None:
        return {
            "policy_check": "insufficient_information",
            "allowed": False,
            "blocked": True,
            "airport_code": airport_code,
            "requested_multiplier": new_multiplier,
            "policy_max_surge": None,
            "reason": (
                "No explicit maximum surge value could be "
                "identified from the retrieved policy evidence. "
                "The action is blocked until policy information "
                "is available."
            ),
        }

    # --------------------------------------------------------
    # Policy violation
    # --------------------------------------------------------

    if new_multiplier > policy_max:
        return {
            "policy_check": "violation",
            "allowed": False,
            "blocked": True,
            "airport_code": airport_code,
            "requested_multiplier": new_multiplier,
            "policy_max_surge": policy_max,
            "reason": (
                f"POLICY VIOLATION: requested surge "
                f"{new_multiplier:.2f}x exceeds the policy "
                f"maximum of {policy_max:.2f}x for {airport_code}."
            ),
        }

    # --------------------------------------------------------
    # Policy compliant
    # --------------------------------------------------------

    return {
        "policy_check": "allowed",
        "allowed": True,
        "blocked": False,
        "airport_code": airport_code,
        "requested_multiplier": new_multiplier,
        "policy_max_surge": policy_max,
        "reason": (
            f"Requested surge {new_multiplier:.2f}x is within "
            f"the policy maximum of {policy_max:.2f}x."
        ),
    }


# ============================================================
# HUMAN APPROVAL
# ============================================================

def request_human_approval(
    action: str,
    risk_level: str,
    reason: str,
    interactive: bool = True,
) -> Dict[str, Any]:
    """
    Request explicit human approval for a high-impact action.

    For automated testing:
        interactive=False

    In non-interactive mode the function does NOT automatically
    approve the action.
    """

    risk_level = validate_severity_level(risk_level)

    approval_required = risk_level in {
        "medium",
        "high",
        "critical",
    }

    if not approval_required:
        return {
            "approval_required": False,
            "approval_decision": "not_required",
            "approved": True,
            "action": action,
            "reason": reason,
        }

    if not interactive:
        return {
            "approval_required": True,
            "approval_decision": "not_provided",
            "approved": False,
            "action": action,
            "reason": reason,
        }

    print()
    print("=" * 60)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print(f"Action : {action}")
    print(f"Risk   : {risk_level.upper()}")
    print(f"Reason : {reason}")
    print()
    print("The AI cannot execute this action without approval.")
    print()

    while True:

        answer = input(
            "Approve this action? [y/n]: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return {
                "approval_required": True,
                "approval_decision": "approved",
                "approved": True,
                "action": action,
                "reason": reason,
            }

        if answer in {"n", "no"}:
            return {
                "approval_required": True,
                "approval_decision": "rejected",
                "approved": False,
                "action": action,
                "reason": reason,
            }

        print("Please enter 'y' or 'n'.")


# ============================================================
# EXECUTION DECISION
# ============================================================

def evaluate_execution(
    recommendation: Dict[str, Any],
    policy_results: Optional[List[Dict[str, Any]]] = None,
    human_approval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete guardrail sequence.

    Flow:

        Output validation
              ↓
        Risk classification
              ↓
        Policy validation
              ↓
        Human approval
              ↓
        Execution decision

    This function only decides whether execution is permitted.
    It does NOT perform a real operational action.
    """

    # --------------------------------------------------------
    # 1. Validate AI output
    # --------------------------------------------------------

    output_validation = validate_recommendation(
        recommendation
    )

    action = output_validation["action"]

    # --------------------------------------------------------
    # 2. Classify risk
    # --------------------------------------------------------

    risk_parameters = dict(recommendation)

    risk = classify_action_risk(
        action=action,
        parameters=risk_parameters,
    )

    # --------------------------------------------------------
    # 3. Policy validation for surge changes
    # --------------------------------------------------------

    policy_check = {
        "policy_check": "not_applicable",
        "allowed": True,
        "blocked": False,
        "reason": "No policy-specific validation required.",
    }

    if action == "increase surge":

        policy_check = validate_policy(
            airport_code=recommendation["airport_code"],
            new_multiplier=recommendation["new_multiplier"],
            policy_results=policy_results,
        )

        if policy_check["blocked"]:
            return {
                "execution_allowed": False,
                "status": "blocked",
                "reason": policy_check["reason"],
                "output_validation": output_validation,
                "risk": risk,
                "policy_check": policy_check,
                "human_approval": {
                    "approval_required": risk["approval_required"],
                    "approval_decision": "not_required",
                    "approved": False,
                },
            }

    # --------------------------------------------------------
    # 4. Approval validation
    # --------------------------------------------------------

    if risk["approval_required"]:

        if human_approval is None:
            return {
                "execution_allowed": False,
                "status": "approval_required",
                "reason": (
                    "Human approval is required before "
                    "this action can execute."
                ),
                "output_validation": output_validation,
                "risk": risk,
                "policy_check": policy_check,
                "human_approval": {
                    "approval_required": True,
                    "approval_decision": "not_provided",
                    "approved": False,
                },
            }

        if not human_approval.get("approved", False):
            return {
                "execution_allowed": False,
                "status": "rejected",
                "reason": (
                    "Human approval was not granted. "
                    "Action will not execute."
                ),
                "output_validation": output_validation,
                "risk": risk,
                "policy_check": policy_check,
                "human_approval": human_approval,
            }

    # --------------------------------------------------------
    # 5. Execution permitted
    # --------------------------------------------------------

    return {
        "execution_allowed": True,
        "status": "approved",
        "reason": (
            "All guardrail checks passed and the action "
            "is permitted to proceed."
        ),
        "output_validation": output_validation,
        "risk": risk,
        "policy_check": policy_check,
        "human_approval": (
            human_approval
            if human_approval is not None
            else {
                "approval_required": False,
                "approval_decision": "not_required",
                "approved": True,
            }
        ),
    }


# ============================================================
# AUDIT RECORD
# ============================================================

def create_audit_record(
    user_request: str,
    agents_invoked: List[str],
    tool_calls: List[Dict[str, Any]],
    retrieved_policies: List[Dict[str, Any]],
    recommendation: Dict[str, Any],
    risk_level: str,
    approval_required: bool,
    approval_decision: str,
    final_action: str,
    execution_result: str,
) -> Dict[str, Any]:
    """
    Create a structured audit record for the complete decision trail.
    """

    return {
        "timestamp": datetime.now().isoformat(),
        "user_request": user_request,
        "agents_invoked": agents_invoked,
        "tool_calls": tool_calls,
        "retrieved_policies": retrieved_policies,
        "recommendation": recommendation,
        "risk_level": risk_level,
        "approval_required": approval_required,
        "approval_decision": approval_decision,
        "final_action": final_action,
        "execution_result": execution_result,
    }


# ============================================================
# TESTS
# ============================================================

def run_guardrail_tests() -> None:
    """
    Run Day 4 guardrail tests.
    """

    print("=" * 70)
    print("DAY 4 - GUARDRAIL TESTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Test 1 - Invalid airport
    # --------------------------------------------------------

    try:
        validate_airport_code("UNKNOWN")
        print("FAIL - Invalid airport was not blocked.")
    except GuardrailError:
        print("PASS - Invalid airport blocked.")

    # --------------------------------------------------------
    # Test 2 - Invalid surge
    # --------------------------------------------------------

    try:
        validate_surge_multiplier(100)
        print("FAIL - Invalid surge was not blocked.")
    except GuardrailError:
        print("PASS - Invalid surge blocked.")

    # --------------------------------------------------------
    # Test 3 - Negative driver count
    # --------------------------------------------------------

    try:
        validate_driver_count(-50)
        print("FAIL - Negative driver count was not blocked.")
    except GuardrailError:
        print("PASS - Negative driver count blocked.")

    # --------------------------------------------------------
    # Test 4 - Invalid severity
    # --------------------------------------------------------

    try:
        validate_severity_level("extreme")
        print("FAIL - Invalid severity was not blocked.")
    except GuardrailError:
        print("PASS - Invalid severity blocked.")

    # --------------------------------------------------------
    # Test 5 - Low-risk action
    # --------------------------------------------------------

    risk = classify_action_risk(
        "read airport metrics"
    )

    assert risk["risk_level"] == "low"
    assert risk["approval_required"] is False

    print("PASS - Read metrics classified as low risk.")

    # --------------------------------------------------------
    # Test 6 - Medium-risk surge
    # --------------------------------------------------------

    risk = classify_action_risk(
        "increase surge",
        {
            "airport_code": "SFO",
            "new_multiplier": 1.2,
        },
    )

    assert risk["risk_level"] == "medium"
    assert risk["approval_required"] is True

    print("PASS - Surge below 1.3x classified as medium risk.")

    # --------------------------------------------------------
    # Test 7 - High-risk surge
    # --------------------------------------------------------

    risk = classify_action_risk(
        "increase surge",
        {
            "airport_code": "SFO",
            "new_multiplier": 1.5,
        },
    )

    assert risk["risk_level"] == "high"
    assert risk["approval_required"] is True

    print("PASS - Surge at/above 1.3x classified as high risk.")

    # --------------------------------------------------------
    # Test 8 - Policy violation
    # --------------------------------------------------------

    policy_results = [
        {
            "source": "sfo_pricing.md",
            "policy_id": "SFO-PRC-001",
            "section": "Section 3",
            "text": (
                "Maximum surge multiplier allowed at SFO "
                "is 1.5x."
            ),
        }
    ]

    recommendation = {
        "action": "increase surge",
        "airport_code": "SFO",
        "new_multiplier": 2.0,
    }

    result = evaluate_execution(
        recommendation=recommendation,
        policy_results=policy_results,
    )

    assert result["execution_allowed"] is False
    assert result["status"] == "blocked"
    assert result["policy_check"]["policy_check"] == "violation"

    print("PASS - Policy-violating recommendation blocked.")

    # --------------------------------------------------------
    # Test 9 - Policy compliant but approval required
    # --------------------------------------------------------

    recommendation = {
        "action": "increase surge",
        "airport_code": "SFO",
        "new_multiplier": 1.5,
    }

    result = evaluate_execution(
        recommendation=recommendation,
        policy_results=policy_results,
    )

    assert result["execution_allowed"] is False
    assert result["status"] == "approval_required"

    print(
        "PASS - Policy-compliant high-risk action "
        "requires human approval."
    )

    # --------------------------------------------------------
    # Test 10 - Explicit rejection
    # --------------------------------------------------------

    result = evaluate_execution(
        recommendation=recommendation,
        policy_results=policy_results,
        human_approval={
            "approval_required": True,
            "approval_decision": "rejected",
            "approved": False,
        },
    )

    assert result["execution_allowed"] is False
    assert result["status"] == "rejected"

    print("PASS - Rejected action was blocked.")

    # --------------------------------------------------------
    # Test 11 - Actual SFO policy wording
    # --------------------------------------------------------

    actual_sfo_policy = [
        {
            "source": "sfo_pricing.md",
            "policy_id": "SFO-PRC-001",
            "section": "4. Surge Above 1.5x",
            "relevant_rule": (
                "Section: 4. Surge Above 1.5x\n\n"
                "Any increase above **1.5x** requires approval "
                "from the **Airport Operations Manager** before "
                "implementation."
            ),
        }
    ]

    extracted_max = extract_policy_max_surge(
        actual_sfo_policy
    )

    assert extracted_max == 1.5

    print(
        "PASS - Actual SFO policy wording correctly "
        "extracts maximum surge of 1.5x."
    )

    # --------------------------------------------------------
    # Test 12 - Actual SFO policy reaches approval layer
    # --------------------------------------------------------

    recommendation = {
        "action": "increase surge",
        "airport_code": "SFO",
        "new_multiplier": 1.5,
    }

    result = evaluate_execution(
        recommendation=recommendation,
        policy_results=actual_sfo_policy,
    )

    assert result["execution_allowed"] is False
    assert result["status"] == "approval_required"
    assert result["policy_check"]["policy_check"] == "allowed"
    assert result["policy_check"]["policy_max_surge"] == 1.5

    print(
        "PASS - Actual SFO policy allows 1.5x and correctly "
        "passes the request to human approval."
    )

    print()
    print("=" * 70)
    print("ALL DAY 4 GUARDRAIL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_guardrail_tests()

