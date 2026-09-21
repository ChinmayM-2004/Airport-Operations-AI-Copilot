
"""
Persistent audit trail for the Airport Operations AI Copilot.

Day 4 responsibilities:
- Record every important decision and execution event.
- Persist audit records as JSONL.
- Capture risk and human-approval metadata.
- Support reading, counting, and retrieving audit records.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIT_DIRECTORY = PROJECT_ROOT / "data" / "audit_logs"
AUDIT_FILE = AUDIT_DIRECTORY / "audit_trail.jsonl"


# -------------------------------------------------------------------
# Audit Logger
# -------------------------------------------------------------------

class AuditLogger:
    """Persist structured audit records to a JSONL file."""

    def __init__(self, audit_file: Optional[Path] = None):
        self.audit_file = Path(audit_file) if audit_file else AUDIT_FILE
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        user_request: str,
        agents_invoked: Optional[List[str]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        retrieved_policies: Optional[List[Dict[str, Any]]] = None,
        recommendation: Optional[Dict[str, Any]] = None,
        risk_level: Optional[str] = None,
        approval_required: bool = False,
        approval_decision: str = "not_required",
        final_action: str = "not_executed",
        execution_result: str = "not_executed",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create and persist one audit record.
        """

        record = {
            "timestamp": datetime.now().isoformat(),
            "user_request": user_request,
            "agents_invoked": agents_invoked or [],
            "tool_calls": tool_calls or [],
            "retrieved_policies": retrieved_policies or [],
            "recommendation": recommendation or {},
            "risk_level": risk_level,
            "approval_required": approval_required,
            "approval_decision": approval_decision,
            "final_action": final_action,
            "execution_result": execution_result,
        }

        if extra:
            record["extra"] = extra

        with self.audit_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all audit records from the JSONL file."""

        if not self.audit_file.exists():
            return []

        records = []

        with self.audit_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return records

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the latest audit record."""

        records = self.read_all()

        if not records:
            return None

        return records[-1]

    def count(self) -> int:
        """Return the number of valid audit records."""

        return len(self.read_all())


# -------------------------------------------------------------------
# Integration helper
# -------------------------------------------------------------------

def create_audit_from_execution(
    user_request: str,
    execution_result: Dict[str, Any],
    agents_invoked: Optional[List[str]] = None,
    previous_tool_calls: Optional[List[Dict[str, Any]]] = None,
    retrieved_policies: Optional[List[Dict[str, Any]]] = None,
    recommendation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a safe-executor result into a complete audit record.

    The safe executor stores risk and human approval information
    inside nested dictionaries:

        execution_result["risk"]
        execution_result["human_approval"]

    This function extracts those values so the persistent audit
    record contains the complete Day 4 decision trail.
    """

    risk_data = execution_result.get("risk", {}) or {}
    human_approval = execution_result.get("human_approval", {}) or {}

    # ---------------------------------------------------------------
    # Extract risk information
    # ---------------------------------------------------------------

    risk_level = risk_data.get("risk_level")

    approval_required = risk_data.get(
        "approval_required",
        human_approval.get("approval_required", False),
    )

    # ---------------------------------------------------------------
    # Extract human approval information
    # ---------------------------------------------------------------

    approval_decision = human_approval.get(
        "approval_decision",
        "not_required",
    )

    # ---------------------------------------------------------------
    # Determine final action
    # ---------------------------------------------------------------

    status = execution_result.get("status", "unknown")

    if status == "executed":
        final_action = execution_result.get(
            "action",
            recommendation.get("action") if recommendation else "executed",
        )

    elif status == "rejected":
        final_action = "execution_stopped_after_rejection"

    elif status == "approval_required":
        final_action = "waiting_for_human_approval"

    elif status == "blocked":
        final_action = "blocked_by_guardrail"

    else:
        final_action = status

    # ---------------------------------------------------------------
    # Execution result
    # ---------------------------------------------------------------

    execution_status = execution_result.get(
        "execution_result",
        status,
    )

    # ---------------------------------------------------------------
    # Additional execution information
    # ---------------------------------------------------------------

    extra = {
        "execution_status": status,
        "policy_check": execution_result.get("policy_check"),
        "human_approval": human_approval,
        "risk_details": risk_data,
    }

    logger = AuditLogger()

    return logger.log(
        user_request=user_request,
        agents_invoked=agents_invoked or [],
        tool_calls=previous_tool_calls or [],
        retrieved_policies=retrieved_policies or [],
        recommendation=recommendation or {},
        risk_level=risk_level,
        approval_required=bool(approval_required),
        approval_decision=approval_decision,
        final_action=final_action,
        execution_result=execution_status,
        extra=extra,
    )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def run_audit_tests() -> None:
    """Run Day 4 audit logger tests."""

    print("=" * 70)
    print("DAY 4 - AUDIT TRAIL TESTS")
    print("=" * 70)

    test_file = AUDIT_DIRECTORY / "audit_test.jsonl"

    # Make sure the temporary test file starts clean.
    if test_file.exists():
        test_file.unlink()

    logger = AuditLogger(test_file)

    # ---------------------------------------------------------------
    # Test 1 - Create audit record
    # ---------------------------------------------------------------

    record = logger.log(
        user_request="Increase SFO surge to 1.4x",
        agents_invoked=[
            "orchestrator",
            "investigator",
            "policy_agent",
            "resolution_agent",
        ],
        tool_calls=[
            {
                "tool_name": "get_airport_metrics",
                "arguments": {"airport_code": "SFO"},
            }
        ],
        retrieved_policies=[
            {
                "source": "sfo_pricing.md",
                "policy_id": "SFO-PRC-001",
                "section": "3",
            }
        ],
        recommendation={
            "action": "increase surge",
            "airport_code": "SFO",
            "new_multiplier": 1.4,
        },
        risk_level="high",
        approval_required=True,
        approval_decision="approved",
        final_action="increase surge",
        execution_result="success",
    )

    assert record["user_request"] == "Increase SFO surge to 1.4x"
    print("PASS - Audit record created.")

    # ---------------------------------------------------------------
    # Test 2 - File created
    # ---------------------------------------------------------------

    assert test_file.exists()
    print("PASS - Audit JSONL file created.")

    # ---------------------------------------------------------------
    # Test 3 - Read record
    # ---------------------------------------------------------------

    records = logger.read_all()

    assert len(records) == 1
    assert records[0]["risk_level"] == "high"
    assert records[0]["approval_required"] is True
    assert records[0]["approval_decision"] == "approved"

    print("PASS - Audit record successfully read from JSONL file.")

    # ---------------------------------------------------------------
    # Test 4 - Latest
    # ---------------------------------------------------------------

    latest = logger.get_latest()

    assert latest is not None
    assert latest["final_action"] == "increase surge"

    print("PASS - Latest audit record retrieved.")

    # ---------------------------------------------------------------
    # Test 5 - Count
    # ---------------------------------------------------------------

    assert logger.count() == 1

    print("PASS - Audit record count verified.")

    # ---------------------------------------------------------------
    # Test 6 - Integration metadata extraction
    # ---------------------------------------------------------------

    integration_result = {
        "status": "executed",
        "action": "increase surge",
        "execution_result": "success",
        "risk": {
            "action": "increase surge",
            "risk_level": "high",
            "approval_required": True,
        },
        "human_approval": {
            "approval_required": True,
            "approval_decision": "approved",
            "approved": True,
        },
        "policy_check": {
            "valid": True,
            "policy_max_surge": 1.5,
        },
    }

    integration_record = create_audit_from_execution(
        user_request="Increase SFO surge to 1.4x",
        execution_result=integration_result,
        agents_invoked=["orchestrator", "investigator", "policy_agent"],
        previous_tool_calls=[
            {
                "tool_name": "get_airport_metrics",
                "arguments": {"airport_code": "SFO"},
            }
        ],
        retrieved_policies=[
            {
                "source": "sfo_pricing.md",
                "policy_id": "SFO-PRC-001",
                "section": "3",
            }
        ],
        recommendation={
            "action": "increase surge",
            "airport_code": "SFO",
            "new_multiplier": 1.4,
        },
    )

    assert integration_record["risk_level"] == "high"
    assert integration_record["approval_required"] is True
    assert integration_record["approval_decision"] == "approved"
    assert integration_record["final_action"] == "increase surge"
    assert integration_record["execution_result"] == "success"

    print("PASS - Integration risk and approval metadata verified.")

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

    if test_file.exists():
        test_file.unlink()

    print("PASS - Temporary audit test file cleaned.")

    print()
    print("=" * 70)
    print("ALL AUDIT TRAIL TESTS PASSED")
    print("=" * 70)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    run_audit_tests()

