"""
Shared state for the Day 3 multi-agent airport operations copilot.

The same state object is passed between agents so that:
    Orchestrator
        -> Investigator
        -> Policy Agent
        -> Resolution Agent
        -> Orchestrator

can share findings without losing context.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


def create_initial_state(user_query: str) -> Dict[str, Any]:
    """
    Create the initial state for a new user request.

    Parameters
    ----------
    user_query : str
        Natural-language operational question from the user.

    Returns
    -------
    dict
        Shared state used by all agents.
    """

    return {
        # Original user request
        "user_query": user_query,

        # Conversation information
        "conversation_history": [],
        "detected_airport": None,

        # Investigator information
        "operational_data": None,
        "investigator_findings": [],
        "severity_level": None,
        "contributing_factors": [],

        # Policy information
        "policy_results": [],
        "policy_decision": None,
        "approval_required": None,

        # Resolution information
        "possible_actions": [],
        "recommended_action": None,
        "recommendation_reasoning": [],

        # ReAct / orchestration information
        "current_agent": "orchestrator",
        "iteration": 0,
        "max_iterations": 5,
        "completed": False,

        # Tool / execution information
        "tool_calls": [],
        "errors": [],

        # Final response
        "final_response": None,

        # Metadata
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def update_state(
    state: Dict[str, Any],
    **updates: Any,
) -> Dict[str, Any]:
    """
    Update shared agent state.

    The function modifies the existing state and returns it so that
    agents can conveniently chain updates.
    """

    state.update(updates)
    state["updated_at"] = datetime.now().isoformat()

    return state


def add_tool_call(
    state: Dict[str, Any],
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    """
    Record a tool execution in shared state.
    """

    state["tool_calls"].append(
        {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
    )


def add_error(
    state: Dict[str, Any],
    agent: str,
    error: str,
) -> None:
    """
    Record an agent or tool error.
    """

    state["errors"].append(
        {
            "agent": agent,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
    )


def increment_iteration(state: Dict[str, Any]) -> int:
    """
    Increment and return the current orchestration iteration.
    """

    state["iteration"] += 1
    state["updated_at"] = datetime.now().isoformat()

    return state["iteration"]


def can_continue(state: Dict[str, Any]) -> bool:
    """
    Determine whether the agentic loop may continue.

    The loop stops when:
    - The issue has been completed.
    - Maximum iterations are reached.
    """

    if state["completed"]:
        return False

    if state["iteration"] >= state["max_iterations"]:
        return False

    return True


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a compact representation of the current state.

    Useful for debugging and displaying the agent handoff flow.
    """

    return {
        "user_query": state["user_query"],
        "detected_airport": state["detected_airport"],
        "severity_level": state["severity_level"],
        "investigator_findings": state["investigator_findings"],
        "contributing_factors": state["contributing_factors"],
        "policy_decision": state["policy_decision"],
        "approval_required": state["approval_required"],
        "possible_actions": state["possible_actions"],
        "recommended_action": state["recommended_action"],
        "current_agent": state["current_agent"],
        "iteration": state["iteration"],
        "max_iterations": state["max_iterations"],
        "tool_calls": len(state["tool_calls"]),
        "errors": state["errors"],
        "completed": state["completed"],
    }


if __name__ == "__main__":

    state = create_initial_state(
        "What's happening at SFO right now?"
    )

    print("Initial Day 3 agent state:")
    print(summarize_state(state))