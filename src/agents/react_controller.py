"""
Controlled ReAct Controller

Provides a lightweight reasoning loop for the Airport
Operations AI Copilot.

The controller does not allow unlimited agent execution.
It uses the shared state iteration counter and enforces
a maximum number of iterations.
"""

from typing import Any, Dict, List


MAX_REACT_ITERATIONS = 5


def evaluate_need_for_more_information(
    state: Dict[str, Any],
) -> bool:
    """
    Determine whether additional investigation is required.

    The decision is deterministic and based on the information
    already available in shared state.

    Returns:
        True if more investigation is required.
        False if enough information is available.
    """

    # ---------------------------------------------------------
    # AIRPORT
    # ---------------------------------------------------------

    if not state.get("detected_airport"):
        return True

    # ---------------------------------------------------------
    # OPERATIONAL DATA
    # ---------------------------------------------------------

    operational_data = state.get(
        "operational_data",
        {},
    )

    if not operational_data:
        return True

    # ---------------------------------------------------------
    # SEVERITY
    # ---------------------------------------------------------

    severity = state.get(
        "severity_level"
    )

    if not severity:
        return True

    # ---------------------------------------------------------
    # CONTRIBUTING FACTORS
    # ---------------------------------------------------------

    contributing_factors = state.get(
        "contributing_factors",
        [],
    )

    if contributing_factors is None:
        contributing_factors = []

    # If the investigator has already produced operational
    # findings, the controller considers the investigation
    # sufficiently complete.
    if state.get("investigator_findings"):
        return False

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------

    return True


def build_react_observation(
    state: Dict[str, Any],
) -> str:
    """
    Build a human-readable observation from shared state.
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

    contributing_factors = state.get(
        "contributing_factors",
        [],
    )

    observations: List[str] = []

    observations.append(
        f"Airport identified: {airport}"
    )

    observations.append(
        f"Severity determined: {severity}"
    )

    if operational_data:

        completion_rate = operational_data.get(
            "completion_rate"
        )

        average_eta = operational_data.get(
            "average_eta_minutes"
        )

        queue_size = operational_data.get(
            "queue_size"
        )

        surge_multiplier = operational_data.get(
            "surge_multiplier"
        )

        if completion_rate is not None:
            observations.append(
                f"Completion rate: "
                f"{completion_rate:.2%}"
            )

        if average_eta is not None:
            observations.append(
                f"Average ETA: "
                f"{average_eta:.2f} minutes"
            )

        if queue_size is not None:
            observations.append(
                f"Queue size: {queue_size}"
            )

        if surge_multiplier is not None:
            observations.append(
                f"Surge multiplier: "
                f"{surge_multiplier:.2f}x"
            )

    if contributing_factors:

        observations.append(
            "Contributing factors identified: "
            + "; ".join(
                str(factor)
                for factor in contributing_factors
            )
        )

    return "\n".join(
        observations
    )


def run_react_iteration(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute one controlled ReAct reasoning step.

    The controller records:

        Observation
        Reasoning
        Action

    in the shared state.

    No external tools are called directly here.
    Tool execution remains the responsibility of
    the specialized agents.
    """

    iteration = state.get(
        "iteration",
        0,
    )

    max_iterations = state.get(
        "max_iterations",
        MAX_REACT_ITERATIONS,
    )

    observation = build_react_observation(
        state
    )

    needs_more_information = (
        evaluate_need_for_more_information(
            state
        )
    )

    if needs_more_information:

        reasoning = (
            "Available information is incomplete. "
            "Additional investigation is required "
            "before proceeding."
        )

        action = (
            "Continue investigation."
        )

    else:

        reasoning = (
            "The current operational investigation "
            "contains sufficient information to proceed "
            "to policy and resolution analysis."
        )

        action = (
            "Proceed to policy and resolution."
        )

    react_record = {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "observation": observation,
        "reasoning": reasoning,
        "action": action,
        "needs_more_information": (
            needs_more_information
        ),
    }

    existing_records = state.get(
        "react_trace"
    )

    if not isinstance(
        existing_records,
        list,
    ):
        existing_records = []

    existing_records.append(
        react_record
    )

    state["react_trace"] = (
        existing_records
    )

    return state


def should_continue_react_loop(
    state: Dict[str, Any],
) -> bool:
    """
    Determine whether another ReAct iteration
    is allowed.

    The loop can continue only when:

        1. More information is required.
        2. The maximum iteration count has not
           been reached.
    """

    if not evaluate_need_for_more_information(
        state
    ):
        return False

    current_iteration = state.get(
        "iteration",
        0,
    )

    max_iterations = state.get(
        "max_iterations",
        MAX_REACT_ITERATIONS,
    )

    return current_iteration < max_iterations


def run_test() -> None:
    """
    Standalone test for the controlled ReAct controller.
    """

    print("\n" + "=" * 70)
    print(
        "CONTROLLED REACT CONTROLLER TEST"
    )
    print("=" * 70)

    state: Dict[str, Any] = {
        "detected_airport": "SFO",
        "severity_level": "low",
        "operational_data": {
            "completion_rate": 0.9466,
            "average_eta_minutes": 6.13,
            "queue_size": 41,
            "surge_multiplier": 1.26,
        },
        "contributing_factors": [
            "High request volume relative to active driver supply"
        ],
        "investigator_findings": [
            "Completion rate: 94.66%",
            "Average ETA: 6.13 minutes",
        ],
        "iteration": 1,
        "max_iterations": 5,
    }

    state = run_react_iteration(
        state
    )

    trace = state.get(
        "react_trace",
        [],
    )

    print("\nObservation:")
    print("-" * 70)
    print(
        trace[0]["observation"]
    )

    print("\nReasoning:")
    print("-" * 70)
    print(
        trace[0]["reasoning"]
    )

    print("\nAction:")
    print("-" * 70)
    print(
        trace[0]["action"]
    )

    print("\nNeed more information:")
    print(
        trace[0][
            "needs_more_information"
        ]
    )

    print("\nContinue ReAct loop:")
    print(
        should_continue_react_loop(
            state
        )
    )

    print("\nIterations:")
    print(
        f"{state['iteration']} / "
        f"{state['max_iterations']}"
    )

    print("\nTrace records:")
    print(
        len(
            state["react_trace"]
        )
    )

    print("\n" + "=" * 70)
    print(
        "CONTROLLED REACT TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_test()