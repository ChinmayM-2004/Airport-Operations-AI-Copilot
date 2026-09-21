"""
Day 3 - Operations Investigator Agent

Responsibilities:
- Identify the airport from the user query.
- Retrieve current airport telemetry using Day 2 tools.
- Analyze operational metrics.
- Identify potential contributing factors.
- Determine operational severity.
- Store findings in shared agent state.

The agent never invents operational telemetry.
All current metrics must come from get_airport_metrics().
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools import get_airport_metrics
from src.agents.state import (
    add_error,
    add_tool_call,
    update_state,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-3.6-flash"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please add it to the project's .env file."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# Airport Detection
# ============================================================

VALID_AIRPORTS = {"SFO", "LAX", "JFK"}


def detect_airport(query: str) -> Optional[str]:
    """
    Detect an airport code from the user's query.

    Parameters
    ----------
    query : str
        User's natural-language question.

    Returns
    -------
    str or None
        Detected airport code.
    """

    query_upper = query.upper()

    for airport in VALID_AIRPORTS:

        if airport in query_upper:
            return airport

    return None


# ============================================================
# Severity Calculation
# ============================================================

def determine_severity(metrics: Dict[str, Any]) -> str:
    """
    Determine operational severity using deterministic rules.

    This keeps severity classification grounded in actual telemetry.

    Rules:
    - Critical:
        completion < 0.75
        OR cancellation >= 0.15
        OR average ETA >= 20 minutes

    - High:
        completion < 0.85
        OR cancellation >= 0.10
        OR average ETA >= 15 minutes
        OR queue >= 100

    - Medium:
        completion < 0.90
        OR cancellation >= 0.07
        OR average ETA >= 10 minutes
        OR queue >= 60

    - Low:
        Otherwise
    """

    completion_rate = metrics["completion_rate"]
    cancellation_rate = metrics["driver_cancellation_rate"]
    average_eta = metrics["average_eta_minutes"]
    queue_size = metrics["queue_size"]

    if (
        completion_rate < 0.75
        or cancellation_rate >= 0.15
        or average_eta >= 20
    ):
        return "critical"

    if (
        completion_rate < 0.85
        or cancellation_rate >= 0.10
        or average_eta >= 15
        or queue_size >= 100
    ):
        return "high"

    if (
        completion_rate < 0.90
        or cancellation_rate >= 0.07
        or average_eta >= 10
        or queue_size >= 60
    ):
        return "medium"

    return "low"


# ============================================================
# Operational Anomaly Detection
# ============================================================

def identify_contributing_factors(
    metrics: Dict[str, Any],
) -> list:
    """
    Identify potential operational contributing factors.

    The factors are based on deterministic thresholds so that the
    agent's observations remain grounded in the telemetry.
    """

    factors = []

    completion_rate = metrics["completion_rate"]
    cancellation_rate = metrics["driver_cancellation_rate"]
    average_eta = metrics["average_eta_minutes"]
    queue_size = metrics["queue_size"]
    request_volume = metrics["request_volume"]
    active_drivers = metrics["active_drivers"]

    if completion_rate < 0.90:
        factors.append(
            f"Low completion rate ({completion_rate:.2%})"
        )

    if cancellation_rate >= 0.07:
        factors.append(
            f"Elevated driver cancellation rate "
            f"({cancellation_rate:.2%})"
        )

    if average_eta >= 10:
        factors.append(
            f"High average ETA ({average_eta:.2f} minutes)"
        )

    if queue_size >= 60:
        factors.append(
            f"Large airport queue ({queue_size} drivers)"
        )

    if request_volume > active_drivers * 2:
        factors.append(
            f"High request volume relative to active driver supply "
            f"({request_volume} requests vs {active_drivers} active drivers)"
        )

    if not factors:
        factors.append(
            "No major anomaly detected using the configured thresholds."
        )

    return factors


# ============================================================
# Gemini Analysis
# ============================================================

def generate_investigation_summary(
    user_query: str,
    metrics: Dict[str, Any],
    severity: str,
    contributing_factors: list,
) -> str:
    """
    Ask Gemini to summarize the investigation.

    Gemini receives only actual tool-derived metrics and detected
    factors. It is explicitly prohibited from inventing values.
    """

    prompt = f"""
You are the Operations Investigator Agent for an airport operations
AI copilot.

Analyze the operational telemetry below.

USER QUERY:
{user_query}

AIRPORT:
{metrics["airport_code"]}

CURRENT TELEMETRY:
{json.dumps(metrics, indent=2)}

DETERMINISTIC SEVERITY:
{severity}

DETECTED CONTRIBUTING FACTORS:
{json.dumps(contributing_factors, indent=2)}

Instructions:

1. Use ONLY the telemetry values provided above.
2. Do not invent metrics or thresholds.
3. Clearly identify whether an operational anomaly is present.
4. Explain the most relevant contributing factors.
5. Keep the summary concise.
6. Do not recommend a specific policy action yet.
   The Policy Agent and Resolution Agent will handle those steps.

Return a concise operational investigation summary.
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            ),
        )

        if response.text:
            return response.text.strip()

    except Exception as exc:

        print(
            f"\nGemini investigation summary unavailable: {exc}"
        )

    # Fallback ensures the investigator still works if Gemini
    # is temporarily unavailable.
    return (
        f"Operational investigation for {metrics['airport_code']}: "
        f"severity is {severity}. "
        f"Detected factors: "
        f"{'; '.join(contributing_factors)}"
    )


# ============================================================
# Main Investigator Agent
# ============================================================

def investigate(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the Operations Investigator Agent.

    Parameters
    ----------
    state : dict
        Shared multi-agent state.

    Returns
    -------
    dict
        Updated shared state.
    """

    user_query = state["user_query"]

    print("\n" + "=" * 70)
    print("OPERATIONS INVESTIGATOR AGENT")
    print("=" * 70)

    print(f"User query: {user_query}")

    # --------------------------------------------------------
    # Step 1: Detect airport
    # --------------------------------------------------------

    airport_code = detect_airport(user_query)

    if not airport_code:

        error_message = (
            "Could not identify an airport from the query. "
            "Supported airports are SFO, LAX, and JFK."
        )

        print(f"\nERROR: {error_message}")

        add_error(
            state,
            "operations_investigator",
            error_message,
        )

        return state

    print(f"\nDetected airport: {airport_code}")

    update_state(
        state,
        current_agent="operations_investigator",
        detected_airport=airport_code,
    )

    # --------------------------------------------------------
    # Step 2: Call operational tool
    # --------------------------------------------------------

    print("\nCalling get_airport_metrics()...")

    try:

        metrics = get_airport_metrics(airport_code)

    except Exception as exc:

        error_message = (
            f"Failed to retrieve operational metrics: {exc}"
        )

        print(f"ERROR: {error_message}")

        add_error(
            state,
            "operations_investigator",
            error_message,
        )

        return state

    if metrics.get("status") != "success":

        error_message = metrics.get(
            "error",
            "Unknown operational tool error.",
        )

        print(f"ERROR: {error_message}")

        add_error(
            state,
            "operations_investigator",
            error_message,
        )

        return state

    add_tool_call(
        state,
        tool_name="get_airport_metrics",
        arguments={"airport_code": airport_code},
        result=metrics,
    )

    # --------------------------------------------------------
    # Step 3: Determine severity
    # --------------------------------------------------------

    severity = determine_severity(metrics)

    print(f"\nDetermined severity: {severity}")

    # --------------------------------------------------------
    # Step 4: Identify contributing factors
    # --------------------------------------------------------

    contributing_factors = identify_contributing_factors(
        metrics
    )

    print("\nContributing factors:")

    for factor in contributing_factors:
        print(f"- {factor}")

    # --------------------------------------------------------
    # Step 5: Build investigation findings
    # --------------------------------------------------------

    findings = [
        (
            f"Completion rate: "
            f"{metrics['completion_rate']:.2%}"
        ),
        (
            f"Average ETA: "
            f"{metrics['average_eta_minutes']:.2f} minutes"
        ),
        (
            f"Active drivers: "
            f"{metrics['active_drivers']}"
        ),
        (
            f"Driver cancellation rate: "
            f"{metrics['driver_cancellation_rate']:.2%}"
        ),
        (
            f"Queue size: "
            f"{metrics['queue_size']}"
        ),
        (
            f"Request volume: "
            f"{metrics['request_volume']}"
        ),
        (
            f"Current surge multiplier: "
            f"{metrics['surge_multiplier']}x"
        ),
    ]

    # --------------------------------------------------------
    # Step 6: Gemini investigation summary
    # --------------------------------------------------------

    summary = generate_investigation_summary(
        user_query=user_query,
        metrics=metrics,
        severity=severity,
        contributing_factors=contributing_factors,
    )

    print("\nInvestigation summary:")
    print("-" * 70)
    print(summary)

    # --------------------------------------------------------
    # Step 7: Update shared state
    # --------------------------------------------------------

    update_state(
        state,
        operational_data=metrics,
        investigator_findings=findings + [summary],
        severity_level=severity,
        contributing_factors=contributing_factors,
        current_agent="operations_investigator",
    )

    return state


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    from src.agents.state import create_initial_state

    test_state = create_initial_state(
        "What's happening at SFO right now?"
    )

    result = investigate(test_state)

    print("\n" + "=" * 70)
    print("INVESTIGATOR TEST COMPLETE")
    print("=" * 70)

    print("\nFinal investigator state:")
    print(
        json.dumps(
            {
                "detected_airport": result["detected_airport"],
                "severity_level": result["severity_level"],
                "operational_data": result["operational_data"],
                "investigator_findings": result[
                    "investigator_findings"
                ],
                "contributing_factors": result[
                    "contributing_factors"
                ],
                "tool_calls": result["tool_calls"],
                "errors": result["errors"],
            },
            indent=2,
        )
    )