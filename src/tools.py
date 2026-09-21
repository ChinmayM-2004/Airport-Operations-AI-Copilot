from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPERATIONAL_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "operational_data"
    / "airport_telemetry.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

VALID_AIRPORTS = {"SFO", "LAX", "JFK"}

VALID_SEVERITY_LEVELS = {
    "low",
    "medium",
    "high",
    "critical",
}

# Incentive per driver, in USD.
INCENTIVE_RATES = {
    "low": 5.0,
    "medium": 10.0,
    "high": 20.0,
    "critical": 30.0,
}

# Maximum surge multiplier allowed by the mock
# Day 2 execution tool.
MOCK_MAX_SURGE_MULTIPLIER = 3.0


# ============================================================
# CUSTOM TOOL EXCEPTION
# ============================================================

class ToolError(Exception):
    """
    Exception raised when a tool cannot complete
    a requested operation.
    """

    pass


# ============================================================
# DATA LOADING
# ============================================================

def load_operational_data():
    """
    Load airport telemetry from the CSV file.

    Returns:
        pandas.DataFrame
    """

    if not OPERATIONAL_DATA_FILE.exists():
        raise ToolError(
            f"Operational data file not found: "
            f"{OPERATIONAL_DATA_FILE}"
        )

    try:
        df = pd.read_csv(
            OPERATIONAL_DATA_FILE,
            parse_dates=["timestamp"],
        )
    except Exception as exc:
        raise ToolError(
            f"Failed to load operational data: {exc}"
        ) from exc

    required_columns = {
        "airport_code",
        "completion_rate",
        "average_eta",
        "active_drivers",
        "driver_cancellation_rate",
        "queue_size",
        "surge_multiplier",
        "request_volume",
        "timestamp",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ToolError(
            f"Operational data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_airport_code(airport_code):
    """
    Validate and normalize an airport code.
    """

    if airport_code is None:
        raise ToolError("airport_code is required.")

    if not isinstance(airport_code, str):
        raise ToolError("airport_code must be a string.")

    airport_code = airport_code.strip().upper()

    if not airport_code:
        raise ToolError("airport_code cannot be empty.")

    if airport_code not in VALID_AIRPORTS:
        raise ToolError(
            f"Invalid airport code '{airport_code}'. "
            f"Valid airports are: "
            f"{', '.join(sorted(VALID_AIRPORTS))}."
        )

    return airport_code


def validate_driver_count(driver_count):
    """
    Validate driver count.
    """

    if driver_count is None:
        raise ToolError("driver_count is required.")

    if isinstance(driver_count, bool):
        raise ToolError("driver_count must be a positive integer.")

    if not isinstance(driver_count, int):
        raise ToolError("driver_count must be an integer.")

    if driver_count <= 0:
        raise ToolError(
            "driver_count must be greater than zero."
        )

    if driver_count > 10000:
        raise ToolError(
            "driver_count cannot exceed 10,000."
        )

    return driver_count


def validate_severity_level(severity_level):
    """
    Validate and normalize severity level.
    """

    if severity_level is None:
        raise ToolError("severity_level is required.")

    if not isinstance(severity_level, str):
        raise ToolError(
            "severity_level must be a string."
        )

    severity_level = severity_level.strip().lower()

    if severity_level not in VALID_SEVERITY_LEVELS:
        raise ToolError(
            f"Invalid severity level '{severity_level}'. "
            f"Valid levels are: "
            f"{', '.join(sorted(VALID_SEVERITY_LEVELS))}."
        )

    return severity_level


def validate_surge_multiplier(new_multiplier):
    """
    Validate the requested surge multiplier.
    """

    if new_multiplier is None:
        raise ToolError(
            "new_multiplier is required."
        )

    if isinstance(new_multiplier, bool):
        raise ToolError(
            "new_multiplier must be a number."
        )

    if not isinstance(
        new_multiplier,
        (int, float),
    ):
        raise ToolError(
            "new_multiplier must be a number."
        )

    if new_multiplier < 1.0:
        raise ToolError(
            "new_multiplier cannot be below 1.0x."
        )

    if new_multiplier > MOCK_MAX_SURGE_MULTIPLIER:
        raise ToolError(
            f"new_multiplier cannot exceed "
            f"{MOCK_MAX_SURGE_MULTIPLIER}x "
            f"for the Day 2 mock tool."
        )

    return round(float(new_multiplier), 2)


# ============================================================
# TOOL 1 — AIRPORT METRICS
# ============================================================

def get_airport_metrics(airport_code):
    """
    Return the latest operational metrics for an airport.

    Args:
        airport_code: Airport code such as SFO, LAX, or JFK.

    Returns:
        Structured dictionary containing airport metrics.
    """

    try:
        airport_code = validate_airport_code(
            airport_code
        )

        df = load_operational_data()

        airport_data = df[
            df["airport_code"] == airport_code
        ].copy()

        if airport_data.empty:
            raise ToolError(
                f"No operational data found for {airport_code}."
            )

        # Select the most recent telemetry record.
        latest_record = airport_data.sort_values(
            "timestamp"
        ).iloc[-1]

        result = {
            "status": "success",
            "airport_code": airport_code,
            "timestamp": latest_record[
                "timestamp"
            ].isoformat(),
            "completion_rate": round(
                float(latest_record["completion_rate"]),
                4,
            ),
            "average_eta_minutes": round(
                float(latest_record["average_eta"]),
                2,
            ),
            "active_drivers": int(
                latest_record["active_drivers"]
            ),
            "driver_cancellation_rate": round(
                float(
                    latest_record[
                        "driver_cancellation_rate"
                    ]
                ),
                4,
            ),
            "queue_size": int(
                latest_record["queue_size"]
            ),
            "surge_multiplier": round(
                float(
                    latest_record[
                        "surge_multiplier"
                    ]
                ),
                2,
            ),
            "request_volume": int(
                latest_record["request_volume"]
            ),
            "source": "airport_telemetry.csv",
        }

        return result

    except ToolError:
        raise

    except Exception as exc:
        raise ToolError(
            f"Airport metrics tool failed: {exc}"
        ) from exc


# ============================================================
# TOOL 2 — DRIVER INCENTIVE CALCULATOR
# ============================================================

def calculate_driver_incentive(
    driver_count,
    severity_level,
):
    """
    Calculate a recommended driver incentive.

    Args:
        driver_count: Number of drivers to incentivize.
        severity_level: low, medium, high, or critical.

    Returns:
        Structured dictionary containing incentive
        recommendation and estimated total cost.
    """

    try:
        driver_count = validate_driver_count(
            driver_count
        )

        severity_level = validate_severity_level(
            severity_level
        )

        incentive_per_driver = INCENTIVE_RATES[
            severity_level
        ]

        estimated_total_cost = (
            driver_count * incentive_per_driver
        )

        result = {
            "status": "success",
            "driver_count": driver_count,
            "severity_level": severity_level,
            "recommended_incentive_per_driver_usd": (
                incentive_per_driver
            ),
            "estimated_total_cost_usd": round(
                estimated_total_cost,
                2,
            ),
            "currency": "USD",
        }

        return result

    except ToolError:
        raise

    except Exception as exc:
        raise ToolError(
            f"Driver incentive calculator failed: {exc}"
        ) from exc


# ============================================================
# TOOL 3 — SURGE OVERRIDE
# ============================================================

def trigger_surge_override(
    airport_code,
    new_multiplier,
    reason,
):
    """
    Mock execution tool for a surge override.

    Day 2 behavior:
        - Validates the airport.
        - Validates the multiplier.
        - Validates the reason.
        - Returns a mock execution result.

    Day 4 will add actual approval and permission controls.
    """

    try:
        airport_code = validate_airport_code(
            airport_code
        )

        new_multiplier = validate_surge_multiplier(
            new_multiplier
        )

        if reason is None:
            raise ToolError(
                "reason is required."
            )

        if not isinstance(reason, str):
            raise ToolError(
                "reason must be a string."
            )

        reason = reason.strip()

        if not reason:
            raise ToolError(
                "reason cannot be empty."
            )

        if len(reason) < 10:
            raise ToolError(
                "reason must contain at least "
                "10 characters."
            )

        result = {
            "status": "success",
            "execution_type": "mock",
            "airport_code": airport_code,
            "new_surge_multiplier": new_multiplier,
            "reason": reason,
            "message": (
                f"Mock surge override triggered for "
                f"{airport_code} at {new_multiplier}x."
            ),
            "approval_required": (
                "Not enforced on Day 2. "
                "Approval controls will be added on Day 4."
            ),
        }

        return result

    except ToolError:
        raise

    except Exception as exc:
        raise ToolError(
            f"Surge override tool failed: {exc}"
        ) from exc


# ============================================================
# TOOL EXECUTION HELPER
# ============================================================

def execute_tool(tool_name, arguments):
    """
    Execute a tool by name using structured arguments.

    This helper will later be used by the LLM
    function-calling workflow.
    """

    if not isinstance(arguments, dict):
        return {
            "status": "error",
            "error": "Tool arguments must be a dictionary.",
        }

    tools = {
        "get_airport_metrics": get_airport_metrics,
        "calculate_driver_incentive": (
            calculate_driver_incentive
        ),
        "trigger_surge_override": (
            trigger_surge_override
        ),
    }

    if tool_name not in tools:
        return {
            "status": "error",
            "error": (
                f"Unknown tool '{tool_name}'. "
                f"Available tools: "
                f"{', '.join(tools.keys())}"
            ),
        }

    try:
        return tools[tool_name](**arguments)

    except ToolError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "tool": tool_name,
        }

    except TypeError as exc:
        return {
            "status": "error",
            "error": (
                f"Invalid or missing tool parameters: "
                f"{exc}"
            ),
            "tool": tool_name,
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": (
                f"Unexpected tool execution error: "
                f"{exc}"
            ),
            "tool": tool_name,
        }


# ============================================================
# TOOL TESTS
# ============================================================

def run_tool_tests():
    """
    Run basic tests for all three tools,
    including error handling.
    """

    print("=" * 70)
    print("OPERATIONAL TOOLS TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Tool 1
    # --------------------------------------------------------

    print("\n1. get_airport_metrics('SFO')")
    print("-" * 70)

    result = get_airport_metrics("SFO")
    print(result)

    # --------------------------------------------------------
    # Tool 2
    # --------------------------------------------------------

    print("\n2. calculate_driver_incentive(50, 'high')")
    print("-" * 70)

    result = calculate_driver_incentive(
        driver_count=50,
        severity_level="high",
    )
    print(result)

    # --------------------------------------------------------
    # Tool 3
    # --------------------------------------------------------

    print("\n3. trigger_surge_override('SFO', 1.8, ...)")
    print("-" * 70)

    result = trigger_surge_override(
        airport_code="SFO",
        new_multiplier=1.8,
        reason=(
            "High request volume with insufficient "
            "driver supply."
        ),
    )
    print(result)

    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    print("\n4. Invalid airport")
    print("-" * 70)

    result = execute_tool(
        "get_airport_metrics",
        {"airport_code": "ABC"},
    )
    print(result)

    print("\n5. Invalid severity")
    print("-" * 70)

    result = execute_tool(
        "calculate_driver_incentive",
        {
            "driver_count": 50,
            "severity_level": "extreme",
        },
    )
    print(result)

    print("\n6. Invalid surge multiplier")
    print("-" * 70)

    result = execute_tool(
        "trigger_surge_override",
        {
            "airport_code": "SFO",
            "new_multiplier": 4.0,
            "reason": (
                "High demand requires surge adjustment."
            ),
        },
    )
    print(result)

    print("\n7. Missing parameter")
    print("-" * 70)

    result = execute_tool(
        "calculate_driver_incentive",
        {
            "driver_count": 50,
        },
    )
    print(result)

    print("\n8. Unknown tool")
    print("-" * 70)

    result = execute_tool(
        "unknown_tool",
        {},
    )
    print(result)

    print("\n" + "=" * 70)
    print("TOOL TESTS COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_tool_tests()