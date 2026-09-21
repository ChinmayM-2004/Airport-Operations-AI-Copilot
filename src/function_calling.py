"""
Day 2 - Function Calling Workflow

This module connects Gemini with the airport operational tools.

Workflow:
    User Query
        ↓
    Gemini decides whether a tool is required
        ↓
    Function Call
        ↓
    Local Tool Execution
        ↓
    Tool Response
        ↓
    Gemini generates grounded final response

Day 2 tools:
1. get_airport_metrics
2. calculate_driver_incentive
3. trigger_surge_override

Note:
The surge override is a MOCK execution on Day 2.
Approval controls will be added on Day 4.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Allow imports when running:
# python src/function_calling.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools import execute_tool


# ============================================================
# Configuration
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-2.5-flash"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please add it to the project's .env file."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# Gemini Function Declarations
# ============================================================

GET_AIRPORT_METRICS_TOOL = types.FunctionDeclaration(
    name="get_airport_metrics",
    description=(
        "Get the latest operational telemetry for an airport. "
        "Use this tool whenever the user asks about current or latest "
        "airport operational conditions such as completion rate, ETA, "
        "active drivers, cancellations, queue size, surge multiplier, "
        "or request volume."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "airport_code": types.Schema(
                type=types.Type.STRING,
                description="Airport code. Supported values: SFO, LAX, JFK.",
            )
        },
        required=["airport_code"],
    ),
)


CALCULATE_DRIVER_INCENTIVE_TOOL = types.FunctionDeclaration(
    name="calculate_driver_incentive",
    description=(
        "Calculate a recommended driver incentive and estimated total "
        "cost based on the number of drivers and operational severity."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "driver_count": types.Schema(
                type=types.Type.INTEGER,
                description="Number of drivers receiving the incentive.",
            ),
            "severity_level": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Operational severity. Supported values: "
                    "low, medium, high, critical."
                ),
            ),
        },
        required=["driver_count", "severity_level"],
    ),
)


TRIGGER_SURGE_OVERRIDE_TOOL = types.FunctionDeclaration(
    name="trigger_surge_override",
    description=(
        "Trigger a MOCK surge multiplier override for an airport. "
        "This does not make a real production change. "
        "Day 2 approval controls are not enforced; approval workflow "
        "will be added on Day 4."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "airport_code": types.Schema(
                type=types.Type.STRING,
                description="Airport code. Supported values: SFO, LAX, JFK.",
            ),
            "new_multiplier": types.Schema(
                type=types.Type.NUMBER,
                description="New surge multiplier. Day 2 maximum is 3.0x.",
            ),
            "reason": types.Schema(
                type=types.Type.STRING,
                description="Reason for requesting the surge override.",
            ),
        },
        required=["airport_code", "new_multiplier", "reason"],
    ),
)


# ============================================================
# Tool Configuration
# ============================================================

TOOLS = types.Tool(
    function_declarations=[
        GET_AIRPORT_METRICS_TOOL,
        CALCULATE_DRIVER_INCENTIVE_TOOL,
        TRIGGER_SURGE_OVERRIDE_TOOL,
    ]
)


# ============================================================
# System Instructions
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are an Airport Operations AI Copilot.

Your job is to help investigate operational issues using the available
airport operational tools.

Available airports:
- SFO
- LAX
- JFK

Important rules:

1. NEVER invent current operational metrics.
   If the user asks for current/latest operational data, use
   get_airport_metrics.

2. Use calculate_driver_incentive when the user asks for a recommended
   driver incentive or estimated incentive cost.

3. Use trigger_surge_override when the user explicitly asks to trigger
   or simulate a surge multiplier override.

4. The surge override is a MOCK execution on Day 2.
   It does NOT change a real production system.

5. If required tool parameters are missing, ask the user for the
   missing information instead of inventing it.

6. If a tool returns an error, clearly report the error.

7. Clearly distinguish:
   - Data returned by operational tools
   - Your interpretation of that data

8. Do not claim that a production operational change occurred when
   using the Day 2 mock surge override.

9. Keep responses concise and operationally useful.
"""


# ============================================================
# Helper: Display Available Tools
# ============================================================

def show_available_tools():
    """
    Display the tools available to the Airport Operations Copilot.

    This is intentionally handled locally instead of calling Gemini,
    because the user is only asking for the tool list.
    """

    print("\n" + "=" * 70)
    print("AVAILABLE OPERATIONAL TOOLS")
    print("=" * 70)

    tools = [
        (
            "get_airport_metrics",
            "Get latest airport telemetry for SFO, LAX, or JFK.",
        ),
        (
            "calculate_driver_incentive",
            "Calculate recommended driver incentive and estimated cost.",
        ),
        (
            "trigger_surge_override",
            "Trigger a mock surge override. Approval is not enforced on Day 2.",
        ),
    ]

    for name, description in tools:
        print(f"\n{name}")
        print(f"  {description}")

    print("=" * 70)


# ============================================================
# Helper: Execute Function Call
# ============================================================

def execute_function_call(function_call):
    """
    Execute a Gemini-generated function call using the local tool layer.

    Parameters
    ----------
    function_call : Gemini function call object

    Returns
    -------
    dict
        Structured tool response.
    """

    tool_name = function_call.name

    try:
        arguments = dict(function_call.args or {})
    except Exception:
        arguments = {}

    print("\n" + "-" * 70)
    print("TOOL SELECTED")
    print("-" * 70)
    print(f"Tool: {tool_name}")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")

    result = execute_tool(tool_name, arguments)

    print("\nTOOL RESULT")
    print("-" * 70)
    print(json.dumps(result, indent=2))

    return result


# ============================================================
# Helper: Extract Function Calls
# ============================================================

def extract_function_calls(response):
    """
    Extract function calls from a Gemini response.

    Returns
    -------
    list
        List of function call objects.
    """

    function_calls = []

    if not response or not response.candidates:
        return function_calls

    for candidate in response.candidates:

        if not candidate.content:
            continue

        if not candidate.content.parts:
            continue

        for part in candidate.content.parts:

            if getattr(part, "function_call", None):
                function_calls.append(part.function_call)

    return function_calls


# ============================================================
# Helper: Gemini API Call
# ============================================================

def call_gemini(contents):
    """
    Call Gemini and gracefully handle API failures.

    This prevents quota/API errors from producing a long traceback
    during the Day 2 demo.
    """

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                tools=[TOOLS],
                temperature=0.1,
            ),
        )

        return response

    except Exception as exc:

        status_code = getattr(exc, "status_code", None)

        print("\n" + "=" * 70)
        print("GEMINI API ERROR")
        print("=" * 70)

        if status_code == 429:

            print(
                "Gemini request quota has been exhausted temporarily."
            )
            print(
                "Please wait for the quota window to reset before "
                "running another Gemini function-calling test."
            )

        else:

            print(f"API error: {exc}")

        print("=" * 70)

        return None


# ============================================================
# Main Agent Function
# ============================================================

def run_agent(user_query):
    """
    Run one user query through the Gemini function-calling workflow.

    Workflow:
        1. Send user query to Gemini.
        2. Gemini decides whether a tool is needed.
        3. Execute selected tool locally.
        4. Send tool result back to Gemini.
        5. Gemini produces final grounded response.

    Parameters
    ----------
    user_query : str
        User's operational question.

    Returns
    -------
    str
        Final assistant response.
    """

    print("\n" + "=" * 70)
    print("USER QUERY")
    print("=" * 70)
    print(user_query)

    # --------------------------------------------------------
    # Step 1: Initial Gemini request
    # --------------------------------------------------------

    user_content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=user_query)
        ],
    )

    response = call_gemini(user_content)

    if response is None:

        return (
            "I could not process this request because the Gemini API "
            "request failed. Please retry after the API issue or quota "
            "window is resolved."
        )

    # --------------------------------------------------------
    # Step 2: Check whether Gemini selected a tool
    # --------------------------------------------------------

    function_calls = extract_function_calls(response)

    if not function_calls:

        print("\n" + "-" * 70)
        print("NO TOOL REQUIRED")
        print("-" * 70)

        try:
            final_text = response.text

            if final_text:
                print("\nFINAL RESPONSE")
                print("-" * 70)
                print(final_text)
                return final_text

        except Exception:
            pass

        return "Gemini did not return a usable response."

    # --------------------------------------------------------
    # Step 3: Execute selected tools
    # --------------------------------------------------------

    tool_response_parts = []

    for function_call in function_calls:

        result = execute_function_call(function_call)

        tool_response_parts.append(
            types.Part.from_function_response(
                name=function_call.name,
                response=result,
            )
        )

    # --------------------------------------------------------
    # Step 4: Send tool result back to Gemini
    # --------------------------------------------------------

    model_function_call_content = types.Content(
        role="model",
        parts=response.candidates[0].content.parts,
    )

    tool_response_content = types.Content(
        role="user",
        parts=tool_response_parts,
    )

    follow_up_contents = [
        user_content,
        model_function_call_content,
        tool_response_content,
    ]

    final_response = call_gemini(follow_up_contents)

    if final_response is None:

        return (
            "The operational tool executed successfully, but Gemini "
            "could not generate the final response because the API "
            "request failed."
        )

    # --------------------------------------------------------
    # Step 5: Return final grounded response
    # --------------------------------------------------------

    try:

        final_text = final_response.text

        if final_text:

            print("\n" + "=" * 70)
            print("FINAL RESPONSE")
            print("=" * 70)
            print(final_text)

            return final_text

    except Exception:
        pass

    return "The tool executed successfully, but no final response was returned."


# ============================================================
# Day 2 Demo
# ============================================================

def run_demo():
    """
    Run the Day 2 function-calling demonstration.

    Tests 1-3 use Gemini because they demonstrate actual
    function-calling behavior.

    Test 4 is handled locally because asking for a static list
    of available tools does not require an LLM request.
    """

    print("\n")
    print("#" * 70)
    print("DAY 2 - FUNCTION CALLING DEMO")
    print("#" * 70)

    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    print("\n\nTEST 1 - Current SFO operational status")

    run_agent(
        "What's happening at SFO right now?"
    )

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    print("\n\nTEST 2 - Driver incentive calculation")

    run_agent(
        "We need incentives for 50 drivers at high severity. "
        "What incentive should we offer and what will it cost?"
    )

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    print("\n\nTEST 3 - Mock surge override")

    run_agent(
        "Trigger a surge override at SFO to 1.8x because request "
        "volume is high and driver supply is insufficient."
    )

    # --------------------------------------------------------
    # Test 4
    # --------------------------------------------------------

    print("\n\nTEST 4 - Available tools")

    show_available_tools()


# ============================================================
# Script Entry Point
# ============================================================

if __name__ == "__main__":
    run_demo()