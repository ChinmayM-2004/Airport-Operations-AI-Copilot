"""
Day 3 - Policy & Compliance Agent

Responsibilities:
- Use the Day 1 RAG system to retrieve relevant airport policies.
- Analyze the operational situation from the Investigator.
- Identify applicable policies and restrictions.
- Determine whether approval may be required.
- Store policy findings in the shared agent state.

This agent does NOT execute operational actions.

Important:
- Gemini is used for policy interpretation when available.
- A deterministic fallback is used when Gemini is unavailable.
- The deterministic fallback relies only on retrieved RAG evidence.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from src.agents.state import update_state, add_error
from src.retriever import PolicyRetriever


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-3.6-flash"

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None


# ============================================================
# POLICY RETRIEVER
# ============================================================

retriever = PolicyRetriever()


# ============================================================
# POLICY SEARCH
# ============================================================

def search_relevant_policies(
    airport_code: str,
    contributing_factors: List[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Search the Day 1 RAG knowledge base for policies relevant
    to the current operational situation.
    """

    airport_code = airport_code.upper()

    factor_text = " ".join(contributing_factors)

    query = (
        f"{airport_code} airport operational policy "
        f"driver supply request volume queue surge "
        f"restrictions approval requirements "
        f"{factor_text}"
    )

    results = retriever.search(
        query=query,
        top_k=top_k,
    )

    return results


# ============================================================
# POLICY RESULT NORMALIZATION
# ============================================================

def normalize_policy_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize a Day 1 retriever result.

    The actual PolicyRetriever returns:
        file_name
        policy_id
        section
        text
        score
        ...

    We expose `source` as an alias of `file_name` so the
    downstream Policy Agent and Streamlit UI can use a
    consistent field name.
    """

    normalized = dict(result)

    normalized["source"] = (
        result.get("file_name")
        or result.get("source")
        or result.get("document")
        or result.get("filename")
        or "unknown"
    )

    normalized["policy_id"] = (
        result.get("policy_id")
        or "unknown"
    )

    normalized["section"] = (
        result.get("section")
        or "unknown"
    )

    normalized["text"] = (
        result.get("text")
        or result.get("content")
        or result.get("chunk")
        or ""
    )

    return normalized


def normalize_policy_results(
    policy_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normalize all retrieved policy results.
    """

    return [
        normalize_policy_result(result)
        for result in policy_results
    ]


# ============================================================
# SURGE REQUEST EXTRACTION
# ============================================================

def extract_requested_surge_multiplier(
    user_query: str,
) -> Optional[float]:
    """
    Extract an explicitly requested surge multiplier.

    Examples:
        "increase surge to 1.5x" -> 1.5
        "raise surge to 1.8x"    -> 1.8
        "set surge at 2x"        -> 2.0

    Returns None when the user did not explicitly request
    a surge multiplier.
    """

    if not user_query:
        return None

    patterns = [
        r"(?:surge|multiplier)[^\d]{0,30}(\d+(?:\.\d+)?)\s*x",
        r"(\d+(?:\.\d+)?)\s*x[^\d]{0,30}(?:surge|multiplier)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            user_query,
            flags=re.IGNORECASE,
        )

        if match:

            try:
                return float(
                    match.group(1)
                )

            except (TypeError, ValueError):
                return None

    return None


# ============================================================
# POLICY SURGE THRESHOLD EXTRACTION
# ============================================================

def extract_policy_max_surge_from_results(
    policy_results: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Extract the surge approval threshold from retrieved
    policy evidence.

    The actual SFO policy retrieved by Day 1 currently uses
    wording such as:

        "Any increase above 1.5x requires approval..."

    Therefore, this function understands both:

        "maximum surge multiplier ... 1.5x"

    and:

        "increase above 1.5x requires approval"

    The value is always extracted from retrieved policy text.
    No airport threshold is hard-coded here.
    """

    normalized_results = normalize_policy_results(
        policy_results
    )

    for result in normalized_results:

        text = str(
            result.get("text", "")
        ).strip()

        if not text:
            continue

        patterns = [

            # ------------------------------------------------
            # Pattern 1
            # "maximum standard surge multiplier ... 1.5x"
            # ------------------------------------------------

            r"maximum\s+standard\s+surge\s+multiplier"
            r"[\s\S]{0,150}?"
            r"(\d+(?:\.\d+)?)\s*x",

            # ------------------------------------------------
            # Pattern 2
            # "maximum surge multiplier ... 1.5x"
            # ------------------------------------------------

            r"maximum\s+surge(?:\s+multiplier)?"
            r"[\s\S]{0,150}?"
            r"(\d+(?:\.\d+)?)\s*x",

            # ------------------------------------------------
            # Pattern 3
            # "max surge ... 1.5x"
            # ------------------------------------------------

            r"max(?:imum)?\s+surge(?:\s+multiplier)?"
            r"[\s\S]{0,150}?"
            r"(\d+(?:\.\d+)?)\s*x",

            # ------------------------------------------------
            # Pattern 4
            # "up to 1.5x"
            # ------------------------------------------------

            r"(?:up\s+to|maximum\s+of)"
            r"\s+(\d+(?:\.\d+)?)\s*x",

            # ------------------------------------------------
            # Pattern 5
            # "above 1.5x requires approval"
            #
            # This matches the ACTUAL SFO policy chunk.
            # ------------------------------------------------

            r"(?:above|over|exceeding)"
            r"\s+(\d+(?:\.\d+)?)\s*x"
            r"[\s\S]{0,100}?"
            r"(?:requires?|need(?:s)?)"
            r"[\s\S]{0,100}?"
            r"approval",

            # ------------------------------------------------
            # Pattern 6
            # "increase above 1.5x requires approval"
            # ------------------------------------------------

            r"increase"
            r"[\s\S]{0,50}?"
            r"(?:above|over|exceeding)"
            r"\s+(\d+(?:\.\d+)?)\s*x"
            r"[\s\S]{0,100}?"
            r"approval",

            # ------------------------------------------------
            # Pattern 7
            # "above 1.5x ... Airport Operations Manager"
            #
            # Useful if the word "requires" is absent.
            # ------------------------------------------------

            r"(?:above|over|exceeding)"
            r"\s+(\d+(?:\.\d+)?)\s*x"
            r"[\s\S]{0,150}?"
            r"Airport\s+Operations\s+Manager",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:

                try:

                    value = float(
                        match.group(1)
                    )

                    if 1.0 <= value <= 10.0:
                        return value

                except (TypeError, ValueError):
                    continue

    return None


# ============================================================
# DETERMINISTIC POLICY FALLBACK
# ============================================================

def deterministic_policy_analysis(
    airport_code: str,
    operational_data: Dict[str, Any],
    contributing_factors: List[str],
    policy_results: List[Dict[str, Any]],
    user_query: str = "",
) -> Dict[str, Any]:
    """
    Deterministic fallback for policy interpretation.

    Used when Gemini is unavailable.

    The fallback:
    - uses only retrieved RAG evidence
    - extracts a policy threshold from retrieved text
    - compares an explicit surge request against that threshold
    - determines whether additional policy approval is required
    - never executes an action
    """

    airport_code = airport_code.upper()

    normalized_results = normalize_policy_results(
        policy_results
    )

    # --------------------------------------------------------
    # Build applicable policy list
    # --------------------------------------------------------

    applicable_policies = []

    for result in normalized_results:

        source = result.get(
            "source",
            "unknown",
        )

        policy_id = result.get(
            "policy_id",
            "unknown",
        )

        section = result.get(
            "section",
            "unknown",
        )

        text = str(
            result.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        # Convert multiline chunk to concise display text.
        rule = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if len(rule) > 300:
            rule = rule[:297] + "..."

        applicable_policies.append(
            {
                "source": source,
                "policy_id": policy_id,
                "section": section,
                "relevant_rule": rule,
            }
        )

    if not applicable_policies:

        return {
            "policy_decision": "Insufficient Information",
            "applicable_policies": [],
            "approval_required": "unknown",
            "restrictions": [],
            "compliance_findings": [
                "No usable policy evidence was available."
            ],
            "reasoning": (
                "The policy retrieval step did not provide "
                "enough evidence to determine compliance."
            ),
        }

    # --------------------------------------------------------
    # Extract requested surge
    # --------------------------------------------------------

    requested_multiplier = (
        extract_requested_surge_multiplier(
            user_query
        )
    )

    # --------------------------------------------------------
    # Extract policy threshold
    # --------------------------------------------------------

    policy_max_surge = (
        extract_policy_max_surge_from_results(
            policy_results
        )
    )

    # --------------------------------------------------------
    # Explicit surge request
    # --------------------------------------------------------

    if requested_multiplier is not None:

        # ----------------------------------------------------
        # No clear policy threshold
        # ----------------------------------------------------

        if policy_max_surge is None:

            return {
                "policy_decision": "Insufficient Information",
                "applicable_policies": applicable_policies,
                "approval_required": "unknown",
                "restrictions": [],
                "compliance_findings": [
                    (
                        "An explicit surge change was requested, "
                        "but the retrieved policy evidence does "
                        "not contain a clear surge limit."
                    )
                ],
                "reasoning": (
                    "Relevant policy documents were retrieved, "
                    "but a surge threshold could not be determined "
                    "reliably from their retrieved text."
                ),
            }

        # ----------------------------------------------------
        # Request is within policy threshold.
        #
        # IMPORTANT:
        # If policy says "above 1.5x requires approval",
        # exactly 1.5x is allowed without that approval.
        # ----------------------------------------------------

        if requested_multiplier <= policy_max_surge:

            return {
                "policy_decision": "Allowed",
                "applicable_policies": applicable_policies,
                "approval_required": False,
                "restrictions": [
                    (
                        f"Surge must remain at or below "
                        f"{policy_max_surge:.2f}x without "
                        f"additional policy approval."
                    )
                ],
                "compliance_findings": [
                    (
                        f"Requested surge "
                        f"{requested_multiplier:.2f}x is within "
                        f"the retrieved {airport_code} policy "
                        f"limit of {policy_max_surge:.2f}x."
                    )
                ],
                "reasoning": (
                    f"The retrieved policy evidence permits "
                    f"standard surge up to "
                    f"{policy_max_surge:.2f}x without additional "
                    f"approval. The requested "
                    f"{requested_multiplier:.2f}x is within "
                    f"that limit."
                ),
            }

        # ----------------------------------------------------
        # Request exceeds policy threshold.
        # ----------------------------------------------------

        return {
            "policy_decision": "Allowed",
            "applicable_policies": applicable_policies,
            "approval_required": True,
            "restrictions": [
                (
                    f"Surge above {policy_max_surge:.2f}x "
                    f"requires the approval specified by "
                    f"the retrieved airport policy."
                )
            ],
            "compliance_findings": [
                (
                    f"Requested surge "
                    f"{requested_multiplier:.2f}x exceeds "
                    f"the retrieved policy limit of "
                    f"{policy_max_surge:.2f}x."
                ),
                (
                    "The requested change requires the "
                    "policy-defined approval before execution."
                ),
            ],
            "reasoning": (
                f"The requested surge of "
                f"{requested_multiplier:.2f}x exceeds the "
                f"retrieved {airport_code} policy limit of "
                f"{policy_max_surge:.2f}x. Additional approval "
                f"is therefore required before execution."
            ),
        }

    # --------------------------------------------------------
    # No explicit surge request
    # --------------------------------------------------------

    current_surge = operational_data.get(
        "surge_multiplier"
    )

    if (
        current_surge is not None
        and policy_max_surge is not None
    ):

        try:

            current_surge = float(
                current_surge
            )

            if current_surge <= policy_max_surge:

                return {
                    "policy_decision": "Allowed",
                    "applicable_policies": applicable_policies,
                    "approval_required": False,
                    "restrictions": [
                        (
                            f"Surge should remain at or below "
                            f"{policy_max_surge:.2f}x without "
                            f"additional policy approval."
                        )
                    ],
                    "compliance_findings": [
                        (
                            f"Current surge "
                            f"{current_surge:.2f}x is within "
                            f"the retrieved policy limit of "
                            f"{policy_max_surge:.2f}x."
                        )
                    ],
                    "reasoning": (
                        f"Current surge is "
                        f"{current_surge:.2f}x, which is within "
                        f"the retrieved {airport_code} policy "
                        f"limit of {policy_max_surge:.2f}x."
                    ),
                }

            return {
                "policy_decision": "Allowed",
                "applicable_policies": applicable_policies,
                "approval_required": True,
                "restrictions": [
                    (
                        f"Current surge "
                        f"{current_surge:.2f}x is above the "
                        f"retrieved policy limit of "
                        f"{policy_max_surge:.2f}x and requires "
                        f"the applicable approval."
                    )
                ],
                "compliance_findings": [
                    (
                        f"Current surge "
                        f"{current_surge:.2f}x exceeds the "
                        f"retrieved policy limit of "
                        f"{policy_max_surge:.2f}x."
                    )
                ],
                "reasoning": (
                    f"Current surge exceeds the retrieved "
                    f"{airport_code} policy limit. The "
                    f"applicable approval requirement must "
                    f"be satisfied."
                ),
            }

        except (TypeError, ValueError):
            pass

    # --------------------------------------------------------
    # General operational question
    # --------------------------------------------------------

    return {
        "policy_decision": "Insufficient Information",
        "applicable_policies": applicable_policies,
        "approval_required": "unknown",
        "restrictions": [],
        "compliance_findings": [
            (
                "Relevant policy evidence was retrieved, "
                "but the current request does not contain "
                "enough specific information to determine "
                "a policy decision."
            )
        ],
        "reasoning": (
            "Policy evidence was retrieved successfully. "
            "A specific compliance decision requires a "
            "clearly defined operational action or condition."
        ),
    }


# ============================================================
# POLICY ANALYSIS WITH GEMINI
# ============================================================

def analyze_policy_with_llm(
    airport_code: str,
    operational_data: Dict[str, Any],
    contributing_factors: List[str],
    policy_results: List[Dict[str, Any]],
    user_query: str = "",
) -> Dict[str, Any]:
    """
    Ask Gemini to interpret retrieved policy evidence.

    If Gemini is unavailable, invalid, or quota-limited,
    deterministic policy analysis is used instead.
    """

    policy_results = normalize_policy_results(
        policy_results
    )

    context_parts = []

    for i, result in enumerate(
        policy_results,
        start=1,
    ):

        context_parts.append(
            f"""
POLICY RESULT {i}

Source: {result.get("source", "unknown")}
Policy ID: {result.get("policy_id", "unknown")}
Section: {result.get("section", "unknown")}
Similarity: {result.get("score", "unknown")}

Content:
{result.get("text", "")}
"""
        )

    policy_context = "\n".join(
        context_parts
    )

    prompt = f"""
You are the Policy & Compliance Agent for an airport
operations AI copilot.

Your job is to analyze the operational situation against
the retrieved company/airport policies.

AIRPORT:
{airport_code}

USER REQUEST:
{user_query}

CURRENT OPERATIONAL DATA:
{json.dumps(operational_data, indent=2)}

CONTRIBUTING FACTORS:
{json.dumps(contributing_factors, indent=2)}

RETRIEVED POLICY EVIDENCE:
{policy_context}

IMPORTANT RULES:

1. Use ONLY the retrieved policy evidence for policy claims.
2. Do not invent policies, thresholds, approvals, or restrictions.
3. Do not invent operational metrics.
4. If the retrieved evidence is insufficient, say so.
5. Distinguish between:
   - what the policy explicitly states
   - what cannot be determined from the available policy evidence
6. Do not execute any operational action.
7. Do not recommend a final operational intervention yet.
8. Focus on compliance and approval requirements.
9. If an explicit surge multiplier is requested, compare it
   directly against the retrieved policy threshold.
10. If the policy says that values ABOVE a threshold require
    approval, the threshold itself does not require that
    additional approval.
11. If the requested multiplier exceeds the stated threshold,
    follow the approval requirement explicitly stated by the
    retrieved policy.
12. Return valid JSON only.

Return this structure:

{{
    "policy_decision": "Allowed" | "Not Allowed" | "Insufficient Information",
    "applicable_policies": [
        {{
            "source": "policy filename",
            "policy_id": "policy ID",
            "section": "section",
            "relevant_rule": "short description"
        }}
    ],
    "approval_required": true | false | "unknown",
    "restrictions": [
        "restriction 1"
    ],
    "compliance_findings": [
        "finding 1"
    ],
    "reasoning": "short explanation"
}}
"""

    # --------------------------------------------------------
    # No Gemini client
    # --------------------------------------------------------

    if client is None:

        print(
            "\nGemini client unavailable."
        )

        print(
            "Using deterministic policy fallback."
        )

        return deterministic_policy_analysis(
            airport_code=airport_code,
            operational_data=operational_data,
            contributing_factors=contributing_factors,
            policy_results=policy_results,
            user_query=user_query,
        )

    # --------------------------------------------------------
    # Gemini request
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        text = (
            response.text or ""
        ).strip()

        if not text:

            raise ValueError(
                "Gemini returned an empty response."
            )

        # Remove Markdown JSON fences.
        if text.startswith("```"):

            text = text.replace(
                "```json",
                "",
            )

            text = text.replace(
                "```",
                "",
            )

            text = text.strip()

        result = json.loads(
            text
        )

        valid_decisions = {
            "Allowed",
            "Not Allowed",
            "Insufficient Information",
        }

        if result.get(
            "policy_decision"
        ) not in valid_decisions:

            raise ValueError(
                "Gemini returned an invalid policy_decision."
            )

        if "approval_required" not in result:

            raise ValueError(
                "Gemini response is missing "
                "approval_required."
            )

        result.setdefault(
            "applicable_policies",
            [],
        )

        result.setdefault(
            "restrictions",
            [],
        )

        result.setdefault(
            "compliance_findings",
            [],
        )

        result.setdefault(
            "reasoning",
            "",
        )

        return result

    except Exception as exc:

        print(
            f"\nGemini policy analysis unavailable: {exc}"
        )

        print(
            "Using deterministic policy fallback "
            "based on retrieved RAG evidence."
        )

        fallback = deterministic_policy_analysis(
            airport_code=airport_code,
            operational_data=operational_data,
            contributing_factors=contributing_factors,
            policy_results=policy_results,
            user_query=user_query,
        )

        fallback.setdefault(
            "compliance_findings",
            [],
        )

        fallback["compliance_findings"].insert(
            0,
            (
                "Gemini policy interpretation was unavailable; "
                "the decision was derived deterministically "
                "from the retrieved RAG evidence."
            ),
        )

        return fallback


# ============================================================
# MAIN POLICY AGENT
# ============================================================

def policy_agent(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the Policy & Compliance Agent.

    Input:
        Shared Day 3 agent state.

    Output:
        Updated shared state.
    """

    airport_code = state.get(
        "detected_airport"
    )

    if not airport_code:

        add_error(
            state,
            "policy_agent",
            (
                "Policy Agent could not determine "
                "the airport from the Investigator state."
            ),
        )

        return state

    operational_data = state.get(
        "operational_data",
        {},
    )

    contributing_factors = state.get(
        "contributing_factors",
        [],
    )

    user_query = state.get(
        "user_query",
        "",
    )

    print("\n" + "=" * 70)
    print("POLICY & COMPLIANCE AGENT")
    print("=" * 70)

    print(
        f"Airport: {airport_code}"
    )

    print(
        "\nSearching policy knowledge base..."
    )

    # --------------------------------------------------------
    # Retrieve policy evidence
    # --------------------------------------------------------

    try:

        policy_results = search_relevant_policies(
            airport_code=airport_code,
            contributing_factors=contributing_factors,
            top_k=3,
        )

        policy_results = normalize_policy_results(
            policy_results
        )

    except Exception as exc:

        print(
            f"\nPolicy retrieval failed: {exc}"
        )

        add_error(
            state,
            "policy_agent",
            f"Policy retrieval failed: {exc}",
        )

        return state

    if not policy_results:

        print(
            "\nNo relevant policy documents were retrieved."
        )

        return update_state(
            state,
            policy_results=[],
            policy_decision="Insufficient Information",
            approval_required="unknown",
        )

    print(
        f"\nRetrieved {len(policy_results)} policy results:"
    )

    for i, result in enumerate(
        policy_results,
        start=1,
    ):

        print(
            f"{i}. "
            f"{result.get('source', 'unknown')} "
            f"| "
            f"{result.get('policy_id', 'unknown')} "
            f"| "
            f"{result.get('section', 'unknown')} "
            f"| "
            f"score={result.get('score', 'unknown')}"
        )

    # --------------------------------------------------------
    # Analyze retrieved policy evidence
    # --------------------------------------------------------

    print(
        "\nAnalyzing policy compliance..."
    )

    analysis = analyze_policy_with_llm(
        airport_code=airport_code,
        operational_data=operational_data,
        contributing_factors=contributing_factors,
        policy_results=policy_results,
        user_query=user_query,
    )

    policy_decision = analysis.get(
        "policy_decision",
        "Insufficient Information",
    )

    approval_required = analysis.get(
        "approval_required",
        "unknown",
    )

    compliance_findings = analysis.get(
        "compliance_findings",
        [],
    )

    restrictions = analysis.get(
        "restrictions",
        [],
    )

    reasoning = analysis.get(
        "reasoning",
        "",
    )

    applicable_policies = analysis.get(
        "applicable_policies",
        [],
    )

    # --------------------------------------------------------
    # Build concise state finding
    # --------------------------------------------------------

    state_finding = (
        f"Policy decision for {airport_code}: "
        f"{policy_decision}. "
        f"Approval required: {approval_required}."
    )

    if reasoning:

        state_finding += (
            f" {reasoning}"
        )

    # --------------------------------------------------------
    # Update shared state
    # --------------------------------------------------------

    updated_state = update_state(
        state,
        policy_results=policy_results,
        policy_decision=policy_decision,
        approval_required=approval_required,
    )

    updated_state["policy_findings"] = (
        compliance_findings
    )

    updated_state["policy_restrictions"] = (
        restrictions
    )

    updated_state["applicable_policies"] = (
        applicable_policies
    )

    updated_state["policy_reasoning"] = (
        reasoning
    )

    findings = updated_state.get(
        "investigator_findings",
        [],
    )

    findings.append(
        state_finding
    )

    updated_state["investigator_findings"] = (
        findings
    )

    return updated_state


# ============================================================
# DISPLAY HELPER
# ============================================================

def print_policy_results(
    state: Dict[str, Any],
) -> None:
    """
    Print the most important Policy Agent results.
    """

    print("\nPolicy decision:")
    print("-" * 70)

    print(
        state.get(
            "policy_decision"
        )
    )

    print("\nApproval required:")
    print("-" * 70)

    print(
        state.get(
            "approval_required"
        )
    )

    print("\nApplicable policies:")
    print("-" * 70)

    policies = state.get(
        "applicable_policies",
        [],
    )

    if not policies:
        print(
            "None identified."
        )

    for policy in policies:

        print(
            f"- {policy.get('source', 'unknown')} | "
            f"{policy.get('policy_id', 'unknown')} | "
            f"{policy.get('section', 'unknown')}"
        )

        rule = policy.get(
            "relevant_rule"
        )

        if rule:

            print(
                f"  Rule: {rule}"
            )

    print("\nRestrictions:")
    print("-" * 70)

    restrictions = state.get(
        "policy_restrictions",
        [],
    )

    if not restrictions:

        print(
            "None identified."
        )

    for restriction in restrictions:

        print(
            f"- {restriction}"
        )

    print("\nCompliance findings:")
    print("-" * 70)

    findings = state.get(
        "policy_findings",
        [],
    )

    if not findings:

        print("None.")

    for finding in findings:

        print(
            f"- {finding}"
        )

    print("\nPolicy reasoning:")
    print("-" * 70)

    print(
        state.get(
            "policy_reasoning",
            "",
        )
    )


# ============================================================
# STANDALONE TEST
# ============================================================

def run_test() -> None:
    """
    Run the Policy Agent independently using the same type
    of state produced by the Investigator.
    """

    from src.agents.state import create_initial_state
    from src.agents.investigator import investigate

    print("\n" + "=" * 70)
    print("POLICY AGENT TEST")
    print("=" * 70)

    state = create_initial_state(
        "What's happening at SFO right now?"
    )

    # Run Investigator first so the Policy Agent receives
    # realistic operational state.
    state = investigate(
        state
    )

    if state.get("errors"):

        print(
            "\nInvestigator errors:"
        )

        for error in state["errors"]:

            print(
                f"- {error}"
            )

    state = policy_agent(
        state
    )

    print_policy_results(
        state
    )

    print("\n" + "=" * 70)
    print("POLICY AGENT TEST COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_test()