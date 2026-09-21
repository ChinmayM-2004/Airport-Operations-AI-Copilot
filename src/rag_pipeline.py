import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from retriever import PolicyRetriever


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash"


# ============================================================
# VALIDATE API KEY
# ============================================================

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Create a .env file in the project root and add:\n"
        "GEMINI_API_KEY=your_api_key"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# RAG PROMPT
# ============================================================

RAG_SYSTEM_INSTRUCTIONS = """
You are an Airport Operations AI Copilot.

Your job is to answer operational questions using ONLY
the supplied airport policy context.

IMPORTANT RULES:

1. Use only the provided policy context.
2. Do not invent policy rules, limits, approvals, or exceptions.
3. If the context does not contain enough information, say:
   "Insufficient information in the available policy documents."
4. When answering, clearly distinguish:
   - What the policy allows
   - What the policy does not allow
   - What approval is required, if applicable
5. Always identify the source policy document.
6. If multiple policy documents are relevant, mention all
   relevant sources.
7. Keep the answer concise and operationally useful.
8. Do not use outside knowledge.
9. Do not make assumptions about airport rules.
"""


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_rag_prompt(query, retrieved_results):
    """
    Build a grounded prompt using the retrieved policy chunks.

    Args:
        query (str): User's question.
        retrieved_results (list): Retrieved policy chunks.

    Returns:
        str: Prompt sent to Gemini.
    """

    if not retrieved_results:
        context = (
            "No relevant policy documents were retrieved."
        )

    else:
        context_parts = []

        for result in retrieved_results:

            context_parts.append(
                f"""
SOURCE DOCUMENT:
{result['file_name']}

POLICY ID:
{result['policy_id']}

AIRPORT:
{result['airport']}

POLICY TYPE:
{result['policy_type']}

SECTION:
{result['section']}

POLICY TEXT:
{result['text']}
"""
            )

        context = "\n".join(context_parts)

    prompt = f"""
{RAG_SYSTEM_INSTRUCTIONS}

============================================================
RETRIEVED POLICY CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{query}

============================================================
RESPONSE FORMAT
============================================================

Decision:
[Allowed / Not Allowed / Approval Required /
Insufficient Information]

Answer:
[Give a concise answer based only on the policy context.]

Reason:
[Explain the relevant policy rule.]

Source:
[Name the policy document used.]
"""

    return prompt


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, top_k=3):
    """
    Retrieve relevant policy chunks and generate a
    grounded answer using Gemini.

    Args:
        query (str): User's policy question.
        top_k (int): Number of policy chunks to retrieve.

    Returns:
        dict: Generated answer and retrieval information.
    """

    # --------------------------------------------------------
    # Retrieve policy context
    # --------------------------------------------------------

    retriever = PolicyRetriever()

    retrieved_results = retriever.search(
        query,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Build grounded prompt
    # --------------------------------------------------------

    prompt = build_rag_prompt(
        query,
        retrieved_results
    )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text.strip()

    return {
        "query": query,
        "answer": answer,
        "retrieved_results": retrieved_results,
    }


# ============================================================
# DISPLAY RAG RESPONSE
# ============================================================

def display_response(result):
    """
    Display the final RAG response and source information.
    """

    print("\n" + "=" * 70)
    print("AIRPORT OPERATIONS AI COPILOT")
    print("=" * 70)

    print(
        f"\nUser Question:\n{result['query']}"
    )

    print("\n" + "-" * 70)

    print("GROUNDED RESPONSE")
    print("-" * 70)

    print(result["answer"])

    print("\n" + "-" * 70)

    print("RETRIEVED SOURCES")
    print("-" * 70)

    for retrieved in result["retrieved_results"]:

        print(
            f"\nRank: {retrieved['rank']}"
        )

        print(
            f"Similarity: "
            f"{retrieved['score']:.4f}"
        )

        print(
            f"Source: "
            f"{retrieved['file_name']}"
        )

        print(
            f"Policy ID: "
            f"{retrieved['policy_id']}"
        )

        print(
            f"Section: "
            f"{retrieved['section']}"
        )

    print("\n" + "=" * 70)


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    "What is the maximum surge multiplier allowed at SFO?",

    "Can drivers abandon the airport queue at SFO?",

    "What approval is required before increasing surge at SFO?",

    "What is the maximum surge multiplier allowed at LAX?",

    "What is the maximum surge multiplier allowed at JFK?",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Run one detailed test
    # --------------------------------------------------------

    query = TEST_QUESTIONS[0]

    result = generate_answer(
        query,
        top_k=3
    )

    display_response(result)


if __name__ == "__main__":
    main()