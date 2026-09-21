"""
Document loader for the Airport Operations AI Copilot.

Loads synthetic airport policy documents from:
data/airport_policies/
"""

from pathlib import Path
import re


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory containing policy documents
POLICY_DIR = PROJECT_ROOT / "data" / "airport_policies"


def extract_metadata(content: str, file_name: str) -> dict:
    """
    Extract basic metadata from a policy document.

    Metadata is read from the document header.
    """

    metadata = {
        "file_name": file_name,
        "policy_id": None,
        "airport": None,
        "policy_type": None,
        "version": None,
        "effective_date": None,
    }

    patterns = {
        "policy_id": r"\*\*Policy ID:\*\*\s*(.+)",
        "airport": r"\*\*Airport:\*\*\s*(.+)",
        "policy_type": r"\*\*Policy Type:\*\*\s*(.+)",
        "version": r"\*\*Version:\*\*\s*(.+)",
        "effective_date": r"\*\*Effective Date:\*\*\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)

        if match:
            metadata[key] = match.group(1).strip()

    return metadata


def clean_document(content: str) -> str:
    """
    Perform basic cleaning on a policy document.

    Cleaning operations:
    - Normalize line endings
    - Remove excessive blank lines
    - Remove leading/trailing whitespace
    """

    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")

    # Remove trailing spaces from each line
    lines = [line.rstrip() for line in content.split("\n")]

    # Remove excessive consecutive blank lines
    cleaned_lines = []

    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank and previous_blank:
            continue

        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def load_documents(policy_dir: Path = POLICY_DIR) -> list[dict]:
    """
    Load all Markdown policy documents.

    Returns:
        List of dictionaries containing:
        - document_id
        - file_name
        - content
        - metadata
    """

    if not policy_dir.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {policy_dir}"
        )

    files = sorted(policy_dir.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"No Markdown policy documents found in: {policy_dir}"
        )

    documents = []

    for file_path in files:

        raw_content = file_path.read_text(
            encoding="utf-8"
        )

        cleaned_content = clean_document(raw_content)

        metadata = extract_metadata(
            cleaned_content,
            file_path.name,
        )

        document = {
            "document_id": file_path.stem,
            "file_name": file_path.name,
            "content": cleaned_content,
            "metadata": metadata,
        }

        documents.append(document)

    return documents


def print_document_summary(documents: list[dict]) -> None:
    """
    Print a readable summary of loaded policy documents.
    """

    print("=" * 70)
    print("AIRPORT POLICY DOCUMENT SUMMARY")
    print("=" * 70)

    print(f"Total documents loaded: {len(documents)}")
    print()

    for index, document in enumerate(documents, start=1):

        metadata = document["metadata"]

        print(f"{index}. {document['file_name']}")
        print(f"   Policy ID: {metadata['policy_id']}")
        print(f"   Airport: {metadata['airport']}")
        print(f"   Policy Type: {metadata['policy_type']}")
        print(f"   Version: {metadata['version']}")
        print(f"   Effective Date: {metadata['effective_date']}")
        print(f"   Characters: {len(document['content'])}")
        print()


if __name__ == "__main__":

    documents = load_documents()

    print_document_summary(documents)