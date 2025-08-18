import json
import re
from typing import Any, Iterable


def normalize_provision(text: str) -> str:
    """
    Normalize a provision string:
    - Convert Windows/Mac line endings to Unix to avoid inconsistencies.
    - Collapse all runs of whitespace (spaces, tabs, newlines) to a single space.
    - Trim leading/trailing whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse arbitrary whitespace (including newlines) into single spaces.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _to_label_list(labels: Any) -> list[str]:
    """
    Coerce 'label' field into a list of strings without changing semantics:
    - If it's already a list, return as-is.
    - If it's a single string, wrap in a list.
    - If it's some other iterable (e.g., set/tuple), cast to list.
    - Otherwise, return an empty list.
    """
    if isinstance(labels, list):
        return labels
    if isinstance(labels, str):
        return [labels]
    if isinstance(labels, Iterable):
        return list(labels)
    return []


def clean_ledgar(input_path: str, output_path: str) -> None:
    """
    Clean and normalize LEDGAR provisions and labels, writing corrected records
    to a new JSONL file. Skips empty/invalid lines.
    """
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed JSON lines

            provision = record.get("provision", "")
            if not str(provision).strip():
                continue  # skip records without a meaningful provision

            cleaned_provision = normalize_provision(str(provision))

            labels = _to_label_list(record.get("label", []))

            cleaned_record = {
                "provision": cleaned_provision,
                "label": labels,
                "source": record.get("source", ""),
            }

            fout.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")


# Function call
clean_ledgar("LEDGAR_2016-2019_clean.jsonl", "ledgar_cleaned.jsonl")
print("Cleaning complete.")
