import json
from pathlib import Path
from typing import List, Tuple, Union


def normalize_text(text: str) -> str:
    """
    Normalize all line endings to Unix style.
    Prevents index misalignment from inconsistent newlines.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_spans(raw_spans: Union[List[int], List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """
    Convert raw span lists into tuples.

    Handles two formats:
    - Flat list of integers [s1, e1, s2, e2, ...]
    - List of [start, end] pairs
    """
    if not raw_spans:
        return []

    # Case: flat integer list
    if isinstance(raw_spans[0], int):
        it = iter(raw_spans)
        return list(zip(it, it))

    # Case: already (start, end) style
    return [(start, end) for start, end in raw_spans]


def clean_contractnli(input_file: str, output_file: str) -> None:
    """
    Normalize text and repair span annotations in ContractNLI JSON format.

    Steps:
    - Normalize document text newlines
    - Re-parse and validate span indices
    - If invalid, attempt to realign using 'hypothesis' text
    - Output cleaned JSON with consistent formatting
    """
    data = json.loads(Path(input_file).read_text(encoding="utf-8"))
    docs = data.get("documents", [])
    labels = data.get("labels", {})

    for doc in docs:
        # Normalize text
        text = normalize_text(doc.get("text", ""))
        doc["text"] = text

        # Extract annotations container
        ann_sets = doc.get("annotation_sets", {})
        annotations = {}

        if isinstance(ann_sets, dict):
            annotations = ann_sets.get("annotations", {})
        else:
            for entry in ann_sets:
                if isinstance(entry, dict) and "annotations" in entry:
                    annotations = entry["annotations"]
                    break

        # Fix spans inside annotations
        for meta in annotations.values():
            raw_spans = meta.get("spans", [])
            spans = parse_spans(raw_spans)
            valid_spans = []
            hypothesis = meta.get("hypothesis", "")

            for start, end in spans:
                if 0 <= start < end <= len(text):
                    valid_spans.append((start, end))
                else:
                    # fallback: try to align hypothesis text if misaligned
                    idx = text.find(hypothesis)
                    if idx != -1:
                        valid_spans.append((idx, idx + len(hypothesis)))

            # Reformat spans into original style
            if raw_spans and isinstance(raw_spans[0], int):
                meta["spans"] = [i for span in valid_spans for i in span]
            else:
                meta["spans"] = valid_spans

    # Save cleaned dataset
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"documents": docs, "labels": labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Function call
if __name__ == "__main__":
    clean_contractnli("train.json", "clean_train.json")
    clean_contractnli("dev.json", "clean_dev.json")
    clean_contractnli("test.json", "clean_test.json")
    print("Cleaning complete.")