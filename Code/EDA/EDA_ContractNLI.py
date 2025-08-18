import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import pandas as pd

SPLITS = {
    "train": "clean_train.json",
    "dev": "clean_dev.json",
    "test": "clean_test.json",
}

PERCENTILES = [10, 25, 50, 75, 90]


def _read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file with UTF-8 and return its parsed dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _get_annotations_container(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return the annotations dict from a ContractNLI document
    """
    ann_sets = doc.get("annotation_sets", {})
    if isinstance(ann_sets, dict):
        return ann_sets.get("annotations", {}) or {}
    # Fallback for list-style containers
    if isinstance(ann_sets, Iterable):
        for entry in ann_sets:
            if isinstance(entry, dict) and "annotations" in entry:
                return entry.get("annotations", {}) or {}
    return {}


def split_stats(split_name: str, path: str) -> Tuple[Dict[str, Any], Counter]:
    """
    Print basic statistics for a split and return.
    Statistics include document length percentiles and hypotheses-per-doc percentiles.
    """
    data = _read_json(path)
    docs = data.get("documents", [])
    total = len(docs)

    # Document lengths (characters)
    doc_lengths = [len(doc.get("text", "")) for doc in docs]
    p10, p25, p50, p75, p90 = np.percentile(doc_lengths, PERCENTILES)

    # Hypotheses count per document and overall label distribution
    hypo_counts: List[int] = []
    label_counts: Counter = Counter()

    for doc in docs:
        annotations = _get_annotations_container(doc)

        hypo_counts.append(len(annotations))

        for meta in annotations.values():
            # 'choice' expected to be one of {"Entailment", "NotMentioned", "Contradiction"}
            label_counts[meta["choice"]] += 1

    h10, h25, h50, h75, h90 = np.percentile(hypo_counts, PERCENTILES)

    print(f"\n=== {split_name.upper()} SPLIT ===")
    print(f"Total documents: {total}")
    print(
        "Document length percentiles (chars): "
        f"10th={p10:.0f}, 25th={p25:.0f}, 50th={p50:.0f}, 75th={p75:.0f}, 90th={p90:.0f}"
    )
    print(
        "Hypotheses per doc percentiles:      "
        f"10th={h10:.0f}, 25th={h25:.0f}, 50th={h50:.0f}, 75th={h75:.0f}, 90th={h90:.0f}"
    )
    print(f"Overall label distribution: {dict(label_counts)}")

    return data, label_counts


def template_stats(data: Dict[str, Any]) -> None:
    """
    Print hypothesis/template length percentiles and per-hypothesis label distribution
    using the dataset.
    """
    labels = data.get("labels", {})
    hyp_texts = [meta["hypothesis"] for meta in labels.values()]
    lengths = [len(h) for h in hyp_texts]
    percentiles = np.percentile(lengths, PERCENTILES)

    print("\nTemplate length percentiles (chars):")
    print(
        f" 10th={percentiles[0]:.1f}, 25th={percentiles[1]:.1f}, 50th={percentiles[2]:.1f}, "
        f"75th={percentiles[3]:.1f}, 90th={percentiles[4]:.1f}"
    )
    print(f"Max template length: {max(lengths)}\n")

    # Per-hypothesis label distribution
    dist: Dict[str, Counter] = {hid: Counter() for hid in labels}

    for doc in data["documents"]:
        annotations = _get_annotations_container(doc)
        for hid, meta in annotations.items():
            dist[hid][meta["choice"]] += 1

    rows = []
    for hid, ctr in dist.items():
        total = sum(ctr.values())
        rows.append(
            {
                "hyp_id": hid,
                "Entailment": ctr["Entailment"],
                "NotMentioned": ctr["NotMentioned"],
                "Contradiction": ctr["Contradiction"],
                "Total": total,
            }
        )

    df = pd.DataFrame(rows).set_index("hyp_id")
    print("Per-hypothesis label distribution:")
    print(df.sort_values("Total", ascending=False))

    df["Entailment_rate"] = df["Entailment"] / df["Total"]
    print("\nTop 5 hypotheses by Entailment rate:")
    print(df.nlargest(5, "Entailment_rate")[["Entailment_rate"]])


def main() -> None:
    print("ContractNLI EDA\n----------------")

    # Split-level EDA
    train_data, train_label_counts = split_stats("train", SPLITS["train"])
    split_stats("dev", SPLITS["dev"])
    split_stats("test", SPLITS["test"])

    # Template-level EDA
    template_stats(train_data)

    # Summary
    print("\nConclusion:")
    train_docs = train_data["documents"]
    dev_docs = _read_json(SPLITS["dev"])["documents"]
    test_docs = _read_json(SPLITS["test"])["documents"]

    print(
        f"- Each split has {len(train_docs)} train, {len(dev_docs)} dev, and {len(test_docs)} test documents."
    )
    median_doc_len = int(np.percentile([len(d["text"]) for d in train_docs], 50))
    median_template_len = int(
        np.percentile([len(m["hypothesis"]) for m in train_data["labels"].values()], 50)
    )
    print(f"- Documents are long (median ~{median_doc_len} chars) with ~17 hypotheses each.")
    print(f"- Template lengths median ~{median_template_len} chars.")
    print(f"- Overall label split (train): {dict(train_label_counts)}")


if __name__ == "__main__":
    main()
