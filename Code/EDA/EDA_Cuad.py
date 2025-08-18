import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np


TRAIN_PATH = "clean_train.jsonl"
TEST_PATH = "clean_test.jsonl"
PERCENTILES = [10, 25, 50, 75, 90]


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load JSONL records from a file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def basic_split_stats(name: str, records: List[Dict[str, Any]]) -> Tuple[int, List[int], List[int]]:
    """
    Print basic stats: total examples, span distribution, context/question lengths.
    Returns (total, ctx_lengths, qst_lengths).
    """
    total = len(records)
    spans = [len(r["answers"]["answer_start"]) for r in records]
    ctx_lens = [len(r["context"]) for r in records]
    qst_lens = [len(r["question"]) for r in records]

    print(f"\n{name} SPLIT ---")
    print(f"Total examples: {total}")
    print("Span count distribution:")
    for count, freq in sorted(Counter(spans).items()):
        print(f"  {count} spans: {freq}")
    print(
        f"Context length (chars): min={min(ctx_lens)}, max={max(ctx_lens)}, "
        f"avg={sum(ctx_lens)/total:.1f}"
    )
    print(
        f"Question length (chars): min={min(qst_lens)}, max={max(qst_lens)}, "
        f"avg={sum(qst_lens)/total:.1f}"
    )

    return total, ctx_lens, qst_lens


def print_length_percentiles(name: str, lengths: List[int], label: str) -> None:
    """Print specified percentiles for a list of lengths."""
    pcts = np.percentile(lengths, PERCENTILES)
    pcts_str = ", ".join(f"{p:.1f}" for p in pcts)
    print(f"{name} {label} length percentiles (p{PERCENTILES}): {pcts_str}")


def span_length_stats(name: str, records: List[Dict[str, Any]]) -> None:
    """Compute and print span-length percentiles and top-5 common span lengths."""
    all_spans: List[int] = []
    for r in records:
        for txt in r["answers"]["text"]:
            all_spans.append(len(txt))

    if not all_spans:
        print(f"{name}: no spans!")
        return

    pcts = np.percentile(all_spans, PERCENTILES)
    print(f"\n{name} span lengths (chars) percentiles:")
    print(" " + ", ".join(f"{p:.1f}" for p in pcts))
    print(f"Max span length: {max(all_spans)}")
    print("Top-5 most common span lengths:", Counter(all_spans).most_common(5))


def question_frequency(path: str, title: str) -> None:
    """Count and print the top-20 most common questions in a split."""
    counter: Counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            counter[rec["question"]] += 1

    print(f"\n{title} unique questions: {len(counter)}")
    print(f"Top 20 most common questions in {title}:")
    for q, cnt in counter.most_common(20):
        print(f"  {cnt:5d} – {q}")


def train_test_question_coverage(train_path: str, test_path: str) -> None:
    """Compute and print coverage of train questions by test questions."""
    with open(train_path, "r", encoding="utf-8") as ft:
        train_qs = {json.loads(l)["question"] for l in ft}
    with open(test_path, "r", encoding="utf-8") as fs:
        test_qs = {json.loads(l)["question"] for l in fs}

    inter = train_qs & test_qs
    print(f"\nTrain questions: {len(train_qs)}")
    print(f"Test questions:  {len(test_qs)}")
    print(f"In both sets:    {len(inter)}")
    print(f"Coverage ratio:  {len(inter)/len(train_qs):.2%}")


def main() -> None:
    # Load splits
    train_recs = load_records(TRAIN_PATH)
    test_recs = load_records(TEST_PATH)

    # 1) Basic train/test stats
    _, train_ctx, train_qst = basic_split_stats("TRAIN", train_recs)
    _, test_ctx, test_qst = basic_split_stats("TEST", test_recs)

    # 2) Context/question length percentiles
    print_length_percentiles("TRAIN", train_ctx, "context")
    print_length_percentiles("TRAIN", train_qst, "question")
    print_length_percentiles("TEST", test_ctx, "context")
    print_length_percentiles("TEST", test_qst, "question")

    # 3) Span length stats
    span_length_stats("TRAIN", train_recs)
    span_length_stats("TEST", test_recs)

    # 4) Question frequencies
    question_frequency(TRAIN_PATH, "TRAIN")
    question_frequency(TEST_PATH, "TEST")

    # 5) Train/test question coverage
    train_test_question_coverage(TRAIN_PATH, TEST_PATH)

    # Conclusion
    print("\n Conclusion:")
    print("- CUAD TRAIN:  spans are roughly balanced 0 vs.1; median span ~197 chars; context avg ~64K chars; questions well covered.")
    print("- CUAD TEST:   heavier long-tail spans (up to 19), context avg ~47K chars; test questions fully overlap train.")


if __name__ == "__main__":
    main()
