import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

SPLITS = {
    'train': 'clean_train.json',
    'dev': 'clean_dev.json',
    'test': 'clean_test.json'
}

PERCENTILES = [10, 25, 50, 75, 90]

def split_stats(split_name, path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    docs = data.get('documents', [])
    total = len(docs)

    # Document lengths
    doc_lengths = [len(doc.get('text', '')) for doc in docs]
    p10, p25, p50, p75, p90 = np.percentile(doc_lengths, PERCENTILES)

    # Hypotheses per document and label counts
    hypo_counts = []
    label_counts = Counter()

    for doc in docs:
        ann_sets = doc.get('annotation_sets', {})
        annotations = {}

        if isinstance(ann_sets, dict):
            annotations = ann_sets.get('annotations', {})
        else:
            for entry in ann_sets:
                if isinstance(entry, dict) and 'annotations' in entry:
                    annotations = entry['annotations']
                    break

        hypo_counts.append(len(annotations))

        for meta in annotations.values():
            label_counts[meta['choice']] += 1

    h10, h25, h50, h75, h90 = np.percentile(hypo_counts, PERCENTILES)

    print(f"\n=== {split_name.upper()} SPLIT ===")
    print(f"Total documents: {total}")
    print(f"Document length percentiles (chars): "
          f"10th={p10:.0f}, 25th={p25:.0f}, 50th={p50:.0f}, 75th={p75:.0f}, 90th={p90:.0f}")
    print(f"Hypotheses per doc percentiles:      "
          f"10th={h10:.0f}, 25th={h25:.0f}, 50th={h50:.0f}, 75th={h75:.0f}, 90th={h90:.0f}")
    print(f"Overall label distribution: {dict(label_counts)}")

    return data, label_counts

def template_stats(data):
    labels = data.get('labels', {})
    hyp_texts = [meta['hypothesis'] for meta in labels.values()]
    lengths = [len(h) for h in hyp_texts]
    percentiles = np.percentile(lengths, PERCENTILES)

    print("\nTemplate length percentiles (chars):")
    print(f" 10th={percentiles[0]:.1f}, 25th={percentiles[1]:.1f}, 50th={percentiles[2]:.1f}, "
          f"75th={percentiles[3]:.1f}, 90th={percentiles[4]:.1f}")
    print(f"Max template length: {max(lengths)}\n")

    # Per-hypothesis label distribution
    dist = {hid: Counter() for hid in labels}

    for doc in data['documents']:
        ann_sets = doc.get('annotation_sets', {})
        annotations = {}

        if isinstance(ann_sets, dict):
            annotations = ann_sets.get('annotations', {})
        else:
            for entry in ann_sets:
                if 'annotations' in entry:
                    annotations = entry['annotations']
                    break

        for hid, meta in annotations.items():
            dist[hid][meta['choice']] += 1

    rows = []
    for hid, ctr in dist.items():
        total = sum(ctr.values())
        rows.append({
            'hyp_id': hid,
            'Entailment': ctr['Entailment'],
            'NotMentioned': ctr['NotMentioned'],
            'Contradiction': ctr['Contradiction'],
            'Total': total
        })

    df = pd.DataFrame(rows).set_index('hyp_id')
    print("Per-hypothesis label distribution:")
    print(df.sort_values('Total', ascending=False))

    df['Entailment_rate'] = df['Entailment'] / df['Total']
    print("\nTop 5 hypotheses by Entailment rate:")
    print(df.nlargest(5, 'Entailment_rate')[['Entailment_rate']])

def main():
    print("ContractNLI EDA\n----------------")

    # Split-level EDA
    train_data, train_label_counts = split_stats('train', SPLITS['train'])
    split_stats('dev', SPLITS['dev'])
    split_stats('test', SPLITS['test'])

    # Template-level EDA (uses training labels)
    template_stats(train_data)

    # Summary
    print("\nConclusion:")
    train_docs = train_data['documents']
    dev_docs = json.loads(Path(SPLITS['dev']).read_text())['documents']
    test_docs = json.loads(Path(SPLITS['test']).read_text())['documents']

    print(f"- Each split has {len(train_docs)} train, {len(dev_docs)} dev, and {len(test_docs)} test documents.")
    median_doc_len = int(np.percentile([len(d['text']) for d in train_docs], 50))
    median_template_len = int(np.percentile([len(h) for h in [m['hypothesis'] for m in train_data['labels'].values()]], 50))
    print(f"- Documents are long (median ~{median_doc_len} chars) with ~17 hypotheses each.")
    print(f"- Template lengths median ~{median_template_len} chars.")
    print(f"- Overall label split (train): {dict(train_label_counts)}")

if __name__ == '__main__':
    main()
