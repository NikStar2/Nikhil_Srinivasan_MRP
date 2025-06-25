import json
import numpy as np
from collections import Counter, defaultdict

LEDGAR_PATH = 'ledgar_cleaned.jsonl'

def main():
    # Step 1: Total provisions, unique labels, top-20 labels
    label_counter = Counter()
    total_provs = 0
    with open(LEDGAR_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            total_provs += 1
            for lbl in rec['label']:
                label_counter[lbl] += 1

    print(f"Total LEDGAR provisions: {total_provs}")
    print(f"Unique labels: {len(label_counter)}")
    print("Top 20 labels:")
    for lbl, cnt in label_counter.most_common(20):
        print(f"  {cnt:7d} – {lbl}")

    # Step 2: Label-per-provision distribution & provision lengths
    label_counts = []
    lengths = []
    with open(LEDGAR_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            label_counts.append(len(rec['label']))
            lengths.append(len(rec['provision']))

    card_dist = Counter(label_counts)
    print("\nLabel count distribution (labels per provision):")
    for num_labels, cnt in sorted(card_dist.items()):
        print(f"  {num_labels:2d} labels: {cnt:7d} ({cnt/total_provs:.2%})")

    pcts = np.percentile(lengths, [10,25,50,75,90])
    print("\nProvision length (chars) percentiles:")
    print(f" 10th: {pcts[0]:.0f}, 25th: {pcts[1]:.0f}, 50th: {pcts[2]:.0f}, "
          f"75th: {pcts[3]:.0f}, 90th: {pcts[4]:.0f}")
    print(f" Max provision length: {max(lengths)}")

    # Step 3: Length buckets
    buckets = [0, 200, 500, 1000, 2000, 5000, 10000, float('inf')]
    bucket_labels = ["≤200","201–500","501–1k","1k–2k","2k–5k","5k–10k","10k+"]
    bucket_counts = Counter({lab:0 for lab in bucket_labels})

    with open(LEDGAR_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            prov = json.loads(line)['provision']
            L = len(prov)
            for i in range(len(buckets)-1):
                if buckets[i] < L <= buckets[i+1]:
                    bucket_counts[bucket_labels[i]] += 1
                    break

    print(f"\nProvision length buckets:")
    for lab in bucket_labels:
        cnt = bucket_counts[lab]
        print(f"  {lab:8s}: {cnt:7d} ({cnt/total_provs:.2%})")

    # Step 4: Provision length vs. label-count
    lengths_by_card = defaultdict(list)
    with open(LEDGAR_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            n = len(rec['label'])
            lengths_by_card[n].append(len(rec['provision']))

    print("\nLabel-Count |   Count  |   Avg Len  |  Median  | 75th pct | 90th pct")
    for n in sorted(lengths_by_card):
        arr = np.array(lengths_by_card[n])
        print(f"{n:11d} | {len(arr):7d} | {arr.mean():10.1f} | {np.median(arr):7.1f} | "
              f"{np.percentile(arr,75):8.1f} | {np.percentile(arr,90):8.1f}")

    # Conclusion

    print("\n Conclusion:")
    print(f"- Total provisions analyzed: {total_provs}")
    print(f"- Unique labels: {len(label_counter)}")
    print(f"- Single-label provisions: {card_dist[1]/total_provs:.2%}")
    print(f"- Median provision length: {pcts[2]:.0f} chars")
    print(f"- 75th percentile length:   {pcts[3]:.0f} chars")
    print(f"- Very long provisions (>10k chars): {bucket_counts['10k+']} examples")



if __name__ == "__main__":
    main()
