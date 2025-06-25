import json
from pathlib import Path

def normalize_text(text: str) -> str:
    """Normalize all line endings to Unix style."""
    return text.replace('\r\n', '\n').replace('\r', '\n')

def parse_spans(raw_spans):
    """Convert raw span list into (start, end) tuples."""
    if not raw_spans:
        return []

    if isinstance(raw_spans[0], int):
        it = iter(raw_spans)
        return list(zip(it, it))

    return [(start, end) for start, end in raw_spans]

def clean_contractnli(input_file: str, output_file: str):
    """Normalize text and fix span annotations in ContractNLI JSON format."""

    data = json.loads(Path(input_file).read_text(encoding='utf-8'))
    docs = data.get('documents', [])
    labels = data.get('labels', {})

    for doc in docs:
        # Normalize document text
        text = normalize_text(doc.get('text', ''))
        doc['text'] = text

        # Extract annotations
        ann_sets = doc.get('annotation_sets', {})
        annotations = {}

        if isinstance(ann_sets, dict):
            annotations = ann_sets.get('annotations', {})
        else:
            for entry in ann_sets:
                if isinstance(entry, dict) and 'annotations' in entry:
                    annotations = entry['annotations']
                    break

        for meta in annotations.values():
            raw_spans = meta.get('spans', [])
            spans = parse_spans(raw_spans)
            valid_spans = []
            hypothesis = meta.get('hypothesis', '')

            for start, end in spans:
                if 0 <= start < end <= len(text):
                    valid_spans.append((start, end))
                else:
                    idx = text.find(hypothesis)
                    if idx != -1:
                        valid_spans.append((idx, idx + len(hypothesis)))

            # Reformat spans in the same format as the input
            if raw_spans and isinstance(raw_spans[0], int):
                meta['spans'] = [i for span in valid_spans for i in span]
            else:
                meta['spans'] = valid_spans

    # Write cleaned data to file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        json.dumps({'documents': docs, 'labels': labels}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

# Function calls for train/dev/test
clean_contractnli('train.json', 'clean_train.json')
clean_contractnli('dev.json',   'clean_dev.json')
clean_contractnli('test.json',  'clean_test.json')
