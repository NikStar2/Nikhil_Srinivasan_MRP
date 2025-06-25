import json
import re

def normalize_provision(text):
    """Normalize whitespace and line breaks in a provision string."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_ledgar(input_path: str, output_path: str):
    """Clean and normalize LEDGAR provisions, labels, and write to a new JSONL file."""

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            provision = record.get('provision', '')
            if not provision.strip():
                continue

            cleaned_provision = normalize_provision(provision)

            labels = record.get('label', [])
            if isinstance(labels, str):
                labels = [labels]
            elif not isinstance(labels, list):
                labels = list(labels) if hasattr(labels, '__iter__') else []

            cleaned_record = {
                'provision': cleaned_provision,
                'label': labels,
                'source': record.get('source', '')
            }

            fout.write(json.dumps(cleaned_record, ensure_ascii=False) + '\n')

# Function call
clean_ledgar('LEDGAR_2016-2019_clean.jsonl', 'ledgar_cleaned.jsonl')
print("Cleaning complete.")
