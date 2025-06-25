import json

def normalize_text(text: str) -> str:
    """Convert all line endings to Unix style (\n)."""
    return text.replace('\r\n', '\n').replace('\r', '\n')

def clean_cuad_file(input_file: str, output_file: str):
    """Normalize text and fix answer start indices in CUAD-formatted JSONL file."""

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            rec = json.loads(line)

            # Normalize line breaks in context and question
            context = normalize_text(rec['context'])
            question = normalize_text(rec['question'])

            texts = rec['answers'].get('text', [])
            starts = rec['answers'].get('answer_start', [])
            fixed_starts = []

            for text, start in zip(texts, starts):
                if context[start:start + len(text)] == text:
                    fixed_starts.append(start)
                else:
                    idx = context.find(text)
                    if idx != -1:
                        fixed_starts.append(idx)

            rec['context'] = context
            rec['question'] = question
            rec['answers']['answer_start'] = fixed_starts

            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

# Process train and test splits
clean_cuad_file('train.jsonl', 'clean_train.jsonl')
clean_cuad_file('test.jsonl',  'clean_test.jsonl')
