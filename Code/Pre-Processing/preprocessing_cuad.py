import json
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Convert all line endings to Unix style.

    This avoids misalignment of answer indices caused by 
    inconsistent newlines.
    """
    return text.replace('\r\n', '\n').replace('\r', '\n')


def clean_cuad_file(input_path: str, output_path: str) -> None:
    """
    Normalize text and fix answer start indices in a CUAD-formatted JSONL file.

    - Ensures consistent line endings in 'context' and 'question'
    - Recomputes answer_start indices if they no longer align
    - Writes corrected records line by line into output_path
    """
    in_file, out_file = Path(input_path), Path(output_path)

    with in_file.open('r', encoding='utf-8') as fin, \
         out_file.open('w', encoding='utf-8') as fout:

        for line in fin:
            record = json.loads(line)

            # Normalize line breaks
            context = normalize_text(record['context'])
            question = normalize_text(record['question'])

            texts = record['answers'].get('text', [])
            starts = record['answers'].get('answer_start', [])
            corrected_starts = []

            for txt, start in zip(texts, starts):
                # If the stored index still matches, keep it
                if context[start:start + len(txt)] == txt:
                    corrected_starts.append(start)
                else:
                    # Otherwise, attempt to find the substring again
                    new_idx = context.find(txt)
                    if new_idx != -1:
                        corrected_starts.append(new_idx)

            record['context'] = context
            record['question'] = question
            record['answers']['answer_start'] = corrected_starts

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')


# Function call
if __name__ == "__main__":
    clean_cuad_file("train.jsonl", "clean_train.jsonl")
    clean_cuad_file("test.jsonl", "clean_test.jsonl")
    print("Cleaning complete.")