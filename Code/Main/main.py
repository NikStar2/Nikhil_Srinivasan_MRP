"""
In this script I run a quick combined evaluation for three models:
LEDGAR (multi-label), ContractNLI (3-way classification), and CUAD (extractive QA).
Each section loads its own checkpoint and test/dev data, computes the usual
metrics, and I aggregate everything into a single CSV at the end.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    LongformerForSequenceClassification,
    AutoModelForQuestionAnswering,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm


# device and a few paths per task (kept casual on purpose)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LEDGAR
LEDGAR_TOKENIZED = "ledgar_tokenized.jsonl"
LEDGAR_CHECKPOINT = "best_ledgar.pt" 
LEDGAR_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
LEDGAR_TOP_K = 5

# ContractNLI
CNLI_TOKENIZED_TEST = "tokenized_test.jsonl"
CNLI_CHECKPOINT = "best_contractnli.pt"
CNLI_MODEL_NAME = "allenai/longformer-base-4096"

# CUAD
CUAD_TEST_PT = "test.pt"
CUAD_CHECKPOINT = "best_cuad.pt"
CUAD_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


# LEDGAR
class LEDGARDataset(Dataset):
    def __init__(self, recs, num_labels, label_key="pruned_labels"):
        self.recs = recs
        self.num_labels = num_labels
        self.label_key = label_key

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        r = self.recs[idx]
        labels = torch.tensor(
            [1 if i in r.get(self.label_key, []) else 0 for i in range(self.num_labels)],
            dtype=torch.int,
        )
        return {
            "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "labels": labels,
        }


def collate_ledgar(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


def evaluate_ledgar():
    # dev split via stream rule: 80/20 where i%10 >= 8
    with open(LEDGAR_TOKENIZED, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    dev_recs = [r for i, r in enumerate(recs) if i % 10 >= 8]

    # figure out the classifier size from the checkpoint
    state = torch.load(LEDGAR_CHECKPOINT, map_location="cpu")
    num_labels = state["classifier.weight"].size(0)

    ds = LEDGARDataset(dev_recs, num_labels)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_ledgar)

    # model
    _ = AutoTokenizer.from_pretrained(LEDGAR_MODEL_NAME, use_fast=True)  # kept for symmetry
    model = AutoModelForSequenceClassification.from_pretrained(
        LEDGAR_MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="LEDGAR Eval"):
            ids = batch["input_ids"].to(DEVICE)
            att = batch["attention_mask"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)

            logits = model(input_ids=ids, attention_mask=att).logits.cpu()
            k = min(LEDGAR_TOP_K, logits.size(1))
            _, topi = logits.topk(k, dim=1)

            preds = torch.zeros_like(logits, dtype=torch.int)
            for i in range(preds.size(0)):
                preds[i, topi[i]] = 1

            all_preds.append(preds)
            all_labels.append(labs.cpu())

    y_pred = torch.cat(all_preds).numpy().flatten()
    y_true = torch.cat(all_labels).numpy().flatten()

    acc = accuracy_score(y_true, y_pred)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "Model": "LEDGAR",
        "Accuracy": acc,
        "Precision_micro": p_micro,
        "Recall_micro": r_micro,
        "F1_micro": f1_micro,
        "Precision_macro": p_mac,
        "Recall_macro": r_mac,
        "F1_macro": f1_mac,
    }


# ContractNLI
class ContractNLITestDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = torch.tensor(rec["input_ids"], dtype=torch.long)
        attn = torch.tensor(rec["attention_mask"], dtype=torch.long)
        gam = torch.tensor([1] + [0] * (len(rec["input_ids"]) - 1), dtype=torch.long)  # CLS global
        label = torch.tensor(rec["label"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "global_attention_mask": gam,
            "labels": label,
        }


def collate_cnli(batch):
    return {
        "input_ids": torch.stack([d["input_ids"] for d in batch]),
        "attention_mask": torch.stack([d["attention_mask"] for d in batch]),
        "global_attention_mask": torch.stack([d["global_attention_mask"] for d in batch]),
        "labels": torch.stack([d["labels"] for d in batch]),
    }


def evaluate_contractnli():
    ds = ContractNLITestDataset(CNLI_TOKENIZED_TEST)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_cnli)

    model = LongformerForSequenceClassification.from_pretrained(
        CNLI_MODEL_NAME, num_labels=3
    ).to(DEVICE)
    model.load_state_dict(torch.load(CNLI_CHECKPOINT, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ContractNLI Eval"):
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                global_attention_mask=batch["global_attention_mask"].to(DEVICE),
            )
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            labs = batch["labels"].numpy()
            all_preds.extend(preds)
            all_labels.extend(labs)

    acc = accuracy_score(all_labels, all_preds)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return {
        "Model": "ContractNLI",
        "Accuracy": acc,
        "Precision_micro": p_micro,
        "Recall_micro": r_micro,
        "F1_micro": f1_micro,
        "Precision_macro": p_mac,
        "Recall_macro": r_mac,
        "F1_macro": f1_mac,
    }


# CUAD
class CUATestDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_cuad(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])[:, :512]
    attention_mask = torch.stack([b["attention_mask"] for b in batch])[:, :512]
    start_positions = torch.tensor([b["start_positions"] for b in batch], dtype=torch.long)
    end_positions = torch.tensor([b["end_positions"] for b in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions,
    }


def make_ttids(input_ids, sep_token_id):
    tt = torch.zeros_like(input_ids)
    for i, seq in enumerate(input_ids):
        sep_idxs = (seq == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_idxs) > 0:
            tt[i, sep_idxs[0] + 1 :] = 1
    return tt


def evaluate_cuad():
    records = torch.load(CUAD_TEST_PT)
    ds = CUATestDataset(records)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_cuad)

    tokenizer = AutoTokenizer.from_pretrained(CUAD_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(CUAD_MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(CUAD_CHECKPOINT, map_location=DEVICE))
    model.eval()

    sep_id = tokenizer.sep_token_id
    total, em_sum, f1_sum = 0, 0, 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="CUAD Eval"):
            ids = batch["input_ids"].to(DEVICE).clamp(0, tokenizer.vocab_size - 1)
            mask = batch["attention_mask"].to(DEVICE)
            tt = make_ttids(ids, sep_id).to(DEVICE)

            gold_s = batch["start_positions"]
            gold_e = batch["end_positions"]

            # filter any spans that fell outside the 512 window
            valid = (gold_s < 512) & (gold_e < 512)
            if valid.sum() == 0:
                continue

            ids_v = ids[valid]
            mask_v = mask[valid]
            tt_v = tt[valid]
            gs = gold_s[valid]
            ge = gold_e[valid]

            out = model(input_ids=ids_v, attention_mask=mask_v, token_type_ids=tt_v)
            start_logits = out.start_logits.cpu()
            end_logits = out.end_logits.cpu()

            for i in range(len(gs)):
                total += 1
                ps, pe = int(start_logits[i].argmax()), int(end_logits[i].argmax())
                if ps == int(gs[i]) and pe == int(ge[i]):
                    em_sum += 1

                pred_set = set(range(ps, pe + 1)) if pe >= ps else set()
                gold_set = (
                    set(range(int(gs[i]), int(ge[i]) + 1)) if ge[i] >= gs[i] else set()
                )
                if pred_set and gold_set:
                    inter = pred_set & gold_set
                    p = len(inter) / len(pred_set)
                    r = len(inter) / len(gold_set)
                    f1_sum += 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    em = em_sum / total if total else 0.0
    f1 = f1_sum / total if total else 0.0
    return {"Model": "CUAD", "EM": em, "F1": f1}


# Main
def main():
    results = [
        evaluate_ledgar(),
        evaluate_contractnli(),
        evaluate_cuad(),
    ]
    df = pd.DataFrame(results)
    out_path = "evaluation_results.csv"
    df.to_csv(out_path, index=False)

    # also print the CSV contents (as requested)
    print("\nSaved metrics to", out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
