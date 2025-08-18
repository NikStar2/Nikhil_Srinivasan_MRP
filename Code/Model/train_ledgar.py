"""
In this script I fine-tune Legal-BERT on the LEDGAR dataset for multi-label
clause classification. To make the label space manageable I keep the most
common clause types and put the rest into an "OTHER" bucket. I train with
focal loss and some oversampling so rare clauses aren’t ignored, and use mixed
precision with AdamW and gradient clipping for efficiency. For evaluation I
force top-K predictions per clause and check precision, recall, and F1 on a
dev split.
"""

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, logging,)
from tqdm import tqdm

logging.set_verbosity_error()

# dataset + model
TOKENIZED_PATH = Path("ledgar_tokenized.jsonl")
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# labels / evaluation
K_TOP_LABELS = 500      # keep only the top-K labels
TOP_K = 5               # force top-K predictions per clause at eval

# training setup
LR = 2e-5
EPOCHS = 6
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2   # gradient accumulation
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0      # clip exploding grads
WARMUP_RATIO = 0.1       # proportion of steps for warmup

SAVE_PATH = "best_ledgar"  


# focal loss for imbalanced multi-label data
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = ce * (1 - pt).pow(self.gamma)
        return loss.mean() if self.reduction == "mean" else loss.sum()


# simple dataset wrapper
class LEDGARDataset(Dataset):
    def __init__(self, records, num_labels: int, label_key: str):
        self.records = records
        self.num_labels = num_labels
        self.label_key = label_key

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        y = torch.zeros(self.num_labels, dtype=torch.float)
        for lab in r.get(self.label_key, []):
            y[lab] = 1.0
        return {
            "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "labels": y,
        }


def collate_ledgar(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


def train_and_evaluate(train_loader, dev_loader, pos_weight, num_labels: int) -> None:
    
    _ = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(DEVICE)
    model.gradient_checkpointing_enable()

    # optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=LR)
    total_steps = EPOCHS * len(train_loader) // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    scaler = GradScaler()
    loss_fn = FocalLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        torch.cuda.empty_cache()
        model.train()
        running = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            with autocast():
                logits = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                ).logits
                loss = loss_fn(logits, batch["labels"].to(DEVICE)) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.item() * ACCUMULATION_STEPS

        print(f"[Epoch {epoch}] Train loss: {running/len(train_loader):.6f}")

        # dev eval with forced top-K predictions per sample
        model.eval()
        tp = pred_pos = true_pos = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch} Dev"):
                with autocast():
                    logits = model(
                        input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                    ).logits.cpu()

                k = min(TOP_K, logits.size(-1))
                _, topi = logits.topk(k, dim=-1)
                preds = torch.zeros_like(logits, dtype=torch.int)
                for i in range(preds.size(0)):
                    preds[i, topi[i]] = 1

                labs = batch["labels"].cpu().int()
                tp += int((preds & labs).sum())
                pred_pos += int(preds.sum())
                true_pos += int(labs.sum())

        prec = tp / pred_pos if pred_pos else 0.0
        rec = tp / true_pos if true_pos else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        print(f"[Epoch {epoch}] Dev micro-F1: {f1:.4f} (P={prec:.4f}, R={rec:.4f}) | Avg labels/sample: {k}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New best model saved (micro-F1={best_f1:.4f})\n")
        else:
            print(f"No improvement (best micro-F1={best_f1:.4f})\n")


if __name__ == "__main__":
    # stream an 80/20 split from tokenized JSONL
    first = TOKENIZED_PATH.read_text(encoding="utf-8").splitlines()[0]
    _ = json.loads(first)  # confirms schema that data uses "label_ids"
    label_key = "label_ids"

    train_recs, dev_recs = [], []
    with open(TOKENIZED_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            (train_recs if (i % 10 < 8) else dev_recs).append(rec)

    # prune labels to top-K and bucket the rest as "__OTHER__"
    freq = Counter(l for r in train_recs for l in r[label_key])
    topk = {lbl for lbl, _ in freq.most_common(K_TOP_LABELS)}
    new_map = {lbl: i for i, lbl in enumerate(sorted(topk))}
    new_map["__OTHER__"] = len(new_map)
    NEW_NUM = len(new_map)

    def remap(r):
        mapped = {new_map.get(l, new_map["__OTHER__"]) for l in r[label_key]}
        r["pruned_labels"] = list(mapped)
        return r

    train_recs = [remap(r) for r in train_recs]
    dev_recs = [remap(r) for r in dev_recs]

    # class weights for focal loss
    total = len(train_recs)
    counts = [0] * NEW_NUM
    for r in train_recs:
        for l in r["pruned_labels"]:
            counts[l] += 1
    neg = [total - c for c in counts]
    pos_weight = torch.tensor([n / (c + 1e-5) for c, n in zip(counts, neg)], device=DEVICE)

    # oversample any record that has at least one positive label
    sample_weights = [5.0 if r["pruned_labels"] else 1.0 for r in train_recs]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = LEDGARDataset(train_recs, NEW_NUM, "pruned_labels")
    dev_ds = LEDGARDataset(dev_recs, NEW_NUM, "pruned_labels")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_ledgar,
        num_workers=0,
        pin_memory=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_ledgar,
        num_workers=0,
        pin_memory=False,
    )

    train_and_evaluate(train_loader, dev_loader, pos_weight, NEW_NUM)
