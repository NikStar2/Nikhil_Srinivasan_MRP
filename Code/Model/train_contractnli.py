"""
In this script I fine-tune a Longformer classifier for ContractNLI. I use mixed
precision and gradient checkpointing, give [CLS] global attention, and train
with AdamW, linear warmup, weight decay, and gradient clipping. Because the
Contradiction class is rare, I weight the loss (and optionally oversample) to
keep it from getting ignored. I also freeze a few lower layers for the first
epochs, then unfreeze. At the end I run a quick error pass on dev to look at
misclassified Contradictions and the CLS attention.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from transformers import (LongformerForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup,)
from sklearn.metrics import (accuracy_score,precision_recall_fscore_support, classification_report, confusion_matrix)
from tqdm import tqdm


# paths / model
TRAIN_FILE = Path("tokenized_train.jsonl")
DEV_FILE = Path("tokenized_dev.jsonl")
TEST_FILE = Path("tokenized_test.jsonl")
MODEL_NAME = "allenai/longformer-base-4096"
NUM_LABELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training setup
LR = 2e-5
EPOCHS = 8
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1

# rebalancing for rare class (index 2 == Contradiction)
LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 519 / 95], device=DEVICE)
OVERSAMPLE_CONTRADICTION = True

# freeze lower layers for a few epochs
FREEZE_LAYERS = 4
FREEZE_EPOCHS = 2


# dataset / loader
class ContractNLISplitDataset(Dataset):
    def __init__(self, path):
        with Path(path).open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids = torch.tensor(rec["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(rec["attention_mask"], dtype=torch.long)
        # global attention on CLS (token 0)
        gam = torch.tensor([1] + [0] * (len(rec["input_ids"]) - 1), dtype=torch.long)
        label = torch.tensor(rec["label"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "global_attention_mask": gam,
            "labels": label,
        }


def cnli_collate(batch):
    return {
        "input_ids": torch.stack([d["input_ids"] for d in batch]),
        "attention_mask": torch.stack([d["attention_mask"] for d in batch]),
        "global_attention_mask": torch.stack([d["global_attention_mask"] for d in batch]),
        "labels": torch.stack([d["labels"] for d in batch]),
    }


def get_contractnli_dataloader(split: str):
    ds = ContractNLISplitDataset({"train": TRAIN_FILE, "dev": DEV_FILE, "test": TEST_FILE}[split])

    if split == "train" and OVERSAMPLE_CONTRADICTION:
        labels = [rec["label"] for rec in ds.records]
        counts = Counter(labels)
        total = len(labels)
        # weight per class = total_samples / (num_classes * count[class])
        class_w = {cls: total / (NUM_LABELS * cnt) for cls, cnt in counts.items()}
        sample_w = [class_w[lbl] for lbl in labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=cnli_collate, num_workers=2, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"), collate_fn=cnli_collate, num_workers=2, pin_memory=True)


# quick error pass for misclassified Contradictions
def error_analysis(model, dev_loader, tokenizer, device, num_samples=5):
    model.eval()
    samples = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            gam = batch["global_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                global_attention_mask=gam,
                output_attentions=True,
            )
            logits, attns = outputs.logits, outputs.attentions
            preds = logits.argmax(dim=-1)

            for i, (pred, true) in enumerate(zip(preds, labels)):
                if true.item() == 2 and pred.item() != 2:
                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    # average CLS attention across heads & layers
                    cls_attns = torch.stack([layer_attn[i, :, 0, :].mean(dim=0) for layer_attn in attns])
                    avg_cls = cls_attns.mean(dim=0).cpu().tolist()
                    samples.append((text, true.item(), pred.item(), avg_cls[:10]))
                    if len(samples) >= num_samples:
                        break
            if len(samples) >= num_samples:
                break

    print("\n=== Error analysis: Contradiction misclassifications ===")
    for idx, (text, true, pred, cls_attn) in enumerate(samples, 1):
        print(f"\nSample {idx}:")
        print(f"Text snippet: {text}")
        print(f"True=Contradiction, Predicted={pred}")
        print(f"Avg CLS attention weights (first 10 tokens): {cls_attn}")
    print("=== End error analysis ===\n")


def train_and_evaluate():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = LongformerForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(DEVICE)

    # gradient checkpointing and initial layer freezing
    model.gradient_checkpointing_enable()
    for layer in model.longformer.encoder.layer[:FREEZE_LAYERS]:
        for p in layer.parameters():
            p.requires_grad = False

    train_loader = get_contractnli_dataloader("train")
    dev_loader = get_contractnli_dataloader("dev")

    # optimizer + scheduler (weight decay only on non-norm/bias)
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped_params, lr=LR)
    total_steps = EPOCHS * len(train_loader) // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    loss_fct = CrossEntropyLoss(weight=LOSS_WEIGHTS)
    scaler = GradScaler()
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        torch.cuda.empty_cache()

        # unfreeze after the freeze window
        if epoch == FREEZE_EPOCHS + 1:
            for layer in model.longformer.encoder.layer[:FREEZE_LAYERS]:
                for p in layer.parameters():
                    p.requires_grad = True

        # train
        model.train()
        running = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            with autocast():
                out = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    global_attention_mask=batch["global_attention_mask"].to(DEVICE),
                )
                logits = out.logits
                labels = batch["labels"].to(DEVICE)
                loss = loss_fct(logits, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.item() * ACCUMULATION_STEPS

        print(f"[Epoch {epoch}] Train loss: {running/len(train_loader):.4f}")

        # validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch} Dev"):
                with autocast():
                    out = model(
                        input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                        global_attention_mask=batch["global_attention_mask"].to(DEVICE),
                    )
                preds = out.logits.argmax(dim=-1).cpu().tolist()
                labels = batch["labels"].cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        _, _, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="micro", zero_division=0
        )
        print(
            f"[Epoch {epoch}] Dev Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
            f"F1(macro): {f1_macro:.4f} | F1(micro): {f1_micro:.4f}"
        )
        print("Classification report:\n", classification_report(all_labels, all_preds, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds), "\n")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), "best_contractnli.pt")
            print(f"New best model saved (macro-F1={best_f1:.4f})\n")
        else:
            print(f"No improvement (best macro-F1={best_f1:.4f})\n")

    # quick look at where it struggles
    error_analysis(model, dev_loader, tokenizer, DEVICE)


if __name__ == "__main__":
    for split, path in [("train", TRAIN_FILE), ("dev", DEV_FILE), ("test", TEST_FILE)]:
        assert Path(path).exists(), f"Missing {split} file: {path}"
    train_and_evaluate()
