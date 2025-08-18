"""
In this script I fine-tune Legal-BERT on CUAD for extractive QA.
I slice an 80/20 dev split from the provided train tensors and keep test separate.
I use gradient checkpointing and mixed precision, train with AdamW and a linear
warmup schedule, and clip gradients. Inputs are truncated to 512 tokens; I clamp
any out-of-range token ids and build token_type_ids manually. I report exact
match (EM) and token-level F1 on dev each epoch and on the test split at the end.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
    logging,
)
from tqdm import tqdm

logging.set_verbosity_error()

# paths / model
TRAIN_PATH = "train.pt"
TEST_PATH = "test.pt"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training setup
SEED = 42
LR = 3e-5
EPOCHS = 3
BATCH_SIZE = 8
ACCUM_STEPS = 1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
MAX_LEN = 512

random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
vocab_size = tokenizer.vocab_size


class CUADDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch):
    # stack then crop to model's max length
    input_ids = torch.stack([b["input_ids"] for b in batch])[:, :MAX_LEN]
    attention_mask = torch.stack([b["attention_mask"] for b in batch])[:, :MAX_LEN]
    start_pos = torch.tensor([b["start_positions"] for b in batch], dtype=torch.long)
    end_pos = torch.tensor([b["end_positions"] for b in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_pos,
        "end_positions": end_pos,
    }


def make_token_type_ids(input_ids):
    
    sep = tokenizer.sep_token_id
    tt = torch.zeros_like(input_ids)
    for i, seq in enumerate(input_ids):
        idxs = (seq == sep).nonzero(as_tuple=True)[0]
        if len(idxs) > 0:
            j = idxs[0].item()
            tt[i, j + 1 :] = 1
    return tt


def compute_metrics(model, loader):
    model.eval()
    total, em_sum, f1_sum = 0, 0, 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            s_pos = batch["start_positions"]
            e_pos = batch["end_positions"]

            # clamp any stray ids
            ids = ids.clamp(0, vocab_size - 1)
            tt = make_token_type_ids(ids).to(DEVICE)

            # drop examples whose gold span fell outside 512
            valid = (s_pos < MAX_LEN) & (e_pos < MAX_LEN)
            if valid.sum() == 0:
                continue
            ids = ids[valid]
            mask = mask[valid]
            tt = tt[valid]
            gold_s = s_pos[valid]
            gold_e = e_pos[valid]

            with autocast():
                out = model(input_ids=ids, attention_mask=mask, token_type_ids=tt)

            start_logits = out.start_logits.cpu()
            end_logits = out.end_logits.cpu()

            for i in range(len(gold_s)):
                total += 1
                ps, pe = int(start_logits[i].argmax()), int(end_logits[i].argmax())
                gs, ge = int(gold_s[i]), int(gold_e[i])

                if ps == gs and pe == ge:
                    em_sum += 1

                # token-level span F1
                pred_set = set(range(ps, pe + 1)) if pe >= ps else set()
                gold_set = set(range(gs, ge + 1)) if ge >= gs else set()
                if pred_set and gold_set:
                    inter = pred_set & gold_set
                    p = len(inter) / len(pred_set)
                    r = len(inter) / len(gold_set)
                    f1_sum += (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    em = em_sum / total if total else 0.0
    f1 = f1_sum / total if total else 0.0
    return em, f1


def train_and_eval():
    # load data and slice a dev split
    all_train = torch.load(TRAIN_PATH)
    random.shuffle(all_train)
    cut = int(0.8 * len(all_train))
    train_records, dev_records = all_train[:cut], all_train[cut:]
    test_records = torch.load(TEST_PATH)

    train_loader = DataLoader(
        CUADDataset(train_records), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        CUADDataset(dev_records), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        CUADDataset(test_records), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # model + optimizer
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    model.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    opt_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(opt_groups, lr=LR)
    total_steps = EPOCHS * len(train_loader) // ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    best_em = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            ids = batch["input_ids"][:, :MAX_LEN].to(DEVICE)
            mask = batch["attention_mask"][:, :MAX_LEN].to(DEVICE)
            s_pos = batch["start_positions"]
            e_pos = batch["end_positions"]

            # clamp ids & build token_type_ids
            ids = ids.clamp(0, vocab_size - 1)
            tt = make_token_type_ids(ids).to(DEVICE)

            # filter OOB spans
            valid = (s_pos < MAX_LEN) & (e_pos < MAX_LEN)
            if valid.sum() == 0:
                continue

            inputs = {
                "input_ids": ids[valid],
                "attention_mask": mask[valid],
                "token_type_ids": tt[valid],
                "start_positions": s_pos[valid].to(DEVICE),
                "end_positions": e_pos[valid].to(DEVICE),
            }

            with autocast():
                out = model(**inputs)
                loss = out.loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS

        avg = running_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train loss: {avg:.4f}")

        em_dev, f1_dev = compute_metrics(model, dev_loader)
        print(f"[Epoch {epoch}] Dev → EM: {em_dev:.4f}, F1: {f1_dev:.4f}")

        if em_dev > best_em:
            best_em = em_dev
            torch.save(model.state_dict(), "best_cuad.pt")
            print(f"New best model (Dev EM={best_em:.4f})")

    # final test
    print("Loading best checkpoint for test")
    model.load_state_dict(torch.load("best_cuad.pt", map_location=DEVICE))
    em_test, f1_test = compute_metrics(model, test_loader)
    print(f"[Test] EM: {em_test:.4f}, F1: {f1_test:.4f}")


if __name__ == "__main__":
    train_and_eval()
