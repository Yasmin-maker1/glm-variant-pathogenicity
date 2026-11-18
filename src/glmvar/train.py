"""Fine-tune (or train a head on a frozen) genomic language model on variant pairs.

Implements the training settings of Module C, Section III.D:
AdamW, mixed precision, class-weighted BCE, batch 32 (with gradient accumulation),
up to 10 epochs, early stopping after 2 epochs without validation-AUROC gain.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .models.tokens import pool_mask
from .utils import ResourceTracker, save_json, set_seed

LOG = logging.getLogger(__name__)


class VariantPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.ref = df["ref_seq"].tolist()
        self.alt = df["alt_seq"].tolist()
        self.y = df["label"].astype(np.float32).to_numpy()
        self.ids = df["variant_id"].tolist()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.ref[i], self.alt[i], self.y[i]


class PairCollator:
    """Tokenizes ref/alt windows and builds the pooling mask around the variant."""

    def __init__(self, tokenizer, pool_radius: int = 32):
        self.tok = tokenizer
        self.radius = pool_radius

    def _encode(self, seqs: list[str]):
        enc = self.tok(seqs, padding="longest", return_tensors="pt", add_special_tokens=True)
        pools = []
        for ids, seq in zip(enc["input_ids"].tolist(), seqs):
            tokens = self.tok.convert_ids_to_tokens(ids)
            pools.append(pool_mask(tokens, center=len(seq) // 2, radius=self.radius))
        attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
        return enc["input_ids"], attn, torch.tensor(pools, dtype=torch.float32)

    def __call__(self, batch):
        ref, alt, y = zip(*batch)
        r_ids, r_attn, r_pool = self._encode(list(ref))
        a_ids, a_attn, a_pool = self._encode(list(alt))
        return {
            "ref_ids": r_ids, "ref_attn": r_attn, "ref_pool": r_pool,
            "alt_ids": a_ids, "alt_attn": a_attn, "alt_pool": a_pool,
            "labels": torch.tensor(y, dtype=torch.float32),
        }


def _to(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _autocast(device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = _to(batch, device)
        batch.pop("labels")
        with _autocast(device):
            logits = model(**batch)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_model(model, tokenizer, df: pd.DataFrame, cfg: dict, out_dir: str | Path, seed: int = 0, device: str | None = None) -> dict:
    """Train with early stopping on validation AUROC; write val/test predictions.

    ``df`` must contain columns: variant_id, split, label, ref_seq, alt_seq.
    Returns a summary dict (also saved as ``summary.json``).
    """
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    t = cfg.get("training", {})
    micro_bs = int(t.get("micro_batch_size", 8))
    eff_bs = int(t.get("batch_size", 32))
    accum = max(1, eff_bs // micro_bs)
    epochs = int(t.get("epochs", 10))
    patience = int(t.get("patience", 2))
    lr = float(t.get("learning_rate", 3e-5))
    head_lr = float(t.get("head_learning_rate", 1e-3))
    wd = float(t.get("weight_decay", 0.01))
    collate = PairCollator(tokenizer, pool_radius=int(t.get("pool_radius", 32)))

    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    train_loader = DataLoader(VariantPairDataset(parts["train"]), batch_size=micro_bs, shuffle=True, collate_fn=collate, num_workers=int(t.get("num_workers", 0)))
    eval_loaders = {s: DataLoader(VariantPairDataset(parts[s]), batch_size=micro_bs * 2, shuffle=False, collate_fn=collate) for s in ("val", "test")}

    # Class weighting for the pathogenic/benign imbalance.
    y = parts["train"]["label"].to_numpy()
    pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=device, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_params, "lr": head_lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": lr})
    opt = torch.optim.AdamW(groups, weight_decay=wd)

    tracker = ResourceTracker()
    best_auc, best_state, bad_epochs, history = -1.0, None, 0, []
    with tracker.track():
        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            running = 0.0
            for step, batch in enumerate(train_loader, 1):
                batch = _to(batch, device)
                labels = batch.pop("labels")
                with _autocast(device):
                    logits = model(**batch)
                loss = loss_fn(logits.float(), labels) / accum
                loss.backward()
                running += loss.item() * accum
                if step % accum == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            val_pred = predict(model, eval_loaders["val"], device)
            val_auc = roc_auc_score(parts["val"]["label"], val_pred) if parts["val"]["label"].nunique() > 1 else float("nan")
            history.append({"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "val_auroc": val_auc})
            LOG.info("epoch %d  loss %.4f  val AUROC %.4f", epoch, history[-1]["train_loss"], val_auc)
            if val_auc > best_auc:
                best_auc, bad_epochs = val_auc, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    LOG.info("early stopping after epoch %d", epoch)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = {}
    for split in ("val", "test"):
        start = time.perf_counter()
        p = predict(model, eval_loaders[split], device)
        elapsed = time.perf_counter() - start
        preds[split] = p
        frame = parts[split][["variant_id", "label"]].copy()
        frame["score"] = p
        frame["split"] = split
        frame.to_csv(out_dir / f"predictions_{split}.csv", index=False)
        if split == "test":
            variants_per_second = len(p) / elapsed if elapsed > 0 else float("nan")

    summary = {
        "seed": seed,
        "best_val_auroc": best_auc,
        "epochs_run": len(history),
        "history": history,
        "variants_per_second": variants_per_second,
        "n_train": len(parts["train"]),
        **tracker.as_dict(),
    }
    save_json(summary, out_dir / "summary.json")
    return summary
