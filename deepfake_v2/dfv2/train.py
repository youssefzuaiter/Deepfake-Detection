"""Train one (variant, seed) run under the shared protocol and save its predictions.

Shared protocol for every variant: Adam, lr 1e-4, label-smoothed cross-entropy,
ReduceLROnPlateau, balanced real/fake subset redrawn each epoch, the best epoch chosen
by validation AUC, and the test split touched exactly once after that choice.
"""
from __future__ import annotations

import contextlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import metrics as M
from .dataset import BalancedEpochSampler, FrameDataset
from .manifest import load_manifest, split_frame
from .models import build_model
from .splits import DEFAULT_DROP_TYPES, audit_identities


@dataclass
class TrainConfig:
    variant: str
    seed: int = 0
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    label_smoothing: float = 0.1
    num_workers: int = 2
    pretrained: bool = True
    amp: bool = True
    device: str = "auto"
    split: str = "clean"           # "clean" = identity-disjoint, "original" = Kaggle assignment
    split_seed: int = 0
    keep_dfd: bool = False         # clean split only: keep DeepFakeDetection (dropped by default)
    max_videos: int | None = None  # smoke runs only

    @property
    def tag(self) -> str:
        return f"{self.variant}_seed{self.seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[np.ndarray, float]:
    """Return (P(fake) per row in loader order, mean unsmoothed cross-entropy)."""
    model.eval()
    probs, loss_sum = [], 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _amp(device, use_amp):
            logits = model(x)
        logits = logits.float()
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    p = torch.cat(probs).numpy().astype(np.float64)
    return p, loss_sum / len(p)


def _prediction_rows(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    out = df[["filename", "split", "video", "fake_type", "label"]].copy()
    out["p_fake"] = p
    return out


def run(cfg: TrainConfig, data_dir: str | Path, out_dir: str | Path) -> dict:
    """Train, pick the best epoch on validation, then write predictions, history and weights."""
    out = Path(out_dir)
    for sub in ("preds", "history", "weights"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = pick_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    set_seed(cfg.seed)

    df = load_manifest(
        data_dir, max_videos=cfg.max_videos, split=cfg.split, split_seed=cfg.split_seed,
        drop_types=() if cfg.keep_dfd else DEFAULT_DROP_TYPES)
    audit = audit_identities(df)  # recorded with the run, so the report can state what was checked
    train_df, val_df, test_df = (split_frame(df, s) for s in ("train", "val", "test"))
    pin = device.type == "cuda"
    sampler = BalancedEpochSampler(train_df["label"].to_numpy(), cfg.seed)
    train_loader = DataLoader(
        FrameDataset(train_df, train=True), batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=cfg.num_workers > 0,
    )

    def eval_loader(frame: pd.DataFrame) -> DataLoader:
        return DataLoader(FrameDataset(frame, train=False), batch_size=128, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=pin)

    val_loader, test_loader = eval_loader(val_df), eval_loader(test_df)

    model = build_model(cfg.variant, pretrained=cfg.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    y_val = val_df["label"].to_numpy()
    history: list[dict] = []
    best: dict = {"val_auc": -1.0}
    started = time.time()
    print(f"[{cfg.tag}] device={device} params={n_params:,} train/epoch={len(sampler)} val={len(val_df)} test={len(test_df)}")

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device, use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val_p, val_loss = predict_probs(model, val_loader, device, use_amp)
        val_auc = M.auc_roc(y_val, val_p)
        scheduler.step(val_auc)
        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_bal_acc": M.balanced_accuracy(y_val, M.predict(val_p)),
        }
        history.append(record)
        print(f"[{cfg.tag}] epoch {epoch + 1}/{cfg.epochs} " + " ".join(
            f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"))
        if val_auc > best["val_auc"]:
            best = {
                "val_auc": val_auc, "epoch": epoch + 1, "val_p": val_p,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    # The test split is read once, with the epoch that validation picked.
    model.load_state_dict(best["state"])
    test_p, test_loss = predict_probs(model, test_loader, device, use_amp)
    preds = pd.concat([_prediction_rows(val_df, best["val_p"]), _prediction_rows(test_df, test_p)], ignore_index=True)
    preds.to_csv(out / "preds" / f"{cfg.tag}.csv", index=False)
    torch.save(best["state"], out / "weights" / f"{cfg.tag}.pt")

    y_test = test_df["label"].to_numpy()
    summary = {
        "tag": cfg.tag,
        "best_epoch": best["epoch"],
        "val_auc": best["val_auc"],
        "test_auc": M.auc_roc(y_test, test_p),
        "test_bal_acc": M.balanced_accuracy(y_test, M.predict(test_p)),
        "test_loss": test_loss,
    }
    meta = {
        "config": asdict(cfg), "n_params": n_params, "device": str(device),
        "torch": torch.__version__, "seconds": round(time.time() - started, 1),
        "history": history, "summary": summary, "split_audit": audit,
    }
    (out / "history" / f"{cfg.tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{cfg.tag}] done in {meta['seconds']}s best_epoch={best['epoch']} "
          f"val_auc={summary['val_auc']:.4f} test_auc={summary['test_auc']:.4f}")
    return summary
