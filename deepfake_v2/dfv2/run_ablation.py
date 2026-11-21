"""Run the matched 2x2 ablation: every variant x every seed under one protocol.

    python -m dfv2.run_ablation --data-dir /content/dataset/dataset_processed_split --out-dir runs

Generalization study: add `--holdouts all --robustness` to train one model per held-out manipulation
type (the type is removed from train and val and only appears at test time) and to score each model on
degraded copies of the test frames. Then `python -m dfv2.generalization --runs-dir runs`.

Runs whose predictions file already exists are skipped, so an interrupted Colab or Kaggle
session can simply be restarted with the same command.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .manifest import MANIPULATIONS
from .models import VARIANTS
from .perturbations import DEFAULT_PERTURBATIONS
from .train import TrainConfig, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder holding dataset_manifest.csv and train/val/test")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=sorted(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--split", choices=["clean", "original"], default="clean",
                   help="clean = identity-disjoint re-split (default); original = Kaggle assignment")
    p.add_argument("--split-seed", type=int, default=0, help="which identity-disjoint split to draw")
    p.add_argument("--keep-dfd", action="store_true", help="clean split only: keep DeepFakeDetection")
    p.add_argument("--holdouts", nargs="*", default=[], choices=[*MANIPULATIONS, "all"],
                   help="train one model per held-out manipulation type; 'all' means every type")
    p.add_argument("--robustness", action="store_true",
                   help="also score the test frames under " + ", ".join(DEFAULT_PERTURBATIONS))
    p.add_argument("--max-videos", type=int, default=None, help="videos per split and label (smoke tests only)")
    p.add_argument("--force", action="store_true", help="re-run even if predictions exist")
    return p


def expand_holdouts(values: list[str]) -> list[str | None]:
    """[] means no hold-out (the usual full-data runs); 'all' means every manipulation type."""
    if not values:
        return [None]
    if "all" in values:
        return list(MANIPULATIONS)
    return list(dict.fromkeys(values))


# Settings that define the experiment. A finished run made under different ones must not be
# silently mixed into a new report.
PROTOCOL_KEYS = ("epochs", "batch_size", "lr", "label_smoothing", "pretrained",
                 "split", "split_seed", "keep_dfd", "holdout", "perturbations", "max_videos")


# Settings added after the first runs were made. A history file without them used the default.
LEGACY_DEFAULTS = {"holdout": None, "perturbations": []}


def check_existing_run(out: Path, cfg: TrainConfig) -> None:
    history = out / "history" / f"{cfg.tag}.json"
    if not history.exists():
        return
    old, new = json.loads(history.read_text())["config"], asdict(cfg)
    changed = {k: (old.get(k, LEGACY_DEFAULTS.get(k)), new[k]) for k in PROTOCOL_KEYS
               if old.get(k, LEGACY_DEFAULTS.get(k)) != new[k]}
    if changed:
        raise SystemExit(f"{cfg.tag}: finished run used different settings {changed} (old, new). "
                         "Use another --out-dir, or --force to overwrite.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    results = []
    holdouts = expand_holdouts(args.holdouts)
    perturbations = list(DEFAULT_PERTURBATIONS) if args.robustness else []
    for seed in args.seeds:  # seed-major, so a partial run still gives every variant the same seeds
        for holdout in holdouts:
            for variant in args.variants:
                cfg = TrainConfig(
                    variant=variant, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                    num_workers=args.num_workers, device=args.device, pretrained=not args.no_pretrained,
                    amp=not args.no_amp, max_videos=args.max_videos,
                    split=args.split, split_seed=args.split_seed, keep_dfd=args.keep_dfd,
                    holdout=holdout, perturbations=perturbations,
                )
                if (out / "preds" / f"{cfg.tag}.csv").exists() and not args.force:
                    check_existing_run(out, cfg)
                    print(f"[{cfg.tag}] predictions exist, skipping")
                    continue
                results.append(run(cfg, args.data_dir, out))
    print("\nfinished runs:")
    for r in results:
        print(f"  {r['tag']:<32} best_epoch={r['best_epoch']} val_auc={r['val_auc']:.4f} "
              f"test_auc={r['test_auc']:.4f} test_bal_acc={r['test_bal_acc']:.4f}")
    report = "dfv2.generalization" if args.holdouts else "dfv2.report"
    print(f"\nnext: python -m {report} --runs-dir {out}")


if __name__ == "__main__":
    main()
