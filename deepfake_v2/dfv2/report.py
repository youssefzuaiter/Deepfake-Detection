"""Turn saved predictions into the ablation report: tables, confidence intervals, figures.

    python -m dfv2.report --runs-dir runs

Reads <runs-dir>/preds/*.csv (and history/*.json for learning curves). No torch needed.
Confidence intervals come from a stratified bootstrap over test *videos*: frames of one
video are strongly correlated, so resampling frames would make the intervals far too narrow.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as M  # noqa: E402

VARIANT_ORDER = ["base", "attn", "freq", "hybrid"]
VARIANT_LABEL = {"base": "Baseline", "attn": "+ Gate", "freq": "+ FFT", "hybrid": "Gate + FFT"}
COLORS = {"base": "#6b7280", "attn": "#2563eb", "freq": "#d97706", "hybrid": "#059669"}
PRED_NAME = re.compile(r"^(?P<variant>[a-z]+)_seed(?P<seed>\d+)\.csv$")
BOOT_KEYS = ("frame_auc", "frame_bal_acc", "video_auc", "video_bal_acc")
KIND = {  # how each metric is printed: share as a percentage, or AUC as a 4-decimal fraction
    "frame_acc": "pct", "frame_bal_acc": "pct", "frame_auc": "auc", "frame_eer": "pct",
    "frame_tpr_at_1pct_fpr": "pct", "frame_real_recall": "pct", "frame_fake_recall": "pct",
    "frame_real_precision": "pct", "frame_bal_acc_val_thr": "pct",
    "video_acc": "pct", "video_bal_acc": "pct", "video_auc": "auc", "video_eer": "pct",
}


@dataclass
class RunData:
    variant: str
    seed: int
    y: np.ndarray          # test frames
    p: np.ndarray
    types: np.ndarray
    codes: np.ndarray      # video index of each test frame
    video_y: np.ndarray    # one row per test video
    video_p: np.ndarray
    thr: float             # equal-error threshold chosen on validation


def load_runs(runs_dir: Path) -> tuple[list[RunData], pd.DataFrame]:
    runs: list[RunData] = []
    ref: pd.DataFrame | None = None
    for path in sorted((runs_dir / "preds").glob("*.csv")):
        m = PRED_NAME.match(path.name)
        if not m or m["variant"] not in VARIANT_LABEL:
            continue
        df = pd.read_csv(path)
        order = ["fake_type", "filename"]
        test = df[df["split"] == "test"].sort_values(order).reset_index(drop=True)
        val = df[df["split"] == "val"]
        if ref is None:
            ref = test[["filename", "fake_type", "video", "label"]]
        elif not ref[["filename", "fake_type"]].equals(test[["filename", "fake_type"]]):
            raise ValueError(f"{path.name}: test rows differ from the other runs")
        videos, codes = np.unique(test["video"].to_numpy(), return_inverse=True)
        ids, video_y, video_p = M.aggregate_by_group(test["video"], test["label"], test["p_fake"])
        assert np.array_equal(ids, videos)
        runs.append(RunData(
            variant=m["variant"], seed=int(m["seed"]),
            y=test["label"].to_numpy(), p=test["p_fake"].to_numpy(), types=test["fake_type"].to_numpy(),
            codes=codes, video_y=video_y, video_p=video_p,
            thr=M.eer_threshold(val["label"].to_numpy(), val["p_fake"].to_numpy()),
        ))
    if not runs or ref is None:
        raise SystemExit(f"no prediction files found in {runs_dir / 'preds'}")
    runs.sort(key=lambda r: (VARIANT_ORDER.index(r.variant), r.seed))
    return runs, ref


def run_stats(r: RunData) -> dict[str, float]:
    """Every reported metric for one run, on the full test set."""
    pred, pred_thr = M.predict(r.p, 0.5), M.predict(r.p, r.thr)
    vpred = M.predict(r.video_p, 0.5)
    return {
        "frame_acc": M.accuracy(r.y, pred),
        "frame_bal_acc": M.balanced_accuracy(r.y, pred),
        "frame_auc": M.auc_roc(r.y, r.p),
        "frame_eer": M.equal_error_rate(r.y, r.p),
        "frame_tpr_at_1pct_fpr": M.tpr_at_fpr(r.y, r.p, 0.01),
        "frame_real_recall": M.real_recall(r.y, pred),
        "frame_fake_recall": M.fake_recall(r.y, pred),
        "frame_real_precision": M.real_precision(r.y, pred),
        "frame_bal_acc_val_thr": M.balanced_accuracy(r.y, pred_thr),
        "video_acc": M.accuracy(r.video_y, vpred),
        "video_bal_acc": M.balanced_accuracy(r.video_y, vpred),
        "video_auc": M.auc_roc(r.video_y, r.video_p),
        "video_eer": M.equal_error_rate(r.video_y, r.video_p),
    }


def boot_stats(r: RunData, counts: np.ndarray) -> np.ndarray:
    """The four headline metrics when each video is weighted by how often it was drawn."""
    wf = counts[r.codes]
    return np.array([
        M.auc_roc(r.y, r.p, wf),
        M.balanced_accuracy(r.y, M.predict(r.p, 0.5), wf),
        M.auc_roc(r.video_y, r.video_p, counts),
        M.balanced_accuracy(r.video_y, M.predict(r.video_p, 0.5), counts),
    ])


# --- formatting ----------------------------------------------------------------

def mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def fmt(value: float, kind: str) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{100 * value:.2f}" if kind == "pct" else f"{value:.4f}"


def fmt_pm(values, kind: str) -> str:
    m, s = mean_std(values)
    return fmt(m, kind) if np.isnan(s) else f"{fmt(m, kind)} ± {fmt(s, kind)}"


def fmt_ci(point: float, lo: float, hi: float, kind: str) -> str:
    return f"{fmt(point, kind)} [{fmt(lo, kind)}, {fmt(hi, kind)}]"


def _nan_to_none(obj):
    """Make results.json strict JSON: NaN is not valid there."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return None if isinstance(obj, float) and np.isnan(obj) else obj


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# --- bootstrap -----------------------------------------------------------------

@dataclass
class Boot:
    point: np.ndarray    # (len(BOOT_KEYS),)
    samples: np.ndarray  # (n_boot, len(BOOT_KEYS))


def bootstrap_all(runs: list[RunData], n_boot: int, seed: int) -> dict[str, Boot]:
    """Seed-averaged headline metrics per variant, resampling test videos jointly for all runs."""
    k = len(BOOT_KEYS)

    def stat(counts: np.ndarray) -> np.ndarray:
        return np.concatenate([boot_stats(r, counts) for r in runs])

    point, samples = M.bootstrap_clusters(len(runs[0].video_y), stat, n_boot, seed, strata=runs[0].video_y)
    point, samples = point.reshape(len(runs), k), samples.reshape(n_boot, len(runs), k)
    out = {}
    for v in sorted({r.variant for r in runs}, key=VARIANT_ORDER.index):
        idx = [i for i, r in enumerate(runs) if r.variant == v]
        out[v] = Boot(point[idx].mean(0), samples[:, idx, :].mean(1))
    return out


def effect_boots(boots: dict[str, Boot]) -> dict[str, Boot]:
    """Differences between variants, with the same resampled videos on both sides."""
    out = {}
    if "base" in boots:
        for v in ("attn", "freq", "hybrid"):
            if v in boots:
                out[f"{VARIANT_LABEL[v]} minus Baseline"] = Boot(
                    boots[v].point - boots["base"].point, boots[v].samples - boots["base"].samples)
    if all(v in boots for v in VARIANT_ORDER):
        b = boots
        out["Gate main effect"] = Boot(
            (b["attn"].point + b["hybrid"].point) / 2 - (b["base"].point + b["freq"].point) / 2,
            (b["attn"].samples + b["hybrid"].samples) / 2 - (b["base"].samples + b["freq"].samples) / 2)
        out["FFT main effect"] = Boot(
            (b["freq"].point + b["hybrid"].point) / 2 - (b["base"].point + b["attn"].point) / 2,
            (b["freq"].samples + b["hybrid"].samples) / 2 - (b["base"].samples + b["attn"].samples) / 2)
        out["Interaction (Gate x FFT)"] = Boot(
            b["hybrid"].point - b["freq"].point - b["attn"].point + b["base"].point,
            b["hybrid"].samples - b["freq"].samples - b["attn"].samples + b["base"].samples)
    return out


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "gain"
    if hi < 0:
        return "loss"
    return "not resolved (interval includes 0)"


METRIC_LABEL = {
    "frame_auc": "frame AUC",
    "frame_bal_acc": "frame balanced acc",
    "video_auc": "video AUC",
    "video_bal_acc": "video balanced acc",
}


def short_verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "gain"
    if hi < 0:
        return "loss"
    return "unresolved"


def effect_text(key: str, point: float, lo: float, hi: float) -> str:
    """A signed difference with its interval. Balanced accuracy is a fraction, shown in percentage points."""
    if KIND[key] == "pct":
        return f"{100 * point:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] percentage points"
    return f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]"


# --- figures -------------------------------------------------------------------

def plot_roc(runs: list[RunData], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, log in zip(axes, (False, True)):
        for v in sorted({r.variant for r in runs}, key=VARIANT_ORDER.index):
            for i, r in enumerate([r for r in runs if r.variant == v]):
                fpr, tpr, _ = M.roc_points(r.y, r.p)
                ax.plot(fpr, tpr, color=COLORS[v], alpha=0.55, lw=1.2, label=VARIANT_LABEL[v] if i == 0 else None)
        ax.set_xlabel("False positive rate (real flagged as fake)")
        ax.set_ylabel("True positive rate (fakes caught)")
        if log:
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1)
            ax.set_title("Same curves, log x-axis (low false-positive region)")
        else:
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_title("ROC on the test set, one line per seed")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_manipulation(per_type_auc: dict[str, dict[str, list[float]]], path: Path) -> None:
    types = sorted({t for v in per_type_auc.values() for t in v})
    variants = [v for v in VARIANT_ORDER if v in per_type_auc]
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, v in enumerate(variants):
        means = [mean_std(per_type_auc[v][t])[0] for t in types]
        stds = [np.nan_to_num(mean_std(per_type_auc[v][t])[1]) for t in types]
        ax.bar(np.arange(len(types)) + i * width, means, width, yerr=stds, color=COLORS[v],
               label=VARIANT_LABEL[v], capsize=2)
    ax.set_xticks(np.arange(len(types)) + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylabel("AUC: this manipulation vs all real frames")
    ax.set_ylim(max(0.5, min(0.95, np.nanmin([mean_std(per_type_auc[v][t])[0] for v in per_type_auc for t in types]) - 0.03)), 1.0)
    ax.set_title("Detection quality by manipulation type (mean and std over seeds)")
    ax.legend(ncol=len(variants), loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_learning_curves(runs_dir: Path, path: Path) -> list[dict]:
    histories = []
    for f in sorted((runs_dir / "history").glob("*.json")):
        meta = json.loads(f.read_text())
        histories.append({"variant": meta["config"]["variant"], "seed": meta["config"]["seed"],
                          "history": meta["history"], "seconds": meta.get("seconds"),
                          "best_epoch": meta["summary"]["best_epoch"],
                          "config": meta["config"], "audit": meta.get("split_audit")})
    if not histories:
        return histories
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for h in histories:
        ep = [r["epoch"] for r in h["history"]]
        c = COLORS[h["variant"]]
        axes[0].plot(ep, [r["train_loss"] for r in h["history"]], color=c, alpha=0.5, label=None)
        axes[1].plot(ep, [r["val_auc"] for r in h["history"]], color=c, alpha=0.5)
    for ax, title in zip(axes, ("Training loss (label-smoothed)", "Validation AUC")):
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for v in sorted({h["variant"] for h in histories}, key=VARIANT_ORDER.index):
        axes[1].plot([], [], color=COLORS[v], label=VARIANT_LABEL[v])
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return histories


# --- report --------------------------------------------------------------------

LEAK_NOTE = (
    "In the Kaggle manifest a fake video is assigned to a split by one of its two source ids, so the other id "
    "often appears in training. Measured on that manifest: 99 of 140 test real videos appear in the training split "
    "as the second id of a fake pair, and all 25 DeepFakeDetection test actors also appear in training.")
DFD_SHORTCUT = (
    "- **DeepFakeDetection has no matching real frames.** The Real folder holds only YouTube-style videos. "
    "{n_dfd:,} of the test fakes come from DeepFakeDetection, so a model could score on them by recognising the "
    "studio scenes. Check the DeepFakeDetection row above against the others.")
DFD_DROPPED = (
    "- **DeepFakeDetection is excluded.** It has fake frames but no real frames from its own studio domain, so a "
    "model could score on it by recognising the scenes, and its actors form linked groups too large to split "
    "cleanly. Re-run with `--keep-dfd` to include it.")


def split_caveats(histories: list[dict], n_dfd: int) -> str:
    """The first two caveat bullets, chosen from what the runs recorded about their split."""
    schemes = {h["config"].get("split") for h in histories}
    dfd = DFD_SHORTCUT.format(n_dfd=n_dfd) if n_dfd else DFD_DROPPED
    if histories and schemes == {"clean"}:
        audit = next((h["audit"] for h in histories if h.get("audit")), None)
        detail = ""
        if audit:
            frames = ", ".join(f"{s} {audit['frames'][s]:,}" for s in ("train", "val", "test"))
            detail = (f" Frames: {frames}. Identities found in more than one split: "
                      f"{sum(audit['shared'].values())}.")
        head = ("- **Identity-disjoint split.** No original video or actor appears in more than one split, "
                "checked when each run started." + detail)
        return head + "\n" + dfd
    if histories and schemes == {"original"}:
        return ("- **Split leakage is present.** " + LEAK_NOTE + " Compare variants with each other, not with other "
                "papers. Re-run with `--split clean` to remove it.\n" + dfd)
    return ("- **Split not recorded.** No history files were found, or the runs used different splits, so this report "
            "cannot say which was used. " + LEAK_NOTE + " The `--split clean` option removes that.\n" + dfd)

def build_report(runs_dir: Path, out_dir: Path, n_boot: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, ref = load_runs(runs_dir)
    variants = sorted({r.variant for r in runs}, key=VARIANT_ORDER.index)
    by_variant = {v: [r for r in runs if r.variant == v] for v in variants}
    seeds_per = {v: len(rs) for v, rs in by_variant.items()}
    stats = {(r.variant, r.seed): run_stats(r) for r in runs}

    y = runs[0].y
    n_real_v, n_fake_v = int((runs[0].video_y == 0).sum()), int((runs[0].video_y == 1).sum())
    base_frame = M.majority_baseline(y)
    base_video = M.majority_baseline(runs[0].video_y)
    n_dfd = int(((runs[0].types == "DeepFakeDetection") & (y == 1)).sum())

    print(f"bootstrapping {len(runs)} runs x {n_boot} resamples ...")
    boots = bootstrap_all(runs, n_boot, seed)
    effects = effect_boots(boots)

    def col(v: str, key: str) -> list[float]:
        return [stats[(r.variant, r.seed)][key] for r in by_variant[v]]

    # tables
    frame_cols = [("frame_acc", "Accuracy @0.5"), ("frame_bal_acc", "Balanced acc @0.5"), ("frame_auc", "AUC"),
                  ("frame_eer", "EER"), ("frame_tpr_at_1pct_fpr", "TPR @1% FPR"),
                  ("frame_real_recall", "Real recall"), ("frame_fake_recall", "Fake recall"),
                  ("frame_real_precision", "Real precision"), ("frame_bal_acc_val_thr", "Bal acc @val-EER thr")]
    video_cols = [("video_acc", "Accuracy @0.5"), ("video_bal_acc", "Balanced acc @0.5"),
                  ("video_auc", "AUC"), ("video_eer", "EER")]

    def table(cols):
        rows = [[VARIANT_LABEL[v], str(seeds_per[v])] + [fmt_pm(col(v, k), KIND[k]) for k, _ in cols] for v in variants]
        return md_table(["Model", "Seeds"] + [h for _, h in cols], rows)

    ci_rows, results_json = [], {"ci": {}, "effects": {}, "per_variant": {}}
    for v in variants:
        b = boots[v]
        lo, hi = M.percentile_ci(b.samples)
        row = [VARIANT_LABEL[v]]
        for j, key in enumerate(BOOT_KEYS):
            row.append(fmt_ci(b.point[j], lo[j], hi[j], KIND[key]))
            results_json["ci"].setdefault(v, {})[key] = [float(b.point[j]), float(lo[j]), float(hi[j])]
        ci_rows.append(row)

    effect_rows = []
    for name, b in effects.items():
        lo, hi = M.percentile_ci(b.samples)
        row = [name]
        for j, key in enumerate(BOOT_KEYS):
            row.append(f"{fmt_ci(b.point[j], lo[j], hi[j], 'pct' if KIND[key] == 'pct' else 'auc')}")
            results_json["effects"].setdefault(name, {})[key] = [float(b.point[j]), float(lo[j]), float(hi[j])]
        row.append("; ".join(f"{METRIC_LABEL[k]}: {short_verdict(lo[j], hi[j])}" for j, k in enumerate(BOOT_KEYS)))
        effect_rows.append(row)

    # per manipulation
    types = sorted({t for t in runs[0].types if t != "Real"})
    per_auc: dict[str, dict[str, list[float]]] = {v: {t: [] for t in types} for v in variants}
    per_acc: dict[str, dict[str, list[float]]] = {v: {t: [] for t in types + ["Real"]} for v in variants}
    for r in runs:
        for t, a in M.per_group_auc(r.y, r.p, r.types).items():
            per_auc[r.variant][t].append(a)
        for t, a in M.per_group_accuracy(r.y, M.predict(r.p), r.types).items():
            per_acc[r.variant][t].append(a)
    auc_rows = [[t] + [fmt_pm(per_auc[v][t], "auc") for v in variants] for t in types]
    acc_rows = [[t] + [fmt_pm(per_acc[v][t], "pct") for v in variants] for t in types + ["Real"]]
    n_per_type = {t: int((runs[0].types == t).sum()) for t in types + ["Real"]}

    plot_roc(runs, out_dir / "roc_frame.png")
    plot_per_manipulation(per_auc, out_dir / "per_manipulation_auc.png")
    histories = plot_learning_curves(runs_dir, out_dir / "learning_curves.png")

    for v in variants:
        results_json["per_variant"][v] = {k: mean_std(col(v, k)) for k in KIND}
    (out_dir / "results.json").write_text(json.dumps(_nan_to_none(results_json), indent=2))

    # headline sentences, generated from the numbers
    # every effect on every headline metric, plus the context needed to read them
    bullets, n_comparisons = [], 0
    for name, b in effects.items():
        lo, hi = M.percentile_ci(b.samples)
        for j, key in enumerate(BOOT_KEYS):
            n_comparisons += 1
            bullets.append(f"- **{name}**, {METRIC_LABEL[key]}: {effect_text(key, b.point[j], lo[j], hi[j])}: "
                           f"{verdict(lo[j], hi[j])}.")
    max_seeds = max(seeds_per.values())
    seed_note = (
        f"Each variant has {max_seeds} training seed{'s' if max_seeds != 1 else ''}. These intervals measure "
        "test-set uncertainty only; training randomness is what the ± values estimate"
        + (", and with fewer than five seeds those estimates are rough." if max_seeds < 5 else "."))
    chance_note = (
        f"This table makes {n_comparisons} comparisons. If none had an effect, the chance that at least one interval "
        f"excludes zero would be about {100 * (1 - 0.95 ** n_comparisons):.0f}% (this assumes independent comparisons, "
        "so it overstates the risk a little). Treat an isolated gain or loss as a lead for more seeds, not a finding.")

    last_epoch_best = [h for h in histories if h["best_epoch"] == len(h["history"])]
    training_note = ""
    if histories:
        training_note = (
            f"{len(last_epoch_best)} of {len(histories)} runs had their best validation AUC in the final epoch"
            + (", so training had not clearly converged and longer runs could change the picture."
               if len(last_epoch_best) > len(histories) / 2 else "."))

    majority_label = "fake" if (y == 1).mean() >= 0.5 else "real"
    split_text = split_caveats(histories, n_dfd)
    variants_line = ", ".join("%s (%d seeds)" % (VARIANT_LABEL[v], seeds_per[v]) for v in variants)
    support_text = "\n".join(bullets) if bullets else "_Only one variant present; no comparison possible._"
    if bullets:
        support_text += "\n\n" + seed_note + ("\n\n" + chance_note if n_comparisons else "")
    ci_table = md_table(["Model", "Frame AUC", "Frame balanced acc", "Video AUC", "Video balanced acc"], ci_rows)
    effect_table = (
        md_table(["Comparison", "Frame AUC", "Frame balanced acc", "Video AUC", "Video balanced acc",
                  "Verdicts (each metric)"], effect_rows)
        if effect_rows else "_No comparisons available._")
    auc_table = md_table(["Manipulation"] + [VARIANT_LABEL[v] for v in variants], auc_rows)
    acc_table = md_table(["Type"] + [VARIANT_LABEL[v] for v in variants], acc_rows)
    per_type_counts = ", ".join("%s %s" % (t, format(n, ",")) for t, n in n_per_type.items())

    md = f"""# Matched 2×2 ablation: report

Variants: {variants_line}.
Every run shares one protocol: same data, same epochs, same loss, same scheduler, same dense head. Only the component under test changes.

**Test set:** {len(y):,} frames from {n_real_v + n_fake_v:,} videos ({n_real_v} real, {n_fake_v} fake); {100 * (y == 1).mean():.1f}% of frames are fake.
Always answering "{majority_label}" would score **{100 * base_frame:.2f}%** accuracy per frame and {100 * base_video:.2f}% per video. Accuracy numbers only mean something relative to that.

## What the data supports

Differences are shown for frame and video AUC (0 to 1 scale) and for balanced accuracy (in percentage points). Each has a 95% interval from resampling test videos. A "gain" or "loss" means the whole interval lies on one side of zero. "Not resolved" means the test set cannot separate the effect from zero; it does not mean the effect is zero.

{support_text}

## How to read the numbers

- **AUC**: probability that a random fake frame scores higher than a random real one. Ignores the threshold and the real/fake class split.
- **Balanced accuracy**: mean of recall on fakes and recall on reals. Not affected by the class split.
- **EER**: error rate at the threshold where wrongly-flagged reals equal missed fakes. Lower is better.
- **TPR @1% FPR**: share of fakes caught while flagging at most 1% of real faces.
- **Real precision**: of faces called real, the share that are. It falls when fakes are common even if the model treats both classes equally, so it is not a measure of bias toward "fake".
- **Bal acc @val-EER thr**: balanced accuracy at a threshold chosen on validation data, not on test.
- **Video-level**: the mean P(fake) over a video's frames, one decision per video.
- **± values** are the standard deviation across training seeds. **Intervals** are test-set uncertainty. They answer different questions.

## 1. Frame-level metrics (mean ± std over seeds; percentages except AUC)

{table(frame_cols)}

## 2. Video-level metrics (mean ± std over seeds)

{table(video_cols)}

## 3. Test-set uncertainty (seed-averaged value and 95% interval over resampled videos)

{ci_table}

## 4. Component effects (same resampled videos on both sides)

Differences are in AUC units (0 to 1) for the AUC columns and in percentage points for the balanced-accuracy columns. The gate and FFT main effects average over the other component; the interaction asks whether the two help more (or less) together than separately.

{effect_table}

## 5. By manipulation type

AUC of each fake type against all real test frames (mean ± std over seeds). Frames per type in the test set: {per_type_counts}.

{auc_table}

Accuracy at threshold 0.5 per type; the Real row is recall on real faces.

{acc_table}

## 6. Training behaviour

{training_note or "No history files found."} See `learning_curves.png`. The best epoch was chosen on validation AUC and the test set was read once afterwards.

## 7. Caveats that these numbers do not remove

{split_text}
- **FaceForensics++ C23 only.** Nothing here tests other datasets or other compression levels.
- **The frequency branch is the original design.** It pools away the frequency positions and reads 224×224 JPEG crops after resizing and augmentation. A negative or null FFT result here says this branch does not help, not that frequency cues cannot.
- **The baseline here is not the original baseline.** It uses the same three-layer dense head as the other variants, so the comparison isolates the gate and the FFT branch.

## Figures

`roc_frame.png`, `per_manipulation_auc.png`, `learning_curves.png`, and every number above as `results.json`.
"""
    report_path = out_dir / "report.md"
    report_path.write_text(md)
    return report_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default=None, help="default: <runs-dir>/report")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    path = build_report(runs_dir, Path(args.out_dir) if args.out_dir else runs_dir / "report", args.n_boot, args.seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
