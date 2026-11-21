"""Generalization report: how detectors do on a manipulation they never saw, and on degraded frames.

    python -m dfv2.generalization --runs-dir runs

Reads the hold-out runs written by `run_ablation --holdouts ... --robustness`:
<runs-dir>/preds/<variant>_hold-<type>_seed<n>.csv and, when present, <runs-dir>/robust/<same name>.

"Unseen" AUC is the held-out manipulation against all real test frames. "Seen" AUC is every other
manipulation against the same real frames. Intervals come from a stratified bootstrap over test
videos, resampled jointly for every run, so comparisons between variants are paired. No torch needed.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as M  # noqa: E402
from .manifest import MANIPULATIONS  # noqa: E402
from .report import COLORS, VARIANT_LABEL, VARIANT_ORDER, _nan_to_none, fmt_pm, md_table, mean_std, verdict  # noqa: E402

HOLD_NAME = re.compile(r"^(?P<variant>[a-z]+)_hold-(?P<holdout>[A-Za-z0-9]+)_seed(?P<seed>\d+)\.csv$")
UNSEEN, SEEN = 0, 1  # the first two metric columns; degraded seen-AUCs follow


@dataclass
class FoldRun:
    variant: str
    holdout: str
    seed: int
    p: np.ndarray                                   # P(fake) for every test frame, in a shared order
    degraded: dict[str, np.ndarray] = field(default_factory=dict)  # degradation name -> P(fake), same order


@dataclass
class TestFrames:
    y: np.ndarray        # label of each test frame
    types: np.ndarray    # manipulation type of each test frame ("Real" for real frames)
    codes: np.ndarray    # index of the test video each frame belongs to
    video_y: np.ndarray  # label of each test video


def _in_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["fake_type", "filename"]).reset_index(drop=True)


def load_runs(runs_dir: Path) -> tuple[list[FoldRun], TestFrames]:
    runs: list[FoldRun] = []
    ref: pd.DataFrame | None = None
    for path in sorted((runs_dir / "preds").glob("*.csv")):
        m = HOLD_NAME.match(path.name)
        if not m or m["variant"] not in VARIANT_LABEL:
            continue
        df = pd.read_csv(path)
        test = _in_order(df[df["split"] == "test"])
        if ref is None:
            ref = test[["filename", "fake_type", "video", "label"]]
        elif not ref[["filename", "fake_type"]].equals(test[["filename", "fake_type"]]):
            raise ValueError(f"{path.name}: test rows differ from the other runs")
        degraded: dict[str, np.ndarray] = {}
        robust_path = runs_dir / "robust" / path.name
        if robust_path.exists():
            for name, part in pd.read_csv(robust_path).groupby("perturbation", sort=False):
                part = _in_order(part)
                if not ref[["filename", "fake_type"]].equals(part[["filename", "fake_type"]]):
                    raise ValueError(f"{robust_path.name}: degraded rows ({name}) differ from the clean test rows")
                degraded[str(name)] = part["p_fake"].to_numpy()
        runs.append(FoldRun(m["variant"], m["holdout"], int(m["seed"]), test["p_fake"].to_numpy(), degraded))
    if not runs or ref is None:
        raise SystemExit(f"no hold-out predictions found in {runs_dir / 'preds'}")
    runs.sort(key=lambda r: (VARIANT_ORDER.index(r.variant), r.holdout, r.seed))
    videos, codes = np.unique(ref["video"].to_numpy(), return_inverse=True)
    video_y = np.zeros(len(videos), dtype=int)
    video_y[codes] = ref["label"].to_numpy()
    return runs, TestFrames(ref["label"].to_numpy(), ref["fake_type"].to_numpy(), codes, video_y)


def bootstrap_all(runs: list[FoldRun], frames: TestFrames, perturbations: list[str], n_boot: int, seed: int):
    """Metrics for every run under resampled test videos.

    Columns: unseen AUC, seen AUC, then the seen AUC under each degradation. Returns the point estimate
    shaped (1, n_runs, K) and the bootstrap samples shaped (n_boot, n_runs, K).
    """
    masks = {h: ((frames.types == h) | (frames.types == "Real"), frames.types != h) for h in {r.holdout for r in runs}}
    evaluators = []
    for r in runs:
        unseen_mask, seen_mask = masks[r.holdout]
        evals = [M.AucEvaluator(frames.y[unseen_mask], r.p[unseen_mask]),
                 M.AucEvaluator(frames.y[seen_mask], r.p[seen_mask])]
        evals += [M.AucEvaluator(frames.y[seen_mask], r.degraded[q][seen_mask]) for q in perturbations]
        evaluators.append(evals)

    def stat(counts: np.ndarray) -> np.ndarray:
        w = counts[frames.codes]
        out = []
        for r, evals in zip(runs, evaluators):
            unseen_mask, seen_mask = masks[r.holdout]
            out.append(evals[UNSEEN](w[unseen_mask]))
            out += [e(w[seen_mask]) for e in evals[SEEN:]]
        return np.array(out)

    point, samples = M.bootstrap_clusters(len(frames.video_y), stat, n_boot, seed, strata=frames.video_y)
    k = 2 + len(perturbations)
    return point.reshape(1, len(runs), k), samples.reshape(n_boot, len(runs), k)


def fold_mean(A: np.ndarray, runs: list[FoldRun], variant: str, holdout: str) -> np.ndarray:
    """Average over seeds for one variant and one fold. A is shaped (draws, runs, metrics)."""
    idx = [i for i, r in enumerate(runs) if r.variant == variant and r.holdout == holdout]
    return A[:, idx, :].mean(axis=1)


def macro(A: np.ndarray, runs: list[FoldRun], variant: str, holdouts: list[str]) -> np.ndarray:
    """Average over folds of the per-fold seed averages."""
    return np.mean([fold_mean(A, runs, variant, h) for h in holdouts], axis=0)


def interval(point: float, samples: np.ndarray) -> list[float]:
    lo, hi = M.percentile_ci(np.asarray(samples)[:, None])
    return [float(point), float(lo[0]), float(hi[0])]


def ci_text(p: float, lo: float, hi: float) -> str:
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def diff_text(p: float, lo: float, hi: float) -> str:
    return f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def gap_verdict(lo: float, hi: float) -> str:
    """A smaller seen-minus-unseen gap is better, so the sign convention is reversed."""
    return verdict(-hi, -lo)


# --- figures -------------------------------------------------------------------

def axis_bottom(values) -> float:
    """Lower y-limit for an AUC plot: tight when every value is good, but it never hides a value."""
    low = float(min(values))
    bottom = low - 0.05
    if low >= 0.5:
        bottom = max(0.5, min(0.9, bottom))
    return max(0.0, bottom)


def plot_seen_unseen(per_fold: dict, variants: list[str], holdouts: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.8 / len(variants)
    x0 = np.arange(len(holdouts))
    for i, v in enumerate(variants):
        x = x0 + i * width
        unseen = [per_fold[v][h]["unseen"][0] for h in holdouts]
        err = [np.nan_to_num(per_fold[v][h]["unseen"][1]) for h in holdouts]
        seen = [per_fold[v][h]["seen"][0] for h in holdouts]
        ax.bar(x, unseen, width, yerr=err, color=COLORS[v], label=f"{VARIANT_LABEL[v]}: unseen type", capsize=2)
        ax.scatter(x, seen, marker="_", s=420, color="black", zorder=3, label="seen types" if i == 0 else None)
    ax.set_xticks(x0 + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(holdouts, rotation=15, ha="right")
    ax.set_ylabel("AUC against real test frames")
    ax.set_xlabel("held-out manipulation")
    shown = [per_fold[v][h][k][0] for v in variants for h in holdouts for k in ("unseen", "seen")]
    ax.set_ylim(axis_bottom(shown), 1.0)
    ax.set_title("Bars: the held-out type, never seen in training. Black marks: the other types, seen in training")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_robustness(robust: dict, variants: list[str], names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(names))
    for i, v in enumerate(variants):
        point = np.array([robust[v][n][0] for n in names])
        lo = np.array([robust[v][n][1] for n in names])
        hi = np.array([robust[v][n][2] for n in names])
        ax.errorbar(x + 0.06 * (i - (len(variants) - 1) / 2), point, yerr=[point - lo, hi - point],
                    color=COLORS[v], marker="o", capsize=3, label=VARIANT_LABEL[v])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("AUC on the seen manipulations")
    ax.set_title("Detection quality as the test frames are degraded")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- report --------------------------------------------------------------------

def build_report(runs_dir: Path, out_dir: Path, n_boot: int = 1000, seed: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, frames = load_runs(runs_dir)
    variants = sorted({r.variant for r in runs}, key=VARIANT_ORDER.index)
    shared = set.intersection(*({r.holdout for r in runs if r.variant == v} for v in variants))
    if not shared:
        raise SystemExit("the variants have no held-out type in common")
    holdouts = sorted(shared, key=lambda h: (MANIPULATIONS.index(h) if h in MANIPULATIONS else len(MANIPULATIONS), h))
    runs = [r for r in runs if r.holdout in shared]
    perturbations = [q for q in runs[0].degraded if all(q in r.degraded for r in runs)]
    seeds_per = {v: max(sum(1 for r in runs if r.variant == v and r.holdout == h) for h in holdouts) for v in variants}

    print(f"bootstrapping {len(runs)} runs x {n_boot} resamples ...")
    point, samples = bootstrap_all(runs, frames, perturbations, n_boot, seed)
    macro_point = {v: macro(point, runs, v, holdouts) for v in variants}   # each (1, K)
    macro_samp = {v: macro(samples, runs, v, holdouts) for v in variants}  # each (draws, K)
    base = "base" if "base" in variants else variants[0]
    others = [v for v in variants if v != base]
    cols = ["clean", *perturbations]  # the seen-AUC columns, in order, after the unseen column

    def seen_col(name: str) -> int:
        return SEEN + cols.index(name)

    def mean_of(v: str, h: str, j: int) -> list[float]:
        return [float(point[0, i, j]) for i, r in enumerate(runs) if r.variant == v and r.holdout == h]

    results: dict = {"folds": holdouts, "variants": variants, "seeds": seeds_per, "macro": {}, "per_fold": {},
                     "comparison": {}, "robustness": {}, "robustness_vs_base": {}}

    # per fold, mean and std over seeds
    per_fold: dict = {v: {} for v in variants}
    for v in variants:
        for h in holdouts:
            unseen, seen = mean_of(v, h, UNSEEN), mean_of(v, h, SEEN)
            gap = [s - u for s, u in zip(seen, unseen)]
            per_fold[v][h] = {"unseen": mean_std(unseen), "seen": mean_std(seen), "gap": mean_std(gap)}
        results["per_fold"][v] = {h: {k: list(x) for k, x in d.items()} for h, d in per_fold[v].items()}

    # macro over folds, with intervals over test videos
    for v in variants:
        P, S = macro_point[v][0], macro_samp[v]
        results["macro"][v] = {
            "unseen_auc": interval(P[UNSEEN], S[:, UNSEEN]),
            "seen_auc": interval(P[SEEN], S[:, SEEN]),
            "gap": interval(P[SEEN] - P[UNSEEN], S[:, SEEN] - S[:, UNSEEN]),
        }
        results["robustness"][v] = {n: interval(P[seen_col(n)], S[:, seen_col(n)]) for n in cols}

    # paired comparisons with the baseline
    bullets, n_comparisons = [], 0

    def add(label: str, what: str, iv: list[float], reverse: bool = False) -> None:
        nonlocal n_comparisons
        n_comparisons += 1
        word = gap_verdict(iv[1], iv[2]) if reverse else verdict(iv[1], iv[2])
        bullets.append(f"- **{label}**, {what}: {diff_text(*iv)}: {word}.")

    for v in others:
        Pv, Sv, Pb, Sb = macro_point[v][0], macro_samp[v], macro_point[base][0], macro_samp[base]
        label = f"{VARIANT_LABEL[v]} minus {VARIANT_LABEL[base]}"
        cmp_ = {
            "unseen_auc": interval(Pv[UNSEEN] - Pb[UNSEEN], Sv[:, UNSEEN] - Sb[:, UNSEEN]),
            "seen_auc": interval(Pv[SEEN] - Pb[SEEN], Sv[:, SEEN] - Sb[:, SEEN]),
            "gap": interval((Pv[SEEN] - Pv[UNSEEN]) - (Pb[SEEN] - Pb[UNSEEN]),
                            (Sv[:, SEEN] - Sv[:, UNSEEN]) - (Sb[:, SEEN] - Sb[:, UNSEEN])),
            "unseen_auc_by_fold": {},
        }
        add(label, "AUC on the unseen manipulation (average over folds)", cmp_["unseen_auc"])
        add(label, "AUC on the seen manipulations", cmp_["seen_auc"])
        add(label, "generalization gap, seen minus unseen (smaller is better)", cmp_["gap"], reverse=True)
        for h in holdouts:
            fv, fb = fold_mean(point, runs, v, h)[0], fold_mean(point, runs, base, h)[0]
            sv, sb = fold_mean(samples, runs, v, h), fold_mean(samples, runs, base, h)
            cmp_["unseen_auc_by_fold"][h] = interval(fv[UNSEEN] - fb[UNSEEN], sv[:, UNSEEN] - sb[:, UNSEEN])
            add(label, f"AUC on unseen {h}", cmp_["unseen_auc_by_fold"][h])
        results["comparison"][v] = cmp_
        results["robustness_vs_base"][v] = {}
        for n in perturbations:
            j = seen_col(n)
            iv = interval(Pv[j] - Pb[j], Sv[:, j] - Sb[:, j])
            results["robustness_vs_base"][v][n] = iv
            add(label, f"AUC on the seen manipulations under {n}", iv)

    plot_seen_unseen(per_fold, variants, holdouts, out_dir / "seen_vs_unseen.png")
    if perturbations:
        plot_robustness(results["robustness"], variants, cols, out_dir / "robustness.png")
    (out_dir / "results.json").write_text(json.dumps(_nan_to_none(results), indent=2))

    # ---- text -----------------------------------------------------------------
    max_seeds = max(seeds_per.values())
    seed_note = (f"Each variant has up to {max_seeds} training seed{'s' if max_seeds != 1 else ''} per fold. These "
                 "intervals measure test-set uncertainty only; training randomness is what the ± values estimate"
                 + (", and with fewer than five seeds those estimates are rough." if max_seeds < 5 else "."))
    chance_note = (f"This report makes {n_comparisons} comparisons. If none had an effect, the chance that at least "
                   f"one interval excludes zero would be about {100 * (1 - 0.95 ** n_comparisons):.0f}% (this assumes "
                   "independent comparisons, so it overstates the risk a little). Treat an isolated gain or loss as "
                   "a lead for more seeds, not a finding.")
    support = "\n".join(bullets) + "\n\n" + seed_note + "\n\n" + chance_note if bullets else (
        "_Only one variant present; no comparison possible._")

    fold_rows = [[h, VARIANT_LABEL[v], fmt_pm([x for x in mean_of(v, h, UNSEEN)], "auc"),
                  fmt_pm(mean_of(v, h, SEEN), "auc"),
                  fmt_pm([s - u for s, u in zip(mean_of(v, h, SEEN), mean_of(v, h, UNSEEN))], "auc")]
                 for h in holdouts for v in variants]
    fold_table = md_table(["Held-out type", "Model", "Unseen AUC", "Seen AUC", "Gap (seen − unseen)"], fold_rows)
    macro_rows = [[VARIANT_LABEL[v], ci_text(*results["macro"][v]["unseen_auc"]), ci_text(*results["macro"][v]["seen_auc"]),
                   diff_text(*results["macro"][v]["gap"])] for v in variants]
    macro_table = md_table(["Model", "Unseen AUC", "Seen AUC", "Gap"], macro_rows)

    if others:
        cmp_rows = []
        for v in others:
            c = results["comparison"][v]
            cmp_rows.append([VARIANT_LABEL[v], diff_text(*c["unseen_auc"]), diff_text(*c["seen_auc"]),
                             diff_text(*c["gap"]), gap_verdict(c["gap"][1], c["gap"][2])])
        cmp_table = md_table(["Comparison with " + VARIANT_LABEL[base], "Unseen AUC", "Seen AUC", "Gap", "Gap verdict"], cmp_rows)
        by_fold_rows = [[h] + [diff_text(*results["comparison"][v]["unseen_auc_by_fold"][h]) for v in others]
                        for h in holdouts]
        by_fold_table = md_table(["Unseen type"] + [f"{VARIANT_LABEL[v]} minus {VARIANT_LABEL[base]}" for v in others],
                                 by_fold_rows)
        comparison_text = cmp_table + "\n\nUnseen AUC difference, fold by fold:\n\n" + by_fold_table
    else:
        comparison_text = "_Only one variant present; no comparison possible._"

    if perturbations:
        rob_rows = [[VARIANT_LABEL[v]] + [ci_text(*results["robustness"][v][n]) for n in cols] for v in variants]
        rob_table = md_table(["Model"] + cols, rob_rows)
        rob_diff = ""
        if others:
            rows = [[VARIANT_LABEL[v]] + [diff_text(*results["robustness_vs_base"][v][n]) for n in perturbations]
                    for v in others]
            rob_diff = (f"\n\nDifference from {VARIANT_LABEL[base]} (same resampled videos on both sides):\n\n"
                        + md_table(["Model"] + perturbations, rows))
        robustness_text = ("AUC on the seen manipulations against real frames, with the test frames degraded "
                           "after training (macro average over folds, 95% interval over test videos). See "
                           "`robustness.png`.\n\n" + rob_table + rob_diff)
    else:
        robustness_text = ("No degraded predictions found. Run the folds with `--robustness` to score the test "
                           "frames under JPEG compression, blur and downscaling.")

    md = f"""# Generalization report: unseen manipulations and degraded frames

Variants: {", ".join("%s (%d seed%s per fold)" % (VARIANT_LABEL[v], seeds_per[v], "" if seeds_per[v] == 1 else "s") for v in variants)}.
Held-out manipulations (one fold each): {", ".join(holdouts)}.

Each fold trains on real faces and every manipulation type except the held-out one, and the held-out type is also removed from validation, so it plays no part in choosing the best epoch. All folds are tested on the same identity-disjoint test frames. **Unseen AUC** compares the held-out type with all real test frames. **Seen AUC** compares the other types with the same real frames. The gap between them is how much a model loses on a manipulation it has not met.

## What the data supports

Differences are in AUC (0 to 1 scale), with a 95% interval from resampling test videos. A "gain" or "loss" means the whole interval lies on one side of zero. "Not resolved" means the test set cannot separate the effect from zero; it does not mean the effect is zero.

{support}

## 1. Seen and unseen manipulations

Mean ± std over seeds, per fold:

{fold_table}

Average over folds, with 95% intervals over test videos:

{macro_table}

![seen versus unseen AUC](seen_vs_unseen.png)

## 2. Comparison with the baseline

{comparison_text}

## 3. Robustness to degradations

{robustness_text}

## Caveats that these numbers do not remove

- **"Unseen" means one manipulation method.** The held-out type's test videos are made from the same kind of source footage as the seen types, so this tests new manipulation artifacts, not new people or new cameras.
- **No other dataset.** Nothing here is tested on Celeb-DF, DFDC or any dataset other than FaceForensics++ C23.
- **The degradations are synthetic.** JPEG re-encoding, Gaussian blur and downscaling are controlled stand-ins for compression and resizing in the wild. They are not real social-media pipelines.
- **The folds share test videos,** so fold-level results are not independent, and the intervals reflect one test set.
- **Few seeds.** The ± values are rough, and the intervals ignore training randomness.
- **The models are the version 2 models:** five epochs, one learning rate, the original FFT branch (global average pooling, which discards frequency position).
"""
    path = out_dir / "generalization.md"
    path.write_text(md)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default=None, help="default: <runs-dir>/generalization")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    path = build_report(runs_dir, Path(args.out_dir) if args.out_dir else runs_dir / "generalization",
                        args.n_boot, args.seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
