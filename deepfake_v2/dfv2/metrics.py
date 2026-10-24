"""Evaluation metrics for binary real/fake detection.

Convention everywhere: label 1 = fake (the positive class), label 0 = real,
and a score is P(fake). Every metric accepts optional per-row weights `w`. The
cluster bootstrap uses that to resample whole videos: a video drawn k times gets
weight k on each of its frames, so no rows are copied.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _weights(n: int, w) -> np.ndarray:
    return np.ones(n) if w is None else np.asarray(w, dtype=float)


def _ranking_inputs(y, score, w):
    """Cast inputs and drop zero-weight rows (they cannot change a ranking metric)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    w = _weights(len(y), w)
    keep = w > 0
    return y[keep], score[keep], w[keep]


def _has_both_classes(y: np.ndarray, w: np.ndarray) -> bool:
    return w[y == 1].sum() > 0 and w[y == 0].sum() > 0


# --- threshold metrics -------------------------------------------------------

def predict(score, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(score) >= threshold).astype(int)


def confusion_counts(y, pred, w=None) -> tuple[float, float, float, float]:
    """Return (tn, fp, fn, tp) as floats."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    w = _weights(len(y), w)
    tn = w[(y == 0) & (pred == 0)].sum()
    fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum()
    tp = w[(y == 1) & (pred == 1)].sum()
    return float(tn), float(fp), float(fn), float(tp)


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def accuracy(y, pred, w=None) -> float:
    tn, fp, fn, tp = confusion_counts(y, pred, w)
    return _ratio(tn + tp, tn + fp + fn + tp)


def balanced_accuracy(y, pred, w=None) -> float:
    """Mean of recall on fakes (TPR) and recall on reals (TNR); insensitive to prevalence."""
    tn, fp, fn, tp = confusion_counts(y, pred, w)
    return (_ratio(tp, tp + fn) + _ratio(tn, tn + fp)) / 2


def fake_recall(y, pred, w=None) -> float:
    tn, fp, fn, tp = confusion_counts(y, pred, w)
    return _ratio(tp, tp + fn)


def real_recall(y, pred, w=None) -> float:
    tn, fp, fn, tp = confusion_counts(y, pred, w)
    return _ratio(tn, tn + fp)


def real_precision(y, pred, w=None) -> float:
    """Of the faces called real, the share that are real. Depends on prevalence."""
    tn, fp, fn, tp = confusion_counts(y, pred, w)
    return _ratio(tn, tn + fn)


def majority_baseline(y, w=None) -> float:
    """Accuracy of always predicting the more common class."""
    y = np.asarray(y).astype(int)
    w = _weights(len(y), w)
    pos, neg = w[y == 1].sum(), w[y == 0].sum()
    return _ratio(max(pos, neg), pos + neg)


# --- ranking metrics ---------------------------------------------------------

def roc_points(y, score, w=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC polygon with exact tie handling: (fpr, tpr, thresholds).

    A face is called fake when score >= threshold. The first point is (0, 0) at an
    infinite threshold. Raises ValueError if one class is missing.
    """
    y, score, w = _ranking_inputs(y, score, w)
    if not _has_both_classes(y, w):
        raise ValueError("ROC needs both classes")
    uniq, inverse = np.unique(score, return_inverse=True)
    pos = np.bincount(inverse, weights=w * (y == 1), minlength=len(uniq))
    neg = np.bincount(inverse, weights=w * (y == 0), minlength=len(uniq))
    tpr = np.concatenate([[0.0], np.cumsum(pos[::-1]) / pos.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(neg[::-1]) / neg.sum()])
    thresholds = np.concatenate([[np.inf], uniq[::-1]])
    return fpr, tpr, thresholds


def auc_roc(y, score, w=None) -> float:
    try:
        fpr, tpr, _ = roc_points(y, score, w)
    except ValueError:
        return float("nan")
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def _eer_crossing(fpr: np.ndarray, tpr: np.ndarray) -> tuple[int, float]:
    """First ROC point at or past the line fpr = 1 - tpr, and the interpolated EER."""
    d = fpr + tpr - 1.0  # starts at -1, ends at +1, never decreases
    i = int(np.argmax(d >= 0))
    t = -d[i - 1] / (d[i] - d[i - 1])
    return i, float(fpr[i - 1] + t * (fpr[i] - fpr[i - 1]))


def equal_error_rate(y, score, w=None) -> float:
    """Error rate at the operating point where false-accept rate equals false-reject rate."""
    try:
        fpr, tpr, _ = roc_points(y, score, w)
    except ValueError:
        return float("nan")
    return _eer_crossing(fpr, tpr)[1]


def eer_threshold(y, score, w=None) -> float:
    """Score threshold at the equal-error point. Choose it on validation data, never on test."""
    fpr, tpr, thresholds = roc_points(y, score, w)
    return float(thresholds[_eer_crossing(fpr, tpr)[0]])


def tpr_at_fpr(y, score, fpr_target: float, w=None) -> float:
    """Share of fakes caught while flagging at most `fpr_target` of real faces."""
    try:
        fpr, tpr, _ = roc_points(y, score, w)
    except ValueError:
        return float("nan")
    return float(tpr[np.searchsorted(fpr, fpr_target, side="right") - 1])


# --- grouping ----------------------------------------------------------------

def aggregate_by_group(groups, y, score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse frames to one row per group (video): mean score, shared label.

    Returns (group_ids sorted, labels, mean_scores).
    """
    ids, codes = np.unique(np.asarray(groups), return_inverse=True)
    counts = np.bincount(codes)
    label_sum = np.bincount(codes, weights=np.asarray(y, dtype=float))
    if not np.all((label_sum == 0) | (label_sum == counts)):
        raise ValueError("labels differ within a group")
    mean_score = np.bincount(codes, weights=np.asarray(score, dtype=float)) / counts
    return ids, (label_sum > 0).astype(int), mean_score


def per_group_auc(y, score, groups, negative_group: str = "Real", w=None) -> dict[str, float]:
    """AUC of each non-negative group against every row of the negative group."""
    y = np.asarray(y)
    score = np.asarray(score)
    groups = np.asarray(groups)
    w = None if w is None else np.asarray(w)
    out = {}
    for g in np.unique(groups):
        if g == negative_group:
            continue
        m = (groups == g) | (groups == negative_group)
        out[str(g)] = auc_roc(y[m], score[m], None if w is None else w[m])
    return out


def per_group_accuracy(y, pred, groups, w=None) -> dict[str, float]:
    """Accuracy inside each group. For the real group this is recall on reals."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    groups = np.asarray(groups)
    w = _weights(len(y), w)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = _ratio(w[m & (y == pred)].sum(), w[m].sum())
    return out


# --- cluster bootstrap -------------------------------------------------------

def bootstrap_clusters(
    n_clusters: int,
    stat_fn: Callable[[np.ndarray], object],
    n_boot: int = 1000,
    seed: int = 0,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample clusters (videos) with replacement and evaluate `stat_fn` on each draw.

    `stat_fn(counts)` receives how many times each cluster was drawn (length
    `n_clusters`) and returns a scalar or 1-D array. With `strata`, clusters are
    resampled within their stratum, which keeps the real/fake video counts fixed.
    Returns (point_estimate, samples) with shapes (k,) and (n_boot, k).
    """
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(np.asarray(stat_fn(np.ones(n_clusters)), dtype=float))
    groups = [np.arange(n_clusters)] if strata is None else [
        np.flatnonzero(strata == s) for s in np.unique(strata)
    ]
    samples = np.empty((n_boot, point.size))
    for b in range(n_boot):
        counts = np.zeros(n_clusters)
        for idx in groups:
            counts[idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
        samples[b] = stat_fn(counts)
    return point, samples


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return np.atleast_1d(lo), np.atleast_1d(hi)
