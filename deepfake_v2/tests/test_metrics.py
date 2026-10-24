import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dfv2 import metrics as M


def toy(n=600, seed=0, round_to=None):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.7).astype(int)
    score = rng.normal(loc=y * 1.0, scale=1.0)
    if round_to is not None:
        score = np.round(score, round_to)
    return y, score


# --- threshold metrics -------------------------------------------------------

def test_confusion_and_derived_metrics():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.8, 0.7, 0.2])
    pred = M.predict(p, 0.5)
    assert M.confusion_counts(y, pred) == (2.0, 1.0, 2.0, 3.0)  # tn, fp, fn, tp
    assert M.accuracy(y, pred) == pytest.approx(5 / 8)
    assert M.balanced_accuracy(y, pred) == pytest.approx((2 / 3 + 3 / 5) / 2)
    assert M.real_precision(y, pred) == pytest.approx(2 / 4)


def test_majority_baseline_is_prevalence_of_larger_class():
    assert M.majority_baseline(np.array([1, 1, 1, 0])) == pytest.approx(0.75)
    assert M.majority_baseline(np.array([0, 0, 0, 1])) == pytest.approx(0.75)


def test_weights_act_like_repeated_rows():
    y = np.array([0, 1, 1, 0])
    pred = np.array([0, 1, 0, 1])
    w = np.array([2.0, 1.0, 3.0, 0.0])
    y_rep = np.repeat(y, w.astype(int))
    pred_rep = np.repeat(pred, w.astype(int))
    assert M.balanced_accuracy(y, pred, w) == pytest.approx(M.balanced_accuracy(y_rep, pred_rep))
    assert M.accuracy(y, pred, w) == pytest.approx(M.accuracy(y_rep, pred_rep))


# --- ranking metrics ---------------------------------------------------------

@pytest.mark.parametrize("round_to", [None, 1])
def test_auc_matches_sklearn(round_to):
    y, s = toy(round_to=round_to)
    assert M.auc_roc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_weighted_auc_matches_sklearn():
    y, s = toy(round_to=1)
    w = np.random.default_rng(1).integers(0, 4, len(y)).astype(float)
    assert M.auc_roc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))


def test_single_class_returns_nan():
    assert np.isnan(M.auc_roc(np.ones(5, int), np.arange(5.0)))
    assert np.isnan(M.equal_error_rate(np.zeros(5, int), np.arange(5.0)))


EX_Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
EX_S = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])


def test_eer_known_case():
    assert M.equal_error_rate(EX_Y, EX_S) == pytest.approx(0.25)


def test_eer_threshold_known_case():
    assert M.eer_threshold(EX_Y, EX_S) == pytest.approx(0.6)


def test_eer_perfect_separation_is_zero():
    y = np.array([0, 0, 1, 1])
    assert M.equal_error_rate(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(0.0)


def test_eer_close_to_bruteforce_on_random_data():
    from sklearn.metrics import roc_curve

    y, s = toy(n=4000, seed=3)
    fpr, tpr, _ = roc_curve(y, s)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    brute = (fpr[i] + 1 - tpr[i]) / 2
    assert M.equal_error_rate(y, s) == pytest.approx(brute, abs=0.01)


def test_tpr_at_fpr_known_case():
    assert M.tpr_at_fpr(EX_Y, EX_S, 0.0) == pytest.approx(0.75)
    assert M.tpr_at_fpr(EX_Y, EX_S, 0.10) == pytest.approx(0.75)
    assert M.tpr_at_fpr(EX_Y, EX_S, 0.25) == pytest.approx(1.0)


# --- grouping ----------------------------------------------------------------

def test_aggregate_by_group_means_scores():
    groups = np.array(["b", "a", "a", "b", "b"])
    y = np.array([0, 1, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.6, 0.3, 0.5])
    ids, y_g, s_g = M.aggregate_by_group(groups, y, s)
    assert list(ids) == ["a", "b"]
    assert list(y_g) == [1, 0]
    assert s_g == pytest.approx([0.4, 0.3])


def test_aggregate_by_group_rejects_mixed_labels():
    with pytest.raises(ValueError):
        M.aggregate_by_group(np.array(["a", "a"]), np.array([0, 1]), np.array([0.1, 0.2]))


def test_per_group_auc_compares_each_fake_type_to_all_real():
    types = np.array(["Real", "Real", "A", "A", "B", "B"])
    y = np.array([0, 0, 1, 1, 1, 1])
    s = np.array([0.2, 0.4, 0.9, 0.8, 0.05, 0.1])  # A perfectly ranked, B entirely below the reals
    out = M.per_group_auc(y, s, types, negative_group="Real")
    assert out["A"] == pytest.approx(1.0)
    assert out["B"] == pytest.approx(0.0)
    assert "Real" not in out


def test_per_group_accuracy():
    types = np.array(["Real", "Real", "A", "A"])
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    out = M.per_group_accuracy(y, pred, types)
    assert out == {"Real": pytest.approx(0.5), "A": pytest.approx(1.0)}


# --- cluster bootstrap -------------------------------------------------------

def test_bootstrap_counts_resample_whole_clusters():
    seen = []

    def stat(counts):
        seen.append(counts.copy())
        return counts.sum()

    point, boots = M.bootstrap_clusters(50, stat, n_boot=20, seed=0)
    assert point[0] == 50
    assert np.all(boots == 50)  # every draw has as many clusters as the original
    assert all(c.shape == (50,) for c in seen)


def test_bootstrap_is_deterministic_for_a_seed():
    values = np.random.default_rng(0).normal(size=100)
    stat = lambda c: np.array([(c * values).sum() / c.sum()])
    a = M.bootstrap_clusters(100, stat, n_boot=30, seed=5)[1]
    b = M.bootstrap_clusters(100, stat, n_boot=30, seed=5)[1]
    assert np.array_equal(a, b)


def test_bootstrap_ci_width_matches_standard_error():
    values = np.random.default_rng(0).normal(size=400)  # standard error of the mean ~ 0.05
    stat = lambda c: np.array([(c * values).sum() / c.sum()])
    point, boots = M.bootstrap_clusters(400, stat, n_boot=1000, seed=0)
    lo, hi = M.percentile_ci(boots)
    assert lo[0] < point[0] < hi[0]
    assert 0.07 < (hi[0] - lo[0]) / 2 < 0.13  # about 1.96 * 0.05


def test_bootstrap_strata_keep_class_sizes():
    strata = np.array([0] * 30 + [1] * 70)
    sums = []

    def stat(counts):
        sums.append((counts[strata == 0].sum(), counts[strata == 1].sum()))
        return 0.0

    M.bootstrap_clusters(100, stat, n_boot=15, seed=0, strata=strata)
    assert all(s == (30, 70) for s in sums)


def test_percentile_ci_handles_multiple_columns():
    samples = np.column_stack([np.arange(101.0), np.arange(101.0) * 2])
    lo, hi = M.percentile_ci(samples, alpha=0.10)
    assert lo == pytest.approx([5, 10])
    assert hi == pytest.approx([95, 190])
