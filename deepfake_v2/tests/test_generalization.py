import json

import numpy as np
import pandas as pd

from dfv2 import generalization as G
from dfv2 import report

TYPES = ("Deepfakes", "Face2Face", "FaceSwap")
VARIANTS = ("base", "freq")
PERTURBATIONS = ("jpeg70", "blur1.5")
SEEN = {"base": 1.6, "freq": 1.6}      # how separable seen manipulations are
UNSEEN = {"base": 0.5, "freq": 1.2}    # the FFT variant is built to generalize better
DEGRADE = {"clean": 1.0, "jpeg70": 0.85, "blur1.5": 0.5}


def layout(frames=3):
    rows = []
    for v in range(30):
        for f in range(frames):
            rows.append((f"r{v}_f{f}.jpg", f"Real/r{v}", "Real", 0))
    for t in TYPES:
        for v in range(20):
            for f in range(frames):
                rows.append((f"{t}{v}_f{f}.jpg", f"{t}/{t}{v}", t, 1))
    return pd.DataFrame(rows, columns=["filename", "video", "fake_type", "label"])


def scores(df, variant, holdout, seed, scale=1.0):
    rng = np.random.default_rng(1000 * seed + 10 * TYPES.index(holdout) + VARIANTS.index(variant))
    strength = np.where(df["fake_type"] == holdout, UNSEEN[variant], SEEN[variant]) * scale
    sign = np.where(df["label"] == 1, 1.0, -1.0)
    return 1 / (1 + np.exp(-rng.normal(loc=sign * strength, scale=1.0)))


def write_runs(runs_dir, with_robust=True, variants=VARIANTS, seeds=(0, 1)):
    (runs_dir / "preds").mkdir(parents=True)
    if with_robust:
        (runs_dir / "robust").mkdir()
    base = layout()
    for variant in variants:
        for holdout in TYPES:
            for seed in seeds:
                tag = f"{variant}_hold-{holdout}_seed{seed}"
                test = base.copy()
                test["split"] = "test"
                test["p_fake"] = scores(test, variant, holdout, seed)
                val = test.head(12).copy()
                val["split"] = "val"
                pd.concat([val, test])[["filename", "split", "video", "fake_type", "label", "p_fake"]].to_csv(
                    runs_dir / "preds" / f"{tag}.csv", index=False)
                if with_robust:
                    parts = []
                    for q in PERTURBATIONS:
                        part = base.copy()
                        part["perturbation"] = q
                        part["p_fake"] = scores(part, variant, holdout, seed, DEGRADE[q])
                        parts.append(part)
                    pd.concat(parts)[["filename", "video", "fake_type", "label", "perturbation", "p_fake"]].to_csv(
                        runs_dir / "robust" / f"{tag}.csv", index=False)


def test_axis_bottom_never_hides_a_value():
    for values in ([0.0952, 0.2305, 0.5], [0.93, 0.97], [0.5, 0.6], [0.75, 0.80], [0.96, 0.98]):
        bottom = G.axis_bottom(values)
        assert 0.0 <= bottom <= min(values) < 1.0
    assert G.axis_bottom([0.96, 0.98]) >= 0.5  # tight when every value is good


def test_the_main_report_does_not_pick_up_hold_out_runs():
    assert report.PRED_NAME.match("base_hold-Deepfakes_seed0.csv") is None
    assert report.PRED_NAME.match("base_seed0.csv") is not None


def test_full_report_has_every_section(tmp_path):
    write_runs(tmp_path / "runs")
    text = G.build_report(tmp_path / "runs", tmp_path / "out", n_boot=60, seed=0).read_text()
    for heading in ("What the data supports", "1. Seen and unseen manipulations", "2. Comparison with the baseline",
                    "3. Robustness to degradations", "Caveats"):
        assert heading in text
    for name in ("seen_vs_unseen.png", "robustness.png", "results.json"):
        assert (tmp_path / "out" / name).stat().st_size > 0
    assert "comparisons. If none had an effect" in text


def test_numbers_follow_the_synthetic_design(tmp_path):
    write_runs(tmp_path / "runs")
    G.build_report(tmp_path / "runs", tmp_path / "out", n_boot=80, seed=0)
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    for v in VARIANTS:
        macro = results["macro"][v]
        assert macro["unseen_auc"][0] < macro["seen_auc"][0]
        assert macro["gap"][0] > 0
    assert results["macro"]["freq"]["unseen_auc"][0] > results["macro"]["base"]["unseen_auc"][0]
    point, lo, hi = results["comparison"]["freq"]["unseen_auc"]  # freq minus base
    assert lo <= point <= hi and lo > 0
    assert results["comparison"]["freq"]["unseen_auc_by_fold"].keys() == set(TYPES)
    robust = results["robustness"]["base"]
    assert robust["blur1.5"][0] < robust["jpeg70"][0] < robust["clean"][0]


def test_seed_counts_are_worded_correctly(tmp_path):
    write_runs(tmp_path / "two", with_robust=False, seeds=(0, 1))
    write_runs(tmp_path / "one", with_robust=False, seeds=(0,))
    two = G.build_report(tmp_path / "two", tmp_path / "out2", n_boot=10, seed=0).read_text()
    one = G.build_report(tmp_path / "one", tmp_path / "out1", n_boot=10, seed=0).read_text()
    assert "(2 seeds per fold)" in two
    assert "(1 seed per fold)" in one and "1 seeds" not in one


def test_report_still_works_without_degraded_predictions(tmp_path):
    write_runs(tmp_path / "runs", with_robust=False)
    text = G.build_report(tmp_path / "runs", tmp_path / "out", n_boot=30, seed=0).read_text()
    assert "No degraded predictions found" in text
    assert not (tmp_path / "out" / "robustness.png").exists()


def test_single_variant_reports_without_a_comparison(tmp_path):
    write_runs(tmp_path / "runs", variants=("base",))
    text = G.build_report(tmp_path / "runs", tmp_path / "out", n_boot=30, seed=0).read_text()
    assert "Only one variant" in text and "+ FFT" not in text


def test_mismatched_test_rows_are_rejected(tmp_path):
    write_runs(tmp_path / "runs", with_robust=False)
    f = tmp_path / "runs" / "preds" / "freq_hold-Face2Face_seed1.csv"
    df = pd.read_csv(f)
    df.loc[df["split"] == "test", "filename"] = "x_" + df.loc[df["split"] == "test", "filename"]
    df.to_csv(f, index=False)
    try:
        G.build_report(tmp_path / "runs", tmp_path / "out", n_boot=5, seed=0)
    except ValueError as e:
        assert "differ" in str(e)
    else:
        raise AssertionError("expected ValueError")
