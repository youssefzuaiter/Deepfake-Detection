import json

import numpy as np
import pandas as pd

from dfv2 import report

SIGNAL = {"base": 0.8, "attn": 1.0, "freq": 1.0, "hybrid": 1.6}
FAKE_TYPES = ("Deepfakes", "Face2Face", "DeepFakeDetection")


def layout(n_real_test=10):
    rows = []
    for split, n_real, n_fake in (("val", 6, 4), ("test", n_real_test, 6)):
        for v in range(n_real):
            for f in range(5):
                rows.append((f"{split}_r{v}_f{f}.jpg", split, f"Real/{split}_r{v}", "Real", 0))
        for t in FAKE_TYPES:
            for v in range(n_fake):
                for f in range(5):
                    rows.append((f"{split}_{t}{v}_f{f}.jpg", split, f"{t}/{split}_{t}{v}", t, 1))
    return pd.DataFrame(rows, columns=["filename", "split", "video", "fake_type", "label"])


def write_runs(runs_dir, variants=("base", "attn", "freq", "hybrid"), seeds=(0, 1), n_real_test=10):
    (runs_dir / "preds").mkdir(parents=True)
    base = layout(n_real_test)
    for vi, variant in enumerate(variants):
        for seed in seeds:
            rng = np.random.default_rng(100 * vi + seed)
            sign = np.where(base["label"] == 1, 1.0, -1.0)
            logit = rng.normal(loc=sign * SIGNAL[variant], scale=1.0, size=len(base))
            df = base.copy()
            df["p_fake"] = 1 / (1 + np.exp(-logit))
            df.to_csv(runs_dir / "preds" / f"{variant}_seed{seed}.csv", index=False)


def test_full_report_from_synthetic_predictions(tmp_path):
    write_runs(tmp_path / "runs")
    path = report.build_report(tmp_path / "runs", tmp_path / "out", n_boot=40, seed=0)
    text = path.read_text()
    for heading in ("What the data supports", "1. Frame-level metrics", "2. Video-level metrics",
                    "3. Test-set uncertainty", "4. Component effects", "5. By manipulation type", "7. Caveats"):
        assert heading in text
    assert "Gate main effect" in text and "FFT main effect" in text and "Interaction" in text
    assert "DeepFakeDetection" in text
    for png in ("roc_frame.png", "per_manipulation_auc.png"):
        assert (tmp_path / "out" / png).stat().st_size > 0

    results = json.loads((tmp_path / "out" / "results.json").read_text())
    auc = {v: results["per_variant"][v]["frame_auc"][0] for v in ("base", "hybrid")}
    assert auc["hybrid"] > auc["base"]  # the synthetic signal is stronger for hybrid
    point, lo, hi = results["effects"]["Gate + FFT minus Baseline"]["frame_auc"]
    assert lo <= point <= hi and lo > 0  # a clear gain should be resolved even with 40 resamples


def test_baseline_sentence_names_the_majority_class(tmp_path):
    write_runs(tmp_path / "fake_majority", variants=("base",), seeds=(0,))
    text = report.build_report(tmp_path / "fake_majority", tmp_path / "a", n_boot=5, seed=0).read_text()
    assert 'Always answering "fake"' in text
    write_runs(tmp_path / "real_majority", variants=("base",), seeds=(0,), n_real_test=60)
    text = report.build_report(tmp_path / "real_majority", tmp_path / "b", n_boot=5, seed=0).read_text()
    assert 'Always answering "real"' in text


def test_summary_covers_all_headline_metrics_and_warns_about_chance_and_seeds(tmp_path):
    write_runs(tmp_path / "runs")  # four variants, two seeds each
    text = report.build_report(tmp_path / "runs", tmp_path / "out", n_boot=40, seed=0).read_text()
    summary = text.split("## What the data supports")[1].split("## How to read")[0]
    assert summary.count("- **") == 6 * 4  # six effects, four metrics each
    for label in ("frame AUC", "frame balanced acc", "video AUC", "video balanced acc"):
        assert label in summary
    assert "percentage points" in summary
    assert "Each variant has 2 training seeds" in summary and "rough" in summary
    assert "comparisons. If none had an effect" in summary
    assert "Verdicts (each metric)" in text


def test_report_says_when_the_split_was_not_recorded(tmp_path):
    write_runs(tmp_path / "runs")
    text = report.build_report(tmp_path / "runs", tmp_path / "out", n_boot=10, seed=0).read_text()
    assert "Split not recorded" in text and "--split clean" in text


def test_single_variant_single_seed_still_reports(tmp_path):
    write_runs(tmp_path / "runs", variants=("base",), seeds=(0,))
    text = report.build_report(tmp_path / "runs", tmp_path / "out", n_boot=10, seed=0).read_text()
    assert "Baseline (1 seeds)" in text
    assert "Gate main effect" not in text
    assert "Fake recall" in text


def test_mismatched_test_rows_are_rejected(tmp_path):
    write_runs(tmp_path / "runs", variants=("base", "attn"), seeds=(0,))
    f = tmp_path / "runs" / "preds" / "attn_seed0.csv"
    df = pd.read_csv(f)
    df.loc[df["split"] == "test", "filename"] = "x_" + df.loc[df["split"] == "test", "filename"]
    df.to_csv(f, index=False)
    try:
        report.build_report(tmp_path / "runs", tmp_path / "out", n_boot=5, seed=0)
    except ValueError as e:
        assert "differ" in str(e)
    else:
        raise AssertionError("expected ValueError")
