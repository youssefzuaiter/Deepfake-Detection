"""End-to-end checks on tiny synthetic datasets: train on CPU, write files, run the report."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from dfv2 import report, run_ablation
from dfv2.splits import audit_identities
from dfv2.manifest import load_manifest
from dfv2.train import TrainConfig, run

COMMON_ARGS = ["--epochs", "1", "--batch-size", "4", "--num-workers", "0",
               "--device", "cpu", "--no-pretrained", "--no-amp"]


def write_image(path, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(path)


def make_dataset(root):
    """Random-noise frames. Fake ids repeat across splits, so only split="original" suits it."""
    rng = np.random.default_rng(0)
    rows = []

    def add(split, fake_type, name, label):
        write_image(root / split / fake_type / name, rng)
        rows.append((name, name.split("_")[0], 0, label, fake_type, split))

    for split in ("train", "val", "test"):
        for vid in range(3):
            for f in range(4):
                add(split, "Real", f"{split}{vid}_f{f}.jpg", "REAL")
        for t in ("Deepfakes", "FaceSwap"):
            for vid in range(2):
                for f in range(4):
                    add(split, t, f"{split}{vid}_{vid + 50}_f{f}.jpg", "FAKE")
    pd.DataFrame(rows, columns=["filename", "source_id", "frame_idx", "label", "fake_type", "split"]).to_csv(
        root / "dataset_manifest.csv", index=False)


def make_pair_dataset(root, n_pairs=40):
    """Pairs of original videos with both real videos and both fake directions, filed under a
    random (leaky) original split, so the clean split has something to fix."""
    rng = np.random.default_rng(1)
    rows = []
    for p in range(n_pairs):
        a, b = f"{2 * p:03d}", f"{2 * p + 1:03d}"
        for fake_type, stem in (("Real", a), ("Real", b), ("Deepfakes", f"{a}_{b}"), ("Deepfakes", f"{b}_{a}")):
            split = str(rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15]))
            name = f"{stem}_f0.jpg"
            write_image(root / split / fake_type / name, rng)
            rows.append((name, stem.split("_")[0], 0, "REAL" if fake_type == "Real" else "FAKE", fake_type, split))
    pd.DataFrame(rows, columns=["filename", "source_id", "frame_idx", "label", "fake_type", "split"]).to_csv(
        root / "dataset_manifest.csv", index=False)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    make_dataset(root)
    return root


@pytest.fixture(scope="module")
def pair_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("pairs")
    make_pair_dataset(root)
    return root


@pytest.mark.parametrize("variant", ["base", "hybrid"])
def test_run_writes_predictions_history_and_weights(variant, data_dir, tmp_path):
    cfg = TrainConfig(variant=variant, seed=0, epochs=2, batch_size=4, num_workers=0,
                      pretrained=False, amp=False, device="cpu", split="original")
    summary = run(cfg, data_dir, tmp_path)
    assert summary["best_epoch"] in (1, 2)

    preds = pd.read_csv(tmp_path / "preds" / f"{variant}_seed0.csv")
    assert list(preds.columns) == ["filename", "split", "video", "fake_type", "label", "p_fake"]
    assert (preds["split"] == "val").sum() == 28 and (preds["split"] == "test").sum() == 28
    assert preds["p_fake"].between(0, 1).all()

    meta = json.loads((tmp_path / "history" / f"{variant}_seed0.json").read_text())
    assert len(meta["history"]) == 2 and meta["config"]["variant"] == variant
    assert meta["config"]["split"] == "original" and "split_audit" in meta
    assert set(torch.load(tmp_path / "weights" / f"{variant}_seed0.pt")) >= {"head.0.weight"}


def test_runs_are_reproducible_for_a_seed(data_dir, tmp_path):
    cfg = TrainConfig(variant="base", seed=3, epochs=1, batch_size=4, num_workers=0,
                      pretrained=False, amp=False, device="cpu", split="original")
    run(cfg, data_dir, tmp_path / "a")
    run(cfg, data_dir, tmp_path / "b")
    a = pd.read_csv(tmp_path / "a" / "preds" / "base_seed3.csv")["p_fake"].to_numpy()
    b = pd.read_csv(tmp_path / "b" / "preds" / "base_seed3.csv")["p_fake"].to_numpy()
    assert np.allclose(a, b, atol=1e-5)


def test_cli_skips_finished_runs_then_report_reads_them(data_dir, tmp_path, capsys):
    args = ["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "runs"), "--variants", "base",
            "--seeds", "0", "--split", "original"] + COMMON_ARGS
    run_ablation.main(args)
    capsys.readouterr()
    run_ablation.main(args)
    assert "skipping" in capsys.readouterr().out

    text = report.build_report(tmp_path / "runs", tmp_path / "report", n_boot=10, seed=0).read_text()
    assert "Baseline (1 seeds)" in text
    assert "Split leakage is present" in text  # the runs recorded the original split
    assert (tmp_path / "report" / "learning_curves.png").exists()


def test_clean_split_end_to_end_has_no_shared_identities(pair_dir, tmp_path):
    out = tmp_path / "runs"
    run_ablation.main(["--data-dir", str(pair_dir), "--out-dir", str(out), "--variants", "base",
                       "--seeds", "0"] + COMMON_ARGS)  # --split defaults to clean
    meta = json.loads((out / "history" / "base_seed0.json").read_text())
    assert meta["config"]["split"] == "clean"
    assert meta["split_audit"]["shared"] == {"train/val": 0, "train/test": 0, "val/test": 0}
    assert meta["split_audit"]["frames"]["train"] > meta["split_audit"]["frames"]["test"] > 0

    preds = pd.read_csv(out / "preds" / "base_seed0.csv")
    assert preds["p_fake"].between(0, 1).all() and set(preds["split"]) == {"val", "test"}

    text = report.build_report(out, tmp_path / "report", n_boot=10, seed=0).read_text()
    assert "Identity-disjoint split" in text and "DeepFakeDetection is excluded" in text

    # the original split of the same files does leak, which is what the clean split removes
    assert sum(audit_identities(load_manifest(pair_dir))["shared"].values()) > 0


def test_cli_refuses_to_mix_runs_made_with_different_settings(tmp_path):
    out = tmp_path / "runs"
    (out / "preds").mkdir(parents=True)
    (out / "history").mkdir()
    (out / "preds" / "base_seed0.csv").write_text("x\n")
    (out / "history" / "base_seed0.json").write_text(json.dumps({"config": {"split": "original", "epochs": 1}}))
    with pytest.raises(SystemExit, match="different settings"):
        run_ablation.main(["--data-dir", str(tmp_path), "--out-dir", str(out), "--variants", "base",
                           "--seeds", "0"] + COMMON_ARGS)  # asks for the clean split
