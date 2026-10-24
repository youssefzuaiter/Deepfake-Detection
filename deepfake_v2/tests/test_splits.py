import numpy as np
import pandas as pd
import pytest

from dfv2 import manifest as mf
from dfv2 import splits as sp


def write_toy_manifest(root, n_pairs=80, frames=3, seed=0, chain=False):
    """Pairs (a, b) of original videos, each with both real videos and both fake directions
    under two manipulations, plus a few DeepFakeDetection actor videos. The original split is
    drawn per video at random, so identities leak across splits on purpose."""
    rng = np.random.default_rng(seed)
    original: dict[str, str] = {}

    def where(key):
        return original.setdefault(key, str(rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])))

    rows = []

    def add(fake_type, stem):
        key = f"{fake_type}/{stem}"
        for k in range(frames):
            rows.append((f"{stem}_f{k}.jpg", 0, k, "REAL" if fake_type == "Real" else "FAKE", fake_type, where(key)))

    for p in range(n_pairs):
        a, b = (f"{p:03d}", f"{p + 1:03d}") if chain else (f"{2 * p:03d}", f"{2 * p + 1:03d}")
        for ident in (a, b):
            add("Real", ident)
        for t in ("Deepfakes", "Face2Face"):
            add(t, f"{a}_{b}")
            add(t, f"{b}_{a}")
    for x, y in (("01", "02"), ("02", "03"), ("04", "05")):
        add("DeepFakeDetection", f"{x}_{y}__walking__AB12")
    pd.DataFrame(rows, columns=["filename", "source_id", "frame_idx", "label", "fake_type", "split"]).to_csv(
        root / "dataset_manifest.csv", index=False)


def shared_total(df):
    return sum(sp.audit_identities(df)["shared"].values())


def test_video_identities():
    assert sp.video_identities("Real/796") == ("yt:796",)
    assert sp.video_identities("Deepfakes/636_578") == ("yt:636", "yt:578")
    assert sp.video_identities("DeepFakeDetection/15_27__outside_talking__WZ4MLUAV") == ("actor:15", "actor:27")


def test_real_and_fake_videos_share_identity_nodes():
    assert sp.video_identities("Real/636")[0] == sp.video_identities("Face2Face/636_578")[0]


def test_audit_detects_leakage_in_the_original_split(tmp_path):
    write_toy_manifest(tmp_path)
    df = mf.load_manifest(tmp_path)
    audit = sp.audit_identities(df)
    assert audit["shared"]["train/test"] > 0
    assert audit["test_identities_seen_in_train"] > 0


def test_clean_split_has_no_shared_identities(tmp_path):
    write_toy_manifest(tmp_path)
    clean = sp.clean_split(mf.load_manifest(tmp_path), seed=0)
    audit = sp.audit_identities(clean)
    assert audit["shared"] == {"train/val": 0, "train/test": 0, "val/test": 0}
    assert audit["test_identities_seen_in_train"] == 0


def test_clean_split_keeps_every_frame_except_dropped_types(tmp_path):
    write_toy_manifest(tmp_path)
    original = mf.load_manifest(tmp_path)
    clean = sp.clean_split(original, seed=0)
    assert len(clean) == (original["fake_type"] != "DeepFakeDetection").sum()
    assert "DeepFakeDetection" not in set(clean["fake_type"])
    assert set(clean["filename"]) == set(original[original["fake_type"] != "DeepFakeDetection"]["filename"])


def test_keep_types_can_retain_deepfakedetection_with_its_actors_together(tmp_path):
    write_toy_manifest(tmp_path)
    clean = sp.clean_split(mf.load_manifest(tmp_path), seed=0, drop_types=())
    assert "DeepFakeDetection" in set(clean["fake_type"])
    assert shared_total(clean) == 0
    actors = clean[clean["fake_type"] == "DeepFakeDetection"]
    assert actors.groupby("video")["split"].nunique().max() == 1


def test_both_directions_and_both_real_videos_of_a_pair_land_together(tmp_path):
    write_toy_manifest(tmp_path)
    clean = sp.clean_split(mf.load_manifest(tmp_path), seed=0)
    clean = clean[clean["fake_type"] != "DeepFakeDetection"]
    for p in range(0, 80, 9):
        a, b = f"{2 * p:03d}", f"{2 * p + 1:03d}"
        keys = [f"Real/{a}", f"Real/{b}", f"Deepfakes/{a}_{b}", f"Deepfakes/{b}_{a}", f"Face2Face/{a}_{b}"]
        assert clean[clean["video"].isin(keys)]["split"].nunique() == 1


def test_split_fractions_are_close_to_targets(tmp_path):
    write_toy_manifest(tmp_path, n_pairs=300)
    clean = sp.clean_split(mf.load_manifest(tmp_path), seed=0)
    share = clean["split"].value_counts(normalize=True)
    for name, target in sp.TARGET_FRACTIONS.items():
        assert share[name] == pytest.approx(target, abs=0.02)


def test_clean_split_is_deterministic_per_seed(tmp_path):
    write_toy_manifest(tmp_path)
    df = mf.load_manifest(tmp_path)
    a, b = sp.clean_split(df, seed=1), sp.clean_split(df, seed=1)
    assert a["split"].tolist() == b["split"].tolist()
    assert a["split"].tolist() != sp.clean_split(df, seed=2)["split"].tolist()


def test_one_giant_identity_component_cannot_be_split(tmp_path):
    write_toy_manifest(tmp_path, n_pairs=40, chain=True)  # every video linked to the next
    with pytest.raises(ValueError, match="cannot be split"):
        sp.clean_split(mf.load_manifest(tmp_path), seed=0)


def test_load_manifest_clean_keeps_paths_in_the_original_folders(tmp_path):
    write_toy_manifest(tmp_path)
    original = mf.load_manifest(tmp_path)
    clean = mf.load_manifest(tmp_path, split="clean", split_seed=0)
    merged = clean.merge(original[["filename", "fake_type", "path", "split"]],
                         on=["filename", "fake_type"], suffixes=("", "_orig"))
    assert len(merged) == len(clean)
    assert (merged["path"] == merged["path_orig"]).all()  # files did not move
    assert (merged["split"] != merged["split_orig"]).any()  # but the split assignment changed


def test_load_manifest_rejects_unknown_split_scheme(tmp_path):
    write_toy_manifest(tmp_path)
    with pytest.raises(ValueError, match="split"):
        mf.load_manifest(tmp_path, split="bogus")
