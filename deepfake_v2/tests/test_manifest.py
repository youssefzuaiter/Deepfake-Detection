import numpy as np
import pandas as pd
import pytest

from dfv2 import manifest as mf


def test_video_key_includes_manipulation_type():
    assert mf.video_key("Deepfakes", "636_578_f12.jpg") == "Deepfakes/636_578"
    assert mf.video_key("Face2Face", "636_578_f12.jpg") == "Face2Face/636_578"
    assert mf.video_key("Real", "796_f34.jpg") == "Real/796"
    dfd = "15_27__outside_talking_still_laughing__WZ4MLUAV_f0.jpg"
    assert mf.video_key("DeepFakeDetection", dfd) == (
        "DeepFakeDetection/15_27__outside_talking_still_laughing__WZ4MLUAV"
    )


def write_manifest(tmp_path, rows):
    pd.DataFrame(rows, columns=["filename", "source_id", "frame_idx", "label", "fake_type", "split"]).to_csv(
        tmp_path / "dataset_manifest.csv", index=False
    )


def toy_rows():
    rows = []
    for split in ("train", "val", "test"):
        for vid in range(5):
            for f in range(4):
                rows.append((f"{split}{vid}_f{f}.jpg", vid, f, "REAL", "Real", split))
        for vid in range(12):
            for f in range(4):
                rows.append((f"{split}{vid}_9_f{f}.jpg", vid, f, "FAKE", "Deepfakes", split))
    return rows


def test_load_manifest_adds_label_video_and_path(tmp_path):
    write_manifest(tmp_path, toy_rows())
    df = mf.load_manifest(tmp_path)
    assert set(df.columns) == {"filename", "split", "fake_type", "label", "video", "path"}
    assert set(df["label"]) == {0, 1}
    assert (df[df.fake_type == "Real"]["label"] == 0).all()
    assert df.loc[0, "path"] == str(tmp_path / "train" / "Real" / "train0_f0.jpg")
    assert df.groupby("video").size().unique().tolist() == [4]


def test_load_manifest_rejects_label_mismatch(tmp_path):
    rows = toy_rows()
    rows[0] = (rows[0][0], rows[0][1], rows[0][2], "FAKE", "Real", rows[0][5])
    write_manifest(tmp_path, rows)
    with pytest.raises(ValueError):
        mf.load_manifest(tmp_path)


def test_max_videos_keeps_whole_videos_per_split_and_label(tmp_path):
    write_manifest(tmp_path, toy_rows())
    df = mf.load_manifest(tmp_path, max_videos=3)
    counts = df.groupby(["split", "label"])["video"].nunique()
    assert (counts == 3).all()
    assert df.groupby("video").size().unique().tolist() == [4]  # no video was cut short


def test_balanced_epoch_uses_all_reals_and_equal_fakes():
    labels = np.array([0] * 10 + [1] * 90)
    idx = mf.balanced_epoch_indices(labels, epoch=0, seed=1)
    assert len(idx) == 20 and len(set(idx)) == 20
    assert (labels[idx] == 0).sum() == 10 and (labels[idx] == 1).sum() == 10
    assert set(np.flatnonzero(labels == 0)) <= set(idx)


def test_balanced_epoch_is_deterministic_and_varies_by_epoch():
    labels = np.array([0] * 10 + [1] * 90)
    a = mf.balanced_epoch_indices(labels, 0, 1)
    assert np.array_equal(a, mf.balanced_epoch_indices(labels, 0, 1))
    assert not np.array_equal(a, mf.balanced_epoch_indices(labels, 1, 1))
    assert not np.array_equal(a, mf.balanced_epoch_indices(labels, 0, 2))


def test_balanced_epochs_cover_more_fakes_over_time():
    labels = np.array([0] * 10 + [1] * 90)
    seen = set()
    for epoch in range(6):
        idx = mf.balanced_epoch_indices(labels, epoch, 0)
        seen |= {i for i in idx if labels[i] == 1}
    assert len(seen) > 10 * 3


def create_frames(root, rows):
    for filename, _, _, _, fake_type, split in rows:
        path = root / split / fake_type / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def frame_keys(df):
    return set(zip(df["split"], df["fake_type"], df["filename"]))


def test_scan_frames_finds_the_same_rows_as_the_manifest(tmp_path):
    write_manifest(tmp_path, toy_rows())
    create_frames(tmp_path, toy_rows())
    from_csv = mf.load_manifest(tmp_path)
    scanned = mf.scan_frames(tmp_path)
    assert len(scanned) == len(from_csv)
    assert frame_keys(scanned) == frame_keys(from_csv)


def test_load_manifest_without_csv_matches_the_csv_route(tmp_path):
    write_manifest(tmp_path, toy_rows())
    create_frames(tmp_path, toy_rows())
    with_csv = mf.load_manifest(tmp_path)
    (tmp_path / "dataset_manifest.csv").unlink()
    without = mf.load_manifest(tmp_path)
    cols = ["split", "fake_type", "filename"]
    pd.testing.assert_frame_equal(
        with_csv.sort_values(cols).reset_index(drop=True),
        without.sort_values(cols).reset_index(drop=True),
    )


def test_scan_frames_rejects_a_tree_without_images(tmp_path):
    for split in ("train", "val", "test"):
        (tmp_path / split / "Real").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        mf.scan_frames(tmp_path)


def test_find_data_dir_skips_decoys_and_finds_the_full_layout(tmp_path):
    (tmp_path / "a-other" / "train" / "Real").mkdir(parents=True)  # no val/ or test/ beside it
    base = tmp_path / "faceforensics" / "dataset_processed_split"
    for split in ("train", "val", "test"):
        (base / split / "Real").mkdir(parents=True)
    assert mf.find_data_dir(tmp_path) == base


def test_find_data_dir_says_what_to_do_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Add Input"):
        mf.find_data_dir(tmp_path)
