"""Read dataset_manifest.csv and derive the video id that every frame belongs to.

No torch here, so reporting and tests can import this anywhere.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import DEFAULT_DROP_TYPES, clean_split

FRAME_SUFFIX = re.compile(r"_f\d+$")
SPLITS = ("train", "val", "test")


def video_key(fake_type: str, filename: str) -> str:
    """Unique id of the video a frame came from.

    The manipulation type is part of the key: the pair "636_578" exists once per
    manipulation method, and those are different videos with different artifacts.
    """
    stem = filename.rsplit(".", 1)[0]
    return f"{fake_type}/{FRAME_SUFFIX.sub('', stem)}"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def scan_frames(data_dir: str | Path) -> pd.DataFrame:
    """Frame table from the folder layout <data_dir>/<split>/<fake_type>/<image>.

    Used when the dataset has no manifest CSV. For the local copy it gives the same rows as the CSV,
    so the rest of the pipeline cannot tell the two routes apart.
    """
    data_dir = Path(data_dir)
    rows = []
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"expected folder {split_dir}")
        for type_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for image in sorted(type_dir.iterdir()):
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    rows.append((image.name, split, type_dir.name))
    if not rows:
        raise FileNotFoundError(f"no images under {data_dir}/<train|val|test>/<type>/")
    return pd.DataFrame(rows, columns=["filename", "split", "fake_type"])


def find_data_dir(search_root: str | Path) -> Path:
    """Return the folder under `search_root` that holds train/, val/ and test/ with a Real/ class folder.

    On Kaggle an attached dataset appears under /kaggle/input/<dataset-name>/ and the depth can
    vary, so this searches rather than hard-coding a path.
    """
    root = Path(search_root)
    for real_dir in sorted(root.glob("**/train/Real")):
        base = real_dir.parent.parent
        if all((base / s).is_dir() for s in SPLITS):
            return base
    raise FileNotFoundError(
        f"no train/, val/ and test/ folders with a Real/ class folder under {root}. "
        "On Kaggle, add the dataset with '+ Add Input' first.")


def subsample_videos(df: pd.DataFrame, max_videos: int, seed: int = 0) -> pd.DataFrame:
    """Keep at most `max_videos` videos per (split, label), with every frame of each."""
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    for (_split, _label), group in df.groupby(["split", "label"], sort=True):
        videos = np.sort(group["video"].unique())
        take = min(max_videos, len(videos))
        keep.extend(rng.choice(videos, size=take, replace=False))
    return df[df["video"].isin(set(keep))].reset_index(drop=True)


def load_manifest(
    data_dir: str | Path,
    max_videos: int | None = None,
    seed: int = 0,
    split: str = "original",
    split_seed: int = 0,
    drop_types: tuple[str, ...] = DEFAULT_DROP_TYPES,
) -> pd.DataFrame:
    """Return one row per frame: filename, split, fake_type, label (0 real, 1 fake), video, path.

    `split="original"` keeps the Kaggle assignment. `split="clean"` reassigns frames so that no
    identity appears in two splits (see splits.py); `drop_types` are removed in that case. Image
    paths always point at the original folders, so no file has to move.

    `max_videos` keeps a small subset of videos per (split, label). It exists for
    quick smoke runs, not for real experiments.
    """
    if split not in ("original", "clean"):
        raise ValueError(f"unknown split scheme {split!r}; use 'original' or 'clean'")
    data_dir = Path(data_dir)
    csv_path = data_dir / "dataset_manifest.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        expected = np.where(df["fake_type"] == "Real", "REAL", "FAKE")
        if (df["label"].to_numpy() != expected).any():
            raise ValueError("manifest label column disagrees with fake_type")
    else:
        df = scan_frames(data_dir)
    df["label"] = (df["fake_type"] != "Real").astype(int)
    df["video"] = [video_key(t, f) for t, f in zip(df["fake_type"], df["filename"])]
    df["path"] = [str(data_dir / s / t / f) for s, t, f in zip(df["split"], df["fake_type"], df["filename"])]
    df = df[["filename", "split", "fake_type", "label", "video", "path"]]
    if split == "clean":
        df = clean_split(df, seed=split_seed, drop_types=drop_types)
    if max_videos is not None:
        df = subsample_videos(df, max_videos, seed)
    return df


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split in a stable order (sorted by filename and type)."""
    out = df[df["split"] == split].sort_values(["fake_type", "filename"])
    return out.reset_index(drop=True)


def balanced_epoch_indices(labels: np.ndarray, epoch: int, seed: int) -> np.ndarray:
    """Indices for one balanced epoch: every row of the smaller class plus an equal
    number drawn without replacement from the larger class, shuffled.

    The draw depends only on (seed, epoch), so a run is reproducible, and different
    epochs see different fakes instead of discarding most of them for good.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng([seed, epoch])
    real = np.flatnonzero(labels == 0)
    fake = np.flatnonzero(labels == 1)
    small, large = (real, fake) if len(real) <= len(fake) else (fake, real)
    drawn = rng.choice(large, size=len(small), replace=False)
    return rng.permutation(np.concatenate([small, drawn]))
