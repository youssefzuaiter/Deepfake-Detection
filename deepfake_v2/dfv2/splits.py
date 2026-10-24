"""Identity-disjoint re-split of the FaceForensics++ frames, plus an audit of any split.

Why: in the Kaggle manifest a fake video "A_B" is put in a split by one of its two ids, so the
other id often also appears in training. A model can then see a person's face (or the video it was
pasted into) in training and meet the same identity again at test time.

How: every original video id (and every DeepFakeDetection actor) is a node, and each fake video
links its two ids. All frames whose videos are connected must live in the same split, so splitting
is done per connected component. Frame counts decide how components are dealt out to train, val and
test; the seed only changes which components land where.

No torch and no import from manifest.py (manifest.py imports this module).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
TARGET_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_DROP_TYPES = ("DeepFakeDetection",)
BALANCE_TOLERANCE = 0.03       # max gap between a split's frame share and its target
BIG_COMPONENT_SHARE = 0.02     # components above this share of all frames are placed first
_DFD_ACTORS = re.compile(r"^(\d+)_(\d+)__")


def video_identities(video: str) -> tuple[str, ...]:
    """Identity nodes behind a video key ("Type/stem"): one for a real video, two for a fake."""
    fake_type, stem = video.split("/", 1)
    if fake_type == "Real":
        return (f"yt:{stem}",)
    if fake_type == "DeepFakeDetection":
        match = _DFD_ACTORS.match(stem)
        if not match:
            raise ValueError(f"unexpected DeepFakeDetection video name {video!r}")
        return tuple(f"actor:{g}" for g in match.groups())
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"unexpected video name {video!r}")
    return (f"yt:{parts[0]}", f"yt:{parts[1]}")


def _components(videos) -> dict[str, int]:
    """Map each video to the index of its connected component of linked identities."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids_of = {v: video_identities(v) for v in videos}
    for ids in ids_of.values():
        for other in ids[1:]:
            parent[find(other)] = find(ids[0])
        find(ids[0])
    members: dict[str, set[str]] = {}
    for ids in ids_of.values():
        members.setdefault(find(ids[0]), set()).update(ids)
    # Name each component by its smallest identity so numbering does not depend on row order.
    order = sorted(members, key=lambda root: min(members[root]))
    index = {root: i for i, root in enumerate(order)}
    return {v: index[find(ids[0])] for v, ids in ids_of.items()}


def _assign(frames: pd.Series, fractions: dict[str, float], seed: int) -> dict[int, str]:
    """Deal components to splits, always to the split furthest below its target share."""
    rng = np.random.default_rng(seed)
    total = float(frames.sum())
    comps = np.sort(frames.index.to_numpy())
    rng.shuffle(comps)
    big = sorted((c for c in comps if frames[c] > BIG_COMPONENT_SHARE * total), key=lambda c: -frames[c])
    rest = [c for c in comps if frames[c] <= BIG_COMPONENT_SHARE * total]
    target = {s: f * total for s, f in fractions.items()}
    filled = dict.fromkeys(fractions, 0.0)
    out: dict[int, str] = {}
    for c in big + rest:
        s = max(fractions, key=lambda k: (target[k] - filled[k]) / target[k])
        out[int(c)] = s
        filled[s] += float(frames[c])
    return out


def _check_result(df: pd.DataFrame, fractions: dict[str, float]) -> None:
    share = df["split"].value_counts(normalize=True)
    for name, target in fractions.items():
        got = float(share.get(name, 0.0))
        if abs(got - target) > BALANCE_TOLERANCE:
            raise ValueError(
                f"identity graph cannot be split into {fractions}: split {name!r} would hold "
                f"{got:.1%} of frames (target {target:.0%}); a few large linked groups dominate")
        if df.loc[df["split"] == name, "label"].nunique() < 2:
            raise ValueError(f"identity graph cannot be split: split {name!r} would lack real or fake frames")


def clean_split(
    df: pd.DataFrame,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
    drop_types: tuple[str, ...] = DEFAULT_DROP_TYPES,
) -> pd.DataFrame:
    """Return `df` with `split` reassigned so that no identity appears in two splits.

    Frames of `drop_types` are removed first. DeepFakeDetection is dropped by default: it has fake
    frames but no real frames from its own domain, and its actors form three linked groups (one of
    18 actors) that cannot be dealt into three balanced splits without a very uneven result.
    Rows keep their `path`, so image files stay where they are.
    """
    fractions = dict(TARGET_FRACTIONS if fractions is None else fractions)
    out = df[~df["fake_type"].isin(drop_types)].copy()
    component = _components(out["video"].unique())
    out["_component"] = out["video"].map(component)
    assignment = _assign(out.groupby("_component").size(), fractions, seed)
    out["split"] = out["_component"].map(assignment)
    _check_result(out, fractions)
    return out.drop(columns="_component").reset_index(drop=True)


def audit_identities(df: pd.DataFrame) -> dict:
    """Count how many identities appear in more than one split. All zeros means no leakage."""
    seen = df[["split", "video"]].drop_duplicates()
    ids: dict[str, set[str]] = {s: set() for s in SPLIT_NAMES}
    for split, video in zip(seen["split"], seen["video"]):
        ids[split].update(video_identities(video))
    frames = df["split"].value_counts()
    videos = seen["split"].value_counts()
    fake_share = df.groupby("split")["label"].mean()
    return {
        "frames": {s: int(frames.get(s, 0)) for s in SPLIT_NAMES},
        "videos": {s: int(videos.get(s, 0)) for s in SPLIT_NAMES},
        "identities": {s: len(ids[s]) for s in SPLIT_NAMES},
        "fake_share": {s: float(fake_share.get(s, float("nan"))) for s in SPLIT_NAMES},
        "shared": {
            "train/val": len(ids["train"] & ids["val"]),
            "train/test": len(ids["train"] & ids["test"]),
            "val/test": len(ids["val"] & ids["test"]),
        },
        "test_identities_seen_in_train": len(ids["test"] & ids["train"]),
        "test_identities": len(ids["test"]),
    }


def format_audit(audit: dict, title: str) -> str:
    lines = [title, f"  {'split':<6}{'frames':>10}{'videos':>9}{'identities':>12}{'fake share':>12}"]
    for s in SPLIT_NAMES:
        lines.append(f"  {s:<6}{audit['frames'][s]:>10,}{audit['videos'][s]:>9,}"
                     f"{audit['identities'][s]:>12,}{audit['fake_share'][s]:>12.1%}")
    shared = ", ".join(f"{k} {v}" for k, v in audit["shared"].items())
    lines.append(f"  identities in more than one split: {shared}")
    lines.append(f"  test identities also present in train: {audit['test_identities_seen_in_train']} "
                 f"of {audit['test_identities']}")
    return "\n".join(lines)
