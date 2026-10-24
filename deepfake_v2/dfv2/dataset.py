"""Torch dataset, transforms and the per-epoch balanced sampler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .manifest import balanced_epoch_indices

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def build_transform(train: bool) -> transforms.Compose:
    """Same transforms as the original notebooks: light augmentation for training only."""
    steps: list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class FrameDataset(Dataset):
    """Rows of a manifest frame -> (normalised image tensor, label). Order follows the frame."""

    def __init__(self, df: pd.DataFrame, train: bool):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy()
        self.transform = build_transform(train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert("RGB")), int(self.labels[i])


class BalancedEpochSampler(Sampler[int]):
    """Yields a fresh balanced real/fake subset each epoch. Call set_epoch(e) before each epoch."""

    def __init__(self, labels: np.ndarray, seed: int):
        self.labels = np.asarray(labels)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        yield from balanced_epoch_indices(self.labels, self.epoch, self.seed).tolist()

    def __len__(self) -> int:
        return 2 * int(min((self.labels == 0).sum(), (self.labels == 1).sum()))
