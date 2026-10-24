"""One model class for the 2x2 ablation: {feature gate off/on} x {FFT branch off/on}.

All four variants share the same backbone, the same dense head (only its input width
differs) and the same frequency branch design as the original notebook, so the only
thing that changes between runs is the component under test.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from torchvision import models

# variant name -> (feature gate, frequency branch)
VARIANTS: dict[str, tuple[bool, bool]] = {
    "base": (False, False),
    "attn": (True, False),
    "freq": (False, True),
    "hybrid": (True, True),
}

BACKBONE_CHANNELS = 1280
FREQ_CHANNELS = 64


def _no_autocast(x: torch.Tensor):
    """cuFFT needs float32 for 224x224 inputs, so keep the FFT out of mixed precision."""
    return torch.autocast("cuda", enabled=False) if x.is_cuda else contextlib.nullcontext()


def frequency_map(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) normalised RGB -> (B, 1, H, W) log-magnitude FFT scaled to [0, 1] per image.

    Matches the original notebook exactly: channel mean, 2-D FFT, fftshift, log1p of the
    magnitude, per-image min-max scaling. It is computed from the augmented, ImageNet-
    normalised tensor, as before, so this ablation tests the original design.
    """
    with _no_autocast(x):
        gray = x.float().mean(dim=1)
        spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(spectrum))
        lo = magnitude.amin(dim=(-2, -1), keepdim=True)
        hi = magnitude.amax(dim=(-2, -1), keepdim=True)
        return ((magnitude - lo) / (hi - lo + 1e-8)).unsqueeze(1)


class DeepfakeNet(nn.Module):
    def __init__(self, gate: bool, frequency: bool, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.features = models.mobilenet_v2(weights=weights).features
        # A learned per-channel, per-position sigmoid gate (1x1 conv over all 1280 channels).
        # This is not squeeze-and-excitation channel attention.
        self.gate = nn.Sequential(nn.Conv2d(BACKBONE_CHANNELS, BACKBONE_CHANNELS, 1), nn.Sigmoid()) if gate else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.freq_branch = (
            nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, FREQ_CHANNELS, 3, padding=1), nn.BatchNorm2d(FREQ_CHANNELS), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            if frequency
            else None
        )
        in_dim = BACKBONE_CHANNELS + (FREQ_CHANNELS if frequency else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        if self.gate is not None:
            feats = feats * self.gate(feats)
        parts = [torch.flatten(self.pool(feats), 1)]
        if self.freq_branch is not None:
            parts.append(torch.flatten(self.freq_branch(frequency_map(x)), 1))
        return self.head(torch.cat(parts, dim=1))


def build_model(variant: str, pretrained: bool = True) -> DeepfakeNet:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    gate, frequency = VARIANTS[variant]
    return DeepfakeNet(gate=gate, frequency=frequency, pretrained=pretrained)
