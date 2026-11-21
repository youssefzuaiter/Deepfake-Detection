"""Test-time degradations: the same frame made worse in a controlled, repeatable way.

Names: "clean" (unchanged), "jpegQ" (re-encode at JPEG quality Q), "blurR" (Gaussian blur of radius R pixels)
and "downKx" (shrink by a factor K, then enlarge back). All of them are deterministic, so two runs see exactly
the same degraded frames. Only PIL is needed, so this module imports anywhere.
"""
from __future__ import annotations

import io
from typing import Callable

from PIL import Image, ImageFilter

CLEAN = "clean"
DEFAULT_PERTURBATIONS = ("jpeg70", "jpeg40", "blur1.5", "down2x")


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def _down(image: Image.Image, factor: int) -> Image.Image:
    width, height = image.size
    small = image.resize((max(1, width // factor), max(1, height // factor)), Image.BILINEAR)
    return small.resize((width, height), Image.BILINEAR)


def parse(name: str) -> Callable[[Image.Image], Image.Image]:
    """Turn a degradation name into a function on PIL images. Raises ValueError for anything else."""
    if name == CLEAN:
        return lambda image: image
    try:
        if name.startswith("jpeg"):
            quality = int(name[4:])
            if 1 <= quality <= 100:
                return lambda image: _jpeg(image, quality)
        elif name.startswith("blur"):
            radius = float(name[4:])
            if radius > 0:
                return lambda image: _blur(image, radius)
        elif name.startswith("down") and name.endswith("x"):
            factor = int(name[4:-1])
            if factor >= 2:
                return lambda image: _down(image, factor)
    except ValueError:
        pass
    raise ValueError(f"unknown degradation {name!r}; use clean, jpeg<quality>, blur<radius> or down<k>x")


def apply(image: Image.Image, name: str) -> Image.Image:
    return parse(name)(image)
