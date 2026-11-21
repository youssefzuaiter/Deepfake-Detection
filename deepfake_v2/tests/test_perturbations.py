import numpy as np
import pytest
from PIL import Image

from dfv2 import perturbations as P


def noise_image(seed=0, size=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def roughness(image):
    """Mean absolute difference between horizontal neighbours: high for noise, low once smoothed."""
    return float(np.abs(np.diff(np.asarray(image).astype(float), axis=1)).mean())


def mean_change(a, b):
    return float(np.abs(np.asarray(a).astype(float) - np.asarray(b).astype(float)).mean())


def test_clean_leaves_the_image_unchanged():
    img = noise_image()
    assert np.array_equal(np.asarray(P.apply(img, "clean")), np.asarray(img))


@pytest.mark.parametrize("name", P.DEFAULT_PERTURBATIONS)
def test_every_default_degradation_keeps_size_and_mode(name):
    out = P.apply(noise_image(), name)
    assert out.size == (64, 64) and out.mode == "RGB"


@pytest.mark.parametrize("name", P.DEFAULT_PERTURBATIONS)
def test_degradations_are_deterministic(name):
    a, b = P.apply(noise_image(), name), P.apply(noise_image(), name)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_lower_jpeg_quality_distorts_more():
    img = noise_image()
    assert 0 < mean_change(img, P.apply(img, "jpeg70")) < mean_change(img, P.apply(img, "jpeg40"))


def test_blur_and_downscaling_remove_fine_detail():
    img = noise_image()
    assert roughness(P.apply(img, "blur1.5")) < roughness(img)
    assert roughness(P.apply(img, "down2x")) < roughness(img)


def test_more_blur_removes_more_detail():
    img = noise_image()
    assert roughness(P.apply(img, "blur3")) < roughness(P.apply(img, "blur1.5"))


@pytest.mark.parametrize("bad", ["sepia", "jpeg", "jpeg0", "jpeg101", "blur", "blur-1", "down0x", "down", "downx"])
def test_unknown_or_out_of_range_names_are_rejected(bad):
    with pytest.raises(ValueError):
        P.apply(noise_image(), bad)
