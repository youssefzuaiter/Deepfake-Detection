import pytest
import torch

from dfv2.models import VARIANTS, build_model, frequency_map


def original_frequency_tensor(image):
    """Verbatim logic of rgb_to_frequency_tensor from the original notebook (one image)."""
    gray = image.mean(dim=0)
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.log1p(torch.abs(fft_shift))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.unsqueeze(0)


def test_frequency_map_matches_original_per_image_code():
    x = torch.randn(4, 3, 64, 64)
    batched = frequency_map(x)
    assert batched.shape == (4, 1, 64, 64)
    for i in range(4):
        assert torch.allclose(batched[i], original_frequency_tensor(x[i]), atol=1e-6)


def test_frequency_map_is_scaled_to_unit_range():
    out = frequency_map(torch.randn(2, 3, 32, 32))
    assert out.min() >= 0 and out.max() <= 1 + 1e-6


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_outputs_two_logits(variant):
    model = build_model(variant, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(3, 3, 224, 224)).shape == (3, 2)


def test_variants_differ_only_in_the_components_under_test():
    gate_flags = {v: build_model(v, pretrained=False).gate is not None for v in VARIANTS}
    freq_flags = {v: build_model(v, pretrained=False).freq_branch is not None for v in VARIANTS}
    assert gate_flags == {"base": False, "attn": True, "freq": False, "hybrid": True}
    assert freq_flags == {"base": False, "attn": False, "freq": True, "hybrid": True}


def test_parameter_counts_grow_with_components():
    n = {v: sum(p.numel() for p in build_model(v, pretrained=False).parameters()) for v in VARIANTS}
    assert n["base"] < n["freq"] < n["hybrid"]
    assert n["base"] < n["attn"] < n["hybrid"]


def test_eval_mode_is_deterministic():
    model = build_model("hybrid", pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        build_model("nope", pretrained=False)
