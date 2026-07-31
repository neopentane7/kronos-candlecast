"""Environment assertions for the training/inference box.

Skipped entirely on CI (no torch installed there, no GPU); meaningful only on the
RTX 4050 laptop the training configs target.
"""

import pytest

torch = pytest.importorskip("torch")


def test_torch_has_cuda_build():
    assert torch.version.cuda is not None, "CPU-only torch installed; expected a CUDA build"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_device_is_visible_and_usable():
    assert torch.cuda.device_count() >= 1
    x = torch.randn(256, 256, device="cuda")
    assert torch.matmul(x, x).isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_has_at_least_6gb():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    assert total_gb >= 5.5, f"expected ~6GB VRAM, saw {total_gb:.1f}GB"
