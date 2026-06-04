"""Integer MAC must be device-invariant: CUDA default == CPU integer reference.

A single int16*int16 product (up to 2**30) exceeds float32's exact-integer
range (2**24), so the legacy float32 CUDA matmul silently diverges from the
CPU integer reference. The default path now uses float64 (bit-exact for int16
operands); float32 is opt-in (`AIMET_RX_MATMUL_FAST_FP32`) for fast previews.
"""

from __future__ import annotations

import pytest
import torch

from aimet_torch.fixed_point.kernels.conv_linear import _int32_matmul

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _operands():
    torch.manual_seed(0)
    # Products exceed 2**24 individually; accumulator stays within int32 range
    # so the CPU integer matmul is itself a valid bit-exact reference.
    lhs = torch.randint(-6000, 6000, (32, 64), dtype=torch.int32)
    rhs = torch.randint(-6000, 6000, (64, 48), dtype=torch.int32)
    return lhs, rhs


@requires_cuda
def test_default_cuda_matmul_is_bit_exact_with_cpu_reference(monkeypatch):
    monkeypatch.delenv("AIMET_RX_MATMUL_FAST_FP32", raising=False)
    monkeypatch.delenv("AIMET_RX_ACC_INT32_SAT", raising=False)
    monkeypatch.delenv("AIMET_RX_HW_REF", raising=False)

    lhs, rhs = _operands()
    exact = (lhs.to(torch.int64) @ rhs.to(torch.int64))
    cuda_out = _int32_matmul(lhs.cuda(), rhs.cuda()).to(torch.int64).cpu()

    assert torch.equal(cuda_out, exact), (
        f"CUDA default matmul diverged: max_err={(cuda_out - exact).abs().max()}"
    )


@requires_cuda
def test_fast_fp32_optin_diverges_from_reference(monkeypatch):
    monkeypatch.setenv("AIMET_RX_MATMUL_FAST_FP32", "1")
    # int32-sat takes the float64 path first; disable it to reach the fp32 branch.
    monkeypatch.setenv("AIMET_RX_ACC_INT32_SAT", "0")
    monkeypatch.delenv("AIMET_RX_HW_REF", raising=False)

    lhs, rhs = _operands()
    exact = (lhs.to(torch.int64) @ rhs.to(torch.int64))
    fast_out = _int32_matmul(lhs.cuda(), rhs.cuda()).to(torch.int64).cpu()

    # Documents (does not endorse) the approximation: it must NOT be bit-exact.
    assert not torch.equal(fast_out, exact)


def test_cpu_matmul_matches_reference_in_range():
    lhs, rhs = _operands()
    exact = (lhs.to(torch.int64) @ rhs.to(torch.int64))
    cpu_out = _int32_matmul(lhs, rhs).to(torch.int64)
    assert torch.equal(cpu_out, exact)
