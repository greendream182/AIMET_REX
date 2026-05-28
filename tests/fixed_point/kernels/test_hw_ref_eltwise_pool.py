"""INT32 accumulator saturation for eltwise / pool under ``AIMET_RX_HW_REF``."""

import pytest
import torch

from aimet_torch.fixed_point.requantize import (
    int32_add_sat,
    int32_sum_sat,
    saturate_mac_accumulator,
)


def test_int32_add_sat_clamps_wrap(monkeypatch):
    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    lhs = torch.tensor([2_000_000_000], dtype=torch.int32)
    rhs = torch.tensor([2_000_000_000], dtype=torch.int32)
    assert int32_add_sat(lhs, rhs).item() == 2_147_483_647


def test_int32_add_sat_default_uses_wrapped_int32(monkeypatch):
    monkeypatch.delenv("AIMET_RX_HW_REF", raising=False)
    monkeypatch.delenv("AIMET_RX_ACC_INT32_SAT", raising=False)
    lhs = torch.tensor([2_000_000_000], dtype=torch.int32)
    rhs = torch.tensor([2_000_000_000], dtype=torch.int32)
    wrapped = (lhs + rhs).to(torch.int32).item()
    assert int32_add_sat(lhs, rhs).item() == wrapped
    assert wrapped != 2_147_483_647


def test_int32_sum_sat_clamps_large_reduce(monkeypatch):
    monkeypatch.setenv("AIMET_RX_HW_REF", "1")
    x = torch.full((100_000,), 32767, dtype=torch.int32)
    assert int32_sum_sat(x, dim=0).item() == 2_147_483_647


def test_saturate_mac_accumulator_on_int64_sum(monkeypatch):
    monkeypatch.setenv("AIMET_RX_ACC_INT32_SAT", "1")
    acc = torch.tensor(3_000_000_000, dtype=torch.int64)
    assert saturate_mac_accumulator(acc).item() == 2_147_483_647
