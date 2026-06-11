import pytest

from aimet_torch.fixed_point.quant_grid import (
    GRID_I32,
    GRID_U32,
    SIM_INT32_QUANT_GRIDS,
    STANDARD_QUANT_GRIDS,
)


def test_standard_grids_cover_requested_dtypes():
    names = {g.name for g in STANDARD_QUANT_GRIDS}
    assert names == {"i8", "u8", "i16", "u16", "i32", "u32"}


def test_sim_int32_grids_exclude_full_u32():
    names = {g.name for g in SIM_INT32_QUANT_GRIDS}
    assert "u32" not in names
    assert "i32" in names
    assert not GRID_U32.fits_sim_int32_container
    assert GRID_I32.fits_sim_int32_container


def test_u32_grid_rejected_by_sim_tensor_container():
    import torch

    from aimet_torch.fixed_point import Int16QuantizedTensor

    with pytest.raises(ValueError, match="int32 sim-tensor"):
        Int16QuantizedTensor(
            int_repr=torch.tensor([0], dtype=torch.int32),
            scale=torch.tensor(1.0, dtype=torch.float32),
            zero_point=torch.tensor(0, dtype=torch.int32),
            qmin=GRID_U32.qmin,
            qmax=GRID_U32.qmax,
        )
