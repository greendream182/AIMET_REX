"""Parity checks against abc_lut-shuai/lut_int_general integer PWL inference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from aimet_torch.fixed_point import InputEncoding, OutputEncoding, generate_pwl_lut
from aimet_torch.fixed_point.kernels.lut import evaluate_pwl_lut_int16
from aimet_torch.fixed_point.offline.lut_gen import pwl_lut_to_json_dict

_ABC_PKG_ROOT = Path(__file__).resolve().parents[4] / "abc_lut-shuai"
_ABC_LUT_ROOT = _ABC_PKG_ROOT / "lut_int_general"
if _ABC_LUT_ROOT.is_dir():
    sys.path.insert(0, str(_ABC_PKG_ROOT))
    _HAS_ABC = True
else:
    _HAS_ABC = False

def _sat_signed_int(value: int, bit_width: int = 32) -> int:
    lo = -(2 ** (bit_width - 1))
    hi = 2 ** (bit_width - 1) - 1
    return max(lo, min(hi, int(value)))


def _arith_right_shift_round_half_up(x: int, shift: int) -> int:
    s = int(shift)
    if s == 0:
        return int(x)
    if s > 0:
        return (int(x) + (1 << (s - 1))) >> s
    return int(x) << (-s)


def _evaluate_pwl_abc_reference(
    q_x: torch.Tensor,
    pwl_lut: dict,
    *,
    acc_bit_width: int = 32,
) -> torch.Tensor:
    """Per-sample path mirroring ``lut_int_general/quantization/lut.py``."""

    thresholds = [int(t) for t in pwl_lut["thresholds"].tolist()]
    q_b = [int(b) for b in pwl_lut["q_b"].tolist()]
    shifts = [int(s) for s in pwl_lut["n_bx_total"].tolist()]
    terms = [int(c) for c in pwl_lut["term_c"].tolist()]
    zp_x = int(pwl_lut["input_zero_point"].item())
    out_qmin = int(pwl_lut["output_qmin"].item())
    out_qmax = int(pwl_lut["output_qmax"].item())
    n_seg = len(thresholds)

    outputs = []
    for q_val in q_x.flatten().tolist():
        q_x_i = int(q_val)
        seg_idx = max(
            0,
            min(
                n_seg - 1,
                __import__("bisect").bisect_right(thresholds, q_x_i) - 1,
            ),
        )
        x_offset = q_x_i - zp_x
        bx = _sat_signed_int(q_b[seg_idx] * x_offset, acc_bit_width)
        n_bx = shifts[seg_idx]
        if n_bx >= 0:
            term_bx = _sat_signed_int(
                _arith_right_shift_round_half_up(bx, n_bx),
                acc_bit_width,
            )
        else:
            term_bx = _sat_signed_int(bx << (-n_bx), acc_bit_width)
        y_acc = _sat_signed_int(term_bx + terms[seg_idx], acc_bit_width)
        outputs.append(max(out_qmin, min(out_qmax, y_acc)))

    return torch.tensor(outputs, dtype=torch.int32, device=q_x.device).view(q_x.shape)


def _encoding(scale: float, zero_point: int = 0, qmin: int = -32768, qmax: int = 32767):
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def _output_encoding(scale: float, zero_point: int = 0, qmin: int = -32768, qmax: int = 32767):
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


@pytest.mark.parametrize(
    "env_name",
    ["AIMET_RX_PWL_HW_REF", "AIMET_RX_HW_REF"],
)
@pytest.mark.parametrize("fn,torch_fn", [("sigmoid", torch.sigmoid), ("tanh", torch.tanh)])
def test_pwl_hw_ref_bit_exact_inline_abc_reference(
    fn: str, torch_fn, env_name: str, monkeypatch
):
    """Strict HW ref env matches abc per-tap INT32 saturation (bit-exact)."""

    monkeypatch.setenv(env_name, "1")
    in_enc = _encoding(scale=8.0 / 32767)
    out_enc = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch_fn, in_enc, out_enc, num_segments=16)
    qx = torch.linspace(in_enc.qmin, in_enc.qmax, 512, dtype=torch.float32).round().to(
        torch.int16
    )
    y_aimet = evaluate_pwl_lut_int16(qx, pwl).to(torch.int32)
    y_abc = _evaluate_pwl_abc_reference(qx, pwl)
    torch.testing.assert_close(y_aimet, y_abc, rtol=0, atol=0)


@pytest.mark.parametrize("fn,torch_fn", [("sigmoid", torch.sigmoid), ("tanh", torch.tanh)])
def test_pwl_aimet_matches_inline_abc_reference_on_grid(fn: str, torch_fn):
    """AIMET default path vs abc per-step INT32 saturation; allow 1 LSB rounding delta."""

    in_enc = _encoding(scale=8.0 / 32767)
    out_enc = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch_fn, in_enc, out_enc, num_segments=16)
    qx = torch.linspace(in_enc.qmin, in_enc.qmax, 512, dtype=torch.float32).round().to(
        torch.int16
    )

    y_aimet = evaluate_pwl_lut_int16(qx, pwl).to(torch.int32)
    y_abc = _evaluate_pwl_abc_reference(qx, pwl)

    max_diff = (y_aimet - y_abc).abs().max().item()
    # Rounding: AIMET uses HALF_AWAY_FROM_ZERO on shift; abc uses half-up bias — allow 1 LSB.
    assert max_diff <= 1, f"{fn}: max |aimet-abc| = {max_diff}"


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai/lut_int_general not in workspace")
def test_pwl_hw_ref_bit_exact_lut_int_general_infer_with_lut(monkeypatch):
    from lut_int_general.quantization.lut import infer_with_lut  # type: ignore[import-not-found]

    monkeypatch.setenv("AIMET_RX_PWL_HW_REF", "1")
    in_enc = _encoding(scale=8.0 / 32767)
    out_enc = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch.sigmoid, in_enc, out_enc, num_segments=16)
    lut_json = pwl_lut_to_json_dict(
        pwl, func_name="sigmoid", input_encoding=in_enc, output_encoding=out_enc
    )
    qx = torch.linspace(in_enc.qmin, in_enc.qmax, 256, dtype=torch.float32).round().to(
        torch.int16
    )
    y_aimet = evaluate_pwl_lut_int16(qx, pwl).to(torch.int32)
    y_abc = torch.tensor(
        infer_with_lut(lut_json, qx.numpy(), input_dtype="int16")["output_quantized"],
        dtype=torch.int32,
    )
    torch.testing.assert_close(y_aimet, y_abc, rtol=0, atol=0)


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai/lut_int_general not in workspace")
def test_pwl_aimet_matches_lut_int_general_infer_with_lut():
    """End-to-end vs packaged ``infer_with_lut`` on the same exported JSON."""

    from lut_int_general.quantization.lut import infer_with_lut  # type: ignore[import-not-found]

    in_enc = _encoding(scale=8.0 / 32767)
    out_enc = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch.sigmoid, in_enc, out_enc, num_segments=16)
    lut_json = pwl_lut_to_json_dict(
        pwl, func_name="sigmoid", input_encoding=in_enc, output_encoding=out_enc
    )
    qx = torch.linspace(in_enc.qmin, in_enc.qmax, 256, dtype=torch.float32).round().to(
        torch.int16
    )
    y_aimet = evaluate_pwl_lut_int16(qx, pwl).to(torch.int32)

    abc_out = infer_with_lut(lut_json, qx.numpy(), input_dtype="int16")
    y_abc = torch.tensor(abc_out["output_quantized"], dtype=torch.int32)

    max_diff = (y_aimet - y_abc).abs().max().item()
    assert max_diff <= 1, f"infer_with_lut parity: max diff = {max_diff}"


@pytest.mark.skipif(not _HAS_ABC, reason="abc_lut-shuai/lut_int_general not in workspace")
def test_pwl_hw_mac_sat_env_closer_to_abc_than_default(monkeypatch):
    """``AIMET_RX_PWL_HW_MAC_SAT=1`` tightens multiply saturation vs abc reference."""

    in_enc = _encoding(scale=1.0 / 128)
    out_enc = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch.sigmoid, in_enc, out_enc, num_segments=16)
    qx = torch.tensor([-32768, -20000, 0, 20000, 32767], dtype=torch.int16)
    y_abc = _evaluate_pwl_abc_reference(qx, pwl)

    monkeypatch.setenv("AIMET_RX_PWL_HW_MAC_SAT", "0")
    y_default = evaluate_pwl_lut_int16(qx, pwl).to(torch.int32)
    diff_default = (y_default - y_abc).abs().max().item()

    monkeypatch.setenv("AIMET_RX_PWL_HW_MAC_SAT", "1")
    y_sat = evaluate_pwl_lut_int16(qx, pwl).to(torch.int32)
    diff_sat = (y_sat - y_abc).abs().max().item()

    assert diff_sat <= diff_default + 1
