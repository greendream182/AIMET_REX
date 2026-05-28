import pytest
import torch
import torch.nn as nn

import aimet_torch.fixed_point.kernels  # noqa: F401
from aimet_torch.fixed_point import (
    InputEncoding,
    Int16QuantizedTensor,
    OutputEncoding,
    generate_lut_int16,
    generate_pwl_lut,
    get_fixed_kernel,
    pwl_lut_from_json_dict,
    pwl_lut_to_json_dict,
)
from aimet_torch.fixed_point.metrics.thresholds import (
    PWL_HARDWARE_NUM_SEGMENTS,
    PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY,
    PWL_VS_ANALYTIC_PER_FN_LIMITS,
)
from aimet_torch.fixed_point.requantize import SIM_TENSOR_DTYPE
from aimet_torch.fixed_point.offline.lut_gen import generate_pwl_lut_for_export
from aimet_torch.fixed_point.kernels.lut import evaluate_pwl_lut_int16, lookup_lut_int16


def _encoding(scale=1.0, zero_point=0, qmin=-32768, qmax=32767):
    return InputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def _output_encoding(scale=1.0, zero_point=0, qmin=-32768, qmax=32767):
    return OutputEncoding(
        scale=torch.tensor(scale, dtype=torch.float32),
        zero_point=torch.tensor(zero_point, dtype=torch.int32),
        qmin=qmin,
        qmax=qmax,
    )


def test_generate_lut_int16_shape_and_dtype():
    lut = generate_lut_int16(
        torch.sigmoid,
        _encoding(scale=1.0, qmin=-4, qmax=4),
        _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767),
        table_size=9,
    )

    assert lut.dtype == torch.int16
    assert lut.shape == (9,)


def test_lookup_lut_int16_maps_range_uniformly():
    x = torch.tensor([-4, 0, 4], dtype=torch.int16)
    lut = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int16)

    output = lookup_lut_int16(x, lut, input_qmin=-4, input_qmax=4)

    assert output.tolist() == [10, 30, 50]


def test_sigmoid_lut_kernel_uses_lut_values():
    x = Int16QuantizedTensor(
        int_repr=torch.tensor([-4, 0, 4], dtype=torch.int16),
        scale=torch.tensor(1.0, dtype=torch.float32),
        zero_point=torch.tensor(0, dtype=torch.int32),
        qmin=-4,
        qmax=4,
    )
    lut = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int16)

    output = get_fixed_kernel(nn.Sigmoid)(
        [x],
        {},
        _output_encoding(scale=1.0, qmin=0, qmax=32767),
        {"lut_int16": lut, "input_qmin": -4, "input_qmax": 4},
    )

    assert output.int_repr.tolist() == [1, 3, 5]


def test_pwl_lut_sigmoid_integer_path_within_hardware_analytic_bound():
    input_encoding = _encoding(scale=8.0 / 32767, qmin=-32768, qmax=32767)
    output_encoding = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl, num_segments, metrics = generate_pwl_lut_for_export(
        torch.sigmoid,
        input_encoding,
        output_encoding,
        enforce_quality=True,
        fn_name="sigmoid",
    )
    assert num_segments == PWL_HARDWARE_NUM_SEGMENTS
    sigmoid_limits = PWL_VS_ANALYTIC_PER_FN_LIMITS["sigmoid"]
    assert metrics["max_lsb"] <= sigmoid_limits["max_lsb"]
    assert metrics["p99_lsb"] <= sigmoid_limits["p99_lsb"]
    assert metrics["rmse_lsb"] <= sigmoid_limits["rmse_lsb"]
    assert metrics["cosine_similarity"] >= PWL_VS_ANALYTIC_MIN_COSINE_SIMILARITY
    qx = torch.tensor([-32768, -10000, 0, 10000, 32767], dtype=torch.int16)

    qy = evaluate_pwl_lut_int16(qx, pwl)
    assert qy.shape == qx.shape
    assert qy.dtype is SIM_TENSOR_DTYPE


def test_pwl_lut_json_roundtrip_preserves_integer_inference():
    input_encoding = _encoding(scale=8.0 / 32767, qmin=-32768, qmax=32767)
    output_encoding = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch.tanh, input_encoding, output_encoding, num_segments=8)
    exported = pwl_lut_to_json_dict(
        pwl,
        func_name="tanh",
        input_encoding=input_encoding,
        output_encoding=output_encoding,
    )
    loaded = pwl_lut_from_json_dict(exported, "tanh")
    qx = torch.tensor([-1000, 0, 1000], dtype=torch.int16)

    torch.testing.assert_close(
        evaluate_pwl_lut_int16(qx, loaded),
        evaluate_pwl_lut_int16(qx, pwl),
        rtol=0,
        atol=0,
    )


def test_pwl_lut_to_json_dict_embeds_quality_metrics():
    input_encoding = _encoding(scale=8.0 / 32767, qmin=-32768, qmax=32767)
    output_encoding = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch.sigmoid, input_encoding, output_encoding, num_segments=16)
    metrics = {
        "max_lsb": 246.0,
        "p99_lsb": 187.05,
        "p999_lsb": 239.91,
        "rmse_lsb": 55.35,
        "cosine_similarity": 0.999997,
    }
    limits = {
        "max_lsb": 320.0,
        "p99_lsb": 260.0,
        "rmse_lsb": 80.0,
        "min_cosine_similarity": 0.9999,
    }

    exported = pwl_lut_to_json_dict(
        pwl,
        func_name="sigmoid",
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        quality_metrics=metrics,
        quality_limits=limits,
    )

    quality = exported["sigmoid"]["quality"]
    assert quality["status"] == "PASS"
    assert quality["metrics"]["cosine_similarity"] == pytest.approx(0.999997)
    assert quality["limits"]["max_lsb"] == 320.0
    assert quality["failures"] == []

    loaded = pwl_lut_from_json_dict(exported, "sigmoid")
    assert loaded["q_b"].numel() == PWL_HARDWARE_NUM_SEGMENTS


def test_pwl_lut_to_json_dict_marks_fail_when_metrics_exceed_limits():
    input_encoding = _encoding(scale=8.0 / 32767, qmin=-32768, qmax=32767)
    output_encoding = _output_encoding(scale=1.0 / 32767, qmin=0, qmax=32767)
    pwl = generate_pwl_lut(torch.sigmoid, input_encoding, output_encoding, num_segments=16)
    exported = pwl_lut_to_json_dict(
        pwl,
        func_name="sigmoid",
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        quality_metrics={"max_lsb": 9999.0, "cosine_similarity": 0.5},
        quality_limits={"max_lsb": 100.0, "min_cosine_similarity": 0.99},
    )

    quality = exported["sigmoid"]["quality"]
    assert quality["status"] == "FAIL"
    fail_names = {f["metric"] for f in quality["failures"]}
    assert fail_names == {"max_lsb", "cosine_similarity"}
