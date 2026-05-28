import pytest

from aimet_torch.fixed_point import (
    ExecutionMode,
    get_quant_execution_mode,
    quant_execution_mode,
    set_quant_execution_mode,
)


def teardown_function():
    set_quant_execution_mode(ExecutionMode.FP32_QDQ)


def test_default_mode_is_fp32_qdq():
    set_quant_execution_mode(ExecutionMode.FP32_QDQ)

    assert get_quant_execution_mode() is ExecutionMode.FP32_QDQ


def test_set_mode_with_string():
    set_quant_execution_mode("fp16_qdq")

    assert get_quant_execution_mode() is ExecutionMode.FP16_QDQ


def test_context_manager_restores_nested_modes():
    set_quant_execution_mode("fp32_qdq")

    with quant_execution_mode("fp16_qdq"):
        assert get_quant_execution_mode() is ExecutionMode.FP16_QDQ

        with quant_execution_mode("int16_fixed_eval"):
            assert get_quant_execution_mode() is ExecutionMode.INT16_FIXED_EVAL

        assert get_quant_execution_mode() is ExecutionMode.FP16_QDQ

    assert get_quant_execution_mode() is ExecutionMode.FP32_QDQ


def test_fixed_scale_qdq_mode_string():
    set_quant_execution_mode("fixed_scale_qdq")
    assert get_quant_execution_mode() is ExecutionMode.FIXED_SCALE_QDQ


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        set_quant_execution_mode("int8_fixed")
