import pytest
import torch.nn as nn

from aimet_torch.fixed_point import (
    KernelNotFoundError,
    get_fixed_kernel,
    list_registered_kernels,
    register_fixed_kernel,
)


class _UnregisteredModule(nn.Module):
    pass


class _RegisteredModule(nn.Module):
    pass


class _DuplicateModule(nn.Module):
    pass


def test_register_and_get_fixed_kernel():
    @register_fixed_kernel(_RegisteredModule)
    class DummyKernel:
        module_type = _RegisteredModule

        def __call__(self, inputs, params, output_encoding, extra):
            return inputs[0]

    kernel = get_fixed_kernel(_RegisteredModule)

    assert isinstance(kernel, DummyKernel)
    assert _RegisteredModule in list_registered_kernels()


def test_duplicate_registration_raises():
    @register_fixed_kernel(_DuplicateModule)
    class FirstKernel:
        module_type = _DuplicateModule

        def __call__(self, inputs, params, output_encoding, extra):
            return inputs[0]

    with pytest.raises(ValueError):

        @register_fixed_kernel(_DuplicateModule)
        class SecondKernel:
            module_type = _DuplicateModule

            def __call__(self, inputs, params, output_encoding, extra):
                return inputs[0]


def test_get_missing_kernel_raises():
    with pytest.raises(KernelNotFoundError):
        get_fixed_kernel(_UnregisteredModule)
