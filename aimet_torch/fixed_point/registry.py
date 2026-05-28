# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Registry for INT16 fixed-point kernels."""

import threading
from typing import Any, Callable, Dict, List, Protocol, Type, runtime_checkable

from aimet_torch.fixed_point.encoding import OutputEncoding
from aimet_torch.fixed_point.tensor import Int16QuantizedTensor


class KernelNotFoundError(RuntimeError):
    """Raised when an INT16 fixed-point kernel is not registered."""


@runtime_checkable
class FixedKernel(Protocol):
    """Protocol implemented by all fixed-point kernels."""

    module_type: Type

    def __call__(
        self,
        inputs: List[Int16QuantizedTensor],
        params: Dict[str, Any],
        output_encoding: OutputEncoding,
        extra: Dict[str, Any],
    ) -> Int16QuantizedTensor:
        """Run the fixed-point kernel."""


_REGISTRY: Dict[Type, FixedKernel] = {}
_LOCK = threading.RLock()


def register_fixed_kernel(
    module_type: Type,
    *,
    overwrite: bool = False,
) -> Callable[[Type], Type]:
    """Register a FixedKernel implementation for a torch module type."""

    def decorator(kernel_cls: Type) -> Type:
        kernel = kernel_cls()
        if not isinstance(kernel, FixedKernel):
            raise TypeError(
                f"{kernel_cls.__qualname__} does not implement FixedKernel."
            )

        with _LOCK:
            if module_type in _REGISTRY and not overwrite:
                raise ValueError(
                    f"Fixed-point kernel for {module_type} is already registered."
                )
            _REGISTRY[module_type] = kernel

        return kernel_cls

    return decorator


def get_fixed_kernel(module_type: Type) -> FixedKernel:
    """Return registered fixed-point kernel for module_type."""

    with _LOCK:
        try:
            return _REGISTRY[module_type]
        except KeyError as exc:
            raise KernelNotFoundError(
                f"INT16 fixed-point kernel is not implemented for {module_type}. "
                "Add a kernel via register_fixed_kernel or mark this module as "
                "float fallback explicitly."
            ) from exc


def list_registered_kernels() -> List[Type]:
    """List module types with registered fixed-point kernels."""

    with _LOCK:
        return list(_REGISTRY)


def clear_fixed_kernel_registry() -> None:
    """Clear registry. Intended for tests only."""

    with _LOCK:
        _REGISTRY.clear()
