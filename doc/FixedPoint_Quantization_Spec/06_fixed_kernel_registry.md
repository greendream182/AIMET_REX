# 06 FixedKernel 协议与 registry

Status: implemented

## 1. 目标

提供与 v1/v2 解耦的 `FixedKernel` 协议与 registry，使 INT16 定点 kernel 可被 v1 wrapper 与 v2 module 共同复用，避免双实现。

## 2. 范围

### 2.1 新增文件

- [aimet_torch/fixed_point/registry.py](../../aimet_torch/fixed_point/registry.py)
- [aimet_torch/v2/quantization/affine/fixed_point/adapter.py](../../aimet_torch/v2/quantization/affine/fixed_point/adapter.py)
- [aimet_torch/v1/fixed_point_adapter.py](../../aimet_torch/v1/fixed_point_adapter.py)

### 2.2 修改文件

- [aimet_torch/fixed_point/__init__.py](../../aimet_torch/fixed_point/__init__.py)：导出 `FixedKernel` / `register_fixed_kernel` / `get_fixed_kernel` / `KernelNotFoundError`。

### 2.3 不在范围

- 具体 kernel 实现（见 spec 07 / 08 / 09）。
- 模型遍历与替换（属于 adapter 内部，按需扩展）。

## 3. 前置依赖

- spec 04（Int16QuantizedTensor）完成。
- spec 05（requantize_int）完成。

## 4. 数据契约

- registry 全局唯一；线程安全（`threading.RLock`）。
- 同一 `module_type` 默认禁止重复注册；`overwrite=True` 时允许。
- 注册键：Python `type` 对象（如 `torch.nn.Conv2d`）。
- adapter 通过 registry 查找 kernel；未找到抛 `KernelNotFoundError`，**禁止 fallback 到 float**。

## 5. API 签名

```python
# aimet_torch/fixed_point/registry.py
from typing import Protocol, runtime_checkable, Callable, Any
from .tensor import Int16QuantizedTensor
from .encoding import OutputEncoding

class KernelNotFoundError(RuntimeError): ...

@runtime_checkable
class FixedKernel(Protocol):
    module_type: type
    def __call__(
        self,
        inputs: list[Int16QuantizedTensor],
        params: dict[str, Int16QuantizedTensor],
        output_encoding: OutputEncoding,
        extra: dict[str, Any],
    ) -> Int16QuantizedTensor: ...

def register_fixed_kernel(
    module_type: type,
    *,
    overwrite: bool = False,
) -> Callable[[type], type]: ...

def get_fixed_kernel(module_type: type) -> FixedKernel: ...
def list_registered_kernels() -> list[type]: ...
```

```python
# aimet_torch/v2/quantization/affine/fixed_point/adapter.py
def patch_v2_module_for_int16(module) -> None:
    """
    覆写 module 的 quantized forward：
      - inputs / params -> Int16QuantizedTensor
      - kernel = get_fixed_kernel(type(module._module_to_wrap))
      - output_encoding 取自 output quantizer
      - 返回 Int16QuantizedTensor 或 dequant 到 sentinel float（仅 debug）
    """
    ...
```

```python
# aimet_torch/v1/fixed_point_adapter.py
def patch_v1_wrapper_for_int16(wrapper) -> None: ...
```

## 6. 算法 / 伪代码

```text
_REGISTRY: dict[type, FixedKernel] = {}
_LOCK = RLock()

def register_fixed_kernel(module_type, overwrite=False):
    def decorator(cls):
        with _LOCK:
            if module_type in _REGISTRY and not overwrite:
                raise ValueError(f"kernel for {module_type} already registered")
            _REGISTRY[module_type] = cls()
        return cls
    return decorator

def get_fixed_kernel(module_type):
    with _LOCK:
        if module_type not in _REGISTRY:
            raise KernelNotFoundError(
                f"INT16 fixed kernel not implemented for {module_type}. "
                "Add a kernel via register_fixed_kernel or mark this module as fp explicitly."
            )
        return _REGISTRY[module_type]
```

adapter 伪代码：

```text
def patch_v2_module_for_int16(module):
    original_forward = module.forward

    def new_forward(*args, **kwargs):
        if get_quant_execution_mode() not in (INT16_FIXED_EVAL, INT16_FIXED_QAT_SIM):
            return original_forward(*args, **kwargs)

        inputs_int16 = [input_quantizer_to_int16(q, t) for q, t in zip(module.input_quantizers, args)]
        params_int16 = {n: param_quantizer_to_int16(pq, p) for n, p, pq in iter_params(module)}
        out_enc = build_output_encoding(module.output_quantizers[0])
        extra = collect_extra(module)
        kernel = get_fixed_kernel(type(module._module_to_wrap))
        out_int16 = kernel(inputs_int16, params_int16, out_enc, extra)
        return out_int16

    module.forward = new_forward
```

## 7. 实施步骤

1. 实现 `registry.py`，包含 `_REGISTRY`、`_LOCK`、装饰器、查找、列表。
2. 实现 v2 adapter：识别量化模块、构造 inputs/params、查 kernel、调用。
3. 实现 v1 adapter：在 `QcQuantizeWrapper.forward` 中分流。
4. 提供 dummy kernel 单元测试 registry 行为。
5. 文档化 adapter 如何处理 `extra`（per op 类型）。

## 8. 验收标准

### 8.1 单元测试

```python
def test_register_and_get():
    @register_fixed_kernel(nn.Conv2d)
    class DummyConv:
        module_type = nn.Conv2d
        def __call__(self, inputs, params, out_enc, extra):
            return inputs[0]
    assert isinstance(get_fixed_kernel(nn.Conv2d), DummyConv)

def test_duplicate_registration_raises():
    with pytest.raises(ValueError):
        @register_fixed_kernel(nn.Conv2d)
        class Another: ...

def test_not_found_raises():
    with pytest.raises(KernelNotFoundError):
        get_fixed_kernel(nn.LSTM)
```

### 8.2 必须通过的现有测试

- 默认 `fp32_qdq` 模式下，注册 kernel 不影响行为。

### 8.3 性能阈值

- 单次 lookup < 1us。

## 9. 不允许做的事

- 不允许把 registry 持久化到磁盘。
- 不允许在 adapter 中包含 kernel 数学逻辑。
- 不允许 silent fallback 到原 float forward（必须显式抛错）。

## 10. 参考

- INTERFACE.md 第 6 节、第 9 节。
- 顶层文档 ADR-005。
