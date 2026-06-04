"""QuantizedSTFT：STFT 作为 FX leaf / 硬件黑盒，默认 fp32 前向。

在 ``prepare_model(..., module_classes_to_exclude=[STFT])`` 下保留原生 ``STFT``，
由 Quantsim 包一层 ``QuantizedSTFT``；诊断时可将 input/output quantizer 置 None，
与板端「STFT 在专用硬件、图内仅接 float 谱」一致。
"""

from __future__ import annotations

import torch
from torch import nn

from aimet_torch.v2.nn import QuantizationMixin
from common.torch_stft import STFT


@QuantizationMixin.implements(STFT)
class QuantizedSTFT(QuantizationMixin, STFT):
    """STFT leaf：可选 I/O QDQ；``--native-trans`` 诊断时 quantizer 全关。"""

    def __quant_init__(self) -> None:
        super().__quant_init__()
        self.input_quantizers = nn.ModuleList([None])
        self.output_quantizers = nn.ModuleList([None])

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        if self.input_quantizers[0] is not None:
            input_data = self.input_quantizers[0](input_data)
        with self._patch_quantized_parameters():
            out = STFT.forward(self, input_data)
        if self.output_quantizers[0] is not None:
            out = self.output_quantizers[0](out)
        return out


def force_native_trans_float(sim_model: nn.Module) -> int:
    """关闭 ``trans``（QuantizedSTFT）全部 quantizer，返回清除的 slot 数。"""
    trans = getattr(sim_model, "trans", None)
    if trans is None:
        return 0
    n = 0
    for attr in ("input_quantizers", "output_quantizers"):
        qlist = getattr(trans, attr, None)
        if qlist is None:
            continue
        for i in range(len(qlist)):
            if qlist[i] is not None:
                qlist[i] = None
                n += 1
    return n
