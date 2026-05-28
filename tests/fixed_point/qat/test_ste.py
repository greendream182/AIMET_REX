import torch

from aimet_torch.fixed_point import fake_quantize_int16_qat


def test_fake_quantize_int16_qat_forward_grid():
    tensor = torch.tensor([-0.12, 0.0, 0.12], dtype=torch.float32)
    scale = torch.tensor(0.1, dtype=torch.float32)
    zero_point = torch.tensor(0, dtype=torch.int32)

    output = fake_quantize_int16_qat(tensor, scale, zero_point, -32768, 32767)

    assert torch.allclose(output, torch.tensor([-0.1, 0.0, 0.1]))


def test_fake_quantize_int16_qat_backward_mask():
    tensor = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32, requires_grad=True)
    scale = torch.tensor(1.0, dtype=torch.float32)
    zero_point = torch.tensor(0, dtype=torch.int32)

    output = fake_quantize_int16_qat(tensor, scale, zero_point, -1, 1)
    output.sum().backward()

    assert tensor.grad.tolist() == [0.0, 1.0, 0.0]
