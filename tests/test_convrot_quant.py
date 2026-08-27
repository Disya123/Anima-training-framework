from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from anima_trainer.quantization import (
    ConvRotLinear,
    _align,
    _quantize_rows,
    apply_convrot,
    build_regular_hadamard,
    rotate_blocks,
    rot_size_for,
)

IN_SMALL = 256
OUT_SMALL = 128


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_regular_hadamard_properties():
    for size in (16, 64, 256):
        m = build_regular_hadamard(size, torch.device("cpu"), torch.float64)
        assert torch.allclose(m @ m.t(), torch.eye(size, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(m, m.t(), atol=1e-12)
        assert torch.allclose(m.sum(dim=1), torch.ones(size, dtype=torch.float64), atol=1e-10)


def test_rotate_blocks_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(5, 7, IN_SMALL, dtype=torch.float64)
    matrix = build_regular_hadamard(IN_SMALL, x.device, torch.float64)
    got = rotate_blocks(x, matrix, IN_SMALL).double()
    want = (x.double().reshape(-1, x.shape[-1] // IN_SMALL, IN_SMALL) @ matrix.t()).reshape_as(x)
    assert torch.allclose(got, want, atol=1e-8)


def test_rot_size_selection():
    assert rot_size_for(2048) == 256
    assert rot_size_for(1024) == 256
    assert rot_size_for(8192) == 256
    assert rot_size_for(2032) == 16


def test_row_quantization_roundtrip():
    torch.manual_seed(1)
    x = torch.randn(33, 512, dtype=torch.float32) * 0.05
    codes, scales = _quantize_rows(x)
    assert codes.dtype == torch.int8
    assert codes.abs().max() <= 127
    rebuilt = codes.float() * scales.unsqueeze(1)
    err = (rebuilt - x).norm() / x.norm()
    assert err.item() < 0.02


def test_align():
    assert _align(1, 32) == 32
    assert _align(32, 32) == 32
    assert _align(33, 32) == 64


def _make_module(device: str, *, in_f: int = IN_SMALL, out_f: int = OUT_SMALL) -> ConvRotLinear:
    torch.manual_seed(2)
    lin = torch.nn.Linear(in_f, out_f, bias=False, device=device, dtype=torch.float32)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    return ConvRotLinear(lin.weight, lin.bias)


def test_weight_roundtrip_error():
    device = _device()
    torch.manual_seed(5)
    lin = torch.nn.Linear(IN_SMALL, OUT_SMALL, bias=False, device=device, dtype=torch.float32)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    module = ConvRotLinear(lin.weight, lin.bias)
    approx_input_space = module.weight_in_original_space(torch.float64).cpu()
    src = lin.weight.detach().double().cpu()
    rel = (approx_input_space.double() - src).norm() / src.norm()
    assert rel.item() < 0.02


def test_dequantized_weight_is_folded_space():
    """codes * scales must reproduce a rotated copy of the source linear."""
    device = _device()
    torch.manual_seed(5)
    lin = torch.nn.Linear(IN_SMALL, OUT_SMALL, bias=False, device=device, dtype=torch.float32)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    module = ConvRotLinear(lin.weight, lin.bias)
    matrix = build_regular_hadamard(IN_SMALL, module.rot.device, torch.float64)
    src = lin.weight.detach().double().cpu()
    out_f, in_f = src.shape
    blocks = in_f // module.rot_size
    want = torch.einsum("o r s, s t -> o r t", src.reshape(out_f, blocks, module.rot_size), matrix.cpu())
    want = want.reshape(out_f, in_f)
    got = module._dequantized_weight(torch.float64).cpu()
    rel = (got.double() - want).norm() / want.norm()
    assert rel.item() < 0.02


def _convrot_output(module: ConvRotLinear, x: torch.Tensor) -> torch.Tensor:
    return module.evaluate(
        rotate_blocks(x.reshape(-1, module.in_features), module.rot, module.rot_size)
    ).reshape(*x.shape[:-1], module.out_features)


def test_forward_parity_against_exact_linear():
    device = _device()
    torch.manual_seed(3)
    lin = torch.nn.Linear(IN_SMALL, OUT_SMALL, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    module = ConvRotLinear(lin.weight, lin.bias).to(dtype=torch.bfloat16)
    x = torch.randn(37, 25, IN_SMALL, device=device, dtype=torch.bfloat16) * 0.5
    got = _convrot_output(module, x)
    flat_x = rotate_blocks(x.reshape(-1, IN_SMALL), module.rot, IN_SMALL)
    want = F.linear(flat_x, module._dequantized_weight(torch.bfloat16)).reshape_as(got)
    err = (got.float() - want.float()).norm() / (want.float().norm() + 1e-12)
    assert err.item() < 0.01


def test_forward_approximates_source_linear():
    device = _device()
    torch.manual_seed(6)
    lin = torch.nn.Linear(IN_SMALL, OUT_SMALL, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    module = ConvRotLinear(lin.weight, lin.bias).to(dtype=torch.bfloat16)
    x = torch.randn(37, 25, IN_SMALL, device=device, dtype=torch.bfloat16) * 0.5
    got = _convrot_output(module, x)
    want = F.linear(x, lin.weight.detach()).reshape_as(got)
    err = (got.float() - want.float()).norm() / (want.float().norm() + 1e-12)
    assert err.item() < 0.05


def test_ste_gradient_matches_analytic_linear():
    device = _device()
    torch.manual_seed(7)
    lin = torch.nn.Linear(IN_SMALL, OUT_SMALL, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.mul_(0.02)
    module = ConvRotLinear(lin.weight, lin.bias).to(dtype=torch.bfloat16)
    weight_dq = module._dequantized_weight(torch.bfloat16).detach()

    def run(system, base_x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = base_x.detach().clone().requires_grad_(True)
        out = system(x)
        (out * target).sum().backward()
        return x.grad.detach()

    def convrot_system(x):
        return module(x)

    def reference_system(x):
        rotated = rotate_blocks(x.reshape(-1, IN_SMALL), module.rot, IN_SMALL)
        out = F.linear(rotated, weight_dq)
        return out.reshape(*x.shape[:-1], OUT_SMALL)

    base_x = torch.randn(29, 101, IN_SMALL, device=device, dtype=torch.bfloat16) * 0.5
    probe_out = convrot_system(base_x.detach().clone())
    target = torch.randn_like(probe_out)
    grad_a = run(convrot_system, base_x, target)
    grad_b = run(reference_system, base_x, target)
    assert grad_a.norm().item() > 0
    gt_a, gt_b = grad_a.flatten(), grad_b.flatten()
    cos = torch.dot(gt_a.float(), gt_b.float()) / (gt_a.norm().float() * gt_b.norm().float() + 1e-12)
    assert cos.item() > 0.99


def test_odd_batch_shape_forward_and_padding():
    device = _device()
    module = _make_module(device).to(dtype=torch.bfloat16)
    for rows in (17, 31, 96):
        x = torch.randn(rows, IN_SMALL, device=device, dtype=torch.bfloat16)
        out = _convrot_output(module, x)
        assert out.shape == (rows, OUT_SMALL)
        assert torch.isfinite(out).all()


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(64, 64, bias=False)})
        self.cross_attn = nn.ModuleDict({"k_proj": nn.Linear(64, 64, bias=False)})
        self.mlp = nn.ModuleDict({"layer1": nn.Linear(64, 64, bias=False)})
        self.llm_adapter_fc = nn.Linear(64, 64, bias=False)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock()])
        self.final_layer = nn.Linear(64, 64, bias=False)


def test_apply_convrot_component_filtering():
    toy = _Toy()
    report = apply_convrot(toy, ["cross_attn", "mlp"])
    names = dict(toy.named_modules())
    assert type(names["blocks.0.self_attn.q_proj"]) is torch.nn.Linear
    assert type(names["blocks.0.cross_attn.k_proj"]) is ConvRotLinear
    assert type(names["blocks.0.mlp.layer1"]) is ConvRotLinear
    assert type(names["blocks.0.llm_adapter_fc"]) is torch.nn.Linear
    assert type(names["final_layer"]) is torch.nn.Linear
    assert report["modules"] >= 2
