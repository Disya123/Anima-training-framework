"""ConvRot-W8A8 dynamic quantization for frozen DiT Linears.

Implements the regular-Hadamard-rotation int8 scheme ("ConvRot", arXiv:2512.03673)
in a minimal form sufficient for AnimaTrainer: symmetric per-output-channel int8
weights, per-token dynamic int8 activations, ``torch._int_mm`` GEMM on Ampere+,
and an analytic straight-through backward (base weights are frozen, only dx flows
through). The rotation folds into the weight offline and applies to activations
at runtime, cancelling in the matmul.

Independent implementation written from the method description; no third-party
training code vendored.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .lora import _parent_and_leaf

CONVROT_QUANT_TYPES = {"convrot8"}

ROT_SIZE_MAX = 256
ROT_SIZE_MIN = 16
ACT_QMAX = 127.0
_ACT_MM_ROWS_ALIGN = 32

_ROT4_BASE = torch.tensor(
    [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]],
    dtype=torch.float64,
) / 2.0

_rotation_cache: dict[tuple[int, str, str], torch.Tensor] = {}
_int_mm_supported_cache: dict[str, bool] = {}


def build_regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Dense regular Hadamard matrix (R4 Kronecker-powered to ``size``).

    The construction is symmetric by definition, so ``matrix.t() == matrix`` and
    the fold/activate conventions below coincide.
    """
    key = (size, str(device), str(dtype))
    cached = _rotation_cache.get(key)
    if cached is not None:
        return cached
    order = int(round(math.log(size, 4)))
    if size != 4**order:
        raise ValueError(f"rotation size {size} must be a power of 4")
    matrix = _ROT4_BASE.clone()
    for _ in range(order - 1):
        matrix = torch.kron(_ROT4_BASE, matrix)
    matrix = matrix.to(device=device, dtype=dtype)
    _rotation_cache[key] = matrix
    return matrix


def rotate_blocks(x: torch.Tensor, matrix: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Multiply every trailing-dim block of ``rot_size`` entries by ``matrix``."""
    if x.shape[-1] == rot_size:
        return x.matmul(matrix.t())
    original_shape = x.shape
    flattened = x.reshape(-1, original_shape[-1])
    width = flattened.shape[-1] // rot_size * rot_size
    head, tail = flattened[:, :width], flattened[:, width:]
    head = head.reshape(-1, rot_size).matmul(matrix.t()).reshape(flattened.shape[0], width)
    return torch.cat([head, tail], dim=-1).reshape(original_shape)


def _fold_rows(weight: torch.Tensor, matrix: torch.Tensor, rot_size: int) -> torch.Tensor:
    out_f, in_f = weight.shape
    reshaped = weight.reshape(out_f, in_f // rot_size, rot_size)
    folded = torch.einsum("o r s, s t -> o r t", reshaped.double(), matrix.double())
    return folded.reshape(out_f, in_f)


def _unfold_rows(folded: torch.Tensor, matrix: torch.Tensor, rot_size: int) -> torch.Tensor:
    out_f = folded.shape[0]
    reshaped = folded.reshape(out_f, folded.shape[1] // rot_size, rot_size)
    unfolded = torch.einsum("o r s, s t -> o r t", reshaped.double(), matrix.double())
    return unfolded.reshape(out_f, folded.shape[1])


def rot_size_for(in_features: int) -> int:
    size = ROT_SIZE_MAX
    while size >= ROT_SIZE_MIN:
        if in_features % size == 0:
            return size
        size //= 4
    return ROT_SIZE_MIN


def _align(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def int_mm_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    key = str(device)
    cached = _int_mm_supported_cache.get(key)
    if cached is None:
        try:
            probe_a = torch.randint(-100, 100, (_ACT_MM_ROWS_ALIGN, 16), dtype=torch.int8, device=device)
            probe_b = torch.randint(-100, 100, (16, 16), dtype=torch.int8, device=device)
            torch._int_mm(probe_a, probe_b)
            cached = True
        except (RuntimeError, TypeError):
            cached = False
        _int_mm_supported_cache[key] = cached
    return cached


def _quantize_rows_eager(x2d: torch.Tensor, qmax: float = ACT_QMAX) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8 quantization; returns (codes, scales)."""
    amax = x2d.float().abs().amax(dim=1)
    scales = (amax / qmax).clamp_min(1e-12)
    codes = torch.round(x2d.float() / scales.unsqueeze(1)).clamp_(-int(qmax), int(qmax)).to(torch.int8)
    return codes, scales


def _row_scale_codes(x: torch.Tensor):
    amax = x.abs().amax(dim=-1)
    scales = (amax / ACT_QMAX).clamp_min(1e-12)
    codes = torch.round(x / scales.unsqueeze(-1)).clamp_(-127, 127)
    return codes.to(torch.int8), scales


_COMPILED: dict[str, object] = {}


def _compiled(name: str):
    fn = _COMPILED.get(name)
    if fn is None:
        if name == "rows":
            base = _row_scale_codes
            probe_args = [(torch.zeros(32, 32, dtype=torch.bfloat16, device="cuda"),)]
        elif name == "epi":

            def base(acc, s_a, s_w, dtype):
                return ((acc.float() * s_a.unsqueeze(-1)) * s_w.unsqueeze(0)).to(dtype)

            probe_args = [
                (
                    torch.zeros(32, 32, dtype=torch.int32, device="cuda"),
                    torch.ones(32, dtype=torch.bfloat16, device="cuda"),
                    torch.ones(32, dtype=torch.bfloat16, device="cuda"),
                    torch.bfloat16,
                )
            ]
        elif name == "scalecols":

            def base(x, s):
                return (x * s.unsqueeze(0)).to(torch.bfloat16)

            probe_args = [
                (
                    torch.zeros(8, 32, dtype=torch.bfloat16, device="cuda"),
                    torch.ones(32, dtype=torch.float32, device="cuda"),
                )
            ]
        else:  # pragma: no cover
            raise KeyError(name)
        try:
            fn = torch.compile(base, dynamic=True)
            for args in probe_args:
                fn(*args)
        except Exception:  # noqa: BLE001
            fn = base
        _COMPILED[name] = fn
    return fn


def _quantize_rows(x2d: torch.Tensor, qmax: float = ACT_QMAX) -> tuple[torch.Tensor, torch.Tensor]:
    if qmax != ACT_QMAX or not x2d.is_cuda:
        return _quantize_rows_eager(x2d, qmax)
    try:
        return _compiled("rows")(x2d)
    except Exception:  # noqa: BLE001
        return _quantize_rows_eager(x2d, qmax)


def _epilogue(acc_i32: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not acc_i32.is_cuda:
        return ((acc_i32.float() * s_a.unsqueeze(-1)) * s_w.unsqueeze(0)).to(dtype)
    return _compiled("epi")(acc_i32, s_a, s_w, dtype)


class ConvRotLinear(nn.Module):
    """Drop-in replacement for a frozen ``nn.Linear`` running W8A8 int8."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, *, rot_size: int | None = None):
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        if self.in_features % 16 != 0:
            raise ValueError(f"in_features={self.in_features} must be divisible by 16 for torch._int_mm")
        if self.out_features % 16 != 0:
            raise ValueError(f"out_features={self.out_features} must be divisible by 16 for torch._int_mm")
        overflow_bound = self.in_features * int(ACT_QMAX) ** 2
        if overflow_bound >= 2**31:
            raise ValueError(f"K={self.in_features} risks int32 accumulator overflow ({overflow_bound})")
        chosen = rot_size or rot_size_for(self.in_features)
        if self.in_features % chosen != 0:
            raise ValueError(f"in_features={self.in_features} not divisible by rotation size {chosen}")
        self.rot_size = int(chosen)

        device = weight.device
        dtype = weight.dtype
        source = weight.detach().float().cpu()
        folded = _fold_rows(source, build_regular_hadamard(self.rot_size, torch.device("cpu"), torch.float64), self.rot_size)
        scales = folded.abs().amax(dim=1).clamp_min(1e-12) / ACT_QMAX
        codes = torch.round(folded / scales.unsqueeze(1)).clamp_(-int(ACT_QMAX), int(ACT_QMAX))
        # Two transposed copies of the packed codes: cuBLASLt picks its fast
        # IMMA kernels only when the B operand of torch._int_mm is column-major,
        # and forward/backward contract along OPPOSITE axes of this weight.
        # cr_w_fwd: [out,in] -> fwd uses .t() view [in,out]-CM (contracts in)
        # cr_w_bwd: [in,out] -> bwd uses it as B [out,in]-CM (contracts out)
        self.register_buffer("cr_w_fwd", codes.to(torch.int8).contiguous().to(device), persistent=False)
        self.register_buffer("cr_w_bwd", codes.to(torch.int8).t().contiguous().to(device), persistent=False)
        self.cr_w_t = self.cr_w_fwd  # legacy alias used by tests/diagnostics
        self.register_buffer("cr_scales", scales.contiguous().to(device), persistent=False)
        self.register_buffer("rot", build_regular_hadamard(self.rot_size, device, dtype))
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None
        self._use_int_mm = int_mm_supported(device)
        self._fallback_cache: dict[torch.dtype, torch.Tensor] = {}

    def _dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Dequantized weight in ROTATED (folded) space: codes * per-row scales.

        Backward (dx) and the fake-quant fallback both operate on rotated
        activations against this folded weight; no unfolding happens here. In
        int-mm mode nothing is cached (reconstruction is elementwise-cheap);
        devices without ``torch._int_mm`` cache the dense form once.
        """
        if self._use_int_mm:
            return (self.cr_w_t.float() * self.cr_scales.unsqueeze(1)).to(dtype)
        cached = self._fallback_cache.get(dtype)
        if cached is None:
            cached = (self.cr_w_t.float() * self.cr_scales.unsqueeze(1)).to(dtype)
            self._fallback_cache[dtype] = cached
        return cached

    def weight_in_original_space(self, dtype: torch.dtype) -> torch.Tensor:
        """Unfolded diagnostic view approximating the pre-quantization weight."""
        folded = self._dequantized_weight(torch.float32)
        reshaped = folded.reshape(self.out_features, self.in_features // self.rot_size, self.rot_size)
        matrix = build_regular_hadamard(self.rot_size, folded.device, torch.float64)
        unfolded = torch.einsum("o r s, s t -> o r t", reshaped.double(), matrix)
        return unfolded.reshape(self.out_features, self.in_features).to(dtype)

    def _gemm(self, x_rot2d: torch.Tensor, ste_act: bool) -> torch.Tensor:
        m, k = x_rot2d.shape
        if self._use_int_mm:
            padded = _align(m, _ACT_MM_ROWS_ALIGN)
            with torch.no_grad():
                codes, scales = _quantize_rows(x_rot2d.detach())
                if padded > m:
                    pad_x = torch.zeros(padded - m, k, dtype=x_rot2d.dtype, device=x_rot2d.device)
                    pad_codes, pad_scales = _quantize_rows(pad_x)
                    codes = torch.cat([codes, pad_codes], dim=0)
                    scales = torch.cat([scales, pad_scales], dim=0)
            acc = torch._int_mm(codes, self.cr_w_t.t())[:m]
            out = _epilogue(acc, scales[:m], self.cr_scales, x_rot2d.dtype)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out
        del k
        with torch.no_grad():
            codes, scales = _quantize_rows(x_rot2d.detach())
            dq = (codes.float() * scales.unsqueeze(1)).to(x_rot2d.dtype)
        active = x_rot2d + (dq - x_rot2d).detach() if ste_act else dq
        out = F.linear(active, self._dequantized_weight(x_rot2d.dtype))
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def evaluate(self, x_rot2d: torch.Tensor) -> torch.Tensor:
        return self._gemm(x_rot2d, ste_act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_rot = rotate_blocks(x.reshape(-1, self.in_features), self.rot, self.rot_size)
        if torch.is_grad_enabled() and x.requires_grad:
            out = _ConvRotSteGemm.apply(x_rot, self)
        else:
            out = self.evaluate(x_rot)
        return out.reshape(*original_shape[:-1], self.out_features)


class _ConvRotSteGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_rot: torch.Tensor, module: "ConvRotLinear") -> torch.Tensor:
        ctx.module = module
        ctx.grad_for_x = x_rot.requires_grad
        return module._gemm(x_rot, ste_act=True)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.grad_for_x:
            return None, None
        module: ConvRotLinear = ctx.module
        grad2d = grad_output.reshape(-1, module.out_features)
        if module._use_int_mm:
            # weight scales fold into dY columns BEFORE quantization so the
            # int32 accumulator needs only a per-row scale afterwards
            with torch.no_grad():
                scaled_dy = _compiled("scalecols")(grad2d.detach().float(), module.cr_scales)
                q_dy, s_dy = _quantize_rows(scaled_dy)
                acc = torch._int_mm(q_dy.contiguous(), module.cr_w_bwd.t())
                dx_rot = (acc.float() * s_dy.unsqueeze(-1)).to(grad_output.dtype)
        else:
            dequant = module._dequantized_weight(grad_output.dtype)
            dx_rot = grad2d.matmul(dequant)
        # return gradient wrt x_rot; the recorded forward rotate composes in
        # autograd automatically (verified against exact composite in
        # test_ste_gradient_matches_analytic_linear / convrot_geometry probe)
        return dx_rot, None


def apply_convrot(
    model: nn.Module,
    components,
    *,
    extent: str = "all",
    below_block: int | None = None,
    exclude_prefixes: tuple[str, ...] = ("llm_adapter",),
) -> dict[str, object]:
    """Replace whitelisted frozen ``nn.Linear`` modules with ConvRotLinear.

    ``extent="below_trainable"`` keeps only blocks strictly below ``below_block``
    (gradient of a mid-stack LoRA never flows through them; they are pure
    feature-conduit, so quantization there cannot affect adapter fidelity).
    ``extent="all"`` quantizes every whitelisted Linear in the DiT.
    """
    from .scopes import block_index, component_of

    allowed = set(components)
    replaced_by_component: dict[str, int] = {}
    for name, module in list(model.named_modules()):
        if type(module) is not nn.Linear:
            continue
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        comp = component_of(name)
        if comp is None or comp not in allowed:
            continue
        if extent == "below_trainable" and below_block is not None:
            idx = block_index(name)
            if idx is None or idx >= below_block:
                continue
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, ConvRotLinear(module.weight, module.bias))
        replaced_by_component[comp] = replaced_by_component.get(comp, 0) + 1
    total = sum(replaced_by_component.values())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"modules": total, "by_component": replaced_by_component}
