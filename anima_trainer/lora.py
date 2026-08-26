from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, dtype=torch.float32):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False, device=base_layer.weight.device, dtype=dtype)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False, device=base_layer.weight.device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.base_layer.requires_grad_(False)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if not self.enabled:
            return base
        adapter_input = self.dropout(x.to(self.lora_down.weight.dtype))
        delta = F.linear(F.linear(adapter_input, self.lora_down.weight), self.lora_up.weight)
        return base + delta.to(base.dtype) * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        delta = self.lora_up.weight.float() @ self.lora_down.weight.float()
        return self.base_layer.weight.float() + delta * self.scaling


def _parent_and_leaf(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, leaf


def resolve_rank(
    name: str,
    *,
    rank: int,
    alpha: float,
    rank_overrides: Mapping[str, Mapping[str, float]] | None,
) -> tuple[int, float] | None:
    """Return (rank, alpha) for a module or None when dropped.

    ``rank_overrides`` maps a substring of the module fqn to a spec
    {"rank": int, "alpha": float}; first matching pattern wins; omitted
    fields fall back to the global values; rank <= 0 drops the module.
    """
    if not rank_overrides:
        return (int(rank), float(alpha))
    for pattern in sorted(rank_overrides, key=len, reverse=True):
        if pattern in name:
            spec = rank_overrides[pattern]
            r = int(spec.get("rank", rank))
            if r <= 0:
                return None
            return (r, float(spec.get("alpha", r)))
    return (int(rank), float(alpha))


def inject_lora(
    model: nn.Module,
    module_names: Iterable[str],
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    dtype=torch.float32,
    rank_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[str, ...]:
    injected: list[str] = []
    for name in sorted(set(module_names)):
        resolved = resolve_rank(name, rank=rank, alpha=alpha, rank_overrides=rank_overrides)
        if resolved is None:
            continue
        r, a = resolved
        module = model.get_submodule(name)
        if isinstance(module, LoRALinear):
            raise ValueError(f"LoRA already injected at {name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {name} is {type(module).__name__}, expected Linear")
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, LoRALinear(module, r, a, dropout, dtype=dtype))
        injected.append(name)
    if not injected:
        raise ValueError("scope did not match any Linear modules")
    return tuple(injected)


def named_lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    modules = [module for _, module in named_lora_modules(model)]
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous, strict=True):
            module.enabled = enabled


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in named_lora_modules(model):
        state[f"{name}.lora_down.weight"] = module.lora_down.weight.detach().cpu()
        state[f"{name}.lora_up.weight"] = module.lora_up.weight.detach().cpu()
        state[f"{name}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    return state


def load_lora_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected: set[str] = set()
    for name, module in named_lora_modules(model):
        down_key = f"{name}.lora_down.weight"
        up_key = f"{name}.lora_up.weight"
        expected.update({down_key, up_key})
        if down_key not in state or up_key not in state:
            raise KeyError(f"LoRA state is missing {down_key} or {up_key}")
        module.lora_down.weight.data.copy_(state[down_key].to(module.lora_down.weight))
        module.lora_up.weight.data.copy_(state[up_key].to(module.lora_up.weight))
    unknown = sorted(k for k in state if k not in expected and not k.endswith(".alpha"))
    if unknown:
        raise KeyError(f"unexpected LoRA keys: {unknown[:5]}")


def kohya_state_dict(model: nn.Module, *, dtype: torch.dtype | None = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Kohya-format export. Weights cast to ``dtype`` (default bf16, matching
    ai-toolkit/ComfyUI conventions and halving file size); alpha stays fp32."""
    state: dict[str, torch.Tensor] = {}
    for name, module in named_lora_modules(model):
        prefix = "lora_unet_" + name.replace(".", "_")
        down = module.lora_down.weight.detach().cpu().contiguous()
        up = module.lora_up.weight.detach().cpu().contiguous()
        if dtype is not None:
            down = down.to(dtype)
            up = up.to(dtype)
        state[f"{prefix}.lora_down.weight"] = down
        state[f"{prefix}.lora_up.weight"] = up
        state[f"{prefix}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    return state


def save_kohya_lora(model: nn.Module, path: str | Path, metadata: Mapping[str, str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    meta.setdefault("format", "pt")
    meta.setdefault("modelspec.architecture", "anima/lora")
    save_file(kohya_state_dict(model), str(path), metadata=meta)

