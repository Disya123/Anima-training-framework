from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file
from torch import nn

from .lora import LoRALinear, lora_state_dict, load_lora_state_dict, named_lora_modules, save_kohya_lora


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config_path: str | Path,
    scope_kind: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "step": int(step),
        "scope_kind": scope_kind,
        "config_path": str(Path(config_path).resolve()),
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _rng_state(),
    }
    torch.save(state, path)
    return path


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = True,
) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    missing = sorted(set(state["model"]) - set(parameters))
    if missing:
        raise KeyError(f"checkpoint parameters do not exist in current scope: {missing[:5]}")
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    absent = sorted(expected - set(state["model"]))
    if absent:
        raise KeyError(f"checkpoint is missing current trainable parameters: {absent[:5]}")
    for name, value in state["model"].items():
        parameters[name].data.copy_(value.to(parameters[name]))
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and "rng" in state:
        _restore_rng_state(state["rng"])
    return int(state["step"])


def export_lora(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    if not any(True for _ in named_lora_modules(model)):
        raise ValueError("model has no injected LoRA modules")
    save_kohya_lora(model, path, metadata)
    return Path(path)


def merged_anima_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Build official `net.*` state without mutating an active LoRA model."""
    lora_names = {name for name, _ in named_lora_modules(model)}
    output: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            output[f"net.{name}.weight"] = module.merged_weight().to(module.base_layer.weight.dtype).cpu().contiguous()
            if module.base_layer.bias is not None:
                output[f"net.{name}.bias"] = module.base_layer.bias.detach().cpu().contiguous()
    for name, value in model.state_dict().items():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in lora_names):
            continue
        output[f"net.{name}"] = value.detach().cpu().contiguous()
    return output


def export_full_model(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {str(key): str(value) for key, value in (metadata or {}).items()}
    meta.setdefault("format", "pt")
    meta.setdefault("modelspec.architecture", "anima")
    save_file(merged_anima_state_dict(model), str(path), metadata=meta)
    return path

