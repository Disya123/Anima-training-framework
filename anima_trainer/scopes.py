from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

from .config import ScopeConfig, torch_dtype
from .lora import inject_lora


_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


@dataclass(frozen=True)
class ScopeReport:
    kind: str
    matched_modules: tuple[str, ...]
    trainable_tensors: int
    trainable_parameters: int
    total_parameters: int


def block_index(name: str) -> int | None:
    match = _BLOCK_RE.match(name)
    return int(match.group(1)) if match else None


def component_of(name: str) -> str | None:
    if name.startswith("llm_adapter"):
        return None
    if ".self_attn." in name:
        return "self_attn"
    if ".cross_attn." in name:
        return "cross_attn"
    if ".mlp." in name:
        return "mlp"
    if ".adaln_modulation_" in name:
        return "modulation"
    if name.startswith("x_embedder"):
        return "x_embedder"
    if name.startswith(("t_embedder", "t_embedding_norm")):
        return "timestep"
    if name.startswith("final_layer"):
        return "final_layer"
    return None


def _component_selected(name: str, components: tuple[str, ...]) -> bool:
    return "all" in components or component_of(name) in components


def _block_selected(name: str, blocks: tuple[int, ...] | None) -> bool:
    idx = block_index(name)
    return blocks is None or idx in blocks


def select_lora_modules(model: nn.Module, scope: ScopeConfig) -> tuple[str, ...]:
    selected: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name.startswith("llm_adapter"):
            continue
        if not _component_selected(name, scope.components):
            continue
        if scope.kind == "selected_blocks_lora":
            if block_index(name) is None or not _block_selected(name, scope.blocks):
                continue
        elif scope.kind == "dit_lora" and scope.blocks is not None and not _block_selected(name, scope.blocks):
            continue
        selected.append(name)
    return tuple(selected)


def apply_train_scope(model: nn.Module, scope: ScopeConfig) -> ScopeReport:
    model.requires_grad_(False)
    matched: tuple[str, ...]
    if scope.kind in {"dit_lora", "selected_blocks_lora"}:
        names = select_lora_modules(model, scope)
        matched = inject_lora(
            model,
            names,
            rank=scope.rank,
            alpha=scope.alpha,
            dropout=scope.dropout,
            dtype=torch_dtype(scope.trainable_dtype),
            rank_overrides=scope.rank_overrides,
        )
    elif scope.kind == "selected_blocks_full":
        matched_list: list[str] = []
        for name, parameter in model.named_parameters():
            if name.startswith("llm_adapter"):
                continue
            idx = block_index(name)
            if idx is None or not _block_selected(name, scope.blocks):
                continue
            if not _component_selected(name, scope.components):
                continue
            parameter.requires_grad_(True)
            matched_list.append(name)
        matched = tuple(matched_list)
    elif scope.kind == "dit_full":
        matched_list = []
        for name, parameter in model.named_parameters():
            if name.startswith("llm_adapter"):
                continue
            if "all" not in scope.components and not _component_selected(name, scope.components):
                continue
            parameter.requires_grad_(True)
            matched_list.append(name)
        matched = tuple(matched_list)
    else:
        raise ValueError(f"unsupported train scope: {scope.kind}")
    if not matched:
        raise ValueError(f"scope {scope.kind} did not match trainable weights")

    target_dtype = torch_dtype(scope.trainable_dtype)
    if scope.kind in {"selected_blocks_full", "dit_full"}:
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype != target_dtype:
                parameter.data = parameter.data.to(target_dtype)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(name.startswith("llm_adapter") and parameter.requires_grad for name, parameter in model.named_parameters()):
        raise AssertionError("llm_adapter must remain frozen in v0.1")
    return ScopeReport(
        kind=scope.kind,
        matched_modules=matched,
        trainable_tensors=len(trainable),
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )

