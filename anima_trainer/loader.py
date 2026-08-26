from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import AnimaDIT, build_anima_dit


_CHECKPOINT_PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.")


def strip_checkpoint_prefix(key: str) -> str:
    for prefix in _CHECKPOINT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def normalize_checkpoint_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {strip_checkpoint_prefix(key): value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError("checkpoint prefix normalization produced duplicate keys")
    return normalized


def _materialize_nonpersistent_buffers(model: AnimaDIT) -> None:
    """Restore small buffers that a meta-device construction intentionally skipped."""
    head_dim = model.model_channels // model.num_heads
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    model.pos_embedder.dim_spatial_range = torch.arange(0, dim_h, 2, dtype=torch.float32)[: dim_h // 2] / dim_h
    model.pos_embedder.dim_temporal_range = torch.arange(0, dim_t, 2, dtype=torch.float32)[: dim_t // 2] / dim_t
    adapter_head_dim = 1024 // 16
    model.llm_adapter.rotary_emb.inv_freq = 1.0 / (
        10000 ** (torch.arange(0, adapter_head_dim, 2, dtype=torch.float32) / adapter_head_dim)
    )


def load_anima_dit(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> AnimaDIT:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # The official model is 2B parameters. Constructing on meta avoids an 8 GB
    # fp32 initialization spike; assign=True adopts the safetensors storage.
    with torch.device("meta"):
        model = build_anima_dit()
    state = normalize_checkpoint_state(load_file(str(checkpoint), device="cpu"))
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Anima checkpoint mismatch: missing={incompatible.missing_keys[:5]}, "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    del state
    _materialize_nonpersistent_buffers(model)
    remaining_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if remaining_meta:
        raise RuntimeError(f"checkpoint left meta parameters: {remaining_meta[:5]}")
    model.to(device=device, dtype=dtype)
    return model


def adapter_state_from_checkpoint(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    prefixes = (
        "net.llm_adapter.",
        "model.diffusion_model.llm_adapter.",
        "diffusion_model.llm_adapter.",
        "llm_adapter.",
    )
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = handle.get_tensor(key)
                    break
    if not state:
        raise KeyError(f"no llm_adapter weights found in {checkpoint}")
    return state

