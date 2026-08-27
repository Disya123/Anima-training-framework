from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Mapping

import torch
from safetensors.torch import save_file

from .lora import LoRALinear, named_lora_modules


_ENERGY_TARGETS = (0.50, 0.90, 0.99)

# module-name tokens used to denormalize kohya `lora_unet_*` names back into
# dot-separated module paths; longest-first so multi-word tokens win
_NAME_TOKENS = sorted(
    {
        "blocks",
        "self_attn",
        "cross_attn",
        "mlp",
        "q_proj",
        "k_proj",
        "v_proj",
        "output_proj",
        "layer1",
        "layer2",
        "final_layer",
        "x_embedder",
        "t_embedder",
        "t_embedding_norm",
        "llm_adapter",
        "linear_1",
        "linear_2",
        "adaln_modulation_",
        "adaln_modulation_self_attn",
        "adaln_modulation_cross_attn",
        "adaln_modulation_mlp",
        "proj",
    },
    key=len,
    reverse=True,
)


def kohya_to_module_name(full: str) -> str:
    """`lora_unet_blocks_14_cross_attn_q_proj` -> `blocks.14.cross_attn.q_proj`.

    A global underscore->dot replacement would corrupt compound component
    names (`cross_attn` -> `cross.attn`), so split on known tokens instead.
    """
    body = full[len("lora_unet_"):] if full.startswith("lora_unet_") else full
    parts: list[str] = []
    i = 0
    while i < len(body):
        match = next((tok for tok in _NAME_TOKENS if body.startswith(tok, i)), None)
        if match is not None:
            parts.append(match)
            i += len(match)
            continue
        j = i + 1
        while j < len(body) and not any(body.startswith(tok, j) for tok in _NAME_TOKENS):
            j += 1
        segment = body[i:j].strip("_")
        if segment:
            parts.append(segment)
        i = j
    return ".".join(parts)


def deltas_from_lora_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, module in named_lora_modules(model):
        delta = module.lora_up.weight.float() @ module.lora_down.weight.float()
        result[name] = delta * module.scaling
    return result


def deltas_from_lora_state(state: Mapping[str, torch.Tensor], alpha: float, rank: int) -> dict[str, torch.Tensor]:
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if key.endswith(".lora_down.weight"):
            modules.setdefault(key[: -len(".lora_down.weight")], {})["down"] = value.float()
        elif key.endswith(".lora_up.weight"):
            modules.setdefault(key[: -len(".lora_up.weight")], {})["up"] = value.float()
        elif key.endswith(".alpha"):
            modules.setdefault(key[: -len(".alpha")], {})["alpha"] = value.float()
    result: dict[str, torch.Tensor] = {}
    for name, parts in modules.items():
        if "alpha" in parts:
            # hetero-rank state: per-module alpha; fall back to module rank
            module_rank = float(parts["down"].shape[0])
            scaling = float(parts["alpha"]) / module_rank
        else:
            scaling = float(alpha) / float(rank)
        result[name] = parts["up"] @ parts["down"] * scaling
    return result


def deltas_from_dense(base_state: Mapping[str, torch.Tensor], trained_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, trained in trained_state.items():
        if name not in base_state:
            raise KeyError(f"trained tensor absent from base checkpoint: {name}")
        base = base_state[name]
        if base.shape != trained.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(base.shape)} vs {tuple(trained.shape)}")
        delta = trained.float() - base.float()
        if delta.abs().max() == 0:
            continue
        result[name] = delta
    return result


def rank_for_energy(singular_values: torch.Tensor, threshold: float) -> int:
    energy = singular_values.square().cumsum(0)
    total = energy[-1]
    if total <= 0:
        return 0
    reached = torch.nonzero(energy / total >= threshold, as_tuple=False)
    return int(reached[0].item()) + 1 if reached.numel() else int(singular_values.numel())


def effective_rank(singular_values: torch.Tensor) -> float:
    p = singular_values.square()
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    entropy = -(p * p.clamp_min(1e-30).log()).sum().item()
    return math.exp(entropy)


@dataclass(frozen=True)
class LayerSpectrum:
    name: str
    shape: tuple[int, int]
    stable_rank: float
    effective_rank: float
    frobenius: float
    spectral_norm: float
    ranks_for_energy: dict[str, int]
    singular_values: list[float]

    def to_json(self) -> dict:
        data = asdict(self)
        return data


def layer_spectrum(name: str, delta: torch.Tensor) -> LayerSpectrum:
    matrix = delta.reshape(delta.shape[0], -1).float()
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
    svals = torch.linalg.svdvals(matrix)
    fro = svals.square().sum().sqrt().item()
    stable = (fro**2 / max(svals[0].item() ** 2, 1e-30)) if svals.numel() else 0.0
    ranks = {f"r{int(t*100)}": rank_for_energy(svals, t) for t in _ENERGY_TARGETS}
    return LayerSpectrum(
        name=name,
        shape=tuple(delta.shape),
        stable_rank=round(stable, 3),
        effective_rank=round(effective_rank(svals), 3),
        frobenius=fro,
        spectral_norm=svals[0].item() if svals.numel() else 0.0,
        ranks_for_energy=ranks,
        singular_values=[round(v, 6) for v in svals.tolist()],
    )


def component_of(name: str) -> str:
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
    return "other"


def block_of(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
        return int(parts[1])
    return None


def analyze_deltas(deltas: Mapping[str, torch.Tensor], *, device: str | torch.device = "cpu") -> dict:
    spectra: dict[str, LayerSpectrum] = {}
    for name, delta in deltas.items():
        matrix = delta.to(device)
        spectra[name] = layer_spectrum(name, matrix)
        del matrix

    by_component: dict[str, dict] = {}
    for target in _ENERGY_TARGETS:
        key = f"r{int(target*100)}"
        for component in ("self_attn", "cross_attn", "mlp", "modulation", "other"):
            values = [s.ranks_for_energy[key] for n, s in spectra.items() if component_of(n) == component]
            if values:
                by_component.setdefault(component, {})[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }
    by_block: dict[str, dict] = {}
    for name, spectrum in spectra.items():
        block = block_of(name)
        if block is None:
            continue
        entry = by_block.setdefault(str(block), {"frobenius_sq": 0.0, "layers": 0})
        entry["frobenius_sq"] += spectrum.frobenius**2
        entry["layers"] += 1
    total_energy = sum(s.frobenius**2 for s in spectra.values()) or 1.0
    for entry in by_block.values():
        entry["energy_share"] = entry["frobenius_sq"] / total_energy

    return {
        "layers": {name: spectrum.to_json() for name, spectrum in sorted(spectra.items())},
        "n_layers": len(spectra),
        "total_frobenius": math.sqrt(total_energy),
        "ranks_by_component": by_component,
        "energy_by_block": {k: round(v["energy_share"], 5) for k, v in sorted(by_block.items(), key=lambda kv: int(kv[0]))},
    }


def truncated_lora_state(
    deltas: Mapping[str, torch.Tensor],
    rank_fn: Callable[[str, torch.Tensor], int],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, delta in deltas.items():
        matrix = delta.to(device)
        rows, cols = matrix.shape
        if rows > cols:
            u, s, v = torch.linalg.svd(matrix.T, full_matrices=False)
            u, v = v, u.T
        else:
            u, s, v = torch.linalg.svd(matrix, full_matrices=False)
        rank = max(0, min(int(rank_fn(name, matrix)), s.numel()))
        if rank == 0:
            continue
        sr = s[:rank]
        sqrt_s = sr.sqrt()
        up = u[:, :rank] * sqrt_s
        down = v[:rank, :] * sqrt_s[:, None]
        prefix = "lora_unet_" + name.replace(".", "_")
        state[f"{prefix}.lora_up.weight"] = up.cpu().float().contiguous()
        state[f"{prefix}.lora_down.weight"] = down.cpu().float().contiguous()
        state[f"{prefix}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
    return state


def save_truncated_loras(
    deltas: Mapping[str, torch.Tensor],
    out_dir: str | Path,
    ranks: tuple[int, ...],
    *,
    device: str | torch.device = "cpu",
    metadata: Mapping[str, str] | None = None,
) -> dict[int, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    for rank in ranks:
        def rank_fn(_name: str, matrix: torch.Tensor, _r: int = rank) -> int:
            return min(_r, min(matrix.shape))

        state = truncated_lora_state(deltas, rank_fn, device=device)
        path = out_dir / f"truncated-rank{rank:03d}.safetensors"
        meta = {str(k): str(v) for k, v in (metadata or {}).items()}
        meta.setdefault("format", "pt")
        meta.setdefault("modelspec.architecture", "anima/lora")
        meta.setdefault("truncation_rank", str(rank))
        save_file(state, str(path), metadata=meta)
        result[rank] = path
    return result


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_orientation(name: str, P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return (A[r, n], B[m, r], transposed_delta) for delta = B @ A."""
    if P.shape[0] == Q.shape[1]:
        return P, Q, False
    if P.shape[1] == Q.shape[1]:
        return P.T.contiguous(), Q, False
    if P.shape[0] == Q.shape[0]:
        return P, Q.T.contiguous(), False
    if P.shape[1] == Q.shape[0]:
        return P.T.contiguous(), Q.T.contiguous(), True
    raise ValueError(f"unrecognized adapter orientation at {name}: {tuple(P.shape)} {tuple(Q.shape)}")


def adapter_factors_from_file(path: str | Path, device: str | torch.device = "cpu") -> dict[str, tuple[str, torch.Tensor, torch.Tensor]]:
    """Load a LoRA safetensors in kohya (`lora_unet_*`) or diffusers (`*.lora_A/B`) layout.

    Returns name -> (canonical_module_name, A[r, in], B[out, r]).
    Names of modules whose delta appears transposed get a "#T" suffix.
    """
    from safetensors.torch import load_file

    sd = load_file(str(path))
    kohya: dict[str, dict[str, torch.Tensor]] = {}
    diffusers: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in sd.items():
        if key.endswith(".lora_down.weight"):
            kohya.setdefault(key[: -len(".lora_down.weight")], {})["down"] = value.float().to(device)
        elif key.endswith(".lora_up.weight"):
            kohya.setdefault(key[: -len(".lora_up.weight")], {})["up"] = value.float().to(device)
        elif key.endswith(".lora_A.weight"):
            diffusers.setdefault(key[: -len(".lora_A.weight")], {})["A"] = value.float().to(device)
        elif key.endswith(".lora_B.weight"):
            diffusers.setdefault(key[: -len(".lora_B.weight")], {})["B"] = value.float().to(device)
    factors: dict[str, tuple[str, torch.Tensor, torch.Tensor]] = {}
    if kohya:
        for full, parts in kohya.items():
            name = kohya_to_module_name(full)
            A, B, transposed = _resolve_orientation(full, parts["down"], parts["up"])
            factors[full + ("#T" if transposed else "")] = (name, A, B)
        return factors
    for full, parts in diffusers.items():
        A, B, transposed = _resolve_orientation(full, parts["A"], parts["B"])
        factors[full + ("#T" if transposed else "")] = (full, A, B)
    return factors


def factor_spectra(factors: Mapping[str, tuple[str, torch.Tensor, torch.Tensor]]) -> dict[str, LayerSpectrum]:
    """Singular spectra of B@A via small-side SVDs (cheap for low-rank adapters)."""
    spectra: dict[str, LayerSpectrum] = {}
    for full, (name, A, B) in factors.items():
        svals, _ = factor_spectrum_pair(A, B)
        fro = svals.square().sum().sqrt().item()
        s0 = svals[0].item() if svals.numel() else 0.0
        stable = fro**2 / max(s0**2, 1e-30)
        spectra[full] = LayerSpectrum(
            name=name,
            shape=(int(B.shape[0]), int(A.shape[1])),
            stable_rank=round(stable, 3),
            effective_rank=round(effective_rank(svals), 3),
            frobenius=fro,
            spectral_norm=s0,
            ranks_for_energy={f"r{int(t*100)}": rank_for_energy(svals, t) for t in _ENERGY_TARGETS},
            singular_values=[round(v, 6) for v in svals.tolist()],
        )
    return spectra


def factor_spectrum_pair(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (singular_values, left_singular_vectors) of B @ A via small SVDs."""
    ua, sa, _ = torch.linalg.svd(A, full_matrices=False)
    buf = B @ ua * sa
    ub, svals, _ = torch.linalg.svd(buf, full_matrices=False)
    return svals, ub


def spectra_report(spectra: Mapping[str, LayerSpectrum]) -> dict:
    total_energy = sum(s.frobenius**2 for s in spectra.values()) or 1.0
    by_component: dict[str, dict] = {}
    for target in _ENERGY_TARGETS:
        key = f"r{int(target*100)}"
        for component in ("self_attn", "cross_attn", "mlp", "modulation", "llm_adapter", "other"):
            values = [s.ranks_for_energy[key] for n, s in spectra.items() if component_of(s.name) == component]
            if values:
                by_component.setdefault(component, {})[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }
    energy_by_component = {
        component: sum(s.frobenius**2 for s in spectra.values() if component_of(s.name) == component) / total_energy
        for component in sorted({component_of(s.name) for s in spectra.values()})
    }
    by_block: dict[str, dict] = {}
    for spectrum in spectra.values():
        block = block_of(spectrum.name)
        if block is None:
            continue
        entry = by_block.setdefault(str(block), {"frobenius_sq": 0.0, "layers": 0})
        entry["frobenius_sq"] += spectrum.frobenius**2
        entry["layers"] += 1
    return {
        "layers": {name: spectrum.to_json() for name, spectrum in sorted(spectra.items())},
        "n_layers": len(spectra),
        "total_frobenius": math.sqrt(total_energy),
        "energy_by_component": {k: round(v, 5) for k, v in energy_by_component.items()},
        "ranks_by_component": by_component,
        "energy_by_block": {
            k: round(v["frobenius_sq"] / total_energy, 5) for k, v in sorted(by_block.items(), key=lambda kv: int(kv[0]))
        },
        "mean_effective_rank": round(sum(s.effective_rank for s in spectra.values()) / max(1, len(spectra)), 3),
    }
