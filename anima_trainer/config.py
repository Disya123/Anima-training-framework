from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_DTYPES = {"bfloat16", "float16", "float32"}
_SCOPE_KINDS = {"dit_lora", "selected_blocks_lora", "selected_blocks_full", "dit_full"}
_COMPONENTS = {
    "all",
    "self_attn",
    "cross_attn",
    "mlp",
    "modulation",
    "x_embedder",
    "timestep",
    "final_layer",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _reject_unknown(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(unknown)}")


def _path(value: str | Path | None, base: Path, *, required: bool = False) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError("required path is missing")
        return None
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = base / result
    return result.resolve()


def _tuple_of_ints(value: Any, name: str) -> tuple[int, ...] | None:
    if value in (None, "all"):
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be 'all' or a list of block indices")
    result = tuple(int(v) for v in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate block indices")
    if any(v < 0 or v > 27 for v in result):
        raise ValueError(f"{name} indices must be in [0, 27]")
    return result


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "anima-run"
    output_dir: Path = Path("runs/anima")
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: Path = Path("anima.safetensors")
    vae: Path = Path("qwen_image_vae.safetensors")
    text_encoder: Path = Path("qwen_3_06b_base.safetensors")
    dtype: str = "bfloat16"
    quantization: str | None = None
    quantize_extent: str = "below_trainable"


_QUANTIZATION_MODES = {None, "convrot8"}
_QUANTIZE_EXTENTS = {"below_trainable", "all"}


@dataclass(frozen=True)
class DataConfig:
    manifest: Path = Path("manifest.jsonl")
    cache_dir: Path = Path("cache")
    prior_manifest: Path | None = None
    prior_cache_dir: Path | None = None
    resolution: int = 768
    aspect_buckets: tuple[float, ...] = (0.5, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 2.0)
    bucket_multiple: int = 16
    num_workers: int = 0
    prior_loss_weight: float = 1.0


@dataclass(frozen=True)
class ConceptConfig:
    mode: str = "style"
    trigger: str | None = None
    trigger_position: str = "prefix"
    hard_tag_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeConfig:
    kind: str = "dit_lora"
    blocks: tuple[int, ...] | None = None
    components: tuple[str, ...] = ("self_attn", "cross_attn", "mlp")
    rank: int = 32
    alpha: float = 32.0
    rank_overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    lr_overrides: Mapping[str, float] = field(default_factory=dict)
    dropout: float = 0.0
    trainable_dtype: str = "float32"


@dataclass(frozen=True)
class TrainConfig:
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    steps: int = 1000
    micro_batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float | None = None
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    optimizer: str = "adamw8bit"
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    timestep_sampling: str = "logit_normal"
    sigmoid_scale: float = 1.0
    sigma_shift: float = 3.0
    gradient_checkpointing: bool = True
    checkpoint_mode: str | None = None
    checkpoint_components: tuple[str, ...] | None = None
    checkpoint_group_size: int = 4
    checkpoint_sac: str | None = None
    gradient_transport: Mapping[str, str] = field(default_factory=dict)
    anchor_no_trigger_weight: float = 0.0
    anchor_every: int = 1
    debug_sync_checks: bool = False
    fast_loss_reduction: bool = False
    log_every: int = 10
    save_every: int = 100
    resume: Path | None = None


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    every: int = 100
    max_samples: int = 16
    fixed_sigmas: tuple[float, ...] = (0.15, 0.5, 0.85)


@dataclass(frozen=True)
class TrainerConfig:
    source: Path
    project: ProjectConfig
    model: ModelConfig
    data: DataConfig
    concept: ConceptConfig
    train: TrainConfig
    validation: ValidationConfig

    @property
    def base_dir(self) -> Path:
        return self.source.parent


def load_config(path: str | Path) -> TrainerConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _mapping(raw, "config")
    _reject_unknown(raw, {"project", "model", "data", "concept", "train", "validation"}, "top-level")
    base = source.parent

    project_raw = _mapping(raw.get("project"), "project")
    _reject_unknown(project_raw, {"name", "output_dir", "seed"}, "project")
    project = ProjectConfig(
        name=str(project_raw.get("name", "anima-run")),
        output_dir=_path(project_raw.get("output_dir", "runs/anima"), base, required=True),
        seed=int(project_raw.get("seed", 0)),
    )

    model_raw = _mapping(raw.get("model"), "model")
    _reject_unknown(model_raw, {"checkpoint", "vae", "text_encoder", "dtype", "quantization", "quantize_extent"}, "model")
    quantization = model_raw.get("quantization")
    quantization = None if quantization in (None, "", "none") else str(quantization)
    if quantization not in _QUANTIZATION_MODES:
        raise ValueError(f"model.quantization must be one of {sorted(str(q) for q in _QUANTIZATION_MODES)}")
    quantize_extent = str(model_raw.get("quantize_extent", "below_trainable"))
    if quantize_extent not in _QUANTIZE_EXTENTS:
        raise ValueError(f"model.quantize_extent must be one of {sorted(_QUANTIZE_EXTENTS)}")
    dtype = str(model_raw.get("dtype", "bfloat16"))
    if dtype not in _DTYPES:
        raise ValueError(f"model.dtype must be one of {sorted(_DTYPES)}")
    model = ModelConfig(
        checkpoint=_path(model_raw.get("checkpoint"), base, required=True),
        vae=_path(model_raw.get("vae"), base, required=True),
        text_encoder=_path(model_raw.get("text_encoder"), base, required=True),
        dtype=dtype,
        quantization=quantization,
        quantize_extent=quantize_extent,
    )

    data_raw = _mapping(raw.get("data"), "data")
    _reject_unknown(
        data_raw,
        {
            "manifest",
            "cache_dir",
            "prior_manifest",
            "prior_cache_dir",
            "resolution",
            "aspect_buckets",
            "bucket_multiple",
            "num_workers",
            "prior_loss_weight",
        },
        "data",
    )
    buckets = tuple(float(v) for v in data_raw.get("aspect_buckets", (0.5, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 2.0)))
    if not buckets or any(v <= 0 for v in buckets):
        raise ValueError("data.aspect_buckets must contain positive ratios")
    bucket_multiple = int(data_raw.get("bucket_multiple", 16))
    if bucket_multiple <= 0 or bucket_multiple % 16 != 0:
        raise ValueError("data.bucket_multiple must be a positive multiple of 16")
    prior_manifest = _path(data_raw.get("prior_manifest"), base)
    cache_dir = _path(data_raw.get("cache_dir", "cache"), base, required=True)
    prior_cache_dir = _path(data_raw.get("prior_cache_dir"), base)
    if prior_manifest is not None and prior_cache_dir is None:
        prior_cache_dir = cache_dir.parent / f"{cache_dir.name}_prior"
    data = DataConfig(
        manifest=_path(data_raw.get("manifest"), base, required=True),
        cache_dir=cache_dir,
        prior_manifest=prior_manifest,
        prior_cache_dir=prior_cache_dir,
        resolution=int(data_raw.get("resolution", 768)),
        aspect_buckets=buckets,
        bucket_multiple=bucket_multiple,
        num_workers=int(data_raw.get("num_workers", 0)),
        prior_loss_weight=float(data_raw.get("prior_loss_weight", 1.0)),
    )
    if data.resolution < 256 or data.resolution % 16:
        raise ValueError("data.resolution must be >=256 and divisible by 16")
    if data.prior_loss_weight < 0:
        raise ValueError("data.prior_loss_weight must be non-negative")

    concept_raw = _mapping(raw.get("concept"), "concept")
    _reject_unknown(concept_raw, {"mode", "trigger", "trigger_position", "hard_tag_weights"}, "concept")
    mode = str(concept_raw.get("mode", "style"))
    if mode not in {"style", "character", "object", "general"}:
        raise ValueError("concept.mode must be style, character, object, or general")
    trigger_position = str(concept_raw.get("trigger_position", "prefix"))
    if trigger_position not in {"prefix", "suffix"}:
        raise ValueError("concept.trigger_position must be prefix or suffix")
    hard_tag_weights = {str(k): float(v) for k, v in _mapping(concept_raw.get("hard_tag_weights"), "concept.hard_tag_weights").items()}
    if any(v <= 0 for v in hard_tag_weights.values()):
        raise ValueError("concept.hard_tag_weights values must be positive")
    concept = ConceptConfig(
        mode=mode,
        trigger=str(concept_raw["trigger"]).strip() if concept_raw.get("trigger") else None,
        trigger_position=trigger_position,
        hard_tag_weights=hard_tag_weights,
    )

    train_raw = _mapping(raw.get("train"), "train")
    _reject_unknown(
        train_raw,
        {
            "scope",
            "steps",
            "micro_batch_size",
            "gradient_accumulation",
            "learning_rate",
            "weight_decay",
            "betas",
            "eps",
            "optimizer",
            "lr_scheduler",
            "warmup_steps",
            "max_grad_norm",
            "timestep_sampling",
            "sigmoid_scale",
            "sigma_shift",
            "gradient_checkpointing",
            "checkpoint_mode",
            "checkpoint_components",
            "checkpoint_group_size",
            "checkpoint_sac",
            "gradient_transport",
            "anchor_no_trigger_weight",
            "anchor_every",
            "debug_sync_checks",
            "fast_loss_reduction",
            "log_every",
            "save_every",
            "resume",
        },
        "train",
    )
    scope_raw = _mapping(train_raw.get("scope"), "train.scope")
    _reject_unknown(
        scope_raw,
        {"kind", "blocks", "components", "rank", "alpha", "rank_overrides", "lr_overrides", "dropout", "trainable_dtype"},
        "train.scope",
    )
    kind = str(scope_raw.get("kind", "dit_lora"))
    if kind not in _SCOPE_KINDS:
        raise ValueError(f"train.scope.kind must be one of {sorted(_SCOPE_KINDS)}")
    blocks = _tuple_of_ints(scope_raw.get("blocks"), "train.scope.blocks")
    if kind.startswith("selected_blocks") and not blocks:
        raise ValueError(f"{kind} requires a non-empty train.scope.blocks list")
    components = tuple(str(v) for v in scope_raw.get("components", ("self_attn", "cross_attn", "mlp")))
    invalid_components = sorted(set(components) - _COMPONENTS)
    if invalid_components:
        raise ValueError(f"unknown scope components: {', '.join(invalid_components)}")
    trainable_dtype = str(scope_raw.get("trainable_dtype", "float32"))
    if trainable_dtype not in _DTYPES:
        raise ValueError(f"train.scope.trainable_dtype must be one of {sorted(_DTYPES)}")
    scope_rank_default = int(scope_raw.get("rank", 32))
    scope_alpha_default = float(scope_raw.get("alpha", scope_rank_default))
    overrides_raw = _mapping(scope_raw.get("rank_overrides"), "train.scope.rank_overrides")
    rank_overrides: dict[str, dict[str, float]] = {}
    for pattern, spec in overrides_raw.items():
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("train.scope.rank_overrides keys must be non-empty strings")
        spec_map = _mapping(spec, f"train.scope.rank_overrides[{pattern}]")
        unknown = sorted(set(spec_map) - {"rank", "alpha"})
        if unknown:
            raise ValueError(f"rank_overrides[{pattern}] unknown keys: {', '.join(unknown)}")
        r = int(spec_map.get("rank", scope_rank_default))
        a = float(spec_map.get("alpha", r if "rank" in spec_map else scope_alpha_default))
        if r < 0:
            raise ValueError(f"rank_overrides[{pattern}].rank must be >= 0 (0 drops the module)")
        if r > 0 and a <= 0:
            raise ValueError(f"rank_overrides[{pattern}].alpha must be positive")
        rank_overrides[pattern] = {"rank": float(r), "alpha": float(a)}
    lr_overrides_raw = _mapping(scope_raw.get("lr_overrides"), "train.scope.lr_overrides")
    lr_overrides: dict[str, float] = {}
    for component, value in lr_overrides_raw.items():
        if component not in _COMPONENTS:
            raise ValueError(f"lr_overrides unknown component: {component}")
        if float(value) <= 0:
            raise ValueError(f"lr_overrides[{component}] must be positive")
        lr_overrides[component] = float(value)
    scope = ScopeConfig(
        kind=kind,
        blocks=blocks,
        components=components,
        rank=scope_rank_default,
        alpha=scope_alpha_default,
        rank_overrides=rank_overrides,
        lr_overrides=lr_overrides,
        dropout=float(scope_raw.get("dropout", 0.0)),
        trainable_dtype=trainable_dtype,
    )
    if scope.rank <= 0 or scope.alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= scope.dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")

    betas_raw = train_raw.get("betas", (0.9, 0.99))
    if len(betas_raw) != 2:
        raise ValueError("train.betas must contain exactly two values")
    timestep_sampling = str(train_raw.get("timestep_sampling", "logit_normal"))
    if timestep_sampling not in {"logit_normal", "uniform", "balanced", "style_balanced", "weighted"}:
        raise ValueError("train.timestep_sampling must be logit_normal, uniform, balanced, style_balanced, or weighted")
    optimizer = str(train_raw.get("optimizer", "adamw8bit"))
    if optimizer not in {"adamw", "adamw8bit"}:
        raise ValueError("train.optimizer must be adamw or adamw8bit")
    gt_raw = _mapping(train_raw.get("gradient_transport"), "train.gradient_transport")
    gradient_transport: dict[str, str] = {}
    _SECTIONS = {"self_attn", "cross_attn", "mlp"}
    for pattern, pol in gt_raw.items():
        if str(pol) not in {"local", "exact"}:
            raise ValueError(f"gradient_transport[{pattern}] must be 'local' or 'exact'")
        head, _, section = str(pattern).rpartition(".")
        if section not in _SECTIONS:
            raise ValueError(f"gradient_transport[{pattern}] section must be one of {sorted(_SECTIONS)}")
        if head.startswith("blocks."):
            rng = head[len("blocks."):]
            try:
                lo, hi = (int(v) for v in rng.split("-", 1)) if "-" in rng else (int(rng),) * 2
            except ValueError:
                raise ValueError(f"gradient_transport[{pattern}] invalid block range") from None
            if not (0 <= lo <= hi <= 27):
                raise ValueError(f"gradient_transport[{pattern}] block range out of [0,27]")
        elif head != "":
            raise ValueError(f"gradient_transport[{pattern}] unknown prefix '{head}' (use 'blocks.N' or nothing)")
        gradient_transport[str(pattern)] = str(pol)
    scheduler = str(train_raw.get("lr_scheduler", "constant"))
    if scheduler not in {"constant", "cosine", "linear"}:
        raise ValueError("train.lr_scheduler must be constant, cosine, or linear")
    _CKPT_COMPONENTS = {"self_attn", "cross_attn", "mlp"}
    cc_raw = train_raw.get("checkpoint_components")
    checkpoint_components = None
    if cc_raw is not None:
        if not isinstance(cc_raw, (list, tuple)):
            raise ValueError("train.checkpoint_components must be a list or null")
        checkpoint_components = tuple(str(v) for v in cc_raw)
        invalid = sorted(set(checkpoint_components) - _CKPT_COMPONENTS)
        if invalid:
            raise ValueError(
                f"train.checkpoint_components allows {sorted(_CKPT_COMPONENTS)}, got {invalid}; "
                "empty list disables section checkpointing"
            )
    train = TrainConfig(
        scope=scope,
        steps=int(train_raw.get("steps", 1000)),
        micro_batch_size=int(train_raw.get("micro_batch_size", 1)),
        gradient_accumulation=int(train_raw.get("gradient_accumulation", 1)),
        learning_rate=float(train_raw["learning_rate"]) if train_raw.get("learning_rate") is not None else None,
        weight_decay=float(train_raw.get("weight_decay", 0.01)),
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        eps=float(train_raw.get("eps", 1e-8)),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        warmup_steps=int(train_raw.get("warmup_steps", 0)),
        max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
        timestep_sampling=timestep_sampling,
        sigmoid_scale=float(train_raw.get("sigmoid_scale", 1.0)),
        sigma_shift=float(train_raw.get("sigma_shift", 1.0)),
        gradient_checkpointing=bool(train_raw.get("gradient_checkpointing", True)),
        checkpoint_mode=(str(train_raw["checkpoint_mode"]) if train_raw.get("checkpoint_mode") is not None else None),
        checkpoint_components=checkpoint_components,
        checkpoint_group_size=int(train_raw.get("checkpoint_group_size", 1)),
        checkpoint_sac=(str(train_raw["checkpoint_sac"]) if train_raw.get("checkpoint_sac") else None),
        gradient_transport=gradient_transport,
        anchor_no_trigger_weight=float(train_raw.get("anchor_no_trigger_weight", 0.0)),
        anchor_every=int(train_raw.get("anchor_every", 1)),
        debug_sync_checks=bool(train_raw.get("debug_sync_checks", False)),
        fast_loss_reduction=bool(train_raw.get("fast_loss_reduction", False)),
        log_every=int(train_raw.get("log_every", 10)),
        save_every=int(train_raw.get("save_every", 100)),
        resume=_path(train_raw.get("resume"), base),
    )
    if min(train.steps, train.micro_batch_size, train.gradient_accumulation, train.log_every, train.save_every) <= 0:
        raise ValueError("steps, batch, accumulation, log_every, and save_every must be positive")
    if train.sigma_shift <= 0 or train.sigmoid_scale <= 0:
        raise ValueError("sigma_shift and sigmoid_scale must be positive")
    if train.anchor_no_trigger_weight < 0:
        raise ValueError("train.anchor_no_trigger_weight must be non-negative")
    _CKPT_COMPONENTS = {"self_attn", "cross_attn", "mlp"}
    if train.anchor_every < 1:
        raise ValueError("train.anchor_every must be >= 1")
    if train.checkpoint_group_size < 1:
        raise ValueError("train.checkpoint_group_size must be >= 1")
    if train.checkpoint_sac is not None and train.checkpoint_sac not in {"sdpa", "gelu", "mm"}:
        raise ValueError("train.checkpoint_sac must be sdpa, gelu, mm, or null")
    if train.checkpoint_sac is not None and train.checkpoint_mode not in (None, "block"):
        raise ValueError("checkpoint_sac requires checkpoint_mode=block (selective policy inside whole-block regions)")
    if train.checkpoint_mode is not None:
        if train.checkpoint_mode not in {"off", "block", "group", "selective"}:
            raise ValueError("train.checkpoint_mode must be off, block, group, selective, or null")
        if train.checkpoint_mode == "group" and train.checkpoint_group_size < 2:
            raise ValueError("checkpoint_mode=group requires checkpoint_group_size >= 2")
        if train.checkpoint_mode == "selective":
            comps = tuple(train.checkpoint_components or ())
            invalid = sorted(set(comps) - _CKPT_COMPONENTS)
            if not comps or invalid:
                raise ValueError(
                    f"checkpoint_mode=selective requires non-empty checkpoint_components from {sorted(_CKPT_COMPONENTS)}"
                    + (f"; unknown: {invalid}" if invalid else "")
                )
    elif train.checkpoint_components is not None and train.checkpoint_group_size > 1:
        raise ValueError("legacy checkpoint_components cannot be combined with checkpoint_group_size>1; set checkpoint_mode")

    val_raw = _mapping(raw.get("validation"), "validation")
    _reject_unknown(val_raw, {"enabled", "every", "max_samples", "fixed_sigmas"}, "validation")
    validation = ValidationConfig(
        enabled=bool(val_raw.get("enabled", True)),
        every=int(val_raw.get("every", 100)),
        max_samples=int(val_raw.get("max_samples", 16)),
        fixed_sigmas=tuple(float(v) for v in val_raw.get("fixed_sigmas", (0.15, 0.5, 0.85))),
    )
    if validation.every <= 0 or validation.max_samples <= 0:
        raise ValueError("validation.every and max_samples must be positive")
    if not validation.fixed_sigmas or any(not 0 <= v <= 1 for v in validation.fixed_sigmas):
        raise ValueError("validation.fixed_sigmas must be non-empty and inside [0, 1]")

    return TrainerConfig(source, project, model, data, concept, train, validation)


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

