from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cache import cache_dataset, records_from_config
from .checkpoints import export_full_model, export_lora, load_training_checkpoint
from .concepts import audit_records
from .config import load_config, torch_dtype
from .loader import load_anima_dit
from .scopes import apply_train_scope
from .training import AnimaTrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anima-trainer",
        description="Architecture-aware native trainer for CircleStone Labs Anima",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="validate concept separation and dataset facet coverage")
    audit.add_argument("--config", required=True)
    audit.add_argument("--prior", action="store_true")
    audit.add_argument("--output", default=None, help="optional JSON report path")

    cache = sub.add_parser("cache", help="cache native Qwen Image VAE latents and Anima conditioning")
    cache.add_argument("--config", required=True)
    cache.add_argument("--prior", action="store_true", help="cache only the regularization manifest")
    cache.add_argument("--include-prior", action="store_true", help="cache target and configured prior manifests")
    cache.add_argument("--force", action="store_true")

    train = sub.add_parser("train", help="run training")
    train.add_argument("--config", required=True)
    train.add_argument("--resume", default=None, help="training-stepXXXXXX.pt")

    validate = sub.add_parser("validate", help="run deterministic held-out validation for a training checkpoint")
    validate.add_argument("--config", required=True)
    validate.add_argument("--checkpoint", required=True)

    export = sub.add_parser("export", help="export a training checkpoint for ComfyUI")
    export.add_argument("--config", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--merged", action="store_true", help="write a merged official net.* model")

    analyze = sub.add_parser("analyze", help="measure update geometry: per-layer spectra, energy vs rank")
    analyze.add_argument("--adapter", default=None, help="LoRA safetensors (kohya or diffusers layout)")
    analyze.add_argument("--config", default=None, help="config for dense checkpoint analysis")
    analyze.add_argument("--checkpoint", default=None, help="training checkpoint with dense/full weights")
    analyze.add_argument("--output", required=True, help="JSON report path")
    analyze.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _audit(args) -> int:
    config = load_config(args.config)
    records = records_from_config(config, prior=args.prior)
    mode = "general" if args.prior else config.concept.mode
    report = audit_records((record.as_audit_mapping() for record in records), mode)
    report["manifest"] = str(config.data.prior_manifest if args.prior else config.data.manifest)
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output:
        destination = Path(args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    return 0


def _cache(args) -> int:
    config = load_config(args.config)
    if args.prior and args.include_prior:
        raise ValueError("choose either --prior or --include-prior")
    if args.prior:
        print(cache_dataset(config, prior=True, force=args.force))
        return 0
    print(cache_dataset(config, prior=False, force=args.force))
    if args.include_prior and config.data.prior_manifest is not None:
        print(cache_dataset(config, prior=True, force=args.force))
    return 0


def _train(args) -> int:
    config = load_config(args.config)
    trainer = AnimaTrainer(config, resume=args.resume)
    trainer.train()
    return 0


def _validate(args) -> int:
    config = load_config(args.config)
    trainer = AnimaTrainer(config, resume=args.checkpoint)
    metrics = trainer.run_validation()
    if metrics is None:
        raise ValueError("validation is disabled or the target manifest has no validation split")
    return 0


def _analyze(args) -> int:
    from .update_geometry import (
        adapter_factors_from_file,
        analyze_deltas,
        deltas_from_dense,
        deltas_from_lora_state,
        factor_spectra,
        spectra_report,
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.adapter:
        if args.config or args.checkpoint:
            raise ValueError("choose either --adapter or --config/--checkpoint")
        factors = adapter_factors_from_file(args.adapter, device=args.device)
        report = spectra_report(factor_spectra(factors))
        report["source"] = str(Path(args.adapter).resolve())
        report["source_kind"] = "adapter_file"
    elif args.config and args.checkpoint:
        config = load_config(args.config)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        scope_kind = state.get("scope_kind", config.train.scope.kind)
        trainable = state["model"]
        if scope_kind in {"dit_lora", "selected_blocks_lora"}:
            deltas = deltas_from_lora_state(trainable, alpha=config.train.scope.alpha, rank=config.train.scope.rank)
            report = analyze_deltas(deltas, device=args.device)
        else:
            from .loader import normalize_checkpoint_state
            from safetensors.torch import load_file

            base = normalize_checkpoint_state(load_file(str(config.model.checkpoint), device="cpu"))
            deltas = deltas_from_dense(base, trainable)
            report = analyze_deltas(deltas, device=args.device)
        report["source"] = str(Path(args.checkpoint).resolve())
        report["source_kind"] = "training_checkpoint"
    else:
        raise ValueError("provide either --adapter or both --config and --checkpoint")
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(output)
    return 0


def _export(args) -> int:
    config = load_config(args.config)
    model = load_anima_dit(config.model.checkpoint, device="cpu", dtype=torch_dtype(config.model.dtype))
    apply_train_scope(model, config.train.scope)
    step = load_training_checkpoint(args.checkpoint, model=model, restore_rng=False)
    metadata = {
        "anima_trainer_version": "0.1.0",
        "concept_mode": config.concept.mode,
        "trigger": config.concept.trigger or "",
        "scope": config.train.scope.kind,
        "step": str(step),
        "base_model": str(config.model.checkpoint),
    }
    if args.merged or config.train.scope.kind in {"selected_blocks_full", "dit_full"}:
        export_full_model(model, args.output, metadata=metadata)
    else:
        export_lora(model, args.output, metadata=metadata)
    print(Path(args.output).resolve())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "audit": _audit,
        "cache": _cache,
        "train": _train,
        "validate": _validate,
        "export": _export,
        "analyze": _analyze,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

