from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

from .conditioning import AnimaConditioner
from .config import TrainerConfig, torch_dtype
from .data import ManifestRecord, choose_bucket, load_bucketed_image, load_manifest


WAN21_MEAN = torch.tensor(
    [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    ]
).view(1, 16, 1, 1, 1)
WAN21_STD = torch.tensor(
    [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    ]
).view(1, 16, 1, 1, 1)


def wan21_process_in(latents: torch.Tensor) -> torch.Tensor:
    return (latents - WAN21_MEAN.to(latents)) / WAN21_STD.to(latents)


class LatentEncoder:
    """Native Qwen Image VAE encoder with Wan21 latent-space normalization."""

    def __init__(self, vae_path: str | Path, device: str = "cuda", dtype=torch.bfloat16):
        from .wan_vae import QwenImageVAEEncoder

        self.encoder = QwenImageVAEEncoder(str(vae_path), device=device, dtype=dtype)

    @torch.inference_mode()
    def encode(self, image_bhwc: torch.Tensor) -> torch.Tensor:
        raw = self.encoder.encode(image_bhwc)
        if raw.ndim != 5 or raw.shape[1] != 16:
            raise RuntimeError(f"Qwen Image VAE returned unexpected latent shape {tuple(raw.shape)}")
        return wan21_process_in(raw)


def records_from_config(config: TrainerConfig, *, prior: bool) -> list[ManifestRecord]:
    if prior:
        if config.data.prior_manifest is None:
            return []
        path = config.data.prior_manifest
    else:
        path = config.data.manifest
    return load_manifest(
        path,
        mode=config.concept.mode,
        global_trigger=config.concept.trigger,
        trigger_position=config.concept.trigger_position,
        hard_tag_weights=config.concept.hard_tag_weights,
        prior=prior,
    )


def _condition_key(prompt: str, content_prompt: str) -> str:
    return hashlib.sha256((prompt + "\0" + content_prompt).encode("utf-8")).hexdigest()[:24]


def _cache_dir(config: TrainerConfig, prior: bool) -> Path:
    result = config.data.prior_cache_dir if prior else config.data.cache_dir
    if result is None:
        raise ValueError("prior cache requested without data.prior_manifest")
    return result


def cache_latents(
    config: TrainerConfig,
    records: Iterable[ManifestRecord],
    *,
    prior: bool,
    force: bool = False,
) -> dict[str, tuple[str, tuple[int, int]]]:
    records = list(records)
    cache_dir = _cache_dir(config, prior)
    latent_dir = cache_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = LatentEncoder(config.model.vae, device=device, dtype=torch_dtype(config.model.dtype))
    result: dict[str, tuple[str, tuple[int, int]]] = {}
    for record in tqdm(records, desc="VAE cache", unit="image"):
        with Image.open(record.image) as probe:
            bucket = choose_bucket(
                probe.width,
                probe.height,
                config.data.resolution,
                config.data.aspect_buckets,
                config.data.bucket_multiple,
            )
        relative = Path("latents") / f"{record.id}.safetensors"
        destination = cache_dir / relative
        if force or not destination.is_file():
            image = load_bucketed_image(record.image, bucket).unsqueeze(0)
            latent = encoder.encode(image)[0].to(torch.bfloat16).cpu().contiguous()
            save_file(
                {"latent": latent},
                str(destination),
                metadata={"source": str(record.image), "format": "pt"},
            )
        result[record.id] = (relative.as_posix(), bucket)
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def cache_conditioning(
    config: TrainerConfig,
    records: Iterable[ManifestRecord],
    *,
    prior: bool,
    force: bool = False,
) -> dict[str, str]:
    records = list(records)
    cache_dir = _cache_dir(config, prior)
    cond_dir = cache_dir / "conditioning"
    cond_dir.mkdir(parents=True, exist_ok=True)
    conditioner = AnimaConditioner(
        config.model.text_encoder,
        config.model.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch_dtype(config.model.dtype),
    )
    key_to_prompts = {
        _condition_key(record.prompt, record.content_prompt): (record.prompt, record.content_prompt)
        for record in records
    }
    for key, (prompt, content_prompt) in tqdm(
        key_to_prompts.items(), desc="conditioning cache", unit="prompt"
    ):
        destination = cond_dir / f"{key}.safetensors"
        if force or not destination.is_file():
            cond = conditioner.encode(prompt).to(torch.bfloat16)
            cond_no_trigger = cond.clone() if content_prompt == prompt else conditioner.encode(content_prompt).to(torch.bfloat16)
            save_file(
                {"cond": cond.contiguous(), "cond_no_trigger": cond_no_trigger.contiguous()},
                str(destination),
                metadata={"prompt": prompt, "content_prompt": content_prompt, "format": "pt"},
            )
    del conditioner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        record.id: (
            Path("conditioning")
            / f"{_condition_key(record.prompt, record.content_prompt)}.safetensors"
        ).as_posix()
        for record in records
    }


def write_cache_index(
    config: TrainerConfig,
    records: Iterable[ManifestRecord],
    latent_map: dict[str, tuple[str, tuple[int, int]]],
    conditioning_map: dict[str, str],
    *,
    prior: bool,
) -> Path:
    records = list(records)
    cache_dir = _cache_dir(config, prior)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "index.jsonl"
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            latent_file, bucket = latent_map[record.id]
            entry = {
                "id": record.id,
                "image": str(record.image),
                "caption": record.caption,
                "prompt": record.prompt,
                "content_prompt": record.content_prompt,
                "trigger": record.trigger,
                "concept_type": record.concept_type,
                "weight": record.weight,
                "hard_tags": list(record.hard_tags),
                "facets": record.facets,
                "split": record.split,
                "bucket": list(bucket),
                "latent_file": latent_file,
                "conditioning_file": conditioning_map[record.id],
                "prior": prior,
            }
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "manifest": str(config.data.prior_manifest if prior else config.data.manifest),
        "checkpoint": str(config.model.checkpoint),
        "vae": str(config.model.vae),
        "text_encoder": str(config.model.text_encoder),
        "resolution": config.data.resolution,
        "prior": prior,
        "records": len(records),
    }
    (cache_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def cache_dataset(config: TrainerConfig, *, prior: bool = False, force: bool = False) -> Path:
    records = records_from_config(config, prior=prior)
    if not records:
        raise ValueError("no prior manifest configured")
    latent_map = cache_latents(config, records, prior=prior, force=force)
    conditioning_map = cache_conditioning(config, records, prior=prior, force=force)
    return write_cache_index(config, records, latent_map, conditioning_map, prior=prior)

