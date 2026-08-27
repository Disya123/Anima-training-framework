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


CACHE_FORMAT_VERSION = 2
_FINGERPRINT_BYTES = 262144


def _file_fingerprint(path: str | Path) -> str:
    """Cheap content-ish fingerprint: name + size + mtime + head/tail bytes."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    with p.open("rb") as handle:
        h.update(handle.read(_FINGERPRINT_BYTES))
        if st.st_size > 2 * _FINGERPRINT_BYTES:
            handle.seek(-_FINGERPRINT_BYTES, 2)
            h.update(handle.read(_FINGERPRINT_BYTES))
    return h.hexdigest()[:20]


def _model_fingerprints(config: TrainerConfig) -> dict[str, str]:
    return {
        "vae_sig": _file_fingerprint(config.model.vae),
        "te_sig": _file_fingerprint(config.model.text_encoder),
        "dit_sig": _file_fingerprint(config.model.checkpoint),
    }


def _hash_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _latent_signature(image_fp: str, bucket: tuple[int, int], vae_fp: str) -> str:
    return _hash_text(f"latent:v1:{image_fp}:{bucket[0]}x{bucket[1]}:{vae_fp}")


def _cond_signature(prompt: str, content_prompt: str, te_fp: str, dit_fp: str) -> str:
    return _hash_text(f"cond:v1:{prompt}\0{content_prompt}\0{te_fp}\0{dit_fp}")


def validate_cache_signatures(config: TrainerConfig) -> None:
    """Raise if an existing cache was built against different base models.

    Legacy caches (format_version missing) only warn: they predate signatures
    and must not hard-block established runs.
    """
    for prior in (False, True):
        if prior and config.data.prior_manifest is None:
            continue
        cache_dir = _cache_dir(config, prior)
        meta_path = cache_dir / "cache_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("format_version", 1)) < CACHE_FORMAT_VERSION:
            print(f"[cache] {cache_dir}: legacy format without signatures; run `anima-trainer cache --force` to adopt")
            continue
        if int(meta.get("resolution", -1)) != config.data.resolution:
            raise ValueError(f"cache {cache_dir}: resolution changed ({meta.get('resolution')} -> {config.data.resolution}); run `anima-trainer cache --force`")
        fps = _model_fingerprints(config)
        for key, current in fps.items():
            if meta.get(key) != current:
                raise ValueError(
                    f"cache {cache_dir}: {key} changed since caching; run `anima-trainer cache --force`"
                )


def _cache_dir(config: TrainerConfig, prior: bool) -> Path:
    result = config.data.prior_cache_dir if prior else config.data.cache_dir
    if result is None:
        raise ValueError("prior cache requested without data.prior_manifest")
    return result


def _read_old_index(cache_dir: Path) -> dict[str, dict]:
    index_path = cache_dir / "index.jsonl"
    if not index_path.is_file():
        return {}
    entries: dict[str, dict] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            entries[entry["id"]] = entry
    return entries


def cache_latents(
    config: TrainerConfig,
    records: Iterable[ManifestRecord],
    *,
    prior: bool,
    force: bool = False,
):
    records = list(records)
    cache_dir = _cache_dir(config, prior)
    latent_dir = cache_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    old_index = {} if force else _read_old_index(cache_dir)
    fps = _model_fingerprints(config)

    def entry_signature(record: ManifestRecord, bucket: tuple[int, int]) -> str:
        return _latent_signature(_file_fingerprint(record.image), bucket, fps["vae_sig"])

    pending: list[ManifestRecord] = []
    signature_of: dict[str, str] = {}
    bucket_of: dict[str, tuple[int, int]] = {}
    for record in records:
        with Image.open(record.image) as probe:
            bucket = choose_bucket(
                probe.width,
                probe.height,
                config.data.resolution,
                config.data.aspect_buckets,
                config.data.bucket_multiple,
            )
        sig = entry_signature(record, bucket)
        signature_of[record.id] = sig
        bucket_of[record.id] = bucket
        destination = cache_dir / "latents" / f"{record.id}.safetensors"
        old = old_index.get(record.id) or {}
        if force or not destination.is_file() or old.get("sig") != sig:
            pending.append(record)

    encoder = None
    for record in tqdm(pending, desc="VAE cache", unit="image"):
        if encoder is None:
            encoder = LatentEncoder(config.model.vae, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch_dtype(config.model.dtype))
        image = load_bucketed_image(record.image, bucket_of[record.id]).unsqueeze(0)
        latent = encoder.encode(image)[0].to(torch.bfloat16).cpu().contiguous()
        destination = cache_dir / "latents" / f"{record.id}.safetensors"
        save_file(
            {"latent": latent},
            str(destination),
            metadata={"source": str(record.image), "format": "pt"},
        )
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {record.id: (f"latents/{record.id}.safetensors", bucket_of[record.id]) for record in records}
    return result, signature_of, len(pending)


def cache_conditioning(
    config: TrainerConfig,
    records: Iterable[ManifestRecord],
    *,
    prior: bool,
    force: bool = False,
):
    records = list(records)
    cache_dir = _cache_dir(config, prior)
    cond_dir = cache_dir / "conditioning"
    cond_dir.mkdir(parents=True, exist_ok=True)
    old_index = {} if force else _read_old_index(cache_dir)
    fps = _model_fingerprints(config)
    key_to_prompts = {
        _condition_key(record.prompt, record.content_prompt): (record.prompt, record.content_prompt)
        for record in records
    }
    signature_of: dict[str, str] = {}
    pending: list[str] = []
    for key, (prompt, content_prompt) in key_to_prompts.items():
        sig = _cond_signature(prompt, content_prompt, fps["te_sig"], fps["dit_sig"])
        signature_of[key] = sig
        destination = cond_dir / f"{key}.safetensors"
        if force or not destination.is_file():
            pending.append(key)
            continue
        # existing file: validate stored signature via sidecar index when present
        stale = not old_index or all(
            entry.get("cond_sig") != sig
            for entry in old_index.values()
            if entry.get("conditioning_file") == f"conditioning/{key}.safetensors"
        )
        if stale:
            pending.append(key)

    conditioner = None
    for key in tqdm(pending, desc="conditioning cache", unit="prompt"):
        if conditioner is None:
            conditioner = AnimaConditioner(
                config.model.text_encoder,
                config.model.checkpoint,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch_dtype(config.model.dtype),
            )
        prompt, content_prompt = key_to_prompts[key]
        cond = conditioner.encode(prompt).to(torch.bfloat16)
        cond_no_trigger = cond.clone() if content_prompt == prompt else conditioner.encode(content_prompt).to(torch.bfloat16)
        save_file(
            {"cond": cond.contiguous(), "cond_no_trigger": cond_no_trigger.contiguous()},
            str(cond_dir / f"{key}.safetensors"),
            metadata={"prompt": prompt, "content_prompt": content_prompt, "format": "pt"},
        )
    del conditioner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cond_map = {
        record.id: f"conditioning/{_condition_key(record.prompt, record.content_prompt)}.safetensors"
        for record in records
    }
    return cond_map, signature_of, len(pending)


def write_cache_index(
    config: TrainerConfig,
    records: Iterable[ManifestRecord],
    latent_map: dict[str, tuple[str, tuple[int, int]]],
    conditioning_map: dict[str, str],
    *,
    prior: bool,
    latent_sigs: dict[str, str] | None = None,
    cond_sigs: dict[str, str] | None = None,
) -> Path:
    records = list(records)
    latent_sigs = latent_sigs or {}
    cond_sigs = cond_sigs or {}
    cache_dir = _cache_dir(config, prior)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "index.jsonl"
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            latent_file, bucket = latent_map[record.id]
            cond_key = _condition_key(record.prompt, record.content_prompt)
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
                "sig": latent_sigs.get(record.id),
                "cond_sig": cond_sigs.get(cond_key),
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
        "format_version": CACHE_FORMAT_VERSION,
        **_model_fingerprints(config),
    }
    (cache_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def cache_dataset(config: TrainerConfig, *, prior: bool = False, force: bool = False) -> Path:
    records = records_from_config(config, prior=prior)
    if not records:
        raise ValueError("no prior manifest configured")
    latent_map, latent_sigs, latents_encoded = cache_latents(config, records, prior=prior, force=force)
    conditioning_map, cond_sigs, conds_encoded = cache_conditioning(config, records, prior=prior, force=force)
    print(f"[cache] latents: {latents_encoded}/{len(records)} encoded, conditioning: {conds_encoded} prompts encoded")
    return write_cache_index(
        config,
        records,
        latent_map,
        conditioning_map,
        prior=prior,
        latent_sigs=latent_sigs,
        cond_sigs=cond_sigs,
    )

