from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torch.utils.data import Dataset, Sampler

from .concepts import ConceptMode, build_prompt, effective_weight


_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_MANIFEST_FIELDS = {
    "id",
    "image",
    "caption",
    "trigger",
    "concept_type",
    "weight",
    "hard_tags",
    "facets",
    "split",
}


@dataclass(frozen=True)
class ManifestRecord:
    id: str
    image: Path
    caption: str
    prompt: str
    content_prompt: str
    trigger: str | None
    concept_type: str
    weight: float
    hard_tags: tuple[str, ...]
    facets: dict[str, Any]
    split: str
    source_line: int

    def as_audit_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "caption": self.caption,
            "trigger": self.trigger,
            "facets": self.facets,
            "hard_tags": list(self.hard_tags),
        }


def _safe_id(value: str) -> str:
    result = _ID_RE.sub("_", value.strip()).strip("._")
    if not result:
        raise ValueError("manifest id becomes empty after sanitization")
    return result


def load_manifest(
    path: str | Path,
    *,
    mode: str,
    global_trigger: str | None = None,
    trigger_position: str = "prefix",
    hard_tag_weights: Mapping[str, float] | None = None,
    prior: bool = False,
    require_images: bool = True,
) -> list[ManifestRecord]:
    path = Path(path).resolve()
    records: list[ManifestRecord] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(raw, dict):
                raise TypeError(f"{path}:{line_number}: each line must be an object")
            unknown = sorted(set(raw) - _MANIFEST_FIELDS)
            if unknown:
                raise ValueError(f"{path}:{line_number}: unknown fields: {', '.join(unknown)}")
            if not raw.get("image"):
                raise ValueError(f"{path}:{line_number}: image is required")
            image = Path(str(raw["image"])).expanduser()
            if not image.is_absolute():
                image = path.parent / image
            image = image.resolve()
            if image.suffix.lower() not in _IMAGE_EXTENSIONS:
                raise ValueError(f"{path}:{line_number}: unsupported image extension: {image.suffix}")
            if require_images and not image.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: {image}")
            record_mode = ConceptMode.GENERAL.value if prior else str(raw.get("concept_type", mode))
            if not prior and record_mode != mode:
                raise ValueError(
                    f"{path}:{line_number}: concept_type={record_mode!r} does not match run mode={mode!r}"
                )
            prompt, content_prompt, trigger = build_prompt(
                str(raw.get("caption", "")),
                mode=record_mode,
                record_trigger=None if prior else raw.get("trigger"),
                global_trigger=None if prior else global_trigger,
                trigger_position=trigger_position,
            )
            raw_id = str(raw.get("id") or f"{image.stem}-{line_number:05d}")
            record_id = _safe_id(raw_id)
            if record_id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            ids.add(record_id)
            hard_tags_raw = raw.get("hard_tags") or []
            if not isinstance(hard_tags_raw, list):
                raise TypeError(f"{path}:{line_number}: hard_tags must be a list")
            hard_tags = tuple(str(tag) for tag in hard_tags_raw)
            weight = effective_weight(float(raw.get("weight", 1.0)), hard_tags, hard_tag_weights or {})
            if weight <= 0:
                raise ValueError(f"{path}:{line_number}: sample weight must be positive (got {weight})")
            split = str(raw.get("split", "train"))
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"{path}:{line_number}: split must be train, validation, or test")
            facets = raw.get("facets") or {}
            if not isinstance(facets, dict):
                raise TypeError(f"{path}:{line_number}: facets must be an object")
            records.append(
                ManifestRecord(
                    id=record_id,
                    image=image,
                    caption=str(raw.get("caption", "")),
                    prompt=prompt,
                    content_prompt=content_prompt,
                    trigger=trigger,
                    concept_type=record_mode,
                    weight=weight,
                    hard_tags=hard_tags,
                    facets=dict(facets),
                    split=split,
                    source_line=line_number,
                )
            )
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def choose_bucket(
    width: int,
    height: int,
    resolution: int,
    aspect_buckets: Sequence[float],
    multiple: int = 16,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    ratio = width / height
    chosen = min(aspect_buckets, key=lambda candidate: abs(math.log(ratio / candidate)))
    area = resolution * resolution
    target_width = max(multiple, round(math.sqrt(area * chosen) / multiple) * multiple)
    target_height = max(multiple, round(math.sqrt(area / chosen) / multiple) * multiple)
    return target_width, target_height


def load_bucketed_image(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(image.height, image.width, 3).float() / 255.0
    return tensor


def read_index(cache_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(cache_dir) / "index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"cache index not found: {path}; run `anima-trainer cache` first")
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                entry = json.loads(line)
                entry["_line"] = line_number
                entries.append(entry)
    return entries


class CachedConceptDataset(Dataset):
    def __init__(self, cache_dir: str | Path, splits: Iterable[str] = ("train",)):
        self.cache_dir = Path(cache_dir).resolve()
        wanted = set(splits)
        self.entries = [entry for entry in read_index(self.cache_dir) if entry["split"] in wanted]
        if not self.entries:
            raise ValueError(f"no cache entries for splits {sorted(wanted)} under {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def bucket(self, index: int) -> tuple[int, int]:
        value = self.entries[index]["bucket"]
        return int(value[0]), int(value[1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        latent_state = load_file(str(self.cache_dir / entry["latent_file"]), device="cpu")
        cond_state = load_file(str(self.cache_dir / entry["conditioning_file"]), device="cpu")
        return {
            "id": entry["id"],
            "latent": latent_state["latent"],
            "cond": cond_state["cond"],
            "cond_no_trigger": cond_state["cond_no_trigger"],
            "weight": float(entry["weight"]),
            "concept_type": entry["concept_type"],
            "split": entry["split"],
            "bucket": tuple(entry["bucket"]),
        }


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: CachedConceptDataset,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        counts: dict[tuple[int, int], int] = {}
        for idx in range(len(self.dataset)):
            bucket = self.dataset.bucket(idx)
            counts[bucket] = counts.get(bucket, 0) + 1
        if self.drop_last:
            return sum(count // self.batch_size for count in counts.values())
        return sum(math.ceil(count / self.batch_size) for count in counts.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        buckets: dict[tuple[int, int], list[int]] = {}
        for idx in range(len(self.dataset)):
            buckets.setdefault(self.dataset.bucket(idx), []).append(idx)
        batches: list[list[int]] = []
        for indices in buckets.values():
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def collate_cached(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets = {tuple(item["bucket"]) for item in batch}
    if len(buckets) != 1:
        raise ValueError(f"mixed latent buckets in one batch: {sorted(buckets)}")
    return {
        "ids": [item["id"] for item in batch],
        "latents": torch.stack([item["latent"] for item in batch]),
        "cond": torch.stack([item["cond"] for item in batch]),
        "cond_no_trigger": torch.stack([item["cond_no_trigger"] for item in batch]),
        "weights": torch.tensor([item["weight"] for item in batch], dtype=torch.float32),
        "concept_types": [item["concept_type"] for item in batch],
        "bucket": next(iter(buckets)),
    }

