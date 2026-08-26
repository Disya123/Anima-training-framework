import json

import torch
from safetensors.torch import save_file

from anima_trainer.data import BucketBatchSampler, CachedConceptDataset, choose_bucket, collate_cached


def _write_cache(root, entries):
    (root / "latents").mkdir(parents=True)
    (root / "conditioning").mkdir()
    lines = []
    for index, (sample_id, bucket) in enumerate(entries):
        latent_rel = f"latents/{sample_id}.safetensors"
        cond_rel = f"conditioning/{sample_id}.safetensors"
        save_file({"latent": torch.zeros(16, 1, bucket[1] // 8, bucket[0] // 8)}, str(root / latent_rel))
        save_file(
            {
                "cond": torch.zeros(512, 1024),
                "cond_no_trigger": torch.zeros(512, 1024),
            },
            str(root / cond_rel),
        )
        lines.append(
            json.dumps(
                {
                    "id": sample_id,
                    "split": "train",
                    "bucket": list(bucket),
                    "latent_file": latent_rel,
                    "conditioning_file": cond_rel,
                    "weight": index + 1,
                    "concept_type": "style",
                }
            )
        )
    (root / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_bucket_dimensions_are_dit_safe():
    width, height = choose_bucket(1000, 600, 768, (0.5, 1.0, 1.5, 2.0), 16)
    assert width % 16 == 0
    assert height % 16 == 0
    assert width > height


def test_batch_sampler_never_mixes_shapes(tmp_path):
    _write_cache(tmp_path, [("a", (768, 768)), ("b", (768, 768)), ("c", (1024, 576))])
    dataset = CachedConceptDataset(tmp_path)
    sampler = BucketBatchSampler(dataset, 2, shuffle=False, drop_last=False, seed=0)
    batches = list(sampler)
    assert sorted(len(batch) for batch in batches) == [1, 2]
    for batch in batches:
        assert len({dataset.bucket(index) for index in batch}) == 1
        collated = collate_cached([dataset[index] for index in batch])
        assert collated["latents"].shape[0] == len(batch)

