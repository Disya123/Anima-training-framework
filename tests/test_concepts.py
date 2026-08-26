import json

import pytest
from PIL import Image

from anima_trainer.concepts import audit_records, build_prompt, effective_weight
from anima_trainer.data import load_manifest


def test_trigger_and_content_are_separated():
    prompt, content, trigger = build_prompt(
        "1girl, black hair, cafe",
        mode="style",
        global_trigger="@kor_lili",
    )
    assert prompt == "@kor_lili, 1girl, black hair, cafe"
    assert content == "1girl, black hair, cafe"
    assert trigger == "@kor_lili"


def test_concept_requires_trigger():
    with pytest.raises(ValueError, match="require a trigger"):
        build_prompt("1girl", mode="character")


def test_hard_weights_multiply_once_per_unique_tag():
    assert effective_weight(2.0, ["dynamic", "dynamic", "overlap"], {"dynamic": 1.5, "overlap": 2}) == 6.0


def test_manifest_builds_prompt_and_weight(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (64, 96), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "one",
                "image": "sample.png",
                "caption": "1girl, sitting",
                "hard_tags": ["dynamic_pose"],
                "facets": {"scene": "cafe"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_manifest(
        manifest,
        mode="style",
        global_trigger="@style",
        hard_tag_weights={"dynamic_pose": 2.5},
    )
    assert records[0].prompt == "@style, 1girl, sitting"
    assert records[0].content_prompt == "1girl, sitting"
    assert records[0].weight == 2.5


def test_audit_reports_missing_factor_diversity():
    report = audit_records(
        [
            {"caption": "a", "trigger": "@alice", "facets": {"pose": "standing"}},
            {"caption": "b", "trigger": "@alice", "facets": {"pose": "standing"}},
        ],
        "character",
    )
    assert any("pose" in warning for warning in report["warnings"])

