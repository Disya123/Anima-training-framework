from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

LEADING_TRIGGER_RE = re.compile(r"^\s*@?\s*TuriSasu(?:\s+style)?\s*,?\s*", re.IGNORECASE)


def strip_leading_trigger(caption: str, trigger: str) -> str:
    content = LEADING_TRIGGER_RE.sub("", caption)
    if trigger:
        content = content.replace(trigger, "").replace(trigger.lower(), "")
    return content


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an AnimaTrainer JSONL manifest from image/.txt pairs")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["style", "character", "object", "general"], required=True)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument(
        "--validation-count",
        type=int,
        default=0,
        help="hold out N random records into the validation split (enables in-train validation metrics)",
    )
    parser.add_argument("--validation-seed", type=int, default=7)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--augment-trigger",
        action="store_true",
        help="emit case/position trigger variants per image (prefix exact, prefix lowercase, suffix, middle); "
        "captions are written content-only and the trigger is re-inserted by the trainer",
    )
    parser.add_argument(
        "--augment-position",
        choices=["prefix", "suffix", "suffix_period", "middle"],
        default="suffix_period",
        help="trigger position used for the single variant emitted by --augment-trigger",
    )
    args = parser.parse_args()

    if args.mode != "general" and not args.trigger:
        parser.error("--trigger is required for style, character, and object manifests")
    if args.validation_count < 0:
        parser.error("--validation-count must be >= 0")
    root = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    pattern = "**/*" if args.recursive else "*"
    images = sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    variant_specs = (
        [(args.trigger.lower(), args.augment_position)]
        if args.augment_trigger
        else [(args.trigger, None)]
    )
    entries = []
    missing = []
    for index, image in enumerate(images, 1):
        caption_path = image.with_suffix(".txt")
        if not caption_path.is_file():
            missing.append(str(image))
            continue
        caption = caption_path.read_text(encoding="utf-8-sig").strip()
        content = strip_leading_trigger(caption, args.trigger) if args.trigger else caption
        content = content.strip(" ,").strip()
        while ", ," in content:
            content = content.replace(", ,", ",")
        for variant_index, (trigger_value, position_value) in enumerate(variant_specs, 1):
            entry = {
                "id": f"{image.stem}-v{variant_index:02d}-{index:05d}",
                "image": image.relative_to(output.parent).as_posix()
                if image.is_relative_to(output.parent)
                else str(image),
                "caption": content,
                "concept_type": args.mode,
                "weight": 1.0,
                "split": args.split,
            }
            if trigger_value:
                entry["trigger"] = trigger_value
            if position_value:
                entry["trigger_position"] = position_value
            entries.append(entry)
    if not entries:
        raise SystemExit(f"no image/.txt pairs found below {root}")
    if args.validation_count > 0:
        if args.split != "train":
            parser.error("--validation-count works together with --split train")
        if args.validation_count >= len(images):
            parser.error(f"--validation-count {args.validation_count} must be smaller than {len(images)} images")
        group_size = len(variant_specs)
        image_count = len(entries) // group_size
        rng = random.Random(args.validation_seed)
        held = set(rng.sample(range(image_count), args.validation_count))
        for image_index in held:
            for entry in entries[image_index * group_size : (image_index + 1) * group_size]:
                entry["split"] = "validation"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} records to {output}")
    held_out = sum(1 for entry in entries if entry["split"] == "validation")
    if held_out:
        print(f"held out {held_out} records into validation split")
    if missing:
        print(f"skipped {len(missing)} images without matching .txt captions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

