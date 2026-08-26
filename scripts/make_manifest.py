from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


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
    args = parser.parse_args()

    if args.mode != "general" and not args.trigger:
        parser.error("--trigger is required for style, character, and object manifests")
    if args.validation_count < 0:
        parser.error("--validation-count must be >= 0")
    root = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    pattern = "**/*" if args.recursive else "*"
    images = sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    entries = []
    missing = []
    for index, image in enumerate(images, 1):
        caption_path = image.with_suffix(".txt")
        if not caption_path.is_file():
            missing.append(str(image))
            continue
        caption = caption_path.read_text(encoding="utf-8-sig").strip()
        entry = {
            "id": f"{image.stem}-{index:05d}",
            "image": image.relative_to(output.parent).as_posix()
            if image.is_relative_to(output.parent)
            else str(image),
            "caption": caption,
            "concept_type": args.mode,
            "weight": 1.0,
            "split": args.split,
        }
        if args.trigger:
            entry["trigger"] = args.trigger
        entries.append(entry)
    if not entries:
        raise SystemExit(f"no image/.txt pairs found below {root}")
    if args.validation_count > 0:
        if args.split != "train":
            parser.error("--validation-count works together with --split train")
        if args.validation_count >= len(entries):
            parser.error(f"--validation-count {args.validation_count} must be smaller than {len(entries)} records")
        rng = random.Random(args.validation_seed)
        for entry in rng.sample(entries, args.validation_count):
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

