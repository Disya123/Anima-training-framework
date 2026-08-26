from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an AnimaTrainer JSONL manifest from image/.txt pairs")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["style", "character", "object", "general"], required=True)
    parser.add_argument("--trigger", default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.mode != "general" and not args.trigger:
        parser.error("--trigger is required for style, character, and object manifests")
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} records to {output}")
    if missing:
        print(f"skipped {len(missing)} images without matching .txt captions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

