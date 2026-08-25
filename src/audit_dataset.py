from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "mouse", 1: "cup"}
PHASH_NEAR_THRESHOLD = 4
ADJACENT_NEAR_THRESHOLD = 8
STEM_PATTERN = re.compile(r"^(mouse|cup)_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: Path, size: int = 32, low: int = 8) -> int:
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float64)
    indices = np.arange(size, dtype=np.float64)
    transform = np.cos(np.pi * (2 * indices[:, None] + 1) * indices[None, :] / (2 * size))
    coefficients = transform.T @ pixels @ transform
    block = coefficients[:low, :low]
    median = float(np.median(block.ravel()[1:]))
    bits = block.ravel() > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def validate_label(path: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    annotation_count = 0
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [f"空标签文件: {path}"], 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"字段数不是5: {path}:{line_number}")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"非数值标签: {path}:{line_number}")
            continue
        if class_id not in CLASS_NAMES:
            errors.append(f"未知类别ID {class_id}: {path}:{line_number}")
        values = (x_center, y_center, width, height)
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"坐标超出[0,1]: {path}:{line_number}")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"边界框宽高非正: {path}:{line_number}")
        epsilon = 1e-6
        if (
            x_center - width / 2 < -epsilon
            or x_center + width / 2 > 1 + epsilon
            or y_center - height / 2 < -epsilon
            or y_center + height / 2 > 1 + epsilon
        ):
            errors.append(f"边界框越过图像边界: {path}:{line_number}")
        annotation_count += 1
    return errors, annotation_count


def audit_dataset(dataset_root: Path, report_path: Path | None = None) -> dict:
    dataset_root = dataset_root.resolve()
    report_path = report_path or dataset_root.parents[1] / "docs" / "dataset_audit.json"
    records: list[dict] = []
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    duplicate_stems: list[str] = []
    label_errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_paths = sorted(
            path for path in image_dir.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_paths = sorted(label_dir.glob("*.txt"))
        image_by_stem: dict[str, Path] = {}
        for image_path in image_paths:
            if image_path.stem in image_by_stem:
                duplicate_stems.append(f"{split}:{image_path.stem}")
            image_by_stem[image_path.stem] = image_path
        label_by_stem = {path.stem: path for path in label_paths}
        missing_labels.extend(f"{split}:{stem}" for stem in sorted(set(image_by_stem) - set(label_by_stem)))
        orphan_labels.extend(f"{split}:{stem}" for stem in sorted(set(label_by_stem) - set(image_by_stem)))
        annotation_count = 0
        for stem in sorted(set(image_by_stem) & set(label_by_stem)):
            image_path = image_by_stem[stem]
            label_path = label_by_stem[stem]
            errors, annotations = validate_label(label_path)
            label_errors.extend(errors)
            annotation_count += annotations
            stem_match = STEM_PATTERN.match(stem)
            records.append(
                {
                    "split": split,
                    "stem": stem,
                    "image": image_path.relative_to(dataset_root).as_posix(),
                    "label": label_path.relative_to(dataset_root).as_posix(),
                    "sha256": sha256_file(image_path),
                    "phash": phash(image_path),
                    "sequence_class": stem_match.group(1) if stem_match else None,
                    "sequence_index": int(stem_match.group(2)) if stem_match else None,
                }
            )
        counts[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "annotations": annotation_count,
        }

    exact_groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        exact_groups[record["sha256"]].append(record)
    exact_duplicates: list[dict] = []
    for digest, members in exact_groups.items():
        if len({member["split"] for member in members}) > 1:
            exact_duplicates.append(
                {"sha256": digest, "members": [f"{member['split']}:{member['stem']}" for member in members]}
            )

    near_duplicates: list[dict] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = hamming(left["phash"], right["phash"])
            if distance <= PHASH_NEAR_THRESHOLD:
                near_duplicates.append(
                    {
                        "left": f"{left['split']}:{left['stem']}",
                        "right": f"{right['split']}:{right['stem']}",
                        "hamming": distance,
                    }
                )

    adjacent_near_duplicates: list[dict] = []
    sequence_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["sequence_class"] is not None:
            sequence_records[record["sequence_class"]].append(record)
    for class_name, members in sequence_records.items():
        members.sort(key=lambda item: item["sequence_index"])
        for left, right in zip(members, members[1:]):
            if right["sequence_index"] != left["sequence_index"] + 1 or left["split"] == right["split"]:
                continue
            distance = hamming(left["phash"], right["phash"])
            if distance <= ADJACENT_NEAR_THRESHOLD:
                adjacent_near_duplicates.append(
                    {
                        "class": class_name,
                        "left": f"{left['split']}:{left['stem']}",
                        "right": f"{right['split']}:{right['stem']}",
                        "hamming": distance,
                    }
                )

    status = "pass"
    if any(
        (
            missing_labels,
            orphan_labels,
            duplicate_stems,
            label_errors,
            exact_duplicates,
            near_duplicates,
            adjacent_near_duplicates,
        )
    ):
        status = "fail"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "status": status,
        "thresholds": {
            "phash_cross_split_hamming_max": PHASH_NEAR_THRESHOLD,
            "adjacent_cross_split_hamming_max": ADJACENT_NEAR_THRESHOLD,
        },
        "counts": counts,
        "pairing": {
            "missing_labels": missing_labels,
            "orphan_labels": orphan_labels,
            "duplicate_stems": duplicate_stems,
        },
        "label_errors": label_errors,
        "cross_split_exact_duplicates": exact_duplicates,
        "cross_split_near_duplicates": near_duplicates,
        "cross_split_adjacent_near_duplicates": adjacent_near_duplicates,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="检查YOLO数据集配对、标签和跨集合重复")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "object_detection",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = audit_dataset(args.dataset, args.report)
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
