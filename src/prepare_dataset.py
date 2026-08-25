from __future__ import annotations

import csv
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from audit_dataset import ADJACENT_NEAR_THRESHOLD, PHASH_NEAR_THRESHOLD, audit_dataset, phash, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "raw_images"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "object_detection"
SPLIT_SEED = 20260825
VAL_PER_CLASS = 24
SOURCE_CLASS_ID = 0
CLASS_CONFIG = {
    "mouse": {
        "class_id": 0,
        "images": RAW_ROOT / "mouse_yolo_annotated" / "images",
        "labels": RAW_ROOT / "mouse_yolo_annotated" / "labels",
    },
    "cup": {
        "class_id": 1,
        "images": RAW_ROOT / "cup_yolo_annotated" / "images",
        "labels": RAW_ROOT / "cup_yolo_annotated" / "labels",
    },
}
INDEX_PATTERN = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class Sample:
    class_name: str
    class_id: int
    index: int
    image: Path
    label: Path
    sha256: str
    phash: int


def validate_and_remap_label(path: Path, target_class_id: int) -> str:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"空标签文件: {path}")
    output: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"标签字段数不是5: {path}:{line_number}")
        if int(parts[0]) != SOURCE_CLASS_ID:
            raise ValueError(f"源标签类别不是0: {path}:{line_number}")
        coordinates = list(map(float, parts[1:]))
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"标签坐标超出[0,1]: {path}:{line_number}")
        x_center, y_center, width, height = coordinates
        if width <= 0 or height <= 0:
            raise ValueError(f"边界框宽高非正: {path}:{line_number}")
        epsilon = 1e-6
        if (
            x_center - width / 2 < -epsilon
            or x_center + width / 2 > 1 + epsilon
            or y_center - height / 2 < -epsilon
            or y_center + height / 2 > 1 + epsilon
        ):
            raise ValueError(f"边界框越过图像边界: {path}:{line_number}")
        output.append(" ".join([str(target_class_id), *parts[1:]]))
    return "\n".join(output) + "\n"


def load_samples(class_name: str, config: dict) -> list[Sample]:
    image_paths = sorted(config["images"].glob("*.jpg"))
    label_paths = sorted(config["labels"].glob("*.txt"))
    image_by_stem = {path.stem: path for path in image_paths}
    label_by_stem = {path.stem: path for path in label_paths}
    missing = sorted(set(image_by_stem) - set(label_by_stem))
    orphan = sorted(set(label_by_stem) - set(image_by_stem))
    if missing or orphan:
        raise ValueError(f"{class_name} 图片/标签不配对: missing={missing}, orphan={orphan}")
    samples: list[Sample] = []
    for stem in sorted(image_by_stem):
        match = INDEX_PATTERN.search(stem)
        if match is None:
            raise ValueError(f"无法解析序号: {stem}")
        label_path = label_by_stem[stem]
        validate_and_remap_label(label_path, config["class_id"])
        image_path = image_by_stem[stem]
        samples.append(
            Sample(
                class_name=class_name,
                class_id=config["class_id"],
                index=int(match.group(1)),
                image=image_path,
                label=label_path,
                sha256=sha256_file(image_path),
                phash=phash(image_path),
            )
        )
    return samples


def build_groups(samples: list[Sample]) -> list[list[Sample]]:
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if (samples[left].phash ^ samples[right].phash).bit_count() <= PHASH_NEAR_THRESHOLD:
                union(left, right)
    ordered = sorted(range(len(samples)), key=lambda index: samples[index].index)
    for left, right in zip(ordered, ordered[1:]):
        if samples[right].index != samples[left].index + 1:
            continue
        distance = (samples[left].phash ^ samples[right].phash).bit_count()
        if distance <= ADJACENT_NEAR_THRESHOLD:
            union(left, right)
    grouped: dict[int, list[Sample]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(find(index), []).append(sample)
    return sorted((sorted(group, key=lambda sample: sample.index) for group in grouped.values()), key=lambda group: group[0].index)


def select_validation_groups(groups: list[list[Sample]], seed: int) -> set[int]:
    randomizer = random.Random(seed)
    order = list(range(len(groups)))
    randomizer.shuffle(order)
    states: dict[tuple[int, int, int], tuple[int, ...]] = {(0, 0, 0): ()}
    for group_index in order:
        contribution = [0, 0, 0]
        for sample in groups[group_index]:
            segment = min((sample.index - 1) // 40, 2)
            contribution[segment] += 1
        updated = dict(states)
        for state, selected in states.items():
            next_state = tuple(state[i] + contribution[i] for i in range(3))
            if sum(next_state) <= VAL_PER_CLASS and next_state not in updated:
                updated[next_state] = (*selected, group_index)
        states = updated
    candidates = [(state, selected) for state, selected in states.items() if sum(state) == VAL_PER_CLASS]
    if not candidates:
        raise RuntimeError(f"无法以完整分组得到{VAL_PER_CLASS}张验证图片")
    target_per_segment = VAL_PER_CLASS // 3
    best_state, best_selected = min(
        candidates,
        key=lambda item: (
            sum(abs(value - target_per_segment) for value in item[0]),
            -len(item[1]),
            tuple(sorted(item[1])),
        ),
    )
    print(f"validation segment counts: {best_state}")
    return set(best_selected)


def ensure_empty_target() -> None:
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            directory = DATASET_ROOT / kind / split
            directory.mkdir(parents=True, exist_ok=True)
            existing = [path for path in directory.iterdir() if path.is_file()]
            if existing:
                raise RuntimeError(f"目标目录非空，停止以避免覆盖: {directory}")
    (DATASET_ROOT / "splits").mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_empty_target()
    rows: list[dict[str, str | int]] = []
    split_lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    summary: dict[str, dict] = {}

    for class_offset, (class_name, config) in enumerate(CLASS_CONFIG.items()):
        samples = load_samples(class_name, config)
        groups = build_groups(samples)
        validation_groups = select_validation_groups(groups, SPLIT_SEED + class_offset)
        class_counts = {"train": 0, "val": 0}
        val_group_names: list[str] = []
        for group_index, group in enumerate(groups):
            group_id = f"{class_name}_g{group_index:03d}"
            split = "val" if group_index in validation_groups else "train"
            if split == "val":
                val_group_names.append(group_id)
            for sample in group:
                destination_image = DATASET_ROOT / "images" / split / sample.image.name
                destination_label = DATASET_ROOT / "labels" / split / sample.label.name
                shutil.copy2(sample.image, destination_image)
                destination_label.write_text(
                    validate_and_remap_label(sample.label, sample.class_id), encoding="utf-8"
                )
                relative_image = destination_image.relative_to(DATASET_ROOT).as_posix()
                split_lists[split].append(relative_image)
                rows.append(
                    {
                        "split": split,
                        "class_name": sample.class_name,
                        "class_id": sample.class_id,
                        "sequence_index": sample.index,
                        "group_id": group_id,
                        "image": relative_image,
                        "label": destination_label.relative_to(DATASET_ROOT).as_posix(),
                        "source_image": sample.image.relative_to(PROJECT_ROOT).as_posix(),
                        "source_label": sample.label.relative_to(PROJECT_ROOT).as_posix(),
                        "sha256": sample.sha256,
                        "phash_hex": f"{sample.phash:016x}",
                    }
                )
                class_counts[split] += 1
        if class_counts != {"train": 96, "val": 24}:
            raise RuntimeError(f"{class_name} 划分数量异常: {class_counts}")
        summary[class_name] = {
            "class_id": config["class_id"],
            "counts": class_counts,
            "group_count": len(groups),
            "validation_groups": val_group_names,
        }

    split_dir = DATASET_ROOT / "splits"
    for split, paths in split_lists.items():
        (split_dir / f"{split}.txt").write_text(
            "\n".join(sorted(paths)) + ("\n" if paths else ""), encoding="utf-8"
        )
    with (split_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["split"]), str(row["class_name"]), int(row["sequence_index"]))))

    split_summary = {
        "seed": SPLIT_SEED,
        "strategy": "pHash<=4 connected groups plus adjacent-frame pHash<=8 groups; whole-group assignment",
        "test_policy": "empty; collect an independent recording session",
        "classes": summary,
    }
    (split_dir / "split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = audit_dataset(DATASET_ROOT, PROJECT_ROOT / "docs" / "dataset_audit.json")
    print(json.dumps({"split_summary": split_summary, "audit_status": report["status"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
