from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="评价训练后的目标检测模型")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    weights = args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights
    YOLO(str(weights)).val(
        data=str(PROJECT_ROOT / "dataset" / "object_detection" / "data.yaml"),
        split=args.split,
        project=str(PROJECT_ROOT / "results"),
        name=f"evaluation_{args.split}",
    )


if __name__ == "__main__":
    main()
