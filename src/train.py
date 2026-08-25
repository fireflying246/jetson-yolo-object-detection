from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_baseline.yaml"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_name = config.pop("model")
    config["data"] = str(PROJECT_ROOT / config["data"])
    config["project"] = str(PROJECT_ROOT / config["project"])
    YOLO(model_name).train(**config)


if __name__ == "__main__":
    main()
