from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_baseline.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 YAML 配置训练 YOLO 模型")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_name = config.pop("model")
    local_model = PROJECT_ROOT / model_name
    if local_model.is_file():
        model_name = str(local_model)
    config["data"] = str(PROJECT_ROOT / config["data"])
    config["project"] = str(PROJECT_ROOT / config["project"])
    YOLO(model_name).train(**config)


if __name__ == "__main__":
    main()
