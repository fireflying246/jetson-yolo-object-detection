"""YOLO 目标检测核心程序。

示例：
    python detection.py --source 0
    python detection.py --source test.jpg
    python detection.py --source test.mp4 --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_source(value: str) -> int | str:
    """把纯数字来源解释为摄像头编号，其余内容按路径或网络流处理。"""
    return int(value) if value.isdecimal() else value


def run_detection(
    weights: Path,
    source: int | str,
    confidence: float,
    iou: float,
    image_size: int,
    device: str | None,
    display: bool,
    save: bool,
) -> None:
    """加载 YOLO 权重，逐帧推理并显示或保存带检测框的结果。"""
    if not weights.is_file():
        raise FileNotFoundError(f"找不到模型权重：{weights}")

    model = YOLO(str(weights))
    predict_options = {
        "source": source,
        "stream": True,
        "conf": confidence,
        "iou": iou,
        "imgsz": image_size,
        "save": save,
        "project": str(PROJECT_ROOT / "runs" / "detect"),
        "name": "prediction",
        "exist_ok": True,
        "verbose": False,
    }
    if device:
        predict_options["device"] = device

    image_source = (
        isinstance(source, str)
        and Path(source).is_file()
        and Path(source).suffix.lower() in IMAGE_SUFFIXES
    )

    try:
        for frame_index, result in enumerate(model.predict(**predict_options), start=1):
            boxes = result.boxes
            detected = 0 if boxes is None else len(boxes)
            print(f"frame={frame_index} detections={detected} {result.verbose().strip()}")

            if display:
                cv2.imshow("YOLO Object Detection - press Q to quit", result.plot())
                delay = 0 if image_source else 1
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("检测已由用户停止。")
    finally:
        if display:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用训练好的 YOLO 模型进行目标检测")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"模型权重路径（默认：{DEFAULT_WEIGHTS}）",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="摄像头编号、图片/视频路径或网络流地址（默认：0）",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument(
        "--device",
        default=None,
        help="运行设备，例如 0、cpu；不填写时由 Ultralytics 自动选择",
    )
    parser.add_argument("--no-display", action="store_true", help="不打开结果窗口")
    parser.add_argument("--save", action="store_true", help="保存带检测框的结果")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_detection(
        weights=args.weights.expanduser().resolve(),
        source=parse_source(args.source),
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
        display=not args.no_display,
        save=args.save,
    )


if __name__ == "__main__":
    main()
