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


# 项目路径与默认配置：默认从项目的 models 目录读取最终权重。
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


# 输入源解析：数字表示摄像头，其余字符串交给 Ultralytics 解析。
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
    # 模型加载：先检查权重是否存在，避免启动后才出现难以定位的错误。
    if not weights.is_file():
        raise FileNotFoundError(f"找不到模型权重：{weights}")

    model = YOLO(str(weights))

    # 推理配置：stream=True 让摄像头和视频按帧处理，避免一次占用大量内存。
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

    # 单张图片需要等待按键；视频、摄像头和网络流则持续刷新窗口。
    image_source = (
        isinstance(source, str)
        and Path(source).is_file()
        and Path(source).suffix.lower() in IMAGE_SUFFIXES
    )

    # 核心检测循环：取得检测结果、输出目标数量，并绘制检测框。
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
        # 无论正常结束还是用户中断，都释放 OpenCV 窗口资源。
        if display:
            cv2.destroyAllWindows()


# 命令行参数：集中管理模型、输入源和推理阈值，便于现场演示时调整。
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


# 程序入口：解析参数后调用核心检测函数。
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
