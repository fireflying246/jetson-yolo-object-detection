"""ROS 2 node for Jetson YOLO mouse/bottle detection."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


class YoloDetector(Node):
    """Run YOLO, publish results, and optionally record annotated video."""

    def __init__(self) -> None:
        super().__init__("yolo_detector")

        self.declare_parameter(
            "model_path", "/home/nvidia/object_detection/models/best.pt"
        )
        self.declare_parameter("source", "0")
        self.declare_parameter("video_path", "")
        self.declare_parameter("record_fps", 25.0)
        self.declare_parameter("record_seconds", 0.0)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("inference_confidence", 0.15)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("mouse_confidence", 0.25)
        self.declare_parameter("bottle_confidence", 0.25)

        self.model_path = str(self.get_parameter("model_path").value)
        source_value = str(self.get_parameter("source").value)
        self.source = int(source_value) if source_value.isdecimal() else source_value
        self.video_path = str(self.get_parameter("video_path").value)
        self.record_fps = float(self.get_parameter("record_fps").value)
        self.record_seconds = float(self.get_parameter("record_seconds").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.inference_confidence = float(
            self.get_parameter("inference_confidence").value
        )
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.class_thresholds = {
            "mouse": float(self.get_parameter("mouse_confidence").value),
            "bottle": float(self.get_parameter("bottle_confidence").value),
        }

        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"Model does not exist: {self.model_path}")
        if self.record_fps <= 0:
            raise ValueError("record_fps must be positive")

        self.publisher_ = self.create_publisher(String, "/detection_result", 10)
        self.model = YOLO(self.model_path)
        if set(self.model.names.values()) != {"mouse", "bottle"}:
            raise RuntimeError(
                f"Unexpected model classes: {self.model.names}; expected mouse/bottle"
            )

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.file_source = isinstance(self.source, str) and Path(self.source).is_file()

        self.video_writer = None
        self.video_frames = 0
        self.record_started_at = None
        self.next_video_time = None
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.stopping = False
        self.source_failures = 0
        self.timer = self.create_timer(0.001, self.detect_callback)

        self.get_logger().info("YOLO ROS2 detector started")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Model classes: {self.model.names}")
        self.get_logger().info(f"Source: {self.source}")
        self.get_logger().info("Publishing topic: /detection_result")
        if self.video_path:
            self.get_logger().info(f"Recording video: {self.video_path}")

    def _open_video_writer(self, frame) -> None:
        if not self.video_path or self.video_writer is not None:
            return
        output = Path(self.video_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self.video_writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.record_fps,
            (width, height),
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Cannot create video: {self.video_path}")
        self.record_started_at = time.perf_counter()
        self.next_video_time = self.record_started_at

    def _write_video_at_fixed_rate(self, frame, now: float) -> None:
        self._open_video_writer(frame)
        if self.video_writer is None or self.next_video_time is None:
            return
        if self.file_source:
            self.video_writer.write(frame)
            self.video_frames += 1
            return
        period = 1.0 / self.record_fps
        while now + 1e-9 >= self.next_video_time:
            self.video_writer.write(frame)
            self.video_frames += 1
            self.next_video_time += period

    def _publish(self, message: String) -> None:
        if self.stopping or not rclpy.ok(context=self.context):
            return
        try:
            self.publisher_.publish(message)
        except Exception:
            if rclpy.ok(context=self.context):
                raise

    def detect_callback(self) -> None:
        if self.stopping:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.source_failures += 1
            self.get_logger().warning("Failed to read source frame")
            if self.source_failures >= 10:
                self.request_stop("video source ended or became unavailable")
            return
        self.source_failures = 0

        result = self.model.predict(
            frame,
            device=0,
            imgsz=self.image_size,
            conf=self.inference_confidence,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.model.names[class_id]
                threshold = self.class_thresholds.get(
                    class_name, self.inference_confidence
                )
                if confidence < threshold:
                    continue
                detections.append(f"{class_name}:{confidence:.2f}")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        self.frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        message = String()
        payload = ", ".join(detections) if detections else "none"
        message.data = f"{payload} | FPS:{fps:.2f}"
        self._publish(message)
        self._write_video_at_fixed_rate(frame, now)

        if (
            self.record_seconds > 0
            and self.record_started_at is not None
            and now - self.record_started_at >= self.record_seconds
        ):
            self.request_stop(f"recording reached {self.record_seconds:.1f} seconds")

    def request_stop(self, reason: str) -> None:
        if self.stopping:
            return
        self.stopping = True
        self.get_logger().info(f"Stopping detector: {reason}")
        if self.timer is not None:
            self.timer.cancel()
        self._release_media()

    def _release_media(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.release()
            self.get_logger().info(
                f"Video closed: {self.video_path}; frames={self.video_frames}"
            )

    def destroy_node(self) -> None:
        self.stopping = True
        if self.timer is not None:
            self.timer.cancel()
        self._release_media()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetector()
        while rclpy.ok() and not node.stopping:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
