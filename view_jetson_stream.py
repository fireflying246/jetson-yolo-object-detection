"""接收 Jetson 发送的长度前缀 JPEG 视频流并显示。"""

from __future__ import annotations

import argparse
import socket
import struct

import cv2
import numpy as np


HEADER_SIZE = 4
MAX_FRAME_BYTES = 20 * 1024 * 1024


def receive_exact(sock: socket.socket, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def view_stream(host: str, port: int) -> None:
    with socket.create_connection((host, port)) as sock:
        print(f"已连接 {host}:{port}，按 Q 退出。")
        while True:
            header = receive_exact(sock, HEADER_SIZE)
            if header is None:
                break

            frame_size = struct.unpack(">I", header)[0]
            if not 0 < frame_size <= MAX_FRAME_BYTES:
                raise ValueError(f"异常帧长度：{frame_size}")

            frame_data = receive_exact(sock, frame_size)
            if frame_data is None:
                break
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            cv2.imshow("Jetson YOLO Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="显示 Jetson 发送的 JPEG 视频流")
    parser.add_argument("--host", default="127.0.0.1", help="Jetson 地址或 SSH 转发地址")
    parser.add_argument("--port", type=int, default=5000, help="视频流端口")
    args = parser.parse_args()
    try:
        view_stream(args.host, args.port)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
