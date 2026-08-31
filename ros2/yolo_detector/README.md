# Jetson ROS2 YOLO 检测节点

该包在 NVIDIA Jetson 上运行 YOLO，并通过 ROS2 话题 `/detection_result`
发布检测类别、置信度和平均 FPS。当前模型类别为：

- `0: mouse`
- `1: bottle`

## 编译

```bash
cd ~/ros2_yolo_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo_detector
source install/setup.bash
```

## 摄像头运行

```bash
ros2 run yolo_detector yolo_node
```

默认使用摄像头 `0` 和模型：

```text
/home/nvidia/object_detection/models/best.pt
```

## 视频文件运行与保存

```bash
ros2 run yolo_detector yolo_node --ros-args \
  -p source:=/path/to/input.mp4 \
  -p video_path:=/path/to/output.mp4 \
  -p record_fps:=25.0 \
  -p record_seconds:=30.0
```

## 第二终端验证

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_yolo_ws/install/setup.bash
ros2 node list
ros2 topic list
ros2 topic info /detection_result
ros2 topic echo /detection_result
```

示例消息：

```text
data: bottle:0.85, mouse:0.71 | FPS:15.46
```

节点支持无图形界面的 SSH 环境，不调用 `cv2.imshow()`。视频保存按固定帧率封装，
停止时会先释放摄像头和视频写入器，避免 MP4 损坏或 ROS2 发布上下文异常。
