# Jetson ROS2 YOLO 最终验证记录（2026-08-31）

## 验证对象

- 平台：NVIDIA Jetson，`aarch64`
- ROS2：Humble
- 功能包：`yolo_detector`
- 节点：`/yolo_detector`
- 话题：`/detection_result`
- 消息类型：`std_msgs/msg/String`
- 类别：`mouse=0`、`bottle=1`
- 模型 SHA-256：`c9a4194643a41d742b3feefe486f4b23179468650751f440c9696a95de5c9f0b`

本地与 Jetson 上的最终 `best.pt` 哈希一致。

## 最终视频测试

输入为本地最新的无旧检测框竖屏视频，使用 Jetson ROS2 节点完成推理、加框和保存。

- 输出：`videos/final_detection_mouse_bottle_20260831.mp4`
- 画面：540×960
- 帧数：495
- 视频帧率：24 FPS
- 时长：20.625 秒
- 输出 SHA-256：`b90e195f0feb53a172932ac92e6d0a82d26615d193574890d461209651695327`
- 板端话题消息中观测速度：20.24–21.88 FPS

视频结束后节点正常停止并关闭输出文件，生成视频可完整读取。

## ROS2 第二终端验证

```text
/yolo_detector
/detection_result
Type: std_msgs/msg/String
Publisher count: 1

data: bottle:0.58 | FPS:20.24
data: mouse:0.79 | FPS:21.01
data: mouse:0.77 | FPS:21.35
data: bottle:0.66 | FPS:21.65
data: bottle:0.90, mouse:0.66 | FPS:21.88
```

原始终端输出保存在 `docs/logs/`。

## 成功与典型错误案例

- 成功案例：`results/cases/success_frame_404.jpg`，同一帧正确识别黑色鼠标和黄绿色瓶子。
- 失败案例：`results/cases/failure_frame_135.jpg`，画面中白色鼠标清晰可见但没有检测框，属于漏检。

两者均来自同一次最终 `best.pt` 视频验证。失败案例不是一段完全识别错误的视频，而是模型正常运行中真实出现的代表性失败帧。

## 说明

本次要求是使用已连接的板子完成视频源推理和 ROS2 发布验证，因此不依赖摄像头设备枚举。最终视频同时显示类别、检测框、置信度与板端 FPS。
