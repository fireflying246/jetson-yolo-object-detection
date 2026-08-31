# Mouse & Bottle YOLO Object Detection

本项目完成 `mouse=0`、`bottle=1` 两类别目标检测模型的训练、测试和 NVIDIA Jetson ROS2 部署。旧数据文件名中可能含 `cup`，但类别 ID 1 已统一解释为 `bottle`。

## 最小交付内容

| 作业要求 | 仓库中的材料 |
|---|---|
| 数据集与标注 | `dataset/dataset_object_detection_final_20260831.zip` |
| 训练代码与配置 | `src/train.py`、`configs/train_finetune_v2.yaml` |
| 核心检测程序 | `detection.py` |
| 最终模型 | `models/best.pt` |
| Jetson ROS2 程序 | `ros2/yolo_detector/` |
| 最终结果视频 | `videos/final_detection_mouse_bottle_20260831.mp4` |
| ROS2 终端验证 | `docs/jetson_ros2_validation_20260831.md`、`docs/logs/` |
| 成功和失败案例 | `results/cases/` |
| 测试指标 | `docs/final_training_summary.json` |

模型、最终视频和数据集压缩包通过 Git LFS 保存；`runs/`、缓存、原始采集压缩包及重复中间结果不上传。

## 数据集

GitHub 原有未压缩基线数据已经与后续扩充数据合并到一个压缩包中，仓库不再重复存放散列图片和标签。

| split | 图片 | 标签 |
|---|---:|---:|
| train | 1082 | 1082 |
| val | 131 | 131 |
| test | 37 | 37 |

解压后保持 `object_detection/` 目录结构，其中包含 `images/`、`labels/`、`splits/` 和 `data.yaml`。压缩包 SHA-256 与文件清单见 `dataset/` 中的 manifest 和 checksum 文件。

## 环境

最后一次训练使用 Python 3.10.20 和 Ultralytics 8.4.127。Jetson 上的 PyTorch/CUDA 应安装与 JetPack 匹配的 NVIDIA 版本。

```bash
python3.10 -m pip install -r requirements.txt
```

## 运行检测

摄像头检测：

```bash
python3.10 detection.py --weights models/best.pt --source 0 --device 0
```

视频检测并保存加框结果：

```bash
python3.10 detection.py \
  --weights models/best.pt \
  --source input.mp4 \
  --save --no-display
```

`--source` 也可以填写图片或网络流地址；按 `Q` 退出显示窗口。

## 训练与评价

```bash
python3.10 src/train.py --config configs/train_finetune_v2.yaml
python3.10 src/evaluate.py --weights models/best.pt --split val
python3.10 src/evaluate.py --weights models/best.pt --split test
```

最终独立测试结果：precision 0.938、recall 0.998、mAP50 0.986、mAP50-95 0.740。模型 SHA-256 为 `c9a4194643a41d742b3feefe486f4b23179468650751f440c9696a95de5c9f0b`。

## Jetson ROS2 运行

```bash
cd ~/ros2_yolo_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo_detector
source install/setup.bash
ros2 run yolo_detector yolo_node
```

节点通过 `/detection_result` 发布类别、置信度和 FPS。第二终端验证：

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_yolo_ws/install/setup.bash
ros2 topic info /detection_result
ros2 topic echo /detection_result
```

板端最终视频测试为 20.625 秒、495 帧、24 FPS；推理消息中观测到 20.24–21.88 FPS。

## 成功和典型错误案例

`results/cases/success_frame_404.jpg` 展示同一帧中正确识别黑色鼠标和黄绿色瓶子；`failure_frame_135.jpg` 展示同一次最终视频验证中白色鼠标漏检。典型错误案例来自最终 `best.pt` 的真实验证过程，不使用一段完全识别错误的视频代替。
