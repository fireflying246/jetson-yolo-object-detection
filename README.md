# Mouse & Bottle YOLO Object Detection

本项目用于训练和部署两类别 YOLO 目标检测模型：`mouse=0`、`bottle=1`。旧数据文件名中可能仍含 `cup`，但类别 ID 1 的当前统一名称以 `data.yaml` 为准，均解释为 `bottle`。

## GitHub 内容范围

仓库保留可供检查和演示的基线数据集，以及训练、评价、审计和检测代码。完整扩充数据集、原始图片、视频、训练缓存和模型权重保留在本地，不直接提交到 Git。

| 内容 | GitHub | 说明 |
|---|---|---|
| 核心代码、配置、依赖和文档 | 是 | 用于助教检查和流程复现 |
| 基线数据集 | 是 | train 192、val 48、test 20 |
| 完整扩充数据集 | 否 | 本地图片约 204 MB |
| `best.pt` 模型权重 | 否 | 使用 Git LFS 或 Release 单独发布 |
| `runs/`、视频、缓存和原始压缩包 | 否 | 均为生成物或原始素材 |

基线数据集的类别数量如下：

| split | mouse | bottle | 总数 |
|---|---:|---:|---:|
| train | 96 | 96 | 192 |
| val | 24 | 24 | 48 |
| test | 10 | 10 | 20 |

`test` 来自独立拍摄会话，只用于最终评价，不应根据其结果调整训练参数。

## 主要文件

```text
detection.py                         # 图片、视频、摄像头和网络流检测
view_jetson_stream.py                # 接收并显示 Jetson 发送的 JPEG 视频流
src/train.py                         # 训练入口
src/evaluate.py                      # val/test 评价入口
src/prepare_dataset.py               # 基线数据准备
src/audit_dataset.py                 # 标签、重复和数据泄漏审计
configs/train_baseline.yaml          # 基线训练配置
configs/train_finetune_v2.yaml       # 最终微调配置
dataset/object_detection/data.yaml   # YOLO 数据集定义
docs/final_training_summary.json     # 最终训练指标摘要
```

## 环境

最后一次训练使用 Python 3.10.20、Ultralytics 8.4.127、NumPy 2.2.6、Pillow 12.3.0 和 PyYAML 6.0.3。PyTorch/CUDA 应根据运行平台单独安装；Jetson 应使用与 JetPack 匹配的 NVIDIA PyTorch 包，不要直接照搬 Windows CUDA wheel。

```bash
python3.10 -m pip install -r requirements.txt
```

## 运行检测

先将训练好的权重放到 `models/best.pt`，然后运行：

```bash
python3.10 detection.py --weights models/best.pt --source 0 --device 0
```

`--source` 可填写摄像头编号、图片、视频或网络流地址；添加 `--save` 可保存带检测框的结果，按 `Q` 退出显示窗口。

接收 Jetson 已编码的视频流时可运行：

```bash
python3.10 view_jetson_stream.py --host 127.0.0.1 --port 5000
```

## 训练与评价

```bash
python3.10 src/train.py --config configs/train_finetune_v2.yaml
python3.10 src/evaluate.py --weights models/best.pt --split val
python3.10 src/evaluate.py --weights models/best.pt --split test
```

数据复核命令：

```bash
python3.10 src/audit_dataset.py
```

审计包含图片/标签配对、YOLO 坐标合法性、跨集合完全重复、近重复和相邻连续帧泄漏检查。模型权重不直接进入 Git；确需从 GitHub 下载时，应通过 Git LFS 或 GitHub Release 发布并记录 SHA-256。
