# Mouse & Cup YOLO Object Detection

本项目用于训练一个两类别目标检测模型：`mouse=0`、`cup=1`。当前已有数据来自两段连续拍摄序列，因此按近重复组而不是逐张随机划分，以降低相邻帧泄漏。

## 数据状态

| split | mouse | cup | 总数 |
|---|---:|---:|---:|
| train | 96 | 96 | 192 |
| val | 24 | 24 | 48 |
| test | 10 | 10 | 20 |

`test` 已由独立拍摄会话补充，共20张（mouse 10张、cup 10张）。该集合只用于最终评估，训练和调参期间不要使用 test 结果。

## 目录

```text
dataset/object_detection/
├─ images/{train,val,test}
├─ labels/{train,val,test}
├─ splits/{train.txt,val.txt,test.txt,manifest.csv,split_summary.json}
└─ data.yaml
src/
├─ prepare_dataset.py
├─ audit_dataset.py
├─ train.py
└─ evaluate.py
configs/train_baseline.yaml
docs/dataset_card.md
docs/experiment_log.md
docs/dataset_audit.json
```

原始 `raw_images` 只读保留。GitHub 默认提交整理后的 `dataset`、标签、划分清单、脚本和实验记录，不重复提交 `raw_images`。

## 划分方法

- 固定种子：`20260825`。
- 对每个类别计算64位感知哈希（pHash）。
- pHash 汉明距离不大于4的图片合并为同一近重复组。
- 序号相邻且 pHash 汉明距离不大于8的连续帧也合并为同组。
- 每个分组整体进入 train 或 val；验证集在序列前、中、后三段尽量均衡抽取。
- `splits/manifest.csv` 保存每张图片的来源、类别、分组、SHA-256 和 pHash。

## 环境

- 推荐 Python 3.12。
- 数据准备脚本已按 `numpy==2.3.5`、`Pillow==12.3.0` 编写。
- 训练基线固定 `ultralytics==8.4.115`。
- PyTorch/CUDA 应根据训练机器显卡和官方安装说明单独安装；安装后将实际环境保存到 `docs/environment-lock.txt`。

```powershell
cd 'D:\object detection'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze | Set-Content -Encoding utf8 docs\environment-lock.txt
```

## 数据复核

重新生成数据集前，目标 `dataset/object_detection/images` 和 `labels` 目录必须为空，脚本不会自动删除已有数据。

```powershell
python src\prepare_dataset.py
python src\audit_dataset.py
```

审计检查：图片/同名标签配对、YOLO坐标合法性、跨集合 SHA-256 完全重复、跨集合 pHash 近重复以及相邻连续帧泄漏。结果写入 `docs/dataset_audit.json`。

## 基线训练

默认参数位于 `configs/train_baseline.yaml`：YOLO11n、100 epochs、640像素、batch 16、patience 20、固定种子并启用确定性训练。

```powershell
python src\train.py
```

显存不足时只调整 `batch`，并在实验记录中注明。不要根据 test 指标调整训练参数。

## 评价

```powershell
python src\evaluate.py --weights results\yolo11n_baseline\weights\best.pt --split val
```

完成训练与调参并通过数据审计后，才运行：

```powershell
python src\evaluate.py --weights results\yolo11n_baseline\weights\best.pt --split test
```

## GitHub 初始化

```powershell
git init
git add .
git commit -m "Prepare grouped YOLO dataset and baseline configuration"
```

首次训练前提交数据划分和配置；训练参数有变化时单独提交。模型权重通常不直接进入 Git，可使用 GitHub Releases 或 Git LFS 保存选定权重。
