# 实验记录

## 2026-08-25 数据准备基线

- 目标：建立 mouse/bottle 两类别YOLO数据集并避免连续帧泄漏。
- 原始数量：mouse 120张、bottle 120张（bottle 原始目录沿用旧名 cup）。
- 类别：mouse=0、bottle=1；bottle 源标签由0重映射为1。
- 划分：每类 train 96张、val 24张；test等待独立采集。
- 固定种子：20260825。
- 分组：pHash<=4近重复组；相邻帧pHash<=8也合并为组。
- 基线模型配置：YOLO11n、epochs=100、imgsz=640、batch=16、patience=20。
- 训练状态：尚未开始。
- 验证状态：数据审计结果以 `dataset_audit.json` 为准。
- 人工审核：待队员复核类别、划分清单和新测试集。

## 2026-08-31 最终微调

- 配置：`configs/train_finetune_v2.yaml`，最多50轮，早停后完成45轮，最佳轮次为35。
- 验证集：Precision 0.981、Recall 0.972、mAP50 0.989、mAP50-95 0.767。
- 独立测试集：Precision 0.938、Recall 0.998、mAP50 0.986、mAP50-95 0.740。
- 权重 SHA-256：`c9a4194643a41d742b3feefe486f4b23179468650751f440c9696a95de5c9f0b`。
- 模型权重和完整扩充数据集仅保存在本地，未直接提交 GitHub。

后续每次训练至少记录：Git提交号、环境锁文件、模型、预训练权重、参数、随机种子、运行命令、硬件、耗时、最佳epoch、Precision、Recall、mAP50、mAP50-95及错误案例。
