# 实验记录

## 2026-08-25 数据准备基线

- 目标：建立 mouse/cup 两类别YOLO数据集并避免连续帧泄漏。
- 原始数量：mouse 120张、cup 120张。
- 类别：mouse=0、cup=1；cup源标签由0重映射为1。
- 划分：每类 train 96张、val 24张；test等待独立采集。
- 固定种子：20260825。
- 分组：pHash<=4近重复组；相邻帧pHash<=8也合并为组。
- 基线模型配置：YOLO11n、epochs=100、imgsz=640、batch=16、patience=20。
- 训练状态：尚未开始。
- 验证状态：数据审计结果以 `dataset_audit.json` 为准。
- 人工审核：待队员复核类别、划分清单和新测试集。

后续每次训练至少记录：Git提交号、环境锁文件、模型、预训练权重、参数、随机种子、运行命令、硬件、耗时、最佳epoch、Precision、Recall、mAP50、mAP50-95及错误案例。
