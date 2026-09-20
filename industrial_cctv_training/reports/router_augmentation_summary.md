# 场景分类 YOLOv8-cls 数据增强对比

最佳实验：`router_exp1_raw_no_aug`，最佳 Top-1 准确率为 **91.11%**。

| 排名 | 实验名 | 数据来源 | 离线增强 | YOLO自带增强 | 训练轮数 | 最佳轮次 | 最佳Top1/% | 最终Top1/% | 最佳Top5/% | 最小验证损失 | 最终验证损失 | 最终训练损失 | 权重路径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | router_exp1_raw_no_aug | 原始数据 | 否 | 否 | 196 | 26 | 91.11 | 88.89 | 100.0 | 0.3198 | 0.5305 | 0.0278 | D:\yolo\Spark-Gated MoE Framework\industrial_cctv_training\runs\classify\router\router_exp1_raw_no_aug\weights\best.pt |
| 2 | router_exp3_dataset_aug_no_aug | 数据集离线增强 | 是 | 否 | 70 | 50 | 91.11 | 88.89 | 100.0 | 0.4473 | 0.7356 | 0.0274 | D:\yolo\Spark-Gated MoE Framework\industrial_cctv_training\runs\classify\router\router_exp3_dataset_aug_no_aug\weights\best.pt |
| 3 | router_exp4_dataset_aug_yolo_aug | 数据集离线增强 | 是 | 是 | 72 | 40 | 88.89 | 84.44 | 100.0 | 0.4884 | 0.6778 | 0.0167 | D:\yolo\Spark-Gated MoE Framework\industrial_cctv_training\runs\classify\router\router_exp4_dataset_aug_yolo_aug\weights\best.pt |
| 4 | router_exp2_raw_yolo_aug | 原始数据 | 否 | 是 | 53 | 23 | 87.78 | 85.56 | 100.0 | 0.4779 | 1.0671 | 0.0171 | D:\yolo\Spark-Gated MoE Framework\industrial_cctv_training\runs\classify\router\router_exp2_raw_yolo_aug\weights\best.pt |

## 说明

- `原始数据` 指 `dataset_gate`。
- `数据集离线增强` 指 `dataset_gate_augmented`。
- `YOLO自带增强` 指训练时启用 YOLO 的在线增强参数。
- 推荐后续主系统优先尝试排名第一实验的 `weights/best.pt`。
