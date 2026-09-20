# 查看各实验最高准确率

import os
import pandas as pd

base_dir = r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router"

experiments = [
    ("实验1", "exp1_raw_no_aug", "原始数据(278张), 无增强"),
    ("实验2", "exp2_raw_yolo_aug", "原始数据(278张), 在线增强"),
    ("实验3", "exp3_augmented_no_aug", "增强数据(826张), 无在线增强"),
    ("实验4", "exp4_augmented_yolo_aug", "增强数据(826张), 在线增强")
]

print("\n")
print("各实验最高准确率")
print("="*60)

for name, folder, desc in experiments:
    csv_path = os.path.join(base_dir, folder, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_acc = df['metrics/accuracy_top1'].max()
        best_epoch = df['metrics/accuracy_top1'].idxmax() + 1
        print(f"{name}: {desc}")
        print(f"  最高准确率: {best_acc:.3f} ({best_acc*100:.1f}%)")
    else:
        print(f"{name}: 未找到实验结果\n")