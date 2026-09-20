# 实验3：增强后数据集直接训练

from ultralytics import YOLO
import os

model = YOLO(r"D:\yolo\Spark-Gated MoE Framework\model\yolov8n-cls.pt")

if __name__ == '__main__':
    results = model.train(
        data=r"D:\yolo\Spark-Gated MoE Framework\dataset\dataset_gate_augmented",
        epochs=150,
        imgsz=224,
        batch=32,
        project=r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router",
        name="exp3_augmented_no_aug",  # 实验3：增强数据，无在线增强
        device=0,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,
        augment=False,  # 关闭在线增强
        dropout=0.2,
        weight_decay=0.0005,
        patience=30,
        workers=2,
        cache=True,
        plots=True,
    )
    print("实验3完成：增强后数据集直接训练")