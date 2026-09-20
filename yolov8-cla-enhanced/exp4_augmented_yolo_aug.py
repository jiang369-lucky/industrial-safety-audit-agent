# 实验4：增强后数据集 + YOLOv8自带增强

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
        name="exp4_augmented_yolo_aug",  # 实验4：增强数据 + YOLO自带增强
        device=0,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,
        augment=True,  # 启用YOLO自带增强
        # 降低增强强度（因为已经做过离线增强）
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.2,
        mosaic=0.2,  # 降低马赛克概率
        mixup=0.1,
        dropout=0.2,
        weight_decay=0.0005,
        patience=30,
        workers=2,
        cache=True,
        plots=True,
    )
    print("实验4完成：增强后数据集 + YOLOv8自带增强")