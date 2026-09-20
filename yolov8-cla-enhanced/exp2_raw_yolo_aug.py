# 实验2：原始数据集 + 优化的YOLOv8自带增强

from ultralytics import YOLO

model = YOLO(r"D:\yolo\Spark-Gated MoE Framework\model\yolov8n-cls.pt")

if __name__ == '__main__':
    results = model.train(
        data=r"D:\yolo\Spark-Gated MoE Framework\dataset\dataset_gate",
        epochs=150,
        imgsz=224,
        batch=32,
        project=r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router",
        name="exp2_raw_yolo_aug",  # 实验2：原始数据 + 优化增强
        device=0,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,
        augment=True,

        # ========== 优化的增强参数 ==========
        # 颜色增强（温和，保护电焊颜色特征）
        hsv_h=0.015,  # 色调变化 (保持默认)
        hsv_s=0.5,  # 饱和度变化 (从0.7降低到0.5，避免电焊颜色失真)
        hsv_v=0.3,  # 明度变化 (从0.4降低到0.3，避免过曝/过暗)

        # 几何变换（适度，帮助模型学习不同角度）
        degrees=15.0,  # 旋转角度 (从0增加到15度，学习不同角度的火花)
        translate=0.1,  # 平移 (保持默认)
        scale=0.3,  # 缩放 (从0.5降低到0.3，保持火花形状)
        shear=2.0,  # 剪切 (从0增加到2，增加一点形变鲁棒性)
        perspective=0.0,  # 透视 (保持关闭)
        flipud=0.0,  # 垂直翻转 (保持关闭，火花有方向性)
        fliplr=0.3,  # 水平翻转 (从0.5降低到0.3)

        # 高级增强
        mosaic=0.2,  # 马赛克增强 (从1.0降低到0.2，避免破坏火花特征)
        mixup=0.1,  # Mixup增强 (从0增加到0.1，增加多样性)
        copy_paste=0.1,  # 复制粘贴 (从0增加到0.1，对小数据集有帮助)

        # 正则化
        dropout=0.2,
        weight_decay=0.0005,

        patience=30,
        workers=2,
        cache=True,
        plots=True,
    )
    print("实验2完成：原始数据集 + 优化的YOLOv8自带增强")