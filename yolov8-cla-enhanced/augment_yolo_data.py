import os
import cv2
import albumentations as A
from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil

# ==================== 配置 ====================
# 路径配置
BASE_DIR = r"D:\yolo\Spark-Gated MoE Framework\dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset_gate", "train")  # 训练原图
VAL_DIR = os.path.join(BASE_DIR, "dataset_gate", "val")  # 验证集
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_gate_augmented")  # 增强后的数据集（新版本）

# 增强倍数（保持2倍，避免过拟合）
AUG_TIMES = {
    'class_cutting_spark': 2,  # 切割 94 → 188 + 原图94 = 282
    'class_welding_spark': 2,  # 电焊 97 → 194 + 原图97 = 291
    'class_idle': 2,  # 空闲 87 → 174 + 原图87 = 261
}


# ==================== 增强后的数据增强策略 ====================
def get_augmentation_pipeline(class_name):
    """
    为不同类别返回不同的增强策略
    新增：RandomResizedCrop（模拟距离）+ Perspective（模拟视角）
    """

    if class_name == 'class_idle':
        # 空闲类：背景为主，主要做光照和模糊变化
        return A.Compose([
            # 新增：模拟不同距离（核心改进）
            A.RandomResizedCrop(
                height=224,
                width=224,
                scale=(0.3, 1.0),  # 30%-100%随机裁剪，模拟远近距离
                ratio=(0.8, 1.2),
                p=0.7
            ),

            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),

            # 新增：模拟不同视角
            A.Perspective(
                scale=(0.05, 0.1),  # 轻微透视变形
                keep_size=True,
                p=0.4
            ),

            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.HorizontalFlip(p=0.3),
            A.RandomScale(scale_limit=(-0.2, 0.2), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Resize(224, 224)
        ])

    elif class_name == 'class_welding_spark':
        # 电焊类：需要保持电弧的亮度和颜色特征
        return A.Compose([
            # 新增：模拟不同距离（核心改进）
            A.RandomResizedCrop(
                height=224,
                width=224,
                scale=(0.3, 1.0),  # 30%-100%随机裁剪
                ratio=(0.9, 1.1),
                p=0.7
            ),

            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                brightness_by_max=True,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-20, 20),
                val_shift_limit=(-20, 20),
                p=0.5
            ),

            # 新增：模拟不同视角
            A.Perspective(
                scale=(0.05, 0.1),
                keep_size=True,
                p=0.4
            ),

            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
            A.HorizontalFlip(p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.RandomScale(scale_limit=(-0.15, 0.15), p=0.3),
            A.Resize(224, 224)
        ])

    else:  # class_cutting_spark
        # 切割类：火花细长，需要保持形状特征
        return A.Compose([
            # 新增：模拟不同距离（核心改进）
            A.RandomResizedCrop(
                height=224,
                width=224,
                scale=(0.3, 1.0),  # 30%-100%随机裁剪
                ratio=(0.8, 1.2),
                p=0.7
            ),

            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.2, 0.2),
                p=0.7
            ),
            A.MotionBlur(blur_limit=(5, 9), p=0.4),

            # 新增：模拟不同视角
            A.Perspective(
                scale=(0.05, 0.1),
                keep_size=True,
                p=0.4
            ),

            A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.HorizontalFlip(p=0.3),
            A.RandomScale(scale_limit=(-0.2, 0.2), p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Resize(224, 224)
        ])


def augment_class_images(class_path, output_class_path, aug_times):
    """
    增强单个类别的图片
    """
    # 获取所有图片
    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"  找到 {len(images)} 张原始图片，将生成 {aug_times} 倍增强")

    # 获取该类别的增强pipeline
    class_name = os.path.basename(class_path)
    aug_pipeline = get_augmentation_pipeline(class_name)

    # 复制原始图片到输出目录
    for img_name in images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(output_class_path, f"original_{img_name}")
        shutil.copy2(src, dst)

    # 生成增强图片
    for img_name in tqdm(images, desc=f"增强 {class_name}"):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取图片 {img_path}，跳过")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        name_without_ext = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]

        # 生成指定数量的增强图片
        for i in range(aug_times):
            try:
                augmented = aug_pipeline(image=img)
                aug_img = augmented['image']

                # 保存增强图片
                aug_img_name = f"{name_without_ext}_aug_{i}{ext}"
                aug_img_path = os.path.join(output_class_path, aug_img_name)

                # 转换回BGR保存
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_img_path, aug_img_bgr)
            except Exception as e:
                print(f"  增强图片 {img_name} 时出错: {e}")
                continue


def create_augmented_dataset():
    """
    创建增强后的数据集
    """
    print("=" * 50)
    print("开始创建增强数据集 (v2版本 - 添加距离+视角模拟)")
    print("=" * 50)

    # 创建输出目录
    train_output = os.path.join(OUTPUT_DIR, "train")
    val_output = os.path.join(OUTPUT_DIR, "val")

    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    # 1. 处理验证集（只复制，不增强）
    print("\n处理验证集（只复制，不增强）...")
    val_classes = ['class_cutting_spark', 'class_welding_spark', 'class_idle']
    for class_name in val_classes:
        val_class_path = os.path.join(VAL_DIR, class_name)
        if os.path.exists(val_class_path):
            output_class_path = os.path.join(val_output, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            # 复制验证集图片
            val_images = [f for f in os.listdir(val_class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in val_images:
                src = os.path.join(val_class_path, img_name)
                dst = os.path.join(output_class_path, img_name)
                shutil.copy2(src, dst)

            print(f"  验证集 {class_name}: {len(val_images)} 张")

    # 2. 处理训练集（增强）
    print("\n处理训练集（进行数据增强）...")
    for class_name, aug_times in AUG_TIMES.items():
        train_class_path = os.path.join(TRAIN_DIR, class_name)
        if os.path.exists(train_class_path):
            output_class_path = os.path.join(train_output, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            augment_class_images(train_class_path, output_class_path, aug_times)

            # 统计增强后的图片数量
            final_count = len([f for f in os.listdir(output_class_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  增强完成 {class_name}: 共 {final_count} 张")

    print("\n" + "=" * 50)
    print("数据集增强完成！")
    print(f"增强后训练集: {train_output}")
    print(f"验证集: {val_output}")
    print("=" * 50)

    # 打印最终统计
    print("\n最终数据集统计：")
    for class_name in AUG_TIMES.keys():
        train_class = os.path.join(train_output, class_name)
        val_class = os.path.join(val_output, class_name)

        train_count = len([f for f in os.listdir(train_class)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_count = len([f for f in os.listdir(val_class)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"  {class_name}: 训练集 {train_count}张, 验证集 {val_count}张")


if __name__ == "__main__":
    create_augmented_dataset()