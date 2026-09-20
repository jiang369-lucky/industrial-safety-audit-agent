# industrial_cctv_training 执行命令

## 1. 进入项目根目录

```bat
cd /d "D:\yolo\Spark-Gated MoE Framework"
```

## 2. 覆盖工作区内 YOLO 训练数据为工业监控风格

默认处理：

- `industrial_cctv_training\dataset\dataset_gate`
- `industrial_cctv_training\dataset\dataset_gate_augmented`
- `industrial_cctv_training\dataset\dataset_detect`

不会处理 `dataset_cot`，因为它属于 Qwen/CoT 微调数据链路。

```bat
python -B industrial_cctv_training\tools\apply_industrial_cctv_style.py --profile paper --delete-cache
```

如果只想先检查图像数量，不覆盖文件：

```bat
python -B industrial_cctv_training\tools\apply_industrial_cctv_style.py --dry-run
```

## 3. 训练四组场景分类 YOLOv8-cls

四组含义：

1. `router_exp1_raw_no_aug`：原始 `dataset_gate`，不启用 YOLO 自带增强。
2. `router_exp2_raw_yolo_aug`：原始 `dataset_gate`，启用 YOLO 自带增强。
3. `router_exp3_dataset_aug_no_aug`：`dataset_gate_augmented`，不启用 YOLO 自带增强。
4. `router_exp4_dataset_aug_yolo_aug`：`dataset_gate_augmented`，启用 YOLO 自带增强。

一次训练四组：

```bat
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target router_all --device 0 --exist-ok
```

Windows 下如果训练初始化阶段长时间不动，优先保持默认 `workers=0` 和不缓存图片；不要额外加 `--router-cache`。

也可以分开训练：

```bat
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target router_raw_no_aug --device 0 --exist-ok
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target router_raw_yolo_aug --device 0 --exist-ok
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target router_dataset_aug_no_aug --device 0 --exist-ok
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target router_dataset_aug_yolo_aug --device 0 --exist-ok
```

## 4. 训练焊接/切割 YOLOv8 检测器

一次训练两个检测器：

```bat
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target detect_all --device 0 --exist-ok
```

分开训练：

```bat
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target weld --device 0 --exist-ok
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target cut --device 0 --exist-ok
```

## 5. 一次训练四组场景分类 + 两个检测器

```bat
python -B industrial_cctv_training\tools\train_yolov8_workspace.py --target all --device 0 --exist-ok
```

## 6. 训练结果位置

```text
industrial_cctv_training\runs\classify\router\router_exp1_raw_no_aug
industrial_cctv_training\runs\classify\router\router_exp2_raw_yolo_aug
industrial_cctv_training\runs\classify\router\router_exp3_dataset_aug_no_aug
industrial_cctv_training\runs\classify\router\router_exp4_dataset_aug_yolo_aug
industrial_cctv_training\runs\detect\yolo_weld_expert
industrial_cctv_training\runs\detect\yolo_cut_expert
```

后续主系统建议优先使用：

```text
industrial_cctv_training\runs\classify\router\router_exp4_dataset_aug_yolo_aug\weights\best.pt
industrial_cctv_training\runs\detect\yolo_weld_expert\weights\best.pt
industrial_cctv_training\runs\detect\yolo_cut_expert\weights\best.pt
```
