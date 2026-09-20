"""
特征提取脚本 - 全合规数据版（优化版）

功能说明：
从电焊作业视频中提取训练门控网络所需的三维特征，自动保存为CSV文件。

提取的特征包括：
1. mask_conf：YOLO检测到的面罩置信度（0-1之间，越高表示检测越可靠）
2. smoke_score：基于HSV颜色空间计算的烟雾浓度分数（0-1之间，越高表示烟雾越浓）
3. temporal_trend：时序趋势，表示面罩置信度的实时变化方向（正值表示置信度上升，负值表示下降）
4. label：标签，从.csv文件里面读取

优化点：
- 采样间隔改为每3帧处理一次，提高时序灵敏度
- 趋势计算改为相邻帧差值，反应更快速

使用场景：
- 您的视频中只有合规场景（工人正确佩戴面罩）
- 门控网络将通过这些特征学习“在不同环境下该信任哪个专家”

输出文件：
- 保存路径：gate_data/real_features.csv
- 文件格式：CSV，包含表头和4列数据

使用方法：
1. 修改下方的视频文件夹路径和YOLO模型路径
2. 直接运行本脚本
3. 等待处理完成，得到CSV文件
"""

import cv2
import numpy as np
import csv
import os
import glob
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# ================= 配置 =================
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
ANNOTATION_FILE = BASE_DIR / "experiments" / "data_preparation" / "resources" / "gt_annotations.csv"
YOLO_MODEL_PATH = REPO_ROOT / "runs" / "detect" / "runs" / "detect" / "yolo_weld_expert" / "weights" / "best.pt"
OUTPUT_CSV = BASE_DIR / "gate_data" / "real_features.csv"

FRAME_SKIP = 3
MASK_CLASS_ID = 2
ALPHA = 0.4  # EMA系数：0.4平衡响应与抗噪

FEATURE_COLUMNS = [
    "mask_conf",
    "smoke_score",
    "temporal_trend",
    "geometry_score",
    "collector_conf",
    "worker_conf",
    "brightness_norm",
    "spark_intensity",
    "conflict_k",
    "mask_area_ratio",
    "worker_mask_distance",
    "label",
]


def compute_geometry_score_local(worker_box, mask_box, mask_conf, frame_w, frame_h):
    if worker_box is None or mask_box is None:
        return 0.0
    wx1, wy1, wx2, wy2 = worker_box
    mx1, my1, mx2, my2 = mask_box
    worker_w = max(1.0, float(wx2 - wx1))
    worker_h = max(1.0, float(wy2 - wy1))
    mask_w = max(1.0, float(mx2 - mx1))
    mask_h = max(1.0, float(my2 - my1))

    inter_x1 = max(wx1, mx1)
    inter_y1 = max(wy1, my1)
    inter_x2 = min(wx2, mx2)
    inter_y2 = min(wy2, my2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    mask_area = mask_w * mask_h
    iou_like = inter_area / max(mask_area, 1.0)

    worker_cx = (wx1 + wx2) / 2.0
    head_cy = wy1 + 0.22 * worker_h
    mask_cx = (mx1 + mx2) / 2.0
    mask_cy = (my1 + my2) / 2.0
    x_score = np.exp(-2.0 * (((mask_cx - worker_cx) / worker_w) ** 2))
    y_score = np.exp(-2.0 * (((mask_cy - head_cy) / worker_h) ** 2))
    position_score = 0.5 * x_score + 0.5 * y_score

    w_ratio = mask_w / worker_w
    h_ratio = mask_h / worker_h
    size_score = 0.5 * np.exp(-5.0 * (w_ratio - 0.35) ** 2) + 0.5 * np.exp(-5.0 * (h_ratio - 0.35) ** 2)
    return float(0.4 * position_score + 0.3 * iou_like + 0.2 * size_score + 0.1 * np.clip(mask_conf, 0.0, 1.0))


def compute_relation_features(worker_box, mask_box):
    if worker_box is None or mask_box is None:
        return 0.0, 1.0
    wx1, wy1, wx2, wy2 = worker_box
    mx1, my1, mx2, my2 = mask_box
    worker_w = max(1.0, float(wx2 - wx1))
    worker_h = max(1.0, float(wy2 - wy1))
    mask_w = max(1.0, float(mx2 - mx1))
    mask_h = max(1.0, float(my2 - my1))
    area_ratio = np.clip((mask_w * mask_h) / (worker_w * worker_h), 0.0, 1.0)
    mask_cx = (mx1 + mx2) / 2.0
    mask_cy = (my1 + my2) / 2.0
    worker_cx = (wx1 + wx2) / 2.0
    head_cy = wy1 + 0.22 * worker_h
    distance = np.sqrt(((mask_cx - worker_cx) / worker_w) ** 2 + ((mask_cy - head_cy) / worker_h) ** 2)
    return float(area_ratio), float(np.clip(distance, 0.0, 1.0))


def compute_spark_intensity(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] > 220) & (hsv[:, :, 1] > 70)
    return float(np.clip(np.mean(bright.astype(float)) * 20.0, 0.0, 1.0))

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
print("加载标注与YOLO模型...")
gt_df = pd.read_csv(ANNOTATION_FILE)
yolo_model = YOLO(str(YOLO_MODEL_PATH))

video_annotations = {}
for _, row in gt_df.iterrows():
    vid = row['video_name']
    if vid not in video_annotations: video_annotations[vid] = []
    video_annotations[vid].append(
        {'start_sec': row['start_sec'], 'end_sec': row['end_sec'], 'label': int(row['violation_gt'])})


def get_label(video_name, sec):
    for ann in video_annotations.get(video_name, []):
        if ann['start_sec'] <= sec <= ann['end_sec']: return ann['label']
    return 0


total_frames = 0
video_count = 0

# 预统计各条件烟雾分位数，用于动态校准（保证单调性）
print("正在扫描视频计算烟雾分位数基准...")
smoke_stats = {"Clean": [], "Light": [], "Medium": [], "Heavy": []}
folder_map = {
    "Clean": os.path.join("clean", "welding"),
    "Light": os.path.join("light", "welding"),
    "Medium": os.path.join("medium", "welding"),
    "Heavy": os.path.join("heavy", "welding"),
}
base_dir = BASE_DIR / "test_video"

for cond, folder in folder_map.items():
    path = os.path.join(base_dir, folder)
    if not os.path.exists(path): continue
    for vid in os.listdir(path):
        if not vid.endswith('.mp4'): continue
        cap = cv2.VideoCapture(os.path.join(path, vid))
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            cnt += 1
            if cnt % 10 != 0: continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 160)
            smoke_stats[cond].append(np.mean(mask.astype(float)))
        cap.release()

# 计算全局校准参数
global_min = min(min(v) for v in smoke_stats.values() if v)
global_max = max(max(v) for v in smoke_stats.values() if v)
print(f"烟雾分数原始范围: {global_min:.3f} ~ {global_max:.3f}")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(FEATURE_COLUMNS)

    for cond, folder in folder_map.items():
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path): continue
        videos = sorted([v for v in os.listdir(path) if v.endswith('.mp4')])
        print(f"\n 提取 {cond} 特征 ({len(videos)} videos)")

        for vid_name in videos:
            cap = cv2.VideoCapture(os.path.join(path, vid_name))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            ema_conf, ema_smoke, prev_ema_conf = None, None, None
            f_idx, count = 0, 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                sec = f_idx / fps
                if f_idx % FRAME_SKIP != 0:
                    f_idx += 1
                    continue

                # 1. YOLO检测特征
                res = yolo_model(frame, verbose=False)
                raw_conf = 0.0
                worker_conf = 0.0
                collector_conf = 0.0
                worker_box = None
                mask_box = None
                for box in res[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                    if cls_id == 0 and conf > collector_conf:
                        collector_conf = conf
                    elif cls_id == 1 and conf > worker_conf:
                        worker_conf = conf
                        worker_box = xyxy
                    elif cls_id == MASK_CLASS_ID and conf > raw_conf:
                        raw_conf = conf
                        mask_box = xyxy

                # 2. 原始烟雾比例
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 160)
                raw_smoke_raw = np.mean(mask.astype(float))

                # 3. 动态线性校准到 0.05~0.95 (保证单调且避开Softmax饱和区)
                raw_smoke = np.clip((raw_smoke_raw - global_min) / (global_max - global_min + 1e-6), 0.05, 0.95)

                # 4. EMA平滑
                if ema_conf is None:
                    ema_conf, ema_smoke = raw_conf, raw_smoke
                    trend = 0.0
                else:
                    prev_ema_conf = ema_conf
                    ema_conf = ALPHA * raw_conf + (1 - ALPHA) * ema_conf
                    ema_smoke = ALPHA * raw_smoke + (1 - ALPHA) * ema_smoke
                    trend = ema_conf - prev_ema_conf

                frame_h, frame_w = frame.shape[:2]
                geometry_score = compute_geometry_score_local(worker_box, mask_box, raw_conf, frame_w, frame_h)
                mask_area_ratio, worker_mask_distance = compute_relation_features(worker_box, mask_box)
                brightness_norm = float(np.clip(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() / 255.0, 0.0, 1.0))
                spark_intensity = compute_spark_intensity(frame)
                conflict_k = float(np.clip(0.20 * raw_smoke + 0.60 * abs(trend) + 0.20 * max(0.0, 0.55 - geometry_score), 0.0, 1.0))
                label = get_label(vid_name, sec)
                writer.writerow([
                    f"{ema_conf:.4f}",
                    f"{ema_smoke:.4f}",
                    f"{trend:.4f}",
                    f"{geometry_score:.4f}",
                    f"{collector_conf:.4f}",
                    f"{worker_conf:.4f}",
                    f"{brightness_norm:.4f}",
                    f"{spark_intensity:.4f}",
                    f"{conflict_k:.4f}",
                    f"{mask_area_ratio:.4f}",
                    f"{worker_mask_distance:.4f}",
                    label,
                ])
                count += 1;
                f_idx += 1
            cap.release()
            print(f"  {vid_name} | {count} frames")
            total_frames += count;
            video_count += 1

print(f"\n 特征提取完成！共 {video_count} 视频, {total_frames} 帧 -> {OUTPUT_CSV}")

