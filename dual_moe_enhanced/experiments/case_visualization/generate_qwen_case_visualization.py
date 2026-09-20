# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent
LOCAL_ULTRALYTICS_DIR = EXPERIMENTS_DIR / ".ultralytics"
LOCAL_ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_ULTRALYTICS_DIR))
os.environ.setdefault("YOLO_VERBOSE", "False")

from ultralytics import YOLO

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from common.experiment_paths import (  # noqa: E402
    CASE_VISUALIZATION_DIR,
    PAPER_FIGURE_DIR,
    TEST_VIDEO_DIR,
    WELDING_ANNOTATION_CSV,
    ensure_dir,
)
import dual_moe_safety_system as sys_mod  # noqa: E402


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_LABELS = {0: "收集器", 1: "工人", 2: "面罩"}
CLASS_COLORS = {0: (38, 160, 218), 1: (40, 180, 90), 2: (240, 140, 30)}
CASE_ORDER = ["CLEAR_PASS", "AMBIGUOUS_MID", "CLEAR_FAIL"]
CASE_TITLES = {
    "CLEAR_PASS": "高置信视觉判定",
    "AMBIGUOUS_MID": "Qwen惰性复核触发",
    "CLEAR_FAIL": "低置信直接告警",
}


@dataclass
class CaseFrame:
    case_type: str
    video: str
    frame_idx: int
    time_sec: float
    gt_violation: int
    reason_gt: str
    frame: np.ndarray
    detections: list[dict[str, object]]
    worker_box: list[int] | None
    mask_box: list[int] | None
    collector_box: list[int] | None
    geometry_score: float
    mask_conf: float
    collector_conf: float
    trigger: bool
    trigger_tag: str
    trigger_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Qwen 惰性复核典型案例中文可视化图。")
    parser.add_argument("--condition", default="clean", choices=["clean", "light", "medium", "heavy"])
    parser.add_argument("--frame-stride", type=int, default=8, help="扫描候选帧的步长。")
    parser.add_argument("--max-videos", type=int, default=0, help="最多扫描多少个焊接视频，0 表示不限制。")
    parser.add_argument("--device", default="0", help="YOLO推理设备，如 0 或 cpu。")
    return parser.parse_args()


def read_frame(video_path: Path, frame_idx: int) -> tuple[np.ndarray | None, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), fps


def collect_active_segments(condition: str) -> pd.DataFrame:
    if not WELDING_ANNOTATION_CSV.exists():
        raise FileNotFoundError(f"找不到焊接标注文件: {WELDING_ANNOTATION_CSV}")
    df = pd.read_csv(WELDING_ANNOTATION_CSV)
    df = df[(df["scene_gt"].astype(str).str.lower() == "welding") & (df["scope"].astype(str).str.lower() == "active")]
    df = df.copy()
    df["video_path"] = df["video_name"].apply(lambda name: TEST_VIDEO_DIR / condition / "welding" / str(name))
    return df[df["video_path"].apply(lambda p: Path(p).exists())]


def pick_boxes(detections: list[dict[str, object]]) -> tuple[list[int] | None, list[int] | None, list[int] | None, float, float]:
    worker_box = mask_box = collector_box = None
    worker_conf = mask_conf = collector_conf = 0.0
    for det in detections:
        cls = int(det["cls"])
        conf = float(det["conf"])
        box = [int(v) for v in det["box"]]
        if cls == 1 and conf > worker_conf:
            worker_conf = conf
            worker_box = box
        elif cls == 2 and conf > mask_conf:
            mask_conf = conf
            mask_box = box
        elif cls == 0 and conf > collector_conf:
            collector_conf = conf
            collector_box = box
    return worker_box, mask_box, collector_box, mask_conf, collector_conf


def classify_case(trigger: bool, tag: str) -> str | None:
    if trigger:
        return "AMBIGUOUS_MID"
    if tag == "CLEAR_PASS":
        return "CLEAR_PASS"
    if tag == "CLEAR_FAIL":
        return "CLEAR_FAIL"
    return None


def scan_cases(args: argparse.Namespace) -> list[CaseFrame]:
    detector = YOLO(sys_mod.CONFIG["weld_detector"])
    segments = collect_active_segments(args.condition)
    if args.max_videos > 0:
        keep = set(segments["video_name"].drop_duplicates().head(args.max_videos))
        segments = segments[segments["video_name"].isin(keep)]

    best: dict[str, CaseFrame] = {}
    score_rank = {"CLEAR_PASS": -1.0, "AMBIGUOUS_MID": -1.0, "CLEAR_FAIL": -1.0}

    for _, row in segments.iterrows():
        video_path = Path(row["video_path"])
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        start_frame = int(float(row["start_sec"]) * fps)
        end_frame = int(float(row["end_sec"]) * fps)
        for frame_idx in range(start_frame, max(start_frame + 1, end_frame), max(1, args.frame_stride)):
            frame, _ = read_frame(video_path, frame_idx)
            if frame is None:
                continue

            detections = sys_mod.detect_single_scale(frame, detector, sys_mod.CONFIG["conf_thres"])
            worker_box, mask_box, collector_box, mask_conf, collector_conf = pick_boxes(detections)
            if worker_box is None:
                continue

            h, w = frame.shape[:2]
            geometry_score = sys_mod.compute_geometry_score(worker_box, mask_box, mask_conf, w, h)
            trigger, tag, reason = sys_mod.should_trigger_qwen_by_geometry(
                geometry_score=geometry_score,
                mask_conf=mask_conf,
                has_smoke=False,
                smoke_score=0.0,
                conflict_k=0.0,
            )
            case_type = classify_case(trigger, tag)
            if case_type is None:
                continue

            gt_violation = int(row.get("violation_gt", 0))
            ranking_score = {
                "CLEAR_PASS": geometry_score if gt_violation == 0 else geometry_score - 0.5,
                "AMBIGUOUS_MID": 1.0 - abs(geometry_score - 0.50),
                "CLEAR_FAIL": 1.0 - geometry_score if gt_violation == 1 else 0.5 - geometry_score,
            }[case_type]
            if ranking_score <= score_rank[case_type]:
                continue

            score_rank[case_type] = ranking_score
            best[case_type] = CaseFrame(
                case_type=case_type,
                video=str(row["video_name"]),
                frame_idx=frame_idx,
                time_sec=frame_idx / fps,
                gt_violation=gt_violation,
                reason_gt=str(row.get("reason_gt", "none")),
                frame=frame,
                detections=detections,
                worker_box=worker_box,
                mask_box=mask_box,
                collector_box=collector_box,
                geometry_score=float(geometry_score),
                mask_conf=float(mask_conf),
                collector_conf=float(collector_conf),
                trigger=bool(trigger),
                trigger_tag=tag,
                trigger_reason=reason,
            )

    cases = [best[key] for key in CASE_ORDER if key in best]
    if not cases:
        raise RuntimeError("没有找到可用于可视化的典型案例，请检查视频和检测模型路径。")
    return cases


def draw_boxes(frame: np.ndarray, detections: list[dict[str, object]], worker_box: list[int] | None) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        cls = int(det["cls"])
        conf = float(det["conf"])
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        canvas = sys_mod.put_chinese_text(
            canvas,
            f"{CLASS_LABELS.get(cls, cls)} {conf:.2f}",
            position=(x1, max(18, y1 - 20)),
            font_size=16,
            color=color,
        )
    if worker_box is not None:
        x1, y1, x2, y2 = worker_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return canvas


def crop_roi(frame: np.ndarray, worker_box: list[int] | None, mask_box: list[int] | None) -> np.ndarray:
    roi = sys_mod.crop_qwen_roi(frame, worker_box, mask_box, expand_ratio=0.35)
    if roi is None or roi.size == 0:
        return frame
    return roi


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decision_text(case: CaseFrame) -> str:
    qwen_line = "调用Qwen复核" if case.trigger else "不调用Qwen"
    if case.case_type == "CLEAR_PASS":
        visual = "视觉证据充分，直接输出合规候选"
    elif case.case_type == "CLEAR_FAIL":
        visual = "视觉低分，直接进入告警候选"
    else:
        visual = "视觉证据处于不确定区间，提交VLM复核"
    gt = "违规" if case.gt_violation else "合规"
    return (
        f"{CASE_TITLES[case.case_type]}\n"
        f"视频: {case.video}\n"
        f"时间: {case.time_sec:.1f}s / 帧: {case.frame_idx}\n"
        f"几何分数: {case.geometry_score:.3f}\n"
        f"面罩置信度: {case.mask_conf:.3f}\n"
        f"收集器置信度: {case.collector_conf:.3f}\n"
        f"触发策略: {qwen_line}\n"
        f"触发原因: {case.trigger_reason}\n"
        f"标注状态: {gt}"
    )


def plot_cases(cases: list[CaseFrame], save_path: Path) -> None:
    n = len(cases)
    fig = plt.figure(figsize=(13.8, 4.15 * n))
    grid = fig.add_gridspec(n, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.08, hspace=0.26)

    for row_idx, case in enumerate(cases):
        original = cv2.resize(case.frame, (520, 320))
        boxed = cv2.resize(draw_boxes(case.frame, case.detections, case.worker_box), (520, 320))
        roi = cv2.resize(crop_roi(case.frame, case.worker_box, case.mask_box), (520, 320))

        ax0 = fig.add_subplot(grid[row_idx, 0])
        ax0.imshow(rgb(original))
        ax0.set_title(f"{CASE_TITLES[case.case_type]}：原始帧", fontsize=11, fontweight="bold")
        ax0.axis("off")

        ax1 = fig.add_subplot(grid[row_idx, 1])
        ax1.imshow(rgb(boxed))
        ax1.set_title("YOLO检测与软几何证据", fontsize=11, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(grid[row_idx, 2])
        ax2.imshow(rgb(roi))
        ax2.set_title("Qwen复核区域与决策摘要", fontsize=11, fontweight="bold")
        ax2.axis("off")
        ax2.text(
            0.02,
            0.98,
            decision_text(case),
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#555555", alpha=0.88),
        )

    fig.suptitle("Qwen惰性复核机制的典型案例可视化", fontsize=16, fontweight="bold", y=0.995)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def image_aspect(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    return max(1.0, float(w)) / max(1.0, float(h))


def resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, target_h), interpolation=interpolation)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def put_label(canvas: np.ndarray, text: str, x: int, y: int, font_size: int = 22) -> np.ndarray:
    return sys_mod.put_chinese_text(
        canvas,
        text,
        position=(x, y),
        font_size=font_size,
        color=(25, 25, 25),
    )


def save_case_triptychs(cases: list[CaseFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, case in enumerate(cases, start=1):
        original = case.frame
        boxed = draw_boxes(case.frame, case.detections, case.worker_box)
        roi = crop_roi(case.frame, case.worker_box, case.mask_box)

        target_h = 720
        images = [resize_to_height(image, target_h) for image in [original, boxed, roi]]
        labels = ["原始帧", "YOLO检测与几何证据", "Qwen复核区域"]
        gap = 28
        header_h = 86
        label_h = 44
        bottom_pad = 14
        total_w = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
        total_h = header_h + label_h + target_h + bottom_pad
        canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

        case_label = {
            "CLEAR_PASS": "高置信视觉判定",
            "AMBIGUOUS_MID": "Qwen惰性复核触发",
            "CLEAR_FAIL": "低置信直接告警",
        }.get(case.case_type, case.case_type)
        canvas = put_label(
            canvas,
            f"{case_label}    几何分数 {case.geometry_score:.3f}    Qwen触发: {'是' if case.trigger else '否'}",
            8,
            12,
            font_size=28,
        )

        x = 0
        y_img = header_h + label_h
        for image, label in zip(images, labels):
            canvas = put_label(canvas, label, x + 8, header_h + 4, font_size=24)
            h, w = image.shape[:2]
            canvas[y_img:y_img + h, x:x + w] = image
            x += w + gap

        save_path = output_dir / f"fig_qwen_case_{idx:02d}_{case.case_type.lower()}_cn.png"
        write_png(save_path, canvas)


def main() -> None:
    args = parse_args()
    cases = scan_cases(args)
    ensure_dir(CASE_VISUALIZATION_DIR)

    rows = []
    for case in cases:
        rows.append(
            {
                "case_type": case.case_type,
                "case_title": CASE_TITLES[case.case_type],
                "video": case.video,
                "frame": case.frame_idx,
                "time_sec": round(case.time_sec, 3),
                "gt_violation": case.gt_violation,
                "geometry_score": round(case.geometry_score, 4),
                "mask_conf": round(case.mask_conf, 4),
                "collector_conf": round(case.collector_conf, 4),
                "trigger_qwen": case.trigger,
                "trigger_tag": case.trigger_tag,
                "trigger_reason": case.trigger_reason,
            }
        )
    out_csv = CASE_VISUALIZATION_DIR / "qwen_case_examples.csv"
    fig_dir = PAPER_FIGURE_DIR / "case_visualization"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_case_triptychs(cases, fig_dir)
    print(f"典型案例CSV已生成: {out_csv}")
    print(f"典型案例图已生成: {fig_dir}")


if __name__ == "__main__":
    main()
