# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
LOCAL_ULTRALYTICS_DIR = EXPERIMENTS_DIR / ".ultralytics"
LOCAL_ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_ULTRALYTICS_DIR))

from ultralytics import YOLO  # noqa: E402

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import (  # noqa: E402
    CUTTING_ANNOTATION_CSV,
    PAPER_TABLE_DIR,
    REPO_ROOT,
    TEST_VIDEO_DIR,
    ensure_dir,
    first_existing,
)


DEFAULT_CUT_MODEL = first_existing(
    REPO_ROOT / "runs" / "detect" / "runs" / "detect" / "yolo_cut_expert" / "weights" / "best.pt",
    REPO_ROOT / "industrial_cctv_training" / "runs" / "detect" / "yolo_cut_expert" / "weights" / "best.pt",
)

SCENE_CN = {
    "clean": "真实低清监控",
    "light": "轻度干扰",
    "medium": "中度干扰",
    "heavy": "重度干扰",
}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def iter_segment_frames(video_path: Path, start_sec: float, end_sec: float, stride: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames, max(start_frame + 1, int(round(end_sec * fps))))

    for frame_idx in range(start_frame, end_frame, max(1, stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            yield frame_idx, frame, fps
    cap.release()


def predict_cutting_frame(
    detector: YOLO,
    frame: np.ndarray,
    conf_thres: float,
    device: str | None,
) -> dict[str, object]:
    kwargs = {"conf": conf_thres, "verbose": False}
    if device:
        kwargs["device"] = device

    result = detector.predict(frame, **kwargs)[0]
    worker_count = 0
    extinguisher_count = 0
    max_worker_conf = 0.0
    max_extinguisher_conf = 0.0

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        for cls_tensor, conf_tensor in zip(boxes.cls, boxes.conf):
            cls_id = int(cls_tensor.item())
            conf = float(conf_tensor.item())
            if cls_id == 0:
                worker_count += 1
                max_worker_conf = max(max_worker_conf, conf)
            elif cls_id == 1:
                extinguisher_count += 1
                max_extinguisher_conf = max(max_extinguisher_conf, conf)

    has_worker = worker_count > 0
    has_extinguisher = extinguisher_count > 0
    pred_violation = bool(has_worker and not has_extinguisher)
    reason = "缺少灭火器" if pred_violation else "合规/无工人"

    return {
        "worker_count": worker_count,
        "extinguisher_count": extinguisher_count,
        "max_worker_conf": max_worker_conf,
        "max_extinguisher_conf": max_extinguisher_conf,
        "pred_violation": int(pred_violation),
        "pred_reason": reason,
    }


def confusion_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for gt, pred in zip(y_true, y_pred) if gt == 1 and pred == 1)
    fp = sum(1 for gt, pred in zip(y_true, y_pred) if gt == 0 and pred == 1)
    tn = sum(1 for gt, pred in zip(y_true, y_pred) if gt == 0 and pred == 0)
    fn = sum(1 for gt, pred in zip(y_true, y_pred) if gt == 1 and pred == 0)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "Accuracy": accuracy,
    }


def build_markdown_table(frame_summary: dict[str, object], segment_summary: dict[str, object]) -> str:
    rows = [
        ("帧级", frame_summary),
        ("片段级", segment_summary),
    ]
    header = "| 评估粒度 | 样本数 | TP | FP | TN | FN | Precision/% | Recall/% | F1/% | FPR/% | Accuracy/% | FPS |\n"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for name, item in rows:
        fps = item.get("FPS", "")
        fps_text = f"{float(fps):.1f}" if fps != "" else "-"
        body.append(
            "| {name} | {n} | {tp} | {fp} | {tn} | {fn} | {precision:.2f} | {recall:.2f} | "
            "{f1:.2f} | {fpr:.2f} | {acc:.2f} | {fps} |".format(
                name=name,
                n=int(item["N"]),
                tp=int(item["TP"]),
                fp=int(item["FP"]),
                tn=int(item["TN"]),
                fn=int(item["FN"]),
                precision=float(item["Precision"]) * 100,
                recall=float(item["Recall"]) * 100,
                f1=float(item["F1"]) * 100,
                fpr=float(item["FPR"]) * 100,
                acc=float(item["Accuracy"]) * 100,
                fps=fps_text,
            )
        )
    note = (
        "\n\n说明：切割分支采用现有规则进行评估，即检测到工人且未检测到灭火器时判定为违规。"
        "片段级结果由同一标注片段内的帧级违规比例聚合得到。"
    )
    return header + sep + "\n".join(body) + note + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估切割分支违规识别效果")
    parser.add_argument("--condition", default="clean", choices=["clean", "light", "medium", "heavy"], help="测试视频条件")
    parser.add_argument("--video-dir", default="", help="自定义切割视频目录；留空时使用 test_video/<condition>/cutting")
    parser.add_argument("--annotation-csv", default=str(CUTTING_ANNOTATION_CSV), help="切割标注CSV")
    parser.add_argument("--model", default=str(DEFAULT_CUT_MODEL), help="切割YOLO专家权重")
    parser.add_argument("--device", default="0", help="推理设备，例如 0 或 cpu")
    parser.add_argument("--stride", type=int, default=4, help="采样步长，1表示逐帧评估")
    parser.add_argument("--conf-thres", type=float, default=0.2, help="YOLO置信度阈值")
    parser.add_argument("--segment-threshold", type=float, default=0.5, help="片段级违规比例阈值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir) if args.video_dir else TEST_VIDEO_DIR / args.condition / "cutting"
    annotation_csv = Path(args.annotation_csv)
    model_path = Path(args.model)

    if not video_dir.exists():
        raise FileNotFoundError(f"找不到切割视频目录: {video_dir}")
    if not annotation_csv.exists():
        raise FileNotFoundError(f"找不到切割标注文件: {annotation_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到切割检测模型: {model_path}")

    print("切割分支补充验证开始")
    print(f"  视频条件: {args.condition} ({SCENE_CN.get(args.condition, args.condition)})")
    print(f"  视频目录: {video_dir}")
    print(f"  标注文件: {annotation_csv}")
    print(f"  检测模型: {model_path}")

    annotations = pd.read_csv(annotation_csv)
    annotations = annotations[
        (annotations["scene_gt"].astype(str).str.lower() == "cutting")
        & (annotations["scope"].astype(str).str.lower() == "active")
    ].copy()
    if annotations.empty:
        raise RuntimeError("切割标注中没有 active cutting 片段。")

    detector = YOLO(str(model_path))
    frame_records: list[dict[str, object]] = []
    segment_records: list[dict[str, object]] = []

    start_time = time.perf_counter()
    inferred_frames = 0

    for segment_id, row in annotations.reset_index(drop=True).iterrows():
        video_path = video_dir / str(row["video_name"])
        if not video_path.exists():
            print(f"[跳过] 找不到视频: {video_path}")
            continue

        segment_preds: list[int] = []
        segment_gt = int(row["violation_gt"])
        segment_frame_count = 0

        for frame_idx, frame, fps in iter_segment_frames(
            video_path,
            float(row["start_sec"]),
            float(row["end_sec"]),
            stride=args.stride,
        ):
            pred = predict_cutting_frame(detector, frame, args.conf_thres, args.device)
            inferred_frames += 1
            segment_frame_count += 1
            segment_preds.append(int(pred["pred_violation"]))

            frame_records.append(
                {
                    "condition": args.condition,
                    "segment_id": segment_id,
                    "video_name": row["video_name"],
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / fps, 3),
                    "gt_violation": segment_gt,
                    **pred,
                }
            )

        violation_ratio = float(np.mean(segment_preds)) if segment_preds else 0.0
        segment_pred = int(violation_ratio >= args.segment_threshold)
        segment_records.append(
            {
                "condition": args.condition,
                "segment_id": segment_id,
                "video_name": row["video_name"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "gt_violation": segment_gt,
                "pred_violation": segment_pred,
                "pred_violation_ratio": round(violation_ratio, 6),
                "sampled_frames": segment_frame_count,
                "reason_gt": row.get("reason_gt", "none"),
            }
        )
        print(
            f"  {row['video_name']} [{row['start_sec']}-{row['end_sec']}s] "
            f"GT={segment_gt} PredRatio={violation_ratio:.3f} Frames={segment_frame_count}"
        )

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    fps_eval = inferred_frames / elapsed

    if not frame_records:
        raise RuntimeError("没有得到任何切割评估帧，请检查视频路径和标注时间段。")

    frame_df = pd.DataFrame(frame_records)
    segment_df = pd.DataFrame(segment_records)

    frame_metrics = confusion_metrics(
        frame_df["gt_violation"].astype(int).tolist(),
        frame_df["pred_violation"].astype(int).tolist(),
    )
    frame_metrics["N"] = len(frame_df)
    frame_metrics["FPS"] = fps_eval

    segment_metrics = confusion_metrics(
        segment_df["gt_violation"].astype(int).tolist(),
        segment_df["pred_violation"].astype(int).tolist(),
    )
    segment_metrics["N"] = len(segment_df)

    out_dir = ensure_dir(SCRIPT_DIR)
    frame_csv = out_dir / "cutting_branch_frame_results.csv"
    segment_csv = out_dir / "cutting_branch_segment_results.csv"
    summary_csv = out_dir / "cutting_branch_summary.csv"
    table_md = ensure_dir(PAPER_TABLE_DIR) / "table_cutting_branch.md"

    frame_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")
    segment_df.to_csv(segment_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"level": "frame", **frame_metrics},
            {"level": "segment", **segment_metrics},
        ]
    ).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    table_md.write_text(build_markdown_table(frame_metrics, segment_metrics), encoding="utf-8")

    print("\n切割分支核心指标")
    print(
        "  帧级: Recall={:.4f} F1={:.4f} FPR={:.4f} Accuracy={:.4f} FPS={:.1f}".format(
            frame_metrics["Recall"],
            frame_metrics["F1"],
            frame_metrics["FPR"],
            frame_metrics["Accuracy"],
            frame_metrics["FPS"],
        )
    )
    print(
        "  片段级: Recall={:.4f} F1={:.4f} FPR={:.4f} Accuracy={:.4f}".format(
            segment_metrics["Recall"],
            segment_metrics["F1"],
            segment_metrics["FPR"],
            segment_metrics["Accuracy"],
        )
    )
    print(f"\n帧级CSV已保存: {frame_csv}")
    print(f"片段级CSV已保存: {segment_csv}")
    print(f"汇总CSV已保存: {summary_csv}")
    print(f"论文表格已保存: {table_md}")


if __name__ == "__main__":
    main()
