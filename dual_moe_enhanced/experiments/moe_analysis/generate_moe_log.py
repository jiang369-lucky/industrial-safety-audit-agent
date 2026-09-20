# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import (  # noqa: E402
    MOE_WEIGHTS_LOG_FALLBACK,
    TEST_VIDEO_DIR,
    add_project_to_syspath,
)

add_project_to_syspath()
import dual_moe_safety_system as sys_mod  # noqa: E402


CONDITIONS = {
    "Clean": TEST_VIDEO_DIR / "clean" / "welding",
    "Light": TEST_VIDEO_DIR / "light" / "welding",
    "Medium": TEST_VIDEO_DIR / "medium" / "welding",
    "Heavy": TEST_VIDEO_DIR / "heavy" / "welding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="备用小 MoE 权重日志生成脚本。")
    parser.add_argument("--frame-skip", type=int, default=6, help="每隔多少帧记录一次。")
    parser.add_argument("--output", type=Path, default=MOE_WEIGHTS_LOG_FALLBACK)
    return parser.parse_args()


def smoke_score(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    smoke_like = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 150)
    return float(np.mean(smoke_like.astype(np.float32)))


def detect_mask_conf(frame: np.ndarray, detector: YOLO) -> float:
    results = detector(frame, verbose=False, classes=[2])
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0
    return float(results[0].boxes.conf.max())


def make_gate_feature(mask_conf: float, smoke: float, prev_mask_conf: float | None) -> np.ndarray:
    temporal_trend = 0.0 if prev_mask_conf is None else mask_conf - prev_mask_conf
    brightness = 0.5
    spark_intensity = 0.0
    conflict_k = float(np.clip(abs(temporal_trend) + smoke * 0.35, 0.0, 1.0))
    return np.array(
        [[
            mask_conf,
            smoke,
            temporal_trend,
            mask_conf,
            0.0,
            0.0,
            brightness,
            spark_intensity,
            conflict_k,
            0.0,
            1.0,
        ]],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate_path = Path(sys_mod.CONFIG.get("micro_gate_path", ""))
    if not gate_path.exists():
        raise FileNotFoundError(f"找不到小 MoE 门控权重: {gate_path}")

    gate = sys_mod.MicroGate().to(device)
    gate.load_state_dict(torch.load(gate_path, map_location=device, weights_only=False))
    gate.eval()

    detector = YOLO(sys_mod.CONFIG["weld_detector"])
    rows: list[dict[str, object]] = []
    start = time.time()

    for condition, folder in CONDITIONS.items():
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            print(f"[跳过] {condition}: 找不到视频 {folder}")
            continue
        print(f"\n{condition}: {len(videos)} 个视频")
        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            kept = 0
            prev_conf: float | None = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % args.frame_skip != 0:
                    frame_idx += 1
                    continue

                conf = detect_mask_conf(frame, detector)
                smoke = smoke_score(frame)
                feature = torch.tensor(make_gate_feature(conf, smoke, prev_conf), device=device)
                with torch.no_grad():
                    weights = gate(feature).squeeze(0).detach().cpu().numpy()
                rows.append(
                    {
                        "Condition": condition,
                        "Video": video_path.name,
                        "Frame": frame_idx,
                        "yolo": round(float(weights[0]), 5),
                        "temporal": round(float(weights[1]), 5),
                        "smoke": round(float(weights[2]), 5),
                    }
                )
                prev_conf = conf
                kept += 1
                frame_idx += 1
            cap.release()
            print(f"  {video_path.name}: {kept} 条权重记录")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n完成: {len(rows)} 条记录 -> {args.output}，耗时 {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
