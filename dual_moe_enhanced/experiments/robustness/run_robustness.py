# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import FRAME_GT_PATH, LOVO_FOLDS_PATH, ROBUSTNESS_DIR  # noqa: E402
from ablation.run_full_ablation import (  # noqa: E402
    DATASETS,
    MODE_NAMES,
    evaluate_one_video,
    get_video_type,
    load_qwen_models,
    shutdown_qwen_executor,
)


DEFAULT_CONDITIONS = ["Clean", "Light", "Medium", "Heavy"]
DEFAULT_MODES = ["A", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行工业低清干扰鲁棒性实验。")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=list(DATASETS.keys()))
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["A", "D"])
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--skip-qwen-load", action="store_true", help="快速检查用，正式实验不要使用。")
    parser.add_argument("--qwen-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not FRAME_GT_PATH.exists() or not LOVO_FOLDS_PATH.exists():
        raise FileNotFoundError("缺少 frame_gt.pkl 或 lovo_folds.pkl，请检查 data_preparation/resources。")

    gt_dict = pickle.load(open(FRAME_GT_PATH, "rb"))
    lovo_folds = pickle.load(open(LOVO_FOLDS_PATH, "rb"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    qwen_loaded = False
    if "D" in args.modes and not args.skip_qwen_load:
        qwen_loaded = load_qwen_models(device, args.qwen_workers)

    rows: list[dict[str, object]] = []
    try:
        for condition in args.conditions:
            folder = DATASETS[condition]
            if not folder.exists():
                print(f"[跳过] {condition}: 找不到目录 {folder}")
                continue
            video_names = list(lovo_folds.keys())
            if args.max_videos > 0:
                video_names = video_names[: args.max_videos]
            for vid_name in video_names:
                vid_key = vid_name if str(vid_name).endswith(".mp4") else f"{vid_name}.mp4"
                vid_path = folder / vid_key
                if not vid_path.exists():
                    print(f"[缺失] {vid_path}")
                    continue
                video_type = get_video_type(gt_dict, vid_key)
                for mode_key in args.modes:
                    print(f"{condition} | {vid_key} | {MODE_NAMES[mode_key]} ...", end=" ", flush=True)
                    row, _ = evaluate_one_video(
                        vid_path=vid_path,
                        vid_key=vid_key,
                        condition=condition,
                        video_type=video_type,
                        mode_key=mode_key,
                        gt_dict=gt_dict,
                        device=device,
                        qwen_loaded=qwen_loaded,
                    )
                    rows.append(row)
                    print(f"Recall:{row['Recall']} F1:{row['F1']} FPR:{row['FPR']:.4f} FPS:{row['FPS']:.1f}")
    finally:
        shutdown_qwen_executor()

    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ROBUSTNESS_DIR / "robustness_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"鲁棒性实验结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
