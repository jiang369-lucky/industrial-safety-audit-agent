# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import (  # noqa: E402
    DATASET_STATISTICS_DIR,
    FRAME_GT_PATH,
    PAPER_TABLE_DIR,
    TEST_VIDEO_DIR,
)
from common.table_utils import save_table  # noqa: E402


CONDITIONS = {
    "Clean": TEST_VIDEO_DIR / "clean",
    "Light": TEST_VIDEO_DIR / "light",
    "Medium": TEST_VIDEO_DIR / "medium",
    "Heavy": TEST_VIDEO_DIR / "heavy",
}


def count_video_frames(folder: Path) -> tuple[int, int]:
    video_count = 0
    frame_count = 0
    for video_path in sorted(folder.glob("*.mp4")):
        video_count += 1
        cap = cv2.VideoCapture(str(video_path))
        frame_count += int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    return video_count, frame_count


def frame_gt_statistics() -> pd.DataFrame:
    if not FRAME_GT_PATH.exists():
        raise FileNotFoundError(f"找不到逐帧标注: {FRAME_GT_PATH}")
    gt_dict = pickle.load(open(FRAME_GT_PATH, "rb"))
    counter: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for item in gt_dict.values():
        scene = str(item.get("scene", "unknown"))
        scope = str(item.get("scope", "unknown"))
        viol = int(item.get("viol", 0))
        key = (scene, scope)
        counter[key]["total"] += 1
        counter[key]["violation" if viol else "compliance"] += 1

    rows = []
    for (scene, scope), c in sorted(counter.items()):
        rows.append(
            {
                "场景": scene,
                "标注范围": scope,
                "逐帧标注数": int(c["total"]),
                "合规帧": int(c["compliance"]),
                "违规帧": int(c["violation"]),
                "违规占比/%": round(c["violation"] / max(1, c["total"]) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def video_statistics() -> pd.DataFrame:
    rows = []
    for condition, root in CONDITIONS.items():
        for scene in ["welding", "cutting"]:
            folder = root / scene
            if not folder.exists():
                continue
            videos, frames = count_video_frames(folder)
            rows.append(
                {
                    "数据条件": condition,
                    "场景": "焊接" if scene == "welding" else "切割",
                    "视频数": videos,
                    "总帧数": frames,
                    "说明": "真实低清监控" if condition == "Clean" else f"{condition}干扰增强",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    DATASET_STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    video_df = video_statistics()
    frame_df = frame_gt_statistics()

    save_table(
        video_df,
        DATASET_STATISTICS_DIR / "dataset_video_statistics.csv",
        PAPER_TABLE_DIR / "table_dataset_video_statistics.md",
        float_digits=2,
    )
    save_table(
        frame_df,
        DATASET_STATISTICS_DIR / "dataset_frame_statistics.csv",
        PAPER_TABLE_DIR / "table_dataset_frame_statistics.md",
        float_digits=2,
    )

    print("数据集视频统计已生成:", DATASET_STATISTICS_DIR / "dataset_video_statistics.csv")
    print("逐帧标注统计已生成:", DATASET_STATISTICS_DIR / "dataset_frame_statistics.csv")
    print("论文 Markdown 表已生成到:", PAPER_TABLE_DIR)


if __name__ == "__main__":
    main()
