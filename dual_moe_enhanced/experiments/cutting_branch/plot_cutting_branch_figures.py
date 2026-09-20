# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import PAPER_FIGURE_DIR, ensure_dir  # noqa: E402


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

FRAME_CSV = SCRIPT_DIR / "cutting_branch_frame_results.csv"
SEGMENT_CSV = SCRIPT_DIR / "cutting_branch_segment_results.csv"
FIG_DIR = PAPER_FIGURE_DIR / "cutting_branch"
DEFAULT_FOCUS_VIDEO = "RWL_VID_00014.mp4"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FRAME_CSV.exists() or not SEGMENT_CSV.exists():
        raise FileNotFoundError(
            "找不到切割分支评估结果。请先运行: "
            "python -B dual_moe_enhanced\\experiments\\cutting_branch\\evaluate_cutting_branch.py "
            "--condition clean --device 0 --stride 1"
        )

    frame_df = pd.read_csv(FRAME_CSV)
    segment_df = pd.read_csv(SEGMENT_CSV)
    required = {
        "gt_violation",
        "pred_violation",
        "video_name",
        "time_sec",
        "max_extinguisher_conf",
        "max_worker_conf",
    }
    missing = required - set(frame_df.columns)
    if missing:
        raise ValueError(f"帧级结果缺少必要列: {sorted(missing)}")
    return frame_df, segment_df


def draw_single_axis(ax, sub: pd.DataFrame, seg_sub: pd.DataFrame, title: str) -> None:
    time = pd.to_numeric(sub["time_sec"], errors="coerce").to_numpy(dtype=float)
    pred_violation = pd.to_numeric(sub["pred_violation"], errors="coerce").fillna(0).to_numpy(dtype=float)
    gt_violation = pd.to_numeric(sub["gt_violation"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ext_conf = pd.to_numeric(sub["max_extinguisher_conf"], errors="coerce").fillna(0).to_numpy(dtype=float)
    worker_conf = pd.to_numeric(sub["max_worker_conf"], errors="coerce").fillna(0).to_numpy(dtype=float)

    for _, row in seg_sub.iterrows():
        if int(row.get("gt_violation", 0)) == 1:
            ax.axvspan(float(row["start_sec"]), float(row["end_sec"]), color="#e74c3c", alpha=0.18, linewidth=0)

    ax.plot(time, ext_conf, color="#2878b5", linewidth=1.8, label="灭火器置信度")
    ax.plot(time, worker_conf, color="#7f8c8d", linewidth=1.35, alpha=0.72, label="工人置信度")
    ax.step(time, gt_violation, where="post", color="#111827", linewidth=2.0, label="真实缺失")
    ax.step(time, pred_violation, where="post", color="#c0392b", linewidth=1.9, linestyle="--", label="规则告警")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.06, 1.08)
    ax.set_ylabel("状态/置信度", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.26)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_focus_video(frame_df: pd.DataFrame, segment_df: pd.DataFrame, video_name: str, save_path: Path) -> None:
    sub = frame_df[frame_df["video_name"] == video_name].copy().sort_values("time_sec")
    seg_sub = segment_df[segment_df["video_name"] == video_name].copy()
    if sub.empty:
        raise ValueError(f"结果中找不到视频: {video_name}")

    positive_segments = seg_sub[seg_sub["gt_violation"].astype(int) == 1]
    if positive_segments.empty:
        zoom_start = max(float(sub["time_sec"].min()), float(sub["time_sec"].max()) - 6.0)
        zoom_end = float(sub["time_sec"].max())
    else:
        zoom_start = max(float(sub["time_sec"].min()), float(positive_segments["start_sec"].min()) - 2.2)
        zoom_end = min(float(sub["time_sec"].max()), float(positive_segments["end_sec"].max()) + 1.2)

    zoom_sub = sub[(sub["time_sec"] >= zoom_start) & (sub["time_sec"] <= zoom_end)].copy()
    gt_rate = pd.to_numeric(sub["gt_violation"], errors="coerce").fillna(0).mean() * 100
    pred_rate = pd.to_numeric(sub["pred_violation"], errors="coerce").fillna(0).mean() * 100

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.0), sharex=False, gridspec_kw={"height_ratios": [1.0, 0.92]})
    draw_single_axis(
        axes[0],
        sub,
        seg_sub,
        f"{video_name} 完整时序：真实缺失帧占比 {gt_rate:.1f}% / 规则告警帧占比 {pred_rate:.1f}%",
    )
    draw_single_axis(
        axes[1],
        zoom_sub,
        seg_sub,
        f"灭火器缺失邻域放大（{zoom_start:.1f}-{zoom_end:.1f} s）",
    )
    axes[1].set_xlabel("视频时间 / s", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("切割分支灭火器缺失判别时序案例", fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.86, hspace=0.42)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_videos(frame_df: pd.DataFrame, segment_df: pd.DataFrame, save_path: Path) -> None:
    videos = sorted(frame_df["video_name"].dropna().unique().tolist())
    if not videos:
        raise ValueError("帧级结果中没有视频名称。")

    fig, axes = plt.subplots(len(videos), 1, figsize=(13.2, max(4.0, len(videos) * 2.25)), sharex=False)
    if len(videos) == 1:
        axes = [axes]

    for ax, video in zip(axes, videos):
        sub = frame_df[frame_df["video_name"] == video].copy().sort_values("time_sec")
        seg_sub = segment_df[segment_df["video_name"] == video].copy()
        if sub.empty:
            continue
        gt_rate = pd.to_numeric(sub["gt_violation"], errors="coerce").fillna(0).mean() * 100
        pred_rate = pd.to_numeric(sub["pred_violation"], errors="coerce").fillna(0).mean() * 100
        draw_single_axis(
            ax,
            sub,
            seg_sub,
            f"{video} 真实缺失帧占比 {gt_rate:.1f}% / 规则告警帧占比 {pred_rate:.1f}%",
        )

    axes[-1].set_xlabel("视频时间 / s", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("切割分支灭火器缺失判别时序结果", fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.90, hspace=0.55)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成切割分支时序图")
    parser.add_argument("--focus-video", default=DEFAULT_FOCUS_VIDEO, help="默认生成单个含违规视频的双层时序图")
    parser.add_argument("--all-videos", action="store_true", help="额外生成所有切割视频的时序图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_df, segment_df = load_results()
    fig_dir = ensure_dir(FIG_DIR)

    focus_path = fig_dir / "fig_cutting_violation_timeline_cn.png"
    plot_focus_video(frame_df, segment_df, args.focus_video, focus_path)
    print(f"切割时序案例图已生成: {focus_path}")

    if args.all_videos:
        all_path = fig_dir / "fig_cutting_violation_timeline_all_cn.png"
        plot_all_videos(frame_df, segment_df, all_path)
        print(f"全视频切割时序图已生成: {all_path}")


if __name__ == "__main__":
    main()
