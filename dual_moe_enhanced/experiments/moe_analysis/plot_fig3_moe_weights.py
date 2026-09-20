# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import (  # noqa: E402
    MOE_WEIGHTS_LOG,
    MOE_WEIGHTS_LOG_FALLBACK,
    PAPER_FIGURE_DIR,
    ensure_dir,
)


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

COND_ORDER = ["Clean", "Light", "Medium", "Heavy"]
COND_LABELS = ["基准", "轻度", "中度", "重度"]
EXPERT_COLUMNS = ["yolo", "temporal", "smoke"]
EXPERT_LABELS = {"yolo": "YOLO专家", "temporal": "时序专家", "smoke": "烟雾专家"}
EXPERT_COLORS = {"yolo": "#1a535c", "temporal": "#35bdb5", "smoke": "#ef8a22"}


def find_log_path() -> Path:
    for path in [MOE_WEIGHTS_LOG, MOE_WEIGHTS_LOG_FALLBACK]:
        if path.exists():
            return path
    raise FileNotFoundError(f"找不到 MoE 权重日志文件: {MOE_WEIGHTS_LOG}")


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Condition", "Video", "Frame", *EXPERT_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"权重日志缺少必要列: {sorted(missing)}")

    df = df[df["Condition"].isin(COND_ORDER)].copy()
    if df.empty:
        raise ValueError("权重日志中没有 Clean/Light/Medium/Heavy 条件数据")

    for column in EXPERT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").clip(0, 1)
    return df.dropna(subset=EXPERT_COLUMNS)


def build_mean_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(EXPERT_COLUMNS), len(COND_ORDER)), dtype=float)
    for row_idx, expert in enumerate(EXPERT_COLUMNS):
        for col_idx, condition in enumerate(COND_ORDER):
            values = df.loc[df["Condition"] == condition, expert]
            matrix[row_idx, col_idx] = float(values.mean()) if len(values) else np.nan
    return matrix


def expert_ylim(df: pd.DataFrame, column: str) -> tuple[float, float]:
    values = df[column].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    pad = max(0.025, (hi - lo) * 0.45)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def draw_distribution_panel(ax, df: pd.DataFrame, column: str) -> None:
    color = EXPERT_COLORS[column]
    rng = np.random.default_rng(42)
    means: list[float] = []

    ax.set_axisbelow(True)
    ax.set_facecolor("#fbfcfd")

    for idx, condition in enumerate(COND_ORDER):
        values = df[df["Condition"] == condition][column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            means.append(np.nan)
            continue

        sample = values if len(values) <= 520 else rng.choice(values, size=520, replace=False)
        x_jitter = idx + rng.normal(0, 0.055, len(sample))
        ax.scatter(
            x_jitter,
            sample,
            alpha=0.10,
            s=11,
            color=color,
            edgecolors="none",
            zorder=1,
        )

        q_low, med, q_high = np.percentile(values, [10, 50, 90])
        mean = float(np.mean(values))
        means.append(mean)

        ax.add_patch(
            plt.Rectangle(
                (idx - 0.28, q_low),
                0.56,
                max(q_high - q_low, 1e-4),
                facecolor=color,
                alpha=0.22,
                edgecolor="#333333",
                linewidth=0.95,
                zorder=2,
            )
        )
        ax.hlines(med, idx - 0.28, idx + 0.28, colors="#111111", linewidth=1.25, zorder=3)
        ax.plot(idx, mean, "o", color="white", markeredgecolor="#111111", markeredgewidth=1.1, markersize=6.4, zorder=4)

    ax.plot(
        range(len(COND_ORDER)),
        means,
        "-",
        color=color,
        linewidth=2.7,
        alpha=0.98,
        zorder=3,
        label="均值趋势",
    )

    ax.set_title(EXPERT_LABELS[column], fontweight="bold", fontsize=12)
    ax.set_xlabel("干扰强度等级", fontsize=10)
    ax.set_ylabel("门控权重", fontsize=10)
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_LABELS)
    ax.set_ylim(*expert_ylim(df, column))
    ax.grid(axis="y", linestyle=":", alpha=0.20, linewidth=0.9)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_heatmap(ax, matrix: np.ndarray) -> None:
    vmax = max(0.70, float(np.nanmax(matrix)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)

    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_LABELS)
    ax.set_yticks(range(len(EXPERT_COLUMNS)))
    ax.set_yticklabels([EXPERT_LABELS[c] for c in EXPERT_COLUMNS])
    ax.set_title("平均权重热力图", fontweight="bold", fontsize=12)

    ax.set_xticks(np.arange(-0.5, len(COND_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(EXPERT_COLUMNS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if value > 0.45 else "#1f2933"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10, color=color)

    cbar = plt.colorbar(image, ax=ax, fraction=0.030, pad=0.020)
    cbar.set_label("平均权重", fontsize=9)


def main() -> None:
    log_path = find_log_path()
    df = load_log(log_path)
    matrix = build_mean_matrix(df)

    print(f"成功加载权重日志: {log_path}")
    print(f"记录数: {len(df)}")

    fig = plt.figure(figsize=(15.2, 8.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.88], wspace=0.28, hspace=0.40)

    for idx, column in enumerate(EXPERT_COLUMNS):
        draw_distribution_panel(fig.add_subplot(grid[0, idx]), df, column)
    draw_heatmap(fig.add_subplot(grid[1, :]), matrix)

    fig.suptitle("图3 可学习小MoE门控的动态专家权重分配", fontweight="bold", fontsize=17, y=0.98)
    save_dir = ensure_dir(PAPER_FIGURE_DIR / "moe")
    save_path = save_dir / "fig3_moe_weights_cn.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"图已生成: {save_path}")


if __name__ == "__main__":
    main()
