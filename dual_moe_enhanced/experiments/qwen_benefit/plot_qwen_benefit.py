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

from common.experiment_paths import ABLATION_RESULT_CSV, PAPER_FIGURE_DIR, QWEN_BENEFIT_DIR, ensure_dir  # noqa: E402


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

METHOD_ORDER = ["A_Baseline", "B_LazyQwen", "C_UATS", "D_FullSystem"]
METHOD_LABELS = {
    "A_Baseline": "A 基线",
    "B_LazyQwen": "B +Qwen",
    "C_UATS": "C +UATS",
    "D_FullSystem": "D 完整系统",
}
METHOD_COLORS = {
    "A_Baseline": "#7f8c8d",
    "B_LazyQwen": "#e67e22",
    "C_UATS": "#2b6cb0",
    "D_FullSystem": "#1f8a70",
}


def load_ablation(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到消融实验结果: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"Condition", "Method", "Recall", "F1", "Qwen_Rate", "FPS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"消融 CSV 缺少必要列: {sorted(missing)}")
    return df[df["Condition"] == "Clean"].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["Method"] == method]
        if sub.empty:
            continue
        rows.append(
            {
                "Method": method,
                "方法": METHOD_LABELS[method],
                "Recall": pd.to_numeric(sub["Recall"], errors="coerce").mean(),
                "F1": pd.to_numeric(sub["F1"], errors="coerce").mean(),
                "Qwen_Rate": pd.to_numeric(sub["Qwen_Rate"], errors="coerce").mean(),
                "FPS": pd.to_numeric(sub["FPS"], errors="coerce").mean(),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("Clean 条件下没有可用消融结果。")

    baseline = summary[summary["Method"] == "A_Baseline"].iloc[0]
    summary["F1提升/百分点"] = (summary["F1"] - float(baseline["F1"])) * 100
    summary["Recall提升/百分点"] = (summary["Recall"] - float(baseline["Recall"])) * 100
    summary["FPS变化"] = summary["FPS"] - float(baseline["FPS"])
    summary["单位调用收益"] = summary.apply(
        lambda row: 0.0 if row["Qwen_Rate"] <= 0 else row["F1提升/百分点"] / row["Qwen_Rate"],
        axis=1,
    )
    return summary


def plot(summary: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.08, 1.0]})
    fig.subplots_adjust(wspace=0.28, top=0.82)

    ax0 = axes[0]
    for _, row in summary.iterrows():
        method = row["Method"]
        size = max(260, float(row["FPS"]) * 9.0)
        ax0.scatter(
            row["Qwen_Rate"],
            row["F1提升/百分点"],
            s=size,
            color=METHOD_COLORS[method],
            alpha=0.78,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )
        dx = 0.10 if method != "A_Baseline" else 0.06
        dy = 0.05 if method != "D_FullSystem" else -0.08
        ax0.text(
            row["Qwen_Rate"] + dx,
            row["F1提升/百分点"] + dy,
            f"{row['方法']}\nF1 +{row['F1提升/百分点']:.2f}",
            fontsize=9,
            va="center",
            ha="left",
        )

    ax0.axhline(0, color="#555555", linewidth=1.0, alpha=0.75)
    ax0.set_title("性能收益-调用成本关系", fontsize=13, fontweight="bold")
    ax0.set_xlabel("Qwen调用率/%", fontsize=10)
    ax0.set_ylabel("相对基线F1提升/百分点", fontsize=10)
    ax0.set_xlim(-0.25, max(5.2, float(summary["Qwen_Rate"].max()) + 0.9))
    ax0.set_ylim(-0.25, max(2.4, float(summary["F1提升/百分点"].max()) + 0.45))
    ax0.grid(linestyle=":", alpha=0.30)

    ax1 = axes[1]
    plot_df = summary[summary["Method"] != "A_Baseline"].copy()
    labels = plot_df["方法"].tolist()
    x = np.arange(len(labels))
    bars = ax1.bar(
        x,
        plot_df["Qwen_Rate"],
        color=[METHOD_COLORS[m] for m in plot_df["Method"]],
        width=0.52,
        alpha=0.86,
        label="Qwen调用率",
    )
    ax1.set_title("惰性复核代价", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Qwen调用率/%", fontsize=10)
    ax1.set_ylim(0, max(5.6, float(plot_df["Qwen_Rate"].max()) + 0.8))
    ax1.grid(axis="y", linestyle=":", alpha=0.28)
    for bar, value in zip(bars, plot_df["Qwen_Rate"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}%", ha="center", fontsize=9)

    ax1b = ax1.twinx()
    ax1b.plot(x, plot_df["FPS"], "-o", color="#4b5563", linewidth=2.3, markersize=6, label="FPS")
    ax1b.set_ylabel("FPS", fontsize=10, color="#4b5563")
    ax1b.tick_params(axis="y", labelcolor="#4b5563")
    ax1b.set_ylim(max(0, float(plot_df["FPS"].min()) - 8), float(plot_df["FPS"].max()) + 8)
    for xi, value in zip(x, plot_df["FPS"]):
        ax1b.text(xi, value - 2.2, f"{value:.1f}", ha="center", fontsize=9, color="#4b5563")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("Qwen惰性复核的收益-代价分析", fontsize=16, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = summarize(load_ablation(ABLATION_RESULT_CSV))
    ensure_dir(QWEN_BENEFIT_DIR)
    out_csv = QWEN_BENEFIT_DIR / "qwen_benefit_summary.csv"
    fig_path = PAPER_FIGURE_DIR / "qwen_benefit" / "fig_qwen_benefit_cn.png"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    plot(summary, fig_path)
    print(f"Qwen收益统计已生成: {out_csv}")
    print(f"Qwen收益图已生成: {fig_path}")


if __name__ == "__main__":
    main()
