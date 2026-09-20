# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
ROUTER_RUN_DIR = WORKSPACE_DIR / "runs" / "classify" / "router"
REPORT_DIR = WORKSPACE_DIR / "reports"


RUNS = [
    ("router_exp1_raw_no_aug", "原始数据", "否", "否"),
    ("router_exp2_raw_yolo_aug", "原始数据", "否", "是"),
    ("router_exp3_dataset_aug_no_aug", "数据集离线增强", "是", "否"),
    ("router_exp4_dataset_aug_yolo_aug", "数据集离线增强", "是", "是"),
]


def pct(value: float) -> float:
    return round(float(value) * 100.0, 2)


def load_one(run_name: str, dataset_aug: str, yolo_aug: str, label: str) -> dict[str, object] | None:
    csv_path = ROUTER_RUN_DIR / run_name / "results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    required = {"epoch", "train/loss", "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {sorted(missing)}")

    top1 = pd.to_numeric(df["metrics/accuracy_top1"], errors="coerce")
    top5 = pd.to_numeric(df["metrics/accuracy_top5"], errors="coerce")
    val_loss = pd.to_numeric(df["val/loss"], errors="coerce")
    train_loss = pd.to_numeric(df["train/loss"], errors="coerce")
    best_idx = int(top1.idxmax())
    last_idx = int(df.index[-1])

    return {
        "实验名": run_name,
        "数据来源": label,
        "离线增强": dataset_aug,
        "YOLO自带增强": yolo_aug,
        "训练轮数": int(len(df)),
        "最佳轮次": int(df.loc[best_idx, "epoch"]),
        "最佳Top1/%": pct(top1.loc[best_idx]),
        "最终Top1/%": pct(top1.loc[last_idx]),
        "最佳Top5/%": pct(top5.loc[best_idx]),
        "最小验证损失": round(float(val_loss.min()), 4),
        "最终验证损失": round(float(val_loss.loc[last_idx]), 4),
        "最终训练损失": round(float(train_loss.loc[last_idx]), 4),
        "权重路径": str((ROUTER_RUN_DIR / run_name / "weights" / "best.pt").resolve()),
    }


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = []
    missing = []
    for run_name, label, dataset_aug, yolo_aug in RUNS:
        row = load_one(run_name, dataset_aug, yolo_aug, label)
        if row is None:
            missing.append(run_name)
        else:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"没有找到任何训练结果，请先运行训练命令。检查目录: {ROUTER_RUN_DIR}")

    df = pd.DataFrame(rows).sort_values("最佳Top1/%", ascending=False).reset_index(drop=True)
    df.insert(0, "排名", range(1, len(df) + 1))
    best = df.iloc[0]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "router_augmentation_summary.csv"
    md_path = REPORT_DIR / "router_augmentation_summary.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    text = [
        "# 场景分类 YOLOv8-cls 数据增强对比",
        "",
        f"最佳实验：`{best['实验名']}`，最佳 Top-1 准确率为 **{best['最佳Top1/%']}%**。",
        "",
        markdown_table(df),
        "",
        "## 说明",
        "",
        "- `原始数据` 指 `dataset_gate`。",
        "- `数据集离线增强` 指 `dataset_gate_augmented`。",
        "- `YOLO自带增强` 指训练时启用 YOLO 的在线增强参数。",
        "- 推荐后续主系统优先尝试排名第一实验的 `weights/best.pt`。",
    ]
    if missing:
        text.extend(["", "## 尚未生成的实验", "", *[f"- `{name}`" for name in missing]])
    md_path.write_text("\n".join(text) + "\n", encoding="utf-8")

    print("场景分类增强对比汇总已生成:")
    print(csv_path)
    print(md_path)
    print(f"当前最佳: {best['实验名']} | Top-1={best['最佳Top1/%']}%")


if __name__ == "__main__":
    main()
