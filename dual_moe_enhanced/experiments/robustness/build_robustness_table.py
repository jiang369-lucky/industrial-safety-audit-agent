# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import PAPER_TABLE_DIR, ROBUSTNESS_DIR  # noqa: E402
from common.table_utils import save_table  # noqa: E402


CONDITION_ORDER = ["Clean", "Light", "Medium", "Heavy"]


def mean_numeric(df: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def main() -> None:
    csv_path = ROBUSTNESS_DIR / "robustness_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到鲁棒性结果: {csv_path}，请先运行 robustness/run_robustness.py")
    df = pd.read_csv(csv_path)

    rows = []
    for condition in CONDITION_ORDER:
        base = df[(df["Condition"] == condition) & (df["Method"] == "A_Baseline")]
        full = df[(df["Condition"] == condition) & (df["Method"] == "D_FullSystem")]
        if base.empty or full.empty:
            continue
        base_v = base[base["Type"] == "violation"]
        full_v = full[full["Type"] == "violation"]
        base_f1 = mean_numeric(base_v, "F1")
        full_f1 = mean_numeric(full_v, "F1")
        rows.append(
            {
                "干扰条件": condition,
                "YOLO基线F1/%": round(base_f1 * 100, 2),
                "完整系统F1/%": round(full_f1 * 100, 2),
                "F1增益/百分点": round((full_f1 - base_f1) * 100, 2),
                "YOLO基线跳变率/%": round(mean_numeric(base, "Oscillation"), 2),
                "完整系统跳变率/%": round(mean_numeric(full, "Oscillation"), 2),
                "YOLO基线误报率/%": round(mean_numeric(base, "FPR") * 100, 2),
                "完整系统误报率/%": round(mean_numeric(full, "FPR") * 100, 2),
                "论文定位": "补充鲁棒性分析，不作为核心消融",
            }
        )

    table = pd.DataFrame(rows)
    save_table(
        table,
        ROBUSTNESS_DIR / "robustness_table.csv",
        PAPER_TABLE_DIR / "table_robustness.md",
        float_digits=2,
    )
    print("鲁棒性论文表已生成:", ROBUSTNESS_DIR / "robustness_table.csv")
    print("Markdown 表已生成:", PAPER_TABLE_DIR / "table_robustness.md")


if __name__ == "__main__":
    main()
