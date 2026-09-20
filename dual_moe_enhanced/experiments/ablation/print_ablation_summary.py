# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import ABLATION_RESULT_CSV  # noqa: E402


METHOD_ORDER = ["A_Baseline", "B_LazyQwen", "C_UATS", "D_FullSystem"]
CONDITION_ORDER = ["Clean", "Light", "Medium", "Heavy"]

CONDITION_TITLE = {
    "Clean": "Clean 真实低清监控条件下详细对比（核心消融结果）",
    "Light": "Light 轻度干扰条件下详细对比（仅当 CSV 中存在时输出）",
    "Medium": "Medium 中度干扰条件下详细对比（仅当 CSV 中存在时输出）",
    "Heavy": "Heavy 重度干扰条件下详细对比（仅当 CSV 中存在时输出）",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="打印 A/B/C/D 消融实验汇总结果。")
    parser.add_argument("--csv", type=Path, default=ABLATION_RESULT_CSV, help="消融实验结果 CSV 路径。")
    parser.add_argument("--condition", choices=CONDITION_ORDER, default="Clean", help="默认输出 Clean。")
    parser.add_argument("--all", action="store_true", help="输出 CSV 中存在的全部条件。")
    return parser.parse_args()


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到结果文件: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"Condition", "Method", "Recall", "F1", "FPR", "Oscillation", "Qwen_Rate", "FPS"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")
    return df


def mean_value(df: pd.DataFrame, column: str) -> float:
    value = pd.to_numeric(df[column], errors="coerce").mean()
    if pd.isna(value):
        return 0.0
    return float(value)


def print_condition_summary(df: pd.DataFrame, condition: str) -> None:
    df_cond = df[df["Condition"] == condition]
    if df_cond.empty:
        print(f"{condition} 条件下没有数据")
        return

    print(CONDITION_TITLE.get(condition, f"{condition} 条件下详细对比"))
    for method in METHOD_ORDER:
        sub = df_cond[df_cond["Method"] == method]
        if sub.empty:
            continue

        print(
            f"  {method:<15} | "
            f"召回率:{mean_value(sub, 'Recall'):.4f} "
            f"F1:{mean_value(sub, 'F1'):.4f} "
            f"误报率:{mean_value(sub, 'FPR'):.4f} "
            f"跳变率:{mean_value(sub, 'Oscillation'):.2f}% "
            f"Qwen调用率:{mean_value(sub, 'Qwen_Rate'):.1f}% "
            f"FPS:{mean_value(sub, 'FPS'):.1f}"
        )


def print_compact_table(df: pd.DataFrame) -> None:
    print("\n核心指标汇总")
    print("  条件       方法             Recall    F1       FPR      Osc%    Qwen%   FPS")
    print("  --------- --------------- -------- -------- -------- ------- ------- -------")
    for condition in CONDITION_ORDER:
        part = df[df["Condition"] == condition]
        if part.empty:
            continue
        for method in METHOD_ORDER:
            sub = part[part["Method"] == method]
            if sub.empty:
                continue
            print(
                f"  {condition:<9} {method:<15} "
                f"{mean_value(sub, 'Recall'):<8.4f} "
                f"{mean_value(sub, 'F1'):<8.4f} "
                f"{mean_value(sub, 'FPR'):<8.4f} "
                f"{mean_value(sub, 'Oscillation'):<7.2f} "
                f"{mean_value(sub, 'Qwen_Rate'):<7.1f} "
                f"{mean_value(sub, 'FPS'):<7.1f}"
            )


def main() -> None:
    args = parse_args()
    df = load_results(args.csv)

    if args.all:
        for condition in CONDITION_ORDER:
            if df[df["Condition"] == condition].empty:
                continue
            print_condition_summary(df, condition)
            print()
        print_compact_table(df)
    else:
        print_condition_summary(df, args.condition)


if __name__ == "__main__":
    main()
