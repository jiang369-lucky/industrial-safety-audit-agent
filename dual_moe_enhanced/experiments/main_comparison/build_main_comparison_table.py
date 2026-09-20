# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import ABLATION_RESULT_CSV, MAIN_COMPARISON_DIR, PAPER_TABLE_DIR  # noqa: E402
from common.table_utils import save_table  # noqa: E402


METHOD_ORDER = ["A_Baseline", "B_LazyQwen", "C_UATS", "D_FullSystem"]
METHOD_LABELS = {
    "A_Baseline": "单视觉YOLO基线",
    "B_LazyQwen": "YOLO+惰性VLM复核",
    "C_UATS": "冲突触发时序融合",
    "D_FullSystem": "本文完整方法",
}


def mean_numeric(df: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def main() -> None:
    if not ABLATION_RESULT_CSV.exists():
        raise FileNotFoundError(f"找不到消融结果: {ABLATION_RESULT_CSV}")
    df = pd.read_csv(ABLATION_RESULT_CSV)

    rows = []
    for method in METHOD_ORDER:
        sub = df[df["Method"] == method]
        viol = sub[sub["Type"] == "violation"]
        rows.append(
            {
                "方法": METHOD_LABELS[method],
                "召回率/%": round(mean_numeric(viol, "Recall") * 100, 2),
                "F1/%": round(mean_numeric(viol, "F1") * 100, 2),
                "误报率/%": round(mean_numeric(sub, "FPR") * 100, 2),
                "跳变率/%": round(mean_numeric(sub, "Oscillation"), 2),
                "Qwen调用率/%": round(mean_numeric(sub, "Qwen_Rate"), 2),
                "FPS": round(mean_numeric(sub, "FPS"), 2),
                "论文解释": {
                    "A_Baseline": "基础视觉感知基线",
                    "B_LazyQwen": "验证VLM低频复核收益",
                    "C_UATS": "验证冲突触发与时序融合收益",
                    "D_FullSystem": "验证完整框架综合性能",
                }[method],
            }
        )

    table = pd.DataFrame(rows)
    save_table(
        table,
        MAIN_COMPARISON_DIR / "main_comparison_table.csv",
        PAPER_TABLE_DIR / "table_main_comparison.md",
        float_digits=2,
    )
    print("主对比论文表已生成:", MAIN_COMPARISON_DIR / "main_comparison_table.csv")
    print("Markdown 表已生成:", PAPER_TABLE_DIR / "table_main_comparison.md")


if __name__ == "__main__":
    main()
