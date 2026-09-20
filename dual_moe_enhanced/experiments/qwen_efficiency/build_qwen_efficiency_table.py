# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import ABLATION_RESULT_CSV, PAPER_TABLE_DIR, QWEN_EFFICIENCY_DIR  # noqa: E402
from common.table_utils import save_table  # noqa: E402


METHOD_ORDER = ["A_Baseline", "B_LazyQwen", "C_UATS", "D_FullSystem"]
METHOD_LABELS = {
    "A_Baseline": "不使用Qwen",
    "B_LazyQwen": "惰性Qwen复核",
    "C_UATS": "冲突触发Qwen",
    "D_FullSystem": "完整系统",
}


def mean_numeric(df: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def main() -> None:
    if not ABLATION_RESULT_CSV.exists():
        raise FileNotFoundError(f"找不到消融结果: {ABLATION_RESULT_CSV}")
    df = pd.read_csv(ABLATION_RESULT_CSV)
    base = df[df["Method"] == "A_Baseline"]
    base_viol = base[base["Type"] == "violation"]
    base_f1 = mean_numeric(base_viol, "F1")
    base_fps = mean_numeric(base, "FPS")

    rows = []
    for method in METHOD_ORDER:
        sub = df[df["Method"] == method]
        viol = sub[sub["Type"] == "violation"]
        f1 = mean_numeric(viol, "F1")
        fps = mean_numeric(sub, "FPS")
        rows.append(
            {
                "方法": METHOD_LABELS[method],
                "Qwen调用率/%": round(mean_numeric(sub, "Qwen_Rate"), 2),
                "F1/%": round(f1 * 100, 2),
                "相对基线F1增益/百分点": round((f1 - base_f1) * 100, 2),
                "FPS": round(fps, 2),
                "相对基线FPS下降": round(max(0.0, base_fps - fps), 2),
                "结论": "基线" if method == "A_Baseline" else "低频调用VLM以换取判别增益",
            }
        )

    table = pd.DataFrame(rows)
    save_table(
        table,
        QWEN_EFFICIENCY_DIR / "qwen_efficiency_table.csv",
        PAPER_TABLE_DIR / "table_qwen_efficiency.md",
        float_digits=2,
    )
    print("Qwen效率论文表已生成:", QWEN_EFFICIENCY_DIR / "qwen_efficiency_table.csv")
    print("Markdown 表已生成:", PAPER_TABLE_DIR / "table_qwen_efficiency.md")


if __name__ == "__main__":
    main()
