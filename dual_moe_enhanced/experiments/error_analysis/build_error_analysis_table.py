# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import ABLATION_RESULT_CSV, ERROR_ANALYSIS_DIR, PAPER_TABLE_DIR  # noqa: E402
from common.table_utils import save_table  # noqa: E402


def main() -> None:
    if not ABLATION_RESULT_CSV.exists():
        raise FileNotFoundError(f"找不到消融结果: {ABLATION_RESULT_CSV}")
    df = pd.read_csv(ABLATION_RESULT_CSV)
    viol = df[df["Type"] == "violation"].copy()

    rows = []
    for video in sorted(viol["Video"].unique()):
        part = viol[viol["Video"] == video].set_index("Method")
        if "A_Baseline" not in part.index or "D_FullSystem" not in part.index:
            continue
        a = part.loc["A_Baseline"]
        d = part.loc["D_FullSystem"]
        f1_gain = (float(d["F1"]) - float(a["F1"])) * 100
        recall_gain = (float(d["Recall"]) - float(a["Recall"])) * 100
        fpr_change = (float(d["FPR"]) - float(a["FPR"])) * 100
        osc_change = float(d["Oscillation"]) - float(a["Oscillation"])

        if f1_gain >= 2.0:
            conclusion = "完整系统显著改善困难片段"
        elif f1_gain > 0:
            conclusion = "完整系统有小幅改善"
        else:
            conclusion = "该片段提升有限，建议作为局限性讨论"

        rows.append(
            {
                "视频": video,
                "基线F1/%": round(float(a["F1"]) * 100, 2),
                "完整系统F1/%": round(float(d["F1"]) * 100, 2),
                "F1增益/百分点": round(f1_gain, 2),
                "召回率增益/百分点": round(recall_gain, 2),
                "误报率变化/百分点": round(fpr_change, 2),
                "跳变率变化/百分点": round(osc_change, 2),
                "论文讨论建议": conclusion,
            }
        )

    table = pd.DataFrame(rows)
    save_table(
        table,
        ERROR_ANALYSIS_DIR / "video_level_error_analysis.csv",
        PAPER_TABLE_DIR / "table_video_level_error_analysis.md",
        float_digits=2,
    )

    discussion = [
        "误差分析建议：",
        "1. 当 F1 增益明显但误报率略升时，可表述为“以可接受误报代价提升违规召回”。",
        "2. 当跳变率下降时，可用于支撑时序稳定性设计。",
        "3. 当个别视频提升有限时，不建议删除，应在局限性中解释强弧光、遮挡或低清导致视觉证据不足。",
        "4. 该表适合放在讨论部分或补充实验部分，不应替代核心消融表。",
    ]
    (ERROR_ANALYSIS_DIR / "error_analysis_notes.md").write_text("\n".join(discussion), encoding="utf-8")

    print("视频级误差分析表已生成:", ERROR_ANALYSIS_DIR / "video_level_error_analysis.csv")
    print("Markdown 表已生成:", PAPER_TABLE_DIR / "table_video_level_error_analysis.md")


if __name__ == "__main__":
    main()
