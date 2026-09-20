# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _format_cell(value, float_digits: int) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{float_digits}f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def to_markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        cells = [_format_cell(row[col], float_digits) for col in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path, float_digits: int = 4) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(to_markdown_table(df, float_digits=float_digits), encoding="utf-8")


def pct(value: float, digits: int = 2) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value) * 100.0, digits)
