# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = COMMON_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

TEST_VIDEO_DIR = PROJECT_DIR / "test_video"
PAPER_FIGURE_DIR = EXPERIMENTS_DIR / "paper_figures"
PAPER_TABLE_DIR = EXPERIMENTS_DIR / "paper_tables"

DATA_PREP_DIR = EXPERIMENTS_DIR / "data_preparation"
RESOURCE_DIR = DATA_PREP_DIR / "resources"
ABLATION_DIR = EXPERIMENTS_DIR / "ablation"
MOE_ANALYSIS_DIR = EXPERIMENTS_DIR / "moe_analysis"
MAIN_COMPARISON_DIR = EXPERIMENTS_DIR / "main_comparison"
ROBUSTNESS_DIR = EXPERIMENTS_DIR / "robustness"
QWEN_EFFICIENCY_DIR = EXPERIMENTS_DIR / "qwen_efficiency"
QWEN_BENEFIT_DIR = EXPERIMENTS_DIR / "qwen_benefit"
CASE_VISUALIZATION_DIR = EXPERIMENTS_DIR / "case_visualization"
SCENE_ROUTER_DIR = EXPERIMENTS_DIR / "scene_router"
DATASET_STATISTICS_DIR = EXPERIMENTS_DIR / "dataset_statistics"
ERROR_ANALYSIS_DIR = EXPERIMENTS_DIR / "error_analysis"

FRAME_GT_PATH = RESOURCE_DIR / "frame_gt.pkl"
LOVO_FOLDS_PATH = RESOURCE_DIR / "lovo_folds.pkl"
WELDING_ANNOTATION_CSV = RESOURCE_DIR / "gt_annotations.csv"
CUTTING_ANNOTATION_CSV = RESOURCE_DIR / "cutting_annotations.csv"

ABLATION_RESULT_CSV = ABLATION_DIR / "ablation_results_4step.csv"
MOE_WEIGHTS_LOG = MOE_ANALYSIS_DIR / "moe_weights_log.csv"
MOE_WEIGHTS_LOG_FALLBACK = MOE_ANALYSIS_DIR / "moe_weights_log_1.csv"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def add_project_to_syspath() -> None:
    import sys

    project = str(PROJECT_DIR)
    if project not in sys.path:
        sys.path.insert(0, project)
