# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import (  # noqa: E402
    ABLATION_DIR,
    ABLATION_RESULT_CSV,
    FRAME_GT_PATH,
    LOVO_FOLDS_PATH,
    MOE_WEIGHTS_LOG,
    PROJECT_DIR,
    TEST_VIDEO_DIR,
    add_project_to_syspath,
)

EXP_DIR = ABLATION_DIR
BASE_VIDEO_DIR = TEST_VIDEO_DIR

LOCAL_YOLO_CONFIG = EXPERIMENTS_DIR / ".ultralytics"
LOCAL_YOLO_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_YOLO_CONFIG))
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

add_project_to_syspath()

import dual_moe_safety_system as sys_mod  # noqa: E402
from ultralytics import YOLO  # noqa: E402


DATASETS = {
    "Clean": BASE_VIDEO_DIR / "clean" / "welding",
    "Light": BASE_VIDEO_DIR / "light" / "welding",
    "Medium": BASE_VIDEO_DIR / "medium" / "welding",
    "Heavy": BASE_VIDEO_DIR / "heavy" / "welding",
}

MODE_CONFIGS = {
    "A": {
        "enable_soft_geometry": True,
        "enable_lazy_qwen": False,
        "enable_qwen": False,
        "enable_ds_fusion": False,
        "enable_k_smooth": False,
        "enable_micro_gate": False,
        "enable_micro_experts": False,
    },
    "B": {
        "enable_soft_geometry": True,
        "enable_lazy_qwen": True,
        "enable_qwen": True,
        "enable_ds_fusion": False,
        "enable_k_smooth": False,
        "enable_micro_gate": False,
        "enable_micro_experts": False,
    },
    "C": {
        "enable_soft_geometry": True,
        "enable_lazy_qwen": True,
        "enable_qwen": True,
        "enable_ds_fusion": True,
        "enable_k_smooth": True,
        "enable_micro_gate": False,
        "enable_micro_experts": False,
    },
    "D": {
        "enable_soft_geometry": True,
        "enable_lazy_qwen": True,
        "enable_qwen": True,
        "enable_ds_fusion": True,
        "enable_k_smooth": True,
        "enable_micro_gate": True,
        "enable_micro_experts": True,
    },
}

MODE_NAMES = {
    "A": "A_Baseline",
    "B": "B_LazyQwen",
    "C": "C_UATS",
    "D": "D_FullSystem",
}

ABLATION_CONDITIONS = {"Clean"}
MOE_LOG_MODE = "D"
RESULT_COLUMNS = [
    "Condition",
    "Video",
    "Type",
    "Method",
    "Recall",
    "F1",
    "FPR",
    "Oscillation",
    "Qwen_Rate",
    "FPS",
]


def main() -> None:
    args = parse_args()
    run_full_ablation(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B/C/D ablation for the Dual-MoE safety system.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="要运行的数据条件，默认 Clean Light Medium Heavy 全部运行。",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODE_CONFIGS.keys()),
        default=list(MODE_CONFIGS.keys()),
        help="要运行的消融模式，默认 A B C D 全部运行。",
    )
    parser.add_argument(
        "--skip-qwen-load",
        action="store_true",
        help="仅用于快速检查流程；正式实验不要使用。使用后 B/C/D 不会加载真实千问。",
    )
    parser.add_argument("--qwen-workers", type=int, default=1, help="千问异步推理线程数，默认 1。")
    parser.add_argument("--max-videos", type=int, default=0, help="调试用，每个条件最多跑多少个视频；0 表示不限制。")
    return parser.parse_args()


def build_run_plan(selected_conditions: list[str], requested_modes: list[str]) -> list[tuple[str, list[str]]]:
    """Clean is the only ablation condition; other levels only generate MoE logs."""
    plan: list[tuple[str, list[str]]] = []
    for condition in selected_conditions:
        if condition in ABLATION_CONDITIONS:
            modes = list(requested_modes)
        else:
            modes = [MOE_LOG_MODE]
        if modes:
            plan.append((condition, modes))
    return plan


def run_full_ablation(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("Dual-MoE A/B/C/D 消融实验")
    print("A: YOLO软几何基线")
    print("B: 几何不确定性触发 Qwen")
    print("C: 几何不确定性 + 历史冲突K 触发 Qwen")
    print("D: C + 小MoE门控；D 不额外影响 Qwen 触发")
    print("=" * 80)

    gt_dict_path = FRAME_GT_PATH
    lovo_path = LOVO_FOLDS_PATH
    if not gt_dict_path.exists() or not lovo_path.exists():
        raise FileNotFoundError("缺少 frame_gt.pkl 或 lovo_folds.pkl，请先准备逐帧标注数据。")

    gt_dict = pickle.load(open(gt_dict_path, "rb"))
    lovo_folds = pickle.load(open(lovo_path, "rb"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    selected_conditions = [c for c in args.conditions if DATASETS[c].exists()]
    run_plan = build_run_plan(selected_conditions, args.modes)
    needs_qwen = any(MODE_CONFIGS[m]["enable_qwen"] for _, modes in run_plan for m in modes)
    qwen_loaded = False
    if needs_qwen and not args.skip_qwen_load:
        qwen_loaded = load_qwen_models(device, args.qwen_workers)
    elif needs_qwen:
        print("警告: 已跳过真实千问加载，本次结果不能作为正式论文实验。")

    all_results: list[dict[str, object]] = []
    log_path = MOE_WEIGHTS_LOG
    if log_path.exists():
        log_path.unlink()

    total_tasks = sum(
        len(modes) * min(len(lovo_folds), args.max_videos or len(lovo_folds))
        for _, modes in run_plan
    )
    task_idx = 0

    try:
        for condition, condition_modes in run_plan:
            folder = DATASETS[condition]
            print(f"\n正在评估条件: {condition} | 视频目录: {folder}")
            video_names = list(lovo_folds.keys())
            if args.max_videos > 0:
                video_names = video_names[: args.max_videos]

            for vid_name in video_names:
                vid_key = vid_name if str(vid_name).endswith(".mp4") else f"{vid_name}.mp4"
                vid_path = folder / vid_key
                if not vid_path.exists():
                    print(f"  跳过缺失视频: {vid_path}")
                    continue

                video_type = get_video_type(gt_dict, vid_key)
                for mode_key in condition_modes:
                    task_idx += 1
                    mode_name = MODE_NAMES[mode_key]
                    print(f"  [{task_idx}/{total_tasks}] {condition} | {vid_key} | {mode_name} ... ", end="", flush=True)
                    row, gate_rows = evaluate_one_video(
                        vid_path=vid_path,
                        vid_key=vid_key,
                        condition=condition,
                        video_type=video_type,
                        mode_key=mode_key,
                        gt_dict=gt_dict,
                        device=device,
                        qwen_loaded=qwen_loaded,
                    )
                    if condition in ABLATION_CONDITIONS:
                        all_results.append(row)
                    append_gate_log(log_path, condition, vid_key, gate_rows)
                    csv_note = "" if condition in ABLATION_CONDITIONS else " | MoE only"
                    print(
                        f"Recall:{fmt(row['Recall'])} F1:{fmt(row['F1'])} "
                        f"FPR:{row['FPR']:.4f} QwenCall:{row['Qwen_Rate']:.2f}% "
                        f"FPS:{row['FPS']:.1f}{csv_note}"
                    )
    finally:
        shutdown_qwen_executor()

    out_csv = ABLATION_RESULT_CSV
    pd.DataFrame(all_results, columns=RESULT_COLUMNS).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n消融实验结果已保存: {out_csv}")
    if log_path.exists():
        print(f"小MoE门控权重日志已保存: {log_path}")
    print_summary(pd.DataFrame(all_results))


def evaluate_one_video(
    vid_path: Path,
    vid_key: str,
    condition: str,
    video_type: str,
    mode_key: str,
    gt_dict: dict,
    device: str,
    qwen_loaded: bool,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    system, _ = init_system(mode_key, device)

    tp = fp = tn = fn = 0
    pred_hist: list[bool] = []
    gate_rows: list[dict[str, float]] = []
    welding_frames = 0
    total_read = 0
    qwen_candidate_frames = 0
    qwen_result_frames = 0
    qwen_trigger_k_values: list[float] = []
    ds_score_values: list[float] = []
    micro_score_values: list[float] = []
    qwen_clear_violation_frames = 0
    qwen_weak_violation_frames = 0
    qwen_compliant_frames = 0
    qwen_decision_frames = 0
    qwen_decision_correct_frames = 0
    qwen_decision_fp = 0
    qwen_decision_fn = 0
    visual_error_frames = 0
    decision_changed_frames = 0
    decision_repair_frames = 0
    decision_break_frames = 0

    cap = cv2.VideoCapture(str(vid_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1e-6:
        video_fps = 25.0
    t0 = time.time()
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        gt_key = f"{vid_key}_{frame_idx}"
        info = gt_dict.get(gt_key)
        if info and info.get("scope") == "active" and info.get("scene") == "welding":
            welding_frames += 1
            gt_violation = int(info.get("viol", 0))
            video_time = frame_idx / video_fps

            _, _, pred_violation, _, mode_str, _ = system._process_welding_enhanced(
                frame, video_time, frame_idx, False, 0, 0
            )
            pred_violation = bool(pred_violation)
            pred_hist.append(pred_violation)

            if str(mode_str).upper() == "LIVE":
                qwen_candidate_frames += 1
            fusion_info = getattr(system, "last_fusion_info", {})
            if fusion_info.get("qwen_available", False):
                qwen_result_frames += 1
                evidence = fusion_info.get("qwen_evidence", "")
                if evidence == "clear_violation":
                    qwen_clear_violation_frames += 1
                elif evidence == "weak_violation":
                    qwen_weak_violation_frames += 1
                elif evidence == "compliant":
                    qwen_compliant_frames += 1
                if evidence in ("clear_violation", "weak_violation", "compliant"):
                    qwen_decision_frames += 1
                    qwen_pred_violation = evidence in ("clear_violation", "weak_violation")
                    if qwen_pred_violation == bool(gt_violation):
                        qwen_decision_correct_frames += 1
                    elif qwen_pred_violation and not bool(gt_violation):
                        qwen_decision_fp += 1
                    elif (not qwen_pred_violation) and bool(gt_violation):
                        qwen_decision_fn += 1
            if "qwen_trigger_K" in fusion_info:
                qwen_trigger_k_values.append(float(fusion_info["qwen_trigger_K"]))
            if "ds_score" in fusion_info:
                ds_score_values.append(float(fusion_info["ds_score"]))
            if "micro_score" in fusion_info:
                micro_score_values.append(float(fusion_info["micro_score"]))
            if "visual_violation" in fusion_info:
                visual_violation = bool(fusion_info["visual_violation"])
                visual_correct = visual_violation == bool(gt_violation)
                final_correct = pred_violation == bool(gt_violation)
                if not visual_correct:
                    visual_error_frames += 1
                if visual_violation != pred_violation:
                    decision_changed_frames += 1
                    if (not visual_correct) and final_correct:
                        decision_repair_frames += 1
                    elif visual_correct and (not final_correct):
                        decision_break_frames += 1

            if mode_key == "D" and hasattr(system, "current_expert_weights"):
                weights = system.current_expert_weights
                if isinstance(weights, dict) and all(k in weights for k in ("yolo", "temporal", "smoke")):
                    gate_rows.append(
                        {
                            "Frame": frame_idx,
                            "yolo": float(weights["yolo"]),
                            "temporal": float(weights["temporal"]),
                            "smoke": float(weights["smoke"]),
                        }
                    )

            if pred_violation == bool(gt_violation):
                if gt_violation:
                    tp += 1
                else:
                    tn += 1
            else:
                if gt_violation:
                    fn += 1
                else:
                    fp += 1
        frame_idx += 1

    cap.release()
    wait_pending_qwen(system)
    elapsed = time.time() - t0

    if video_type == "violation" and (tp + fn) > 0:
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    else:
        recall = np.nan
        f1 = np.nan

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    oscillation = transition_rate(pred_hist)
    fps = total_read / elapsed if elapsed > 0 else 0.0
    qwen_calls = int(getattr(system, "trigger_count", 0))

    row = {
        "Condition": condition,
        "Video": vid_key,
        "Type": video_type,
        "Method": MODE_NAMES[mode_key],
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "Oscillation": round(oscillation, 2),
        "Qwen_Rate": round(qwen_calls / max(1, welding_frames) * 100, 2),
        "FPS": round(fps, 1),
    }
    return row, gate_rows


def init_system(mode_key: str, device: str):
    cfg = MODE_CONFIGS[mode_key].copy()
    sys_mod.FEATURE_CONFIG.clear()
    sys_mod.FEATURE_CONFIG.update(cfg)
    sys_mod.CONFIG.update(cfg)

    detector_weld = YOLO(sys_mod.CONFIG["weld_detector"]).to(device)
    detector_cut = YOLO(sys_mod.CONFIG["cut_detector"]).to(device)
    router = YOLO(sys_mod.CONFIG["router_model"]).to(device)
    macro_router = sys_mod.MacroMoERouter(router, sys_mod.CONFIG)
    system = sys_mod.DualMoESafetySystem(sys_mod.CONFIG, detector_weld, detector_cut, macro_router)

    tag = "[no gate]"
    if cfg["enable_micro_gate"]:
        gate = sys_mod.MicroGate()
        gate_path = Path(sys_mod.CONFIG.get("micro_gate_path", ""))
        if gate_path.exists():
            try:
                gate.load_state_dict(torch.load(gate_path, map_location=device, weights_only=False))
                gate.to(device).eval()
                system.set_gate_network(gate, device)
                tag = "[gate loaded]"
            except RuntimeError as exc:
                tag = f"[gate incompatible, please retrain: {gate_path.name}]"
                print(tag, end=" ")
        else:
            tag = f"[gate missing: {gate_path}]"
            print(tag, end=" ")
    return system, tag


def load_qwen_models(device: str, max_workers: int) -> bool:
    if sys_mod.model_container.chat_model is not None and sys_mod.model_container.processor is not None:
        return True

    base_model_path = Path(sys_mod.CONFIG["base_model"])
    lora_path = Path(sys_mod.CONFIG["lora_path"])
    if not base_model_path.exists():
        raise FileNotFoundError(f"找不到千问基座模型: {base_model_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"找不到千问LoRA: {lora_path}")

    print("正在加载真实 Qwen 模型，这一步会占用显存并耗时较久...")
    sys_mod.model_container.executor = ThreadPoolExecutor(max_workers=max_workers)
    sys_mod.model_container.processor = sys_mod.AutoProcessor.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )

    if device == "cuda" and sys_mod.CONFIG.get("use_8bit", True):
        quantization_config = sys_mod.BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False
        )
        base_model = sys_mod.Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_model_path),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        base_model = sys_mod.Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_model_path),
            device_map="cpu" if device == "cpu" else "auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    chat_model = sys_mod.PeftModel.from_pretrained(base_model, str(lora_path))
    chat_model.eval()
    sys_mod.model_container.chat_model = chat_model
    sys_mod.model_container.device = device
    print("真实 Qwen 模型加载完成。")
    return True


def shutdown_qwen_executor() -> None:
    executor = getattr(sys_mod.model_container, "executor", None)
    if executor is not None:
        executor.shutdown(wait=True)
        sys_mod.model_container.executor = None


def wait_pending_qwen(system) -> None:
    future = getattr(system.qwen_async, "pending_future", None)
    if future is None:
        return
    try:
        future.result(timeout=sys_mod.CONFIG.get("qwen_timeout", 10) + 60)
    except Exception as exc:
        print(f"[Qwen pending warning: {exc}]", end=" ")


def append_gate_log(log_path: Path, condition: str, video: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["Condition"] = condition
    df["Video"] = video
    columns = ["Condition", "Video", "Frame", "yolo", "temporal", "smoke"]
    df = df[columns]
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", index=False, header=header, encoding="utf-8-sig")


def get_video_type(gt_dict: dict, vid_name: str) -> str:
    for key, val in gt_dict.items():
        if key.startswith(vid_name) and int(val.get("viol", 0)) == 1:
            return "violation"
    return "compliance"


def transition_rate(history: list[bool]) -> float:
    if not history:
        return 0.0
    transitions = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])
    return transitions / len(history) * 100


def print_summary(df: pd.DataFrame) -> None:
    print("\n消融实验摘要:")
    for condition in ["Clean", "Light", "Medium", "Heavy"]:
        part = df[df["Condition"] == condition]
        if part.empty:
            continue
        print(f"\n--- {condition} ---")
        for method in MODE_NAMES.values():
            sub = part[part["Method"] == method]
            if sub.empty:
                continue
            print(
                f"  {method:<15} | Recall:{fmt(sub['Recall'].mean())} "
                f"F1:{fmt(sub['F1'].mean())} FPR:{sub['FPR'].mean():.4f} "
                f"Osc:{sub['Oscillation'].mean():.2f}% "
                f"QwenCall:{sub['Qwen_Rate'].mean():.2f}% "
                f"FPS:{sub['FPS'].mean():.1f}"
            )


def fmt(value) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
