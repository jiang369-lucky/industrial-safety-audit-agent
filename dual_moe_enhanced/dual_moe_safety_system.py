"""
安全监控系统 - 基于双层MoE架构的智能焊接/切割作业安全监控系统
【软评分几何判断版 - 集成高斯衰减 + 分级触发 + 加权融合】

使用技术
1. YOLOv8目标检测 - 检测工人、面罩、烟雾收集器、灭火器
2. Qwen3-VL-4B视觉语言模型 - 处理模糊情况的违规判断
3. 混合专家架构(MoE) - 大MoE场景路由 + 小MoE专家集群
4. D-S证据理论 - 融合YOLO和Qwen的检测结果，输出冲突系数K
5. 可学习门控网络 - 2层MLP动态计算专家权重
6. K-自适应平滑 - 根据冲突系数K动态调整平滑窗口大小
7. 异步推理 - Qwen异步调用，不阻塞主流程
8. 多级平滑机制 - 检测平滑、状态平滑、场景平滑
9. 风险感知缓存 - 动态TTL策略缓存Qwen结果
10. 软评分几何判断 - 高斯衰减代替硬截断，适应工人姿态变化
11. 分级触发策略 - 高分跳过Qwen、低分直接判违规、中等分数触发Qwen
12. 加权融合机制 - Qwen占70%权重，几何占30%权重

系统流程
视频输入
    ↓
步骤1：帧采样
- 每4帧处理一次，降低计算负载
    ↓
步骤2：场景识别（大MoE）
- 每5帧执行一次场景分类
- 识别结果：焊接/切割/空闲
- 场景平滑：5帧中至少2帧为同一场景才切换
    ↓
    ├──→ 切割场景 ──→ YOLO检测(工人/灭火器) ──→ 违规判断 ──→ 输出
    ├──→ 空闲场景 ──→ 低功耗监控 ──→ 输出
    └──→ 焊接场景 ──→ 进入焊接检测流程
                          ↓
步骤3：YOLO目标检测（焊接场景）
- 检测目标：工人(cls=1)、烟雾收集器(cls=0)、面罩(cls=2)
- 输出：检测框坐标、置信度
    ↓
步骤4：K-自适应检测平滑
- 根据冲突系数K动态调整窗口大小：W = 5 × (1 + 10×K)
- 窗口内至少3帧检测到才判定存在
    ↓
步骤5：小MoE专家集群
- YOLO专家：基于面罩置信度输出分数
- 时序专家：基于置信度变化趋势输出分数
- 烟雾专家：基于烟雾浓度输出分数
    ↓
步骤6：可学习门控网络
- 输入：[面罩置信度, 烟雾分数, 时序趋势]
- 2层MLP计算动态权重
- 输出：[w_yolo, w_temporal, w_smoke]
    ↓
步骤7：专家加权融合
- 融合分数 = w_yolo×YOLO专家 + w_temporal×时序专家 + w_smoke×烟雾专家
    ↓
步骤8：软评分几何判断（核心改进）
- 使用高斯衰减计算几何置信度分数（0-1连续值）
- 考虑IoU、X/Y位置、尺寸比例、YOLO置信度
- 高分(>0.75)且高置信度 → 跳过Qwen
- 中等分数(0.25-0.75) → 触发Qwen
- 低分(<0.25)且无烟雾遮挡 → 直接判违规
- 低分但有烟雾遮挡 → 触发Qwen（可能是遮挡）
    ↓
步骤9：Qwen异步推理
- 仅在可疑时调用（几何分数0.25-0.75或烟雾遮挡低分）
- 异步执行，不阻塞主检测流程
- 结果缓存2秒，避免重复调用
    ↓
步骤10：加权融合判定
- 有Qwen结果：最终分 = 0.7×Qwen分 + 0.3×几何分
- 无Qwen结果：最终分 = 几何分
- 最终分 ≥ 0.5 → 合规，< 0.5 → 违规
    ↓
步骤11：违规判断与原因生成
- 组合规则：缺少面罩/缺少收集器/未正确佩戴面罩
- 输出：合规/违规 + 具体原因
    ↓
步骤12：D-S证据融合
- 融合YOLO检测结果和Qwen判断结果
- 计算折扣因子（基于熵）和动态可靠性
- 输出：融合置信度 + 冲突系数K
- K用于驱动自适应平滑
    ↓
步骤13：状态平滑
- 窗口大小：10帧
- 10帧中至少6帧违规才输出违规，防止状态跳变
- 冷却机制：状态切换后15帧冷却
    ↓
步骤14：风险感知缓存
- 强合规(置信度>0.85)：缓存2.0秒
- 强违规(置信度<0.3)：缓存1.0秒
- 模糊状态：缓存0.3秒，高频检查
- 安全守卫：YOLO证据与缓存冲突时否决缓存
    ↓
步骤15：视频输出
- 绘制检测框和置信度
- 显示信息面板（场景、状态、K值、几何分、最终分、专家权重、FPS等）
- 保存输出视频
    ↓
输出结果

关键公式
1. K-自适应平滑窗口：
   W_dynamic = W_base × (1 + λ × K)
   其中：W_base=5, λ=10, K∈[0,0.5]

2. 软评分几何分数：
   geometry_score = 0.4×position_score + 0.3×iou_score + 0.2×size_score + 0.1×conf_score
   position_score = 0.5×exp(-2×X_offset²) + 0.5×exp(-2×Y_dist²)
   iou_score = clamp((iou - 0.2) / 0.3, 0, 1)

3. 熵权折扣因子：
   α = 0.5 + 0.2 / (1 + e^(5×(H-1)))
   H = -p×log(p) - (1-p)×log(1-p)

4. D-S证据融合：
   belief = (m1_compliant × m2_compliant) / (1 - K)
   K = m1_compliant×m2_omega + m1_omega×m2_compliant

5. 加权融合最终分数：
   final_score = 0.7×qwen_score + 0.3×geometry_score

6. 时序趋势：
   trend = mean(最近3帧) - mean(前3帧)

7. 烟雾分数：
   smoke = 低饱和度(<50)且高亮度(>150)的像素比例
"""

import cv2
import torch
import torch.nn as nn
import os
import time
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

os.environ["YOLO_VERBOSE"] = "False"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import threading
import queue
import gc
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future

# ================= 实验模式配置（默认值，可由外部函数修改） =================
# 消融实验配置说明:
# A: 基线系统 (YOLO检测 + 软几何评分，不调用Qwen)
# B: +惰性Qwen触发 (A + 几何触发Qwen)
# C: +UATS模块 (B + D-S证据融合 + K-自适应平滑)  [创新点2：不确定性感知的时序稳定]
# D: 完整系统 (C + 小MoE门控网络)              [创新点3：领域知识引导的门控]

_EXPERIMENT_MODE = "D"  # 内部变量，不要直接修改，使用 set_experiment_mode()

# 根据实验模式配置功能开关
_FEATURE_CONFIG = {
    "enable_soft_geometry": True,
    "enable_lazy_qwen": False,
    "enable_qwen": False,
    "enable_ds_fusion": False,
    "enable_k_smooth": False,
    "enable_micro_gate": False,
    "enable_micro_experts": False,
}


def set_experiment_mode(mode):
    """
    从外部设置实验模式
    参数:
        mode: "A", "B", "C", "D"
    返回:
        FEATURE_CONFIG 字典
    """
    global _EXPERIMENT_MODE, _FEATURE_CONFIG

    _EXPERIMENT_MODE = mode

    if _EXPERIMENT_MODE == "A":
        # 基线: YOLO + 软几何评分
        _FEATURE_CONFIG = {
            "enable_soft_geometry": True,
            "enable_lazy_qwen": False,
            "enable_qwen": False,
            "enable_ds_fusion": False,
            "enable_k_smooth": False,
            "enable_micro_gate": False,
            "enable_micro_experts": False,
        }
    elif _EXPERIMENT_MODE == "B":
        # +惰性Qwen触发
        _FEATURE_CONFIG = {
            "enable_soft_geometry": True,
            "enable_lazy_qwen": True,
            "enable_qwen": True,
            "enable_ds_fusion": False,
            "enable_k_smooth": False,
            "enable_micro_gate": False,
            "enable_micro_experts": False,
        }
    elif _EXPERIMENT_MODE == "C":
        # +UATS模块 (D-S证据融合 + K-自适应平滑)
        _FEATURE_CONFIG = {
            "enable_soft_geometry": True,
            "enable_lazy_qwen": True,
            "enable_qwen": True,
            "enable_ds_fusion": True,
            "enable_k_smooth": True,
            "enable_micro_gate": False,
            "enable_micro_experts": False,
        }
    else:  # _EXPERIMENT_MODE == "D"
        # 完整系统 (UATS + 小MoE门控网络)
        _FEATURE_CONFIG = {
            "enable_soft_geometry": True,
            "enable_lazy_qwen": True,
            "enable_qwen": True,
            "enable_ds_fusion": True,
            "enable_k_smooth": True,
            "enable_micro_gate": True,
            "enable_micro_experts": True,
        }

    print(f"[系统] 实验模式已设置为: {_EXPERIMENT_MODE}")
    return _FEATURE_CONFIG


def get_experiment_mode():
    """获取当前实验模式"""
    return _EXPERIMENT_MODE


def get_feature_config():
    """获取当前功能配置"""
    return _FEATURE_CONFIG.copy()


# 初始化默认配置（D模式）
set_experiment_mode("D")

# 为了兼容原有代码，创建 FEATURE_CONFIG 变量（指向内部配置）
FEATURE_CONFIG = _FEATURE_CONFIG
EXPERIMENT_MODE = _EXPERIMENT_MODE

# ================= 配置区域 =================
CONFIG = {
    "router_model": r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router\exp2_raw_yolo_aug\weights\best.pt",
    "weld_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_weld_expert\weights\best.pt",
    "cut_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_cut_expert\weights\best.pt",
    "base_model": r"D:\yolo\Spark-Gated MoE Framework\model\Qwen3-VL-4B-Instruct",
    "lora_path": r"D:\yolo\Spark-Gated MoE Framework\LlamaFactory-main\saves\qwen3-vl-weld-3epoch",

    "video_source": r"D:\yolo\Spark-Gated MoE Framework\dual_moe_enhanced\test_video\clean\welding\RWL_VID_00009.mp4",
    "conf_thres": 0.2,
    "router_interval": 5,
    "save_dir": "violations",
    "skip_frames": 4,
    "max_detection_size": 640,
    "max_display_height": 900,
    "info_panel_width": 380,

    "qwen_timeout": 10,
    "qwen_image_size": 512,
    "qwen_roi_expand": 0.35,
    "use_8bit": True,
    "check_interval": 2.0,
    "check_interval_min": 1.0,
    "check_interval_max": 3.0,
    "cache_duration": 2.0,
    "mask_conf_thresh": 0.6,

    "detection_smooth_window": 5,
    "detection_smooth_thresh": 3,
    "state_smooth_window": 10,
    "state_smooth_thresh": 6,
    "state_cooldown_frames": 15,

    "enable_smoke_detection": True,
    "smoke_threshold": 0.06,
    "smoke_min_frames": 3,
    "smoke_cooldown": 3.0,
    "silent_frames": 60,

    "enable_lowlight_enhance": True,
    "brightness_threshold": 45,
    "clahe_clip_limit": 2.0,

    "cut_threshold": 0.35,
    "weld_threshold_high": 0.65,
    "weld_threshold_low": 0.1,

    "output_folder": r"D:\yolo\Spark-Gated MoE Framework\dual_moe_enhanced\dual_moe_demo\welding_heavy",
    "save_video": True,
    "show_display": True,

    "font_paths": ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"],

    "micro_gate_path": r"D:\yolo\Spark-Gated MoE Framework\dual_moe_enhanced\pth\micro_gate_final.pth",
    "micro_gate_input_dim": 11,
    "k_adaptive_lambda": 10.0,
    "k_adaptive_max_window": 20,

    # 门控输入归一化系数
    "smoke_scale_factor": 10.0,
    "trend_scale_factor": 2.0,
    "micro_gate_clean_score_margin": 0.05,

    # 软评分几何判断参数
    "geometry_clear_pass_threshold": 0.75,
    "geometry_clear_fail_threshold": 0.25,
    "qwen_conflict_force_threshold": 0.25,
    "qwen_conflict_skip_threshold": 0.06,
    "qwen_low_conflict_safe_geometry": 0.55,
    "qwen_low_conflict_safe_mask_conf": 0.45,
    "qwen_trigger_conflict_decay": 0.85,
    "qwen_trigger_conflict_momentum": 0.6,
    "qwen_clear_visual_guard_geometry": 0.58,
    "qwen_clear_visual_guard_mask_conf": 0.45,
    "qwen_weak_violation_score": 0.45,
    "qwen_clear_violation_score": 0.1,
    "qwen_compliant_score": 0.84,
    "qwen_direct_weight": 0.0,
    "qwen_ds_weight": 0.48,
    "qwen_min_geometry_for_compliance": 0.55,
    "qwen_min_mask_conf_for_compliance": 0.30,
    "qwen_low_geometry_compliance_cap": 0.48,
    "qwen_observe_only_without_ds": True,
    "qwen_direct_override_geometry": 0.68,
    "qwen_direct_override_mask_conf": 0.42,
    "qwen_override_missing_mask_score": 0.70,
    "qwen_override_missing_collector_score": 0.70,
    "fusion_qwen_weight": 0.48,
    "fusion_geometry_weight": 0.52,
    "ds_final_weight": 0.25,
    "ds_final_weight_when_qwen": 0.35,
    "micro_final_weight": 0.25,
    "micro_final_weight_conflict_boost": 0.15,
    "micro_final_safe_margin": 0.04,
    "micro_gate_release_smoke_start": 0.38,
    "micro_gate_release_smoke_span": 0.35,
    "micro_gate_release_trend_norm": 0.35,
    "micro_gate_nonheavy_score_margin": 0.08,
    "micro_gate_heavy_smoke_threshold": 0.50,
    "final_compliance_threshold": 0.5,
}

WINDOW_TITLE = f"Dual_MoE_Mode_{EXPERIMENT_MODE}"


@dataclass
class ModelContainer:
    processor: Any = None
    chat_model: Any = None
    device: str = "cuda"
    executor: ThreadPoolExecutor = None


model_container = ModelContainer()


def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def enhance_low_light(frame):
    if not CONFIG["enable_lowlight_enhance"]:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def compute_frame_hash(frame):
    small = cv2.resize(frame, (32, 32))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


# ================= 软评分几何判断函数 =================
def compute_geometry_score(worker_box, mask_box, mask_conf, frame_w, frame_h):
    """
    计算面罩佩戴的几何置信度分数 (0.0 - 1.0)
    使用高斯衰减代替硬截断，适应姿态变化
    分数越高，表示越像"正确佩戴"
    """
    if worker_box is None or mask_box is None:
        return 0.0

    wx1, wy1, wx2, wy2 = worker_box
    mx1, my1, mx2, my2 = mask_box

    worker_w = wx2 - wx1
    worker_h = wy2 - wy1
    mask_w = mx2 - mx1
    mask_h = my2 - my1

    if worker_w <= 0 or worker_h <= 0 or mask_w <= 0 or mask_h <= 0:
        return 0.0

    # 1. IoU 分数
    inter_x1 = max(wx1, mx1)
    inter_y1 = max(wy1, my1)
    inter_x2 = min(wx2, mx2)
    inter_y2 = min(wy2, my2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        iou = 0.0
    else:
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        mask_area = mask_w * mask_h
        iou = intersection / mask_area if mask_area > 0 else 0.0

    iou_score = np.clip((iou - 0.2) / 0.3, 0.0, 1.0)

    # 2. 相对位置分数
    worker_cx = wx1 + worker_w / 2
    worker_cy = wy1 + worker_h / 2
    mask_cx = mx1 + mask_w / 2
    mask_cy = my1 + mask_h / 2

    x_offset_norm = abs(mask_cx - worker_cx) / (worker_w / 2)
    x_score = np.exp(-2.0 * (x_offset_norm ** 2))

    head_top = wy1 + worker_h * 0.15
    head_bottom = wy1 + worker_h * 0.55
    head_center = (head_top + head_bottom) / 2
    head_height = head_bottom - head_top

    y_dist = abs(mask_cy - head_center)
    y_score = np.exp(-2.0 * ((y_dist / max(head_height, 1)) ** 2))

    position_score = 0.5 * x_score + 0.5 * y_score

    # 3. 尺寸合理性分数
    w_ratio = mask_w / worker_w
    h_ratio = mask_h / worker_h
    w_score = np.exp(-5.0 * (w_ratio - 0.35) ** 2)
    h_score = np.exp(-5.0 * (h_ratio - 0.35) ** 2)
    size_score = 0.5 * w_score + 0.5 * h_score

    # 4. 综合几何分数
    geometry_score = (0.4 * position_score +
                      0.3 * iou_score +
                      0.2 * size_score +
                      0.1 * np.clip(mask_conf, 0.0, 1.0))

    return float(geometry_score)


def should_trigger_qwen_by_geometry(
        geometry_score, mask_conf, has_smoke, smoke_score,
        conflict_k=0.0, conflict_aware=False):
    """
    基于几何分数决定是否需要 Qwen 介入
    """
    if (conflict_aware
            and conflict_k >= CONFIG.get("qwen_conflict_force_threshold", 0.25)
            and geometry_score < 0.88):
        return True, "CONFLICT_RECHECK", f"history conflict K={conflict_k:.2f}"

    if geometry_score > CONFIG.get("geometry_clear_pass_threshold", 0.75):
        if mask_conf > 0.5:
            return False, "CLEAR_PASS", "高分+高置信度"
        elif geometry_score > 0.85:
            return False, "CLEAR_PASS", "极高几何分数"

    if conflict_aware:
        low_k = CONFIG.get("qwen_conflict_skip_threshold", 0.06)
        safe_geo = CONFIG.get("qwen_low_conflict_safe_geometry", 0.55)
        safe_mask = CONFIG.get("qwen_low_conflict_safe_mask_conf", 0.45)
        if (conflict_k <= low_k and geometry_score >= safe_geo and mask_conf >= safe_mask
                and not (has_smoke and smoke_score > 0.3)):
            return False, "LOW_CONFLICT_SKIP", f"low conflict K={conflict_k:.2f}"

    if geometry_score < CONFIG.get("geometry_clear_fail_threshold", 0.25):
        if has_smoke and smoke_score > 0.3:
            return True, "AMBIGUOUS_SMOKE", f"烟雾遮挡+低分({geometry_score:.2f})"
        return False, "CLEAR_FAIL", f"低分({geometry_score:.2f})"

    return True, "AMBIGUOUS_MID", f"中等分数({geometry_score:.2f})"


def parse_qwen_safety_response(response: str) -> Dict[str, Any]:
    """
    Conservative parsing for VLM evidence.
    Explicit missing-object claims are strong; generic violation is weak.
    """
    text = (response or "").strip()
    compact = text.replace(" ", "").replace("\n", "")
    prefix = compact[:12]

    negative_terms = ("不违规", "无违规", "未发现违规", "没有违规")
    clear_violation_terms = (
        "缺少面罩", "缺失面罩", "没有面罩", "未戴面罩", "未佩戴面罩",
        "缺少烟雾收集器", "缺失烟雾收集器", "没有烟雾收集器", "无烟雾收集器",
        "缺少收集器", "缺失收集器", "没有收集器",
    )

    has_negative = any(term in compact for term in negative_terms)
    has_clear_violation = any(term in compact for term in clear_violation_terms)
    starts_compliant = prefix.startswith("合规")
    starts_violation = prefix.startswith("违规")

    if starts_compliant or has_negative:
        return {
            "violation": False,
            "compliant_prob": 0.95,
            "is_compliant": True,
            "qwen_evidence": "compliant",
        }

    if has_clear_violation:
        return {
            "violation": True,
            "compliant_prob": 0.0,
            "is_compliant": False,
            "qwen_evidence": "clear_violation",
        }

    if starts_violation or ("违规" in compact and not has_negative):
        return {
            "violation": True,
            "compliant_prob": 0.35,
            "is_compliant": False,
            "qwen_evidence": "weak_violation",
        }

    return {
        "violation": False,
        "compliant_prob": 0.5,
        "is_compliant": None,
        "qwen_evidence": "uncertain",
    }


def crop_worker_roi(frame, worker_box, expand_ratio=0.3):
    if worker_box is None:
        return frame
    x1, y1, x2, y2 = worker_box
    h = y2 - y1
    w = x2 - x1
    expand_h = int(h * expand_ratio)
    expand_w = int(w * expand_ratio)
    roi_x1 = max(0, x1 - expand_w)
    roi_y1 = max(0, y1 - expand_h)
    roi_x2 = min(frame.shape[1], x2 + expand_w)
    roi_y2 = min(frame.shape[0], y2 + expand_h)
    return frame[roi_y1:roi_y2, roi_x1:roi_x2]


def crop_qwen_roi(frame, worker_box, mask_box=None, expand_ratio=0.35):
    """
    Crop the worker-centered region for VLM verification.
    The VLM sees fewer irrelevant objects than in the full frame.
    """
    if worker_box is None:
        return frame
    h, w = frame.shape[:2]
    boxes = [worker_box]
    if mask_box is not None:
        boxes.append(mask_box)
    x1 = min(int(b[0]) for b in boxes)
    y1 = min(int(b[1]) for b in boxes)
    x2 = max(int(b[2]) for b in boxes)
    y2 = max(int(b[3]) for b in boxes)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * expand_ratio)
    pad_y = int(bh * expand_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def compute_box_relation_features(worker_box, mask_box, frame_w, frame_h):
    if worker_box is None or mask_box is None:
        return 0.0, 1.0

    wx1, wy1, wx2, wy2 = worker_box
    mx1, my1, mx2, my2 = mask_box
    worker_w = max(1.0, float(wx2 - wx1))
    worker_h = max(1.0, float(wy2 - wy1))
    mask_w = max(1.0, float(mx2 - mx1))
    mask_h = max(1.0, float(my2 - my1))

    area_ratio = np.clip((mask_w * mask_h) / (worker_w * worker_h), 0.0, 1.0)
    mask_cx = (mx1 + mx2) / 2.0
    mask_cy = (my1 + my2) / 2.0
    worker_cx = (wx1 + wx2) / 2.0
    head_cy = wy1 + 0.22 * worker_h
    distance = np.sqrt(((mask_cx - worker_cx) / worker_w) ** 2 + ((mask_cy - head_cy) / worker_h) ** 2)
    return float(area_ratio), float(np.clip(distance, 0.0, 1.0))


def build_micro_gate_features(
        mask_conf, smoke_score, temporal_trend, geometry_score,
        collector_conf, worker_conf, brightness, spark_intensity,
        conflict_k, mask_area_ratio, worker_mask_distance):
    brightness_norm = float(np.clip(brightness / 255.0, 0.0, 1.0))
    return [
        float(np.clip(mask_conf, 0.0, 1.0)),
        float(np.clip(smoke_score, 0.0, 1.0)),
        float(np.clip(temporal_trend, -0.5, 0.5)),
        float(np.clip(geometry_score, 0.0, 1.0)),
        float(np.clip(collector_conf, 0.0, 1.0)),
        float(np.clip(worker_conf, 0.0, 1.0)),
        brightness_norm,
        float(np.clip(spark_intensity, 0.0, 1.0)),
        float(np.clip(conflict_k, 0.0, 1.0)),
        float(np.clip(mask_area_ratio, 0.0, 1.0)),
        float(np.clip(worker_mask_distance, 0.0, 1.0)),
    ]


def detect_single_scale(frame, detector, conf_thres=0.2):
    h, w = frame.shape[:2]
    if max(h, w) > CONFIG["max_detection_size"]:
        scale = CONFIG["max_detection_size"] / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        detection_frame = cv2.resize(frame, (new_w, new_h))
    else:
        detection_frame = frame

    results = detector(detection_frame, conf=conf_thres, verbose=False)
    detections = []

    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        h_scale = h / detection_frame.shape[0]
        w_scale = w / detection_frame.shape[1]
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            orig_box = box.clone()
            orig_box[0] *= w_scale
            orig_box[1] *= h_scale
            orig_box[2] *= w_scale
            orig_box[3] *= h_scale
            detections.append({
                'box': orig_box.cpu().numpy(),
                'conf': conf.item(),
                'cls': int(cls.item())
            })
    return detections


def put_chinese_text(img, text, position, font_size=30, color=(255, 255, 255), max_width=None):
    if not text:
        return img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    draw.text((position[0], position[1]), text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ================= 轻量可学习门控网络定义 =================
class MicroGate(nn.Module):
    def __init__(self, input_dim=None):
        super().__init__()
        input_dim = input_dim or CONFIG.get("micro_gate_input_dim", 11)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# ================= 小MoE专家池 =================
class MicroExpertPool:
    def __init__(self, config):
        self.config = config
        self.expert_weights = {"yolo_expert": 0.6, "temporal_expert": 0.25, "smoke_expert": 0.15}
        self.confidence_history = deque(maxlen=10)
        self.expert_calls = {k: 0 for k in self.expert_weights}
        self.micro_gate_net = None
        self.use_learnable_gate = False
        self.micro_gate_device = "cpu"

    def set_gate_network(self, gate_net, device):
        self.micro_gate_net = gate_net
        self.micro_gate_device = device
        self.use_learnable_gate = gate_net is not None

    def _compute_temporal_trend(self, current_conf):
        self.confidence_history.append(current_conf)
        if len(self.confidence_history) < 3:
            return 0.0
        recent = np.mean(list(self.confidence_history)[-3:])
        earlier = np.mean(list(self.confidence_history)[-6:-3]) if len(self.confidence_history) >= 6 else recent
        trend = recent - earlier
        return np.clip(trend, -0.5, 0.5)

    def yolo_expert(self, mask_conf, has_mask, has_collector):
        self.expert_calls["yolo_expert"] += 1
        if not has_mask:
            return 0.15
        if not has_collector:
            return 0.35
        return min(0.95, max(0.2, mask_conf * 0.8 + 0.2))

    def temporal_expert(self, current_mask_conf):
        self.expert_calls["temporal_expert"] += 1
        if len(self.confidence_history) < 3:
            return 0.5
        recent = np.mean(list(self.confidence_history)[-3:])
        earlier = np.mean(list(self.confidence_history)[-6:-3]) if len(self.confidence_history) >= 6 else recent
        trend = recent - earlier
        if trend < -0.15:
            return 0.3
        elif trend > 0.15:
            return 0.7
        return 0.5

    def smoke_expert(self, smoke_score: float, has_smoke: bool) -> float:
        """
        烟雾专家：输出环境干扰下的视觉可信度基准
        - smoke_score: 0.0(无烟雾) ~ 1.0(浓烟)
        - has_smoke: 是否检测到烟雾区域
        输出范围: [0.25, 0.85]，连续可微，提供强梯度信号
        """
        if not has_smoke or smoke_score < 0.05:
            return 0.85  # 环境清晰，专家给出高置信度基准

        # 指数衰减模拟视觉信息丢失：烟雾越浓，可信度越低
        confidence = 0.85 * np.exp(-3.0 * smoke_score)
        return np.clip(confidence, 0.25, 0.85)

    def fuse_experts(self, yolo_score, temporal_score, smoke_score,
                     yolo_conf=None, smoke_score_raw=None, temporal_trend=None,
                     gate_features=None):
        if not FEATURE_CONFIG.get("enable_micro_experts", False):
            return yolo_score

        use_dynamic = (self.use_learnable_gate and
                       FEATURE_CONFIG.get("enable_micro_gate", False) and
                       self.micro_gate_net is not None and
                       yolo_conf is not None and
                       smoke_score_raw is not None and
                       temporal_trend is not None)

        if use_dynamic:
            with torch.no_grad():
                if gate_features is None:
                    gate_features = [
                        yolo_conf, smoke_score_raw, temporal_trend,
                        yolo_conf, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0
                    ]
                gate_input = torch.FloatTensor([gate_features]).to(self.micro_gate_device)
                raw_weights = self.micro_gate_net(gate_input).squeeze().cpu().numpy().astype(float)

            # Keep D as an adaptive fusion module: trust YOLO more in clean scenes,
            # and release the learned gate mainly under smoke or temporal jumps.
            conservative_weights = np.array([0.72, 0.18, 0.10], dtype=float)
            smoke_start = self.config.get("micro_gate_release_smoke_start", 0.38)
            smoke_span = self.config.get("micro_gate_release_smoke_span", 0.35)
            trend_norm = self.config.get("micro_gate_release_trend_norm", 0.35)
            smoke_stress = np.clip((float(smoke_score_raw) - smoke_start) / max(smoke_span, 1e-6), 0.0, 1.0)
            trend_stress = np.clip(abs(float(temporal_trend)) / max(trend_norm, 1e-6), 0.0, 1.0) * 0.45
            stress = max(float(smoke_stress), float(trend_stress))
            if yolo_conf > 0.65 and smoke_score_raw < 0.45:
                stress *= 0.5

            weights = (1.0 - stress) * conservative_weights + stress * raw_weights
            weights = weights / max(float(weights.sum()), 1e-6)
            w_yolo, w_temporal, w_smoke = weights[0], weights[1], weights[2]
            self.current_dynamic_weights = {"yolo": w_yolo, "temporal": w_temporal, "smoke": w_smoke}
        else:
            w_yolo = self.expert_weights["yolo_expert"]
            w_temporal = self.expert_weights["temporal_expert"]
            w_smoke = self.expert_weights["smoke_expert"]
            self.current_dynamic_weights = {"yolo": w_yolo, "temporal": w_temporal, "smoke": w_smoke}

        fused = (w_yolo * yolo_score + w_temporal * temporal_score + w_smoke * smoke_score)
        if (use_dynamic and smoke_score_raw < 0.35 and yolo_conf is not None and yolo_conf > 0.6):
            fused = max(fused, yolo_score - self.config.get("micro_gate_clean_score_margin", 0.05))
        if use_dynamic and smoke_score_raw < self.config.get("micro_gate_heavy_smoke_threshold", 0.50):
            fused = min(fused, yolo_score + self.config.get("micro_gate_nonheavy_score_margin", 0.08))
        return min(0.95, max(0.1, fused))

    def get_current_weights(self):
        return getattr(self, 'current_dynamic_weights', {"yolo": 0.6, "temporal": 0.25, "smoke": 0.15})

    def get_stats(self):
        total = sum(self.expert_calls.values())
        return {k: v / total if total > 0 else 0 for k, v in self.expert_calls.items()}


# ================= 大MoE场景路由器 =================
class MacroMoERouter:
    def __init__(self, router_model, config):
        self.router = router_model
        self.config = config
        self.smoke_detector = None
        self.scene_history = deque(maxlen=10)
        self.expert_calls = {"weld": 0, "cut": 0, "idle": 0}
        self.entropy_history = deque(maxlen=20)

    def _get_smoke_detector(self):
        if self.smoke_detector is None:
            self.smoke_detector = SmokeMotionDetector(self.config)
        return self.smoke_detector

    def _compute_routing_entropy(self, probs_dict):
        probs = list(probs_dict.values())
        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p + 1e-8)
        return entropy

    def detect_scene(self, frame, current_time):
        brightness = get_brightness(frame)
        is_low_light = brightness < self.config["brightness_threshold"]
        smoke_detector = self._get_smoke_detector()
        smoke_score, has_smoke = smoke_detector.detect(frame, current_time)

        if is_low_light:
            enhanced_frame = enhance_low_light(frame)
        else:
            enhanced_frame = frame

        if enhanced_frame.shape[1] > 640:
            scale = 640 / enhanced_frame.shape[1]
            new_w = 640
            new_h = int(enhanced_frame.shape[0] * scale)
            small_frame = cv2.resize(enhanced_frame, (new_w, new_h))
        else:
            small_frame = enhanced_frame

        results = self.router(small_frame, verbose=False)
        probs = results[0].probs

        weld_prob = 0.0
        cut_prob = 0.0
        for i, name in self.router.names.items():
            name_lower = name.lower()
            if 'weld' in name_lower:
                weld_prob = probs.data[i].item()
            elif 'cut' in name_lower:
                cut_prob = probs.data[i].item()

        routing_entropy = self._compute_routing_entropy({"weld": weld_prob, "cut": cut_prob})
        self.entropy_history.append(routing_entropy)

        if cut_prob > self.config["cut_threshold"]:
            scene = "cut"
            confidence = cut_prob
        elif weld_prob > self.config["weld_threshold_high"]:
            scene = "weld"
            confidence = weld_prob
        elif has_smoke and weld_prob > self.config["weld_threshold_low"]:
            scene = "weld"
            confidence = weld_prob + 0.1
        elif weld_prob > 0.5:
            scene = "weld"
            confidence = weld_prob
        elif cut_prob > 0.3:
            scene = "cut"
            confidence = cut_prob
        else:
            scene = "idle"
            confidence = max(probs.data).item()

        self.expert_calls[scene] = self.expert_calls.get(scene, 0) + 1
        return scene, confidence, has_smoke, smoke_score, routing_entropy

    def get_load_balance(self):
        total = sum(self.expert_calls.values())
        return {k: v / total for k, v in self.expert_calls.items()} if total > 0 else {}

    def get_avg_entropy(self):
        return np.mean(self.entropy_history) if self.entropy_history else 0


# ================= 几何惰性触发器 =================
class GeometricLazyTrigger:
    def __init__(self, config):
        self.config = config
        self.trigger_count = 0
        self.clear_pass_count = 0
        self.clear_fail_count = 0

    def check_ambiguity(self, geometry_score, mask_conf, has_smoke, smoke_score):
        if geometry_score > self.config.get("geometry_clear_pass_threshold", 0.75):
            if mask_conf > 0.5 or geometry_score > 0.85:
                self.clear_pass_count += 1
                return "CLEAR_PASS", f"高分({geometry_score:.2f})"

        if geometry_score < self.config.get("geometry_clear_fail_threshold", 0.25):
            if not (has_smoke and smoke_score > 0.3):
                self.clear_fail_count += 1
                return "CLEAR_FAIL", f"低分({geometry_score:.2f})"

        self.trigger_count += 1
        return "AMBIGUOUS", f"中分({geometry_score:.2f})"

    def get_stats(self):
        total = self.clear_pass_count + self.clear_fail_count + self.trigger_count
        trigger_rate = self.trigger_count / total if total > 0 else 0
        return {"clear_pass": self.clear_pass_count, "clear_fail": self.clear_fail_count,
                "ambiguous": self.trigger_count, "trigger_rate": trigger_rate}


# ================= D-S证据融合 =================
class UncertaintyDSFusion:
    def __init__(self, config):
        self.config = config
        self.fusion_history = deque(maxlen=10)
        self.mask_history = deque(maxlen=5)
        self.base_reliability = {"yolo_mask": 0.85, "yolo_collector": 0.80, "smoke": 0.70, "spark": 0.60}

    def compute_entropy(self, prob):
        if prob <= 0 or prob >= 1:
            return 0.0
        return -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))

    def compute_discount_factor(self, entropy):
        return 0.5 + 0.2 / (1.0 + np.exp(5 * (entropy - 1.0)))

    def compute_dynamic_reliability(self, spark_intensity, brightness, has_smoke):
        reli = self.base_reliability.copy()
        is_dark = brightness < self.config.get("brightness_threshold", 45)

        if is_dark:
            spark_factor = 1.2 if spark_intensity > 0.6 else (1.1 if spark_intensity > 0.3 else 1.0)
        else:
            spark_factor = 0.8 if spark_intensity > 0.6 else (0.9 if spark_intensity > 0.3 else 1.0)

        reli["yolo_mask"] *= spark_factor
        if is_dark and spark_intensity < 0.3:
            reli["yolo_mask"] *= 0.7
        if has_smoke:
            reli["yolo_mask"] *= 0.7

        for k in reli:
            reli[k] = max(0.2, min(0.95, reli[k]))
        return reli

    def dempster_combination(self, m1, m2):
        K = m1["Compliant"] * m2["Omega"] + m1["Omega"] * m2["Compliant"]
        if K >= 1.0:
            return 0.5, K
        belief = (m1["Compliant"] * m2["Compliant"]) / (1 - K)
        return belief, K

    def fuse_uncertainty(self, yolo_conf, qwen_compliant_prob, qwen_available,
                         spark_intensity, brightness, has_smoke):
        self.mask_history.append(yolo_conf)
        reli = self.compute_dynamic_reliability(spark_intensity, brightness, has_smoke)

        h_yolo = self.compute_entropy(yolo_conf)
        alpha_yolo = self.compute_discount_factor(h_yolo)
        m_yolo_compliant = yolo_conf * alpha_yolo * reli["yolo_mask"]
        m_yolo = {"Compliant": m_yolo_compliant, "Omega": 1 - m_yolo_compliant}

        if qwen_available and qwen_compliant_prob is not None:
            h_qwen = 0.1
            alpha_qwen = self.compute_discount_factor(h_qwen)
            m_qwen_compliant = qwen_compliant_prob * alpha_qwen
            m_qwen = {"Compliant": m_qwen_compliant, "Omega": 1 - m_qwen_compliant}
        else:
            m_qwen = {"Compliant": 0.0, "Omega": 1.0}

        fused_belief, conflict_K = self.dempster_combination(m_yolo, m_qwen)

        self.fusion_history.append(fused_belief)
        smoothed_belief = np.mean(list(self.fusion_history)[-3:]) if len(self.fusion_history) >= 3 else fused_belief

        return smoothed_belief, conflict_K

    def fuse_basic(self, yolo_conf, spark_intensity, brightness, has_smoke):
        reli = self.compute_dynamic_reliability(spark_intensity, brightness, has_smoke)
        return yolo_conf * reli["yolo_mask"], 0.0

    def should_trigger_qwen(self, yolo_conf, fused_conf, spark_intensity, has_smoke, brightness):
        diff = abs(yolo_conf - fused_conf)
        is_dark = brightness < self.config.get("brightness_threshold", 45)
        if diff > 0.25:
            return True, f"YOLO与融合差异大(Δ={diff:.2f})"
        if fused_conf < 0.5 and has_smoke:
            return True, f"烟雾+低置信度({fused_conf:.2f})"
        if yolo_conf < 0.4:
            return True, f"YOLO置信度低({yolo_conf:.2f})"
        if not is_dark and spark_intensity > 0.6 and 0.4 < yolo_conf < 0.7:
            return True, "亮光强火花需确认"
        return False, ""


# ================= K-自适应检测平滑器 =================
class KAdaptiveDetectionSmoother:
    def __init__(self, base_window_size=5, min_positive=3, lambda_k=10.0, max_window=20):
        self.base_window_size = base_window_size
        self.min_positive_base = min_positive
        self.lambda_k = lambda_k
        self.max_window = max_window
        self.mask_history = deque(maxlen=max_window)
        self.collector_history = deque(maxlen=max_window)
        for _ in range(base_window_size):
            self.mask_history.append(True)
            self.collector_history.append(True)

    def update(self, has_mask, has_collector, current_conflict_K=0.0):
        self.mask_history.append(has_mask)
        self.collector_history.append(has_collector)

        if FEATURE_CONFIG.get("enable_k_smooth", False):
            dynamic_window = int(self.base_window_size * (1 + self.lambda_k * current_conflict_K))
            dynamic_window = max(self.base_window_size, min(dynamic_window, self.max_window))
        else:
            dynamic_window = self.base_window_size

        recent_masks = list(self.mask_history)[-dynamic_window:]
        recent_collectors = list(self.collector_history)[-dynamic_window:]
        dynamic_thresh = max(1, int(dynamic_window * (self.min_positive_base / self.base_window_size)))
        smoothed_mask = sum(recent_masks) >= dynamic_thresh
        smoothed_collector = sum(recent_collectors) >= dynamic_thresh
        return smoothed_mask, smoothed_collector, dynamic_window


# ================= 状态平滑器 =================
class StateSmoother:
    def __init__(self, window_size=10, min_positive=6, cooldown_frames=15):
        self.window_size = window_size
        self.min_positive = min_positive
        self.cooldown_frames = cooldown_frames
        self.violation_history = deque(maxlen=window_size)
        self.reason_history = deque(maxlen=window_size)
        self.current_state = False
        self.cooldown_counter = 0
        self.consecutive_violations = 0
        self.pending_violations = []
        for _ in range(window_size):
            self.violation_history.append(False)
            self.reason_history.append("")

    def update(self, raw_is_violation, reason=""):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.violation_history.append(raw_is_violation)
            self.reason_history.append(reason if raw_is_violation else "")
            if raw_is_violation:
                self.pending_violations.append((time.time(), reason))
            smoothed_reason = self._get_smoothed_reason()
            return self.current_state, False, smoothed_reason

        self.violation_history.append(raw_is_violation)
        self.reason_history.append(reason if raw_is_violation else "")
        violation_count = sum(self.violation_history)
        old_state = self.current_state

        if violation_count >= self.min_positive:
            self.consecutive_violations += 1
            if self.consecutive_violations >= 2:
                self.current_state = True
                self.consecutive_violations = min(self.consecutive_violations, 3)
        else:
            if self.current_state:
                self.consecutive_violations -= 1
                if self.consecutive_violations <= 0:
                    self.current_state = False
                    self.cooldown_counter = self.cooldown_frames
            else:
                self.consecutive_violations = max(0, self.consecutive_violations - 1)

        state_changed = (old_state != self.current_state)
        smoothed_reason = self._get_smoothed_reason()
        return self.current_state, state_changed, smoothed_reason

    def _get_smoothed_reason(self):
        from collections import Counter
        recent_reasons = list(self.reason_history)[-self.window_size:]
        valid_reasons = [r for r in recent_reasons if r]
        if valid_reasons:
            return Counter(valid_reasons).most_common(1)[0][0]
        return ""


# ================= 火花强度检测器 =================
class SparkIntensityDetector:
    def __init__(self, config):
        self.config = config
        self.prev_frame_gray = None
        self.spark_history = deque(maxlen=30)
        self.current_intensity = 0.0
        self._last_compute_time = 0
        self._cached_intensity = 0.0

    def detect_spark_intensity(self, frame):
        current_time = time.time()
        if current_time - self._last_compute_time < 0.033:
            return self._cached_intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray.copy()
            return 0.0
        frame_diff = cv2.absdiff(gray, self.prev_frame_gray)
        _, diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        diff_ratio = cv2.countNonZero(diff_thresh) / (diff_thresh.shape[0] * diff_thresh.shape[1])
        high_brightness_mask = gray > 200
        high_brightness_ratio = np.sum(high_brightness_mask) / (gray.shape[0] * gray.shape[1])
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
        spark_intensity = (0.4 * min(1.0, diff_ratio * 10) +
                           0.3 * min(1.0, high_brightness_ratio * 20) +
                           0.3 * min(1.0, edge_ratio * 5))
        spark_intensity = min(1.0, spark_intensity)
        self.prev_frame_gray = gray.copy()
        self.spark_history.append(spark_intensity)
        if len(self.spark_history) >= 5:
            self.current_intensity = np.mean(list(self.spark_history)[-5:])
        else:
            self.current_intensity = spark_intensity
        self._cached_intensity = self.current_intensity
        self._last_compute_time = current_time
        return self.current_intensity


# ================= 烟雾检测器 =================
class SmokeMotionDetector:
    def __init__(self, config):
        self.config = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=True)
        self.frame_count = 0
        self.smoke_frame_count = 0
        self.smoke_history = deque(maxlen=10)
        self.last_smoke_time = 0

    def detect(self, frame, current_time):
        if not self.config["enable_smoke_detection"]:
            return 0.0, False
        self.frame_count += 1
        if self.frame_count < self.config["silent_frames"]:
            self.bg_subtractor.apply(frame)
            return 0.0, False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        motion_ratio = cv2.countNonZero(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1])
        texture_score = 0.0
        if motion_ratio > 0.001:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()
            texture_score = min(1.0, laplacian_var / 800.0)
        smoke_score = 0.8 * motion_ratio * 100 + 0.2 * texture_score
        smoke_score = min(1.0, smoke_score)
        self.smoke_history.append(smoke_score)
        smoothed_score = np.mean(self.smoke_history)
        if smoothed_score > self.config["smoke_threshold"]:
            self.smoke_frame_count += 1
        else:
            self.smoke_frame_count = max(0, self.smoke_frame_count - 1)
        has_smoke = self.smoke_frame_count >= self.config["smoke_min_frames"]
        if has_smoke:
            self.last_smoke_time = current_time
        return smoothed_score, has_smoke


# ================= Qwen异步推理 =================
class QwenAsyncInference:
    def __init__(self, timeout=10):
        self.result_queue = queue.Queue()
        self.timeout = timeout
        self.is_running = False
        self.lock = threading.Lock()
        self.last_result = None
        self.pending_future = None
        self.cache = {"valid": False, "expire_time": 0, "data": None, "needs_timestamp": False}

    def call_async(self, frame, context_text=""):
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True

        def _inference():
            try:
                result = self._run_inference(frame, context_text=context_text)
                with self.lock:
                    self.cache = {
                        "valid": True,
                        "expire_time": time.time() + CONFIG["cache_duration"],
                        "data": result,
                        "needs_timestamp": True
                    }
                    self.last_result = result
                self.result_queue.put(result)
            except Exception as e:
                print(f"Qwen推理错误: {e}")
                self.result_queue.put({"violation": False, "reason": f"推理失败: {e}"})
            finally:
                with self.lock:
                    self.is_running = False
                    self.pending_future = None

        if model_container.executor:
            self.pending_future = model_container.executor.submit(_inference)
        else:
            thread = threading.Thread(target=_inference, daemon=True)
            thread.start()
        return True

    def get_result_safe(self, current_time=None):
        if self.pending_future and self.pending_future.done():
            try:
                _ = self.pending_future.result()
                if current_time is not None and self.cache.get("valid", False):
                    self.cache["expire_time"] = current_time + CONFIG["cache_duration"]
                    self.cache["needs_timestamp"] = False
            except Exception as e:
                print(f"Qwen任务异常: {e}")
            finally:
                self.pending_future = None

        if current_time is None:
            current_time = time.time()
        with self.lock:
            if self.cache.get("valid", False) and self.cache.get("needs_timestamp", False):
                self.cache["expire_time"] = current_time + CONFIG["cache_duration"]
                self.cache["needs_timestamp"] = False
            if self.cache["valid"] and current_time < self.cache["expire_time"]:
                return self.cache["data"]
        return None

    def get_cached_result(self, current_time=None):
        if current_time is None:
            current_time = time.time()
        with self.lock:
            if self.cache.get("valid", False) and self.cache.get("needs_timestamp", False):
                self.cache["expire_time"] = current_time + CONFIG["cache_duration"]
                self.cache["needs_timestamp"] = False
            if self.cache["valid"] and current_time < self.cache["expire_time"]:
                return self.cache["data"]
        return None

    def _run_inference(self, frame, context_text=""):
        if model_container.chat_model is None or model_container.processor is None:
            return {
                "violation": False,
                "reason": "Qwen模型未加载",
                "compliant_prob": 0.5,
                "is_compliant": None,
                "qwen_evidence": "not_loaded",
            }

        try:
            target_size = CONFIG["qwen_image_size"]
            h, w = frame.shape[:2]
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            new_h = (new_h // 28) * 28
            new_w = (new_w // 28) * 28
            if new_h == 0:
                new_h = 28
            if new_w == 0:
                new_w = 28

            small_frame = cv2.resize(frame, (new_w, new_h))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            prompt = """电焊作业安全判断规则：
- 合规：工人正确佩戴面罩(你只要看见面罩的框基本包裹住面部即可，只要不是拿在手上或者顶在头顶上或者放在地上这种完全远离工人的场景都算合规)，并且有烟雾收集器
- 违规：缺少面罩 或 缺少烟雾收集器
请直接回答"合规"或"违规" ，如果违规请给出缺少面罩或者缺失烟雾收集器"""
            if context_text:
                prompt += "\n\n视觉检测提示：" + context_text

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

            with torch.no_grad():
                text_inputs = model_container.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = model_container.processor(
                    text=[text_inputs], images=[frame_rgb],
                    return_tensors="pt", padding=True
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                generated_ids = model_container.chat_model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                    num_beams=1, use_cache=True
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in
                    zip(inputs['input_ids'], generated_ids)
                ]
                response = model_container.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0]

            parsed = parse_qwen_safety_response(response)
            return {
                "violation": parsed["violation"],
                "reason": response.strip(),
                "compliant_prob": parsed["compliant_prob"],
                "is_compliant": parsed["is_compliant"],
                "qwen_evidence": parsed["qwen_evidence"],
            }

        except Exception as e:
            print(f"Qwen推理异常: {e}")
            return {
                "violation": False,
                "reason": f"推理异常: {e}",
                "compliant_prob": 0.5,
                "is_compliant": None,
                "qwen_evidence": "error",
            }

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


# ================= 场景平滑器 =================
class SceneSmoother:
    def __init__(self, window_size=5, min_threshold=2):
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.scene_history = deque(maxlen=window_size)
        self.current_scene = "idle"
        for _ in range(window_size):
            self.scene_history.append("idle")

    def update(self, new_scene):
        self.scene_history.append(new_scene)
        counts = {}
        for scene in self.scene_history:
            counts[scene] = counts.get(scene, 0) + 1
        max_count = 0
        max_scene = self.current_scene
        for scene, count in counts.items():
            if count > max_count:
                max_count = count
                max_scene = scene
        if max_count >= self.min_threshold:
            self.current_scene = max_scene
        return self.current_scene

    def get_scene(self):
        return self.current_scene


# ================= 主系统类 =================
class DualMoESafetySystem:
    def __init__(self, config, detector_weld, detector_cut, macro_router):
        self.config = config
        self.detector_weld = detector_weld
        self.detector_cut = detector_cut
        self.macro_router = macro_router

        self.micro_experts = MicroExpertPool(config)

        self.qwen_cache = {
            "valid": False,
            "expire_time": 0,
            "is_violation": False,
            "reason": "",
            "compliant_prob": 0.5,
            "qwen_evidence": "none",
        }
        self.last_qwen_call_time = 0
        self.qwen_async = QwenAsyncInference(timeout=config["qwen_timeout"])
        self.trigger_count = 0
        self.last_trigger_reason = ""
        self._last_trigger_frame = 0

        self.spark_detector = SparkIntensityDetector(config)
        self.current_spark_intensity = 0.0

        self.detection_smoother = KAdaptiveDetectionSmoother(
            base_window_size=config.get("detection_smooth_window", 5),
            min_positive=config.get("detection_smooth_thresh", 3),
            lambda_k=config.get("k_adaptive_lambda", 10.0),
            max_window=config.get("k_adaptive_max_window", 20)
        )
        self.state_smoother = StateSmoother(
            window_size=config.get("state_smooth_window", 10),
            min_positive=config.get("state_smooth_thresh", 6),
            cooldown_frames=config.get("state_cooldown_frames", 15)
        )

        self.geometric_trigger = GeometricLazyTrigger(config)
        self.uncertainty_fusion = UncertaintyDSFusion(config)
        self.fused_mask_conf = 0.0
        self.fusion_trigger_count = 0

        self.last_qwen_result = None
        self.last_qwen_triggered = False
        self.qwen_future = None

        self.current_conflict_K = 0.0
        self.current_qwen_trigger_K = 0.0
        self.current_dynamic_window = 5
        self.current_expert_weights = {"yolo": 0.6, "temporal": 0.25, "smoke": 0.15}
        self.conf_history = deque(maxlen=10)

    def set_gate_network(self, gate_net, device):
        self.micro_experts.set_gate_network(gate_net, device)

    def _get_violation_reason(self, has_mask, mask_is_proper, has_collector):
        reasons = []
        if not has_mask:
            reasons.append("缺少面罩")
        elif not mask_is_proper:
            reasons.append("未正确佩戴面罩")
        if not has_collector:
            reasons.append("缺少收集器")
        if reasons:
            return "和".join(reasons)
        return ""

    def process_frame(self, frame, current_scene, current_time, frame_count,
                      has_smoke=False, smoke_score=0, routing_entropy=0):
        if current_scene == 'cut':
            return self._process_cutting(frame)
        elif current_scene == 'weld':
            return self._process_welding_enhanced(frame, current_time, frame_count,
                                                  has_smoke, smoke_score, routing_entropy)
        else:
            return self._process_idle(frame)

    def _process_idle(self, frame):
        return frame, "监控中", False, {}, "IDLE", ""

    def _process_cutting(self, frame):
        results = detect_single_scale(frame, self.detector_cut, self.config["conf_thres"])
        display_frame = frame.copy()
        counts = {"worker": 0, "extinguisher": 0}

        for det in results:
            cls, conf, box = det['cls'], det['conf'], det['box']
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if cls == 0:
                counts["worker"] += 1
            elif cls == 1:
                counts["extinguisher"] += 1

        has_worker = counts["worker"] > 0
        has_extinguisher = counts["extinguisher"] > 0

        if has_worker and not has_extinguisher:
            return display_frame, "违规", True, counts, "RULE", "缺少灭火器"
        elif has_worker:
            return display_frame, "合规", False, counts, "RULE", ""
        else:
            return display_frame, "无工人", False, counts, "RULE", ""

    def _process_welding_enhanced(self, frame, current_time, frame_count, has_smoke, smoke_score, routing_entropy):
        frame_h, frame_w = frame.shape[:2]
        results = detect_single_scale(frame, self.detector_weld, self.config["conf_thres"])

        display_frame = frame.copy()
        counts = {"worker": 0, "collector": 0, "mask": 0}
        max_mask_conf = 0.0
        max_worker_conf = 0.0
        max_collector_conf = 0.0
        raw_has_mask = False
        raw_has_collector = False
        worker_box = None
        mask_box = None

        for det in results:
            cls, conf, box = det['cls'], det['conf'], det['box']
            x1, y1, x2, y2 = map(int, box)
            if cls == 1:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cls == 0:
                counts["collector"] += 1
                raw_has_collector = True
                max_collector_conf = max(max_collector_conf, float(conf))
            elif cls == 1:
                counts["worker"] += 1
                if conf > max_worker_conf:
                    max_worker_conf = float(conf)
                    worker_box = [x1, y1, x2, y2]
            elif cls == 2:
                counts["mask"] += 1
                raw_has_mask = True
                if conf > max_mask_conf:
                    max_mask_conf = conf
                    mask_box = [x1, y1, x2, y2]

        has_worker = counts["worker"] > 0
        if not has_worker:
            return display_frame, "无工人", False, counts, "RULE", ""

        # 检测平滑
        smoothed_mask, smoothed_collector, dynamic_window = self.detection_smoother.update(
            raw_has_mask, raw_has_collector, self.current_conflict_K
        )
        self.current_dynamic_window = dynamic_window
        has_mask = smoothed_mask
        has_collector = smoothed_collector

        brightness = get_brightness(frame)
        self.current_spark_intensity = self.spark_detector.detect_spark_intensity(frame)

        self.conf_history.append(max_mask_conf)
        temporal_trend = 0.0
        if len(self.conf_history) >= 2:
            temporal_trend = self.conf_history[-1] - self.conf_history[-2]
        temporal_trend = np.clip(temporal_trend, -0.5, 0.5)

        # 几何分数
        geometry_score = compute_geometry_score(
            worker_box, mask_box, max_mask_conf, frame_w, frame_h
        )
        mask_area_ratio, worker_mask_distance = compute_box_relation_features(
            worker_box, mask_box, frame_w, frame_h
        )
        gate_features = build_micro_gate_features(
            max_mask_conf, smoke_score, temporal_trend, geometry_score,
            max_collector_conf, max_worker_conf, brightness, self.current_spark_intensity,
            self.current_conflict_K, mask_area_ratio, worker_mask_distance
        )

        # 小MoE专家
        yolo_score = self.micro_experts.yolo_expert(max_mask_conf, has_mask, has_collector)
        temporal_score = self.micro_experts.temporal_expert(max_mask_conf)
        smoke_score_val = self.micro_experts.smoke_expert(has_smoke, smoke_score)

        micro_fused_score = self.micro_experts.fuse_experts(
            yolo_score, temporal_score, smoke_score_val,
            yolo_conf=max_mask_conf,
            smoke_score_raw=smoke_score,
            temporal_trend=temporal_trend,
            gate_features=gate_features
        )
        self.current_expert_weights = self.micro_experts.get_current_weights()

        # Qwen触发逻辑
        should_trigger_qwen = False
        trigger_reason = ""
        skip_tag = ""

        if FEATURE_CONFIG.get("enable_lazy_qwen", False) and worker_box is not None:
            should_trigger, tag, reason = should_trigger_qwen_by_geometry(
                geometry_score, max_mask_conf, has_smoke, smoke_score,
                conflict_k=self.current_qwen_trigger_K,
                conflict_aware=FEATURE_CONFIG.get("enable_ds_fusion", False)
            )
            should_trigger_qwen = should_trigger
            trigger_reason = reason
            skip_tag = tag

        # Qwen推理
        qwen_data = None
        if should_trigger_qwen and FEATURE_CONFIG.get("enable_qwen", False):
            cache_valid = current_time < self.qwen_cache.get("expire_time", 0)
            if not cache_valid:
                self.trigger_count += 1
                self.last_trigger_reason = trigger_reason
                self._last_trigger_frame = frame_count
                qwen_frame = crop_qwen_roi(
                    frame, worker_box, mask_box,
                    expand_ratio=self.config.get("qwen_roi_expand", 0.35)
                )
                qwen_context = (
                    f"工人置信度{max_worker_conf:.2f}，面罩置信度{max_mask_conf:.2f}，"
                    f"收集器置信度{max_collector_conf:.2f}，几何分数{geometry_score:.2f}，"
                    f"烟雾分数{smoke_score:.2f}。请以图像为准，检测提示只能作为参考。"
                )
                self.qwen_async.call_async(qwen_frame, context_text=qwen_context)
                self.last_qwen_triggered = True
            qwen_data = self.qwen_async.get_result_safe(current_time=current_time)
            if qwen_data:
                self.qwen_cache = {
                    "valid": True,
                    "expire_time": current_time + self.config["cache_duration"],
                    "is_violation": qwen_data.get("violation", False),
                    "reason": qwen_data.get("reason", ""),
                    "compliant_prob": qwen_data.get("compliant_prob", 0.5),
                    "qwen_evidence": qwen_data.get("qwen_evidence", "unknown"),
                }
        else:
            qwen_data = self.qwen_async.get_cached_result(current_time=current_time)
            if qwen_data:
                self.qwen_cache = {
                    "valid": True,
                    "expire_time": current_time + self.config["cache_duration"],
                    "is_violation": qwen_data.get("violation", False),
                    "reason": qwen_data.get("reason", ""),
                    "compliant_prob": qwen_data.get("compliant_prob", 0.5),
                    "qwen_evidence": qwen_data.get("qwen_evidence", "unknown"),
                }

        if self.qwen_cache.get("valid", False) and current_time >= self.qwen_cache.get("expire_time", 0):
            self.qwen_cache["valid"] = False

        qwen_compliant_prob = self.qwen_cache.get("compliant_prob", 0.5) if self.qwen_cache.get("valid") else None
        qwen_available = self.qwen_cache.get("valid", False) and qwen_compliant_prob is not None
        qwen_state = None
        if qwen_available:
            evidence_for_state = self.qwen_cache.get("qwen_evidence", "unknown")
            qwen_state = {
                "violation": self.qwen_cache.get("is_violation", False),
                "reason": self.qwen_cache.get("reason", ""),
                "compliant_prob": qwen_compliant_prob,
                "is_compliant": (
                    True if evidence_for_state == "compliant"
                    else False if evidence_for_state in ("clear_violation", "weak_violation")
                    else None
                ),
                "qwen_evidence": evidence_for_state,
            }

        if FEATURE_CONFIG.get("enable_ds_fusion", False):
            if qwen_available and qwen_compliant_prob is not None:
                instant_trigger_k = abs(float(geometry_score) - float(qwen_compliant_prob))
                momentum = self.config.get("qwen_trigger_conflict_momentum", 0.6)
                self.current_qwen_trigger_K = (
                    momentum * self.current_qwen_trigger_K
                    + (1.0 - momentum) * instant_trigger_k
                )
            else:
                self.current_qwen_trigger_K *= self.config.get("qwen_trigger_conflict_decay", 0.85)
        else:
            self.current_qwen_trigger_K = 0.0

        # DS证据融合
        if FEATURE_CONFIG.get("enable_ds_fusion", False):
            self.fused_mask_conf, self.current_conflict_K = self.uncertainty_fusion.fuse_uncertainty(
                yolo_conf=micro_fused_score,
                qwen_compliant_prob=qwen_compliant_prob,
                qwen_available=qwen_available,
                spark_intensity=self.current_spark_intensity,
                brightness=brightness,
                has_smoke=has_smoke
            )
        else:
            self.fused_mask_conf = micro_fused_score
            self.current_conflict_K = 0.0

        has_mask_bool = has_mask
        has_collector_bool = has_collector
        compliance_threshold = self.config.get("final_compliance_threshold", 0.5)
        visual_reason = self._get_violation_reason(
            has_mask, geometry_score >= compliance_threshold, has_collector
        )
        visual_violation = bool(visual_reason)

        # 最终融合判定
        qwen_evidence = self.qwen_cache.get("qwen_evidence", "none")
        qwen_can_clear_visual_alarm = (
            geometry_score >= self.config.get("qwen_min_geometry_for_compliance", 0.45)
            or max_mask_conf >= self.config.get("qwen_min_mask_conf_for_compliance", 0.20)
        )
        if qwen_available and qwen_state and FEATURE_CONFIG.get("enable_qwen", False):
            if qwen_state.get("violation", False):
                if qwen_evidence == "clear_violation":
                    qwen_compliant_score = self.config.get("qwen_clear_violation_score", 0.1)
                else:
                    qwen_compliant_score = self.config.get("qwen_weak_violation_score", 0.45)
                    clear_visual = (
                        geometry_score >= self.config.get("qwen_clear_visual_guard_geometry", 0.58)
                        and max_mask_conf >= self.config.get("qwen_clear_visual_guard_mask_conf", 0.45)
                        and not (has_smoke and smoke_score > 0.3)
                    )
                    if clear_visual:
                        qwen_compliant_score = max(qwen_compliant_score, 0.75)
            elif qwen_state.get("is_compliant") is None:
                qwen_compliant_score = geometry_score
            else:
                qwen_compliant_score = self.config.get("qwen_compliant_score", 0.92)
                if not qwen_can_clear_visual_alarm:
                    qwen_compliant_score = min(
                        qwen_compliant_score,
                        self.config.get("qwen_low_geometry_compliance_cap", 0.48)
                    )

            if FEATURE_CONFIG.get("enable_ds_fusion", False):
                qwen_weight = self.config.get("qwen_ds_weight", self.config.get("fusion_qwen_weight", 0.48))
            elif self.config.get("qwen_observe_only_without_ds", True):
                qwen_weight = 0.0
            else:
                qwen_weight = self.config.get("qwen_direct_weight", 0.28)
            if qwen_evidence == "compliant" and not qwen_can_clear_visual_alarm:
                qwen_weight = min(qwen_weight, 0.12)
            geo_weight = 1.0 - qwen_weight
            final_compliant_score = qwen_weight * qwen_compliant_score + geo_weight * geometry_score
        else:
            final_compliant_score = geometry_score

        if FEATURE_CONFIG.get("enable_ds_fusion", False):
            ds_weight = self.config.get("ds_final_weight_when_qwen", 0.35) if qwen_available else self.config.get("ds_final_weight", 0.25)
            final_compliant_score = (
                (1.0 - ds_weight) * final_compliant_score
                + ds_weight * float(np.clip(self.fused_mask_conf, 0.0, 1.0))
            )

        if FEATURE_CONFIG.get("enable_micro_experts", False):
            micro_weight = self.config.get("micro_final_weight", 0.25)
            if self.current_conflict_K > 0.15:
                micro_weight += self.config.get("micro_final_weight_conflict_boost", 0.15)
            micro_weight = float(np.clip(micro_weight, 0.0, 0.45))
            final_compliant_score = (
                (1.0 - micro_weight) * final_compliant_score
                + micro_weight * float(np.clip(micro_fused_score, 0.0, 1.0))
            )
            if micro_fused_score >= 0.70 and qwen_evidence != "clear_violation":
                final_compliant_score = max(
                    final_compliant_score,
                    min(0.95, micro_fused_score - self.config.get("micro_final_safe_margin", 0.04))
                )

        qwen_direct_override_allowed = (
            not self.config.get("qwen_observe_only_without_ds", True)
            and not FEATURE_CONFIG.get("enable_ds_fusion", False)
            and geometry_score >= self.config.get("qwen_direct_override_geometry", 0.68)
            and max_mask_conf >= self.config.get("qwen_direct_override_mask_conf", 0.42)
            and not (has_smoke and smoke_score > 0.45)
        )
        qwen_ds_override_allowed = FEATURE_CONFIG.get("enable_ds_fusion", False)
        if (qwen_available and qwen_evidence == "compliant" and qwen_can_clear_visual_alarm
                and (qwen_ds_override_allowed or qwen_direct_override_allowed)):
            if not has_mask_bool and final_compliant_score >= self.config.get("qwen_override_missing_mask_score", 0.62):
                has_mask_bool = True
            if not has_collector_bool and final_compliant_score >= self.config.get("qwen_override_missing_collector_score", 0.62):
                has_collector_bool = True

        mask_looks_proper = final_compliant_score >= compliance_threshold

        current_reason = self._get_violation_reason(has_mask_bool, mask_looks_proper, has_collector_bool)
        is_violation_now = bool(current_reason)

        smoothed_is_violation, state_changed, smoothed_reason = self.state_smoother.update(
            is_violation_now, current_reason
        )

        if smoothed_is_violation and not smoothed_reason:
            smoothed_reason = current_reason if current_reason else "检测到违规"

        # 模式标识
        if should_trigger_qwen and FEATURE_CONFIG.get("enable_qwen", False):
            mode = "LIVE"
        elif self.qwen_cache.get("valid", False):
            mode = "CACHED"
        else:
            mode = "RULE"

        status = "违规" if smoothed_is_violation else "合规"

        self.last_fusion_info = {
            "geometry_score": geometry_score,
            "qwen_available": qwen_available,
            "qwen_trigger_K": self.current_qwen_trigger_K,
            "qwen_compliant": qwen_state.get("is_compliant") if qwen_state else None,
            "qwen_evidence": qwen_evidence,
            "final_score": final_compliant_score,
            "ds_score": self.fused_mask_conf,
            "micro_score": micro_fused_score,
            "worker_conf": max_worker_conf,
            "collector_conf": max_collector_conf,
            "brightness": brightness,
            "spark_intensity": self.current_spark_intensity,
            "mask_area_ratio": mask_area_ratio,
            "worker_mask_distance": worker_mask_distance,
            "visual_violation": visual_violation,
            "visual_reason": visual_reason,
            "is_proper": mask_looks_proper,
            "skip_tag": skip_tag
        }

        return display_frame, status, smoothed_is_violation, counts, mode, smoothed_reason


# ================= 主函数 =================
def main():
    print("=" * 60)
    print(f"双层MoE安全监控系统 - 实验模式: {EXPERIMENT_MODE}")
    print("消融实验配置:")
    if EXPERIMENT_MODE == "A":
        print("  A: 基线系统 (YOLO检测 + 软几何评分，不调用Qwen)")
    elif EXPERIMENT_MODE == "B":
        print("  B: +惰性Qwen触发 (A + 几何触发Qwen)")
    elif EXPERIMENT_MODE == "C":
        print("  C: +UATS模块 (B + D-S证据融合 + K-自适应平滑)")
    else:
        print("  D: 完整系统 (C + 小MoE门控网络)")
    print("=" * 60)

    os.makedirs(CONFIG["output_folder"], exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    model_container.executor = ThreadPoolExecutor(max_workers=2)

    try:
        router = YOLO(CONFIG["router_model"]).to(device)
        detector_weld = YOLO(CONFIG["weld_detector"]).to(device)
        detector_cut = YOLO(CONFIG["cut_detector"]).to(device)
        print("YOLO模型加载成功")
    except Exception as e:
        print(f"YOLO加载失败: {e}")
        return

    micro_gate_net = None
    if FEATURE_CONFIG.get("enable_micro_gate", False):
        try:
            micro_gate_net = MicroGate()
            micro_gate_path = CONFIG.get("micro_gate_path")
            if micro_gate_path and os.path.exists(micro_gate_path):
                try:
                    micro_gate_net.load_state_dict(torch.load(micro_gate_path, map_location=device, weights_only=False))
                    micro_gate_net.to(device)
                    micro_gate_net.eval()
                    print("门控网络加载成功")
                except RuntimeError:
                    print(f"门控网络维度不匹配，请重新训练: {micro_gate_path}")
                    micro_gate_net = None
            else:
                print(f"门控网络路径不存在: {micro_gate_path}")
        except Exception as e:
            print(f"门控网络加载失败: {e}，使用固定权重")

    macro_router = MacroMoERouter(router, CONFIG)

    if FEATURE_CONFIG.get("enable_qwen", False):
        print("正在加载Qwen模型...")
        try:
            model_container.processor = AutoProcessor.from_pretrained(
                CONFIG["base_model"], trust_remote_code=True
            )
            print("Processor加载成功")

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False
            )
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                CONFIG["base_model"], quantization_config=quantization_config,
                device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
            )
            print("基础模型加载成功")

            chat_model = PeftModel.from_pretrained(base_model, CONFIG["lora_path"])
            print("LoRA加载成功")

            chat_model.eval()
            model_container.chat_model = chat_model
            model_container.device = device
            print("Qwen模型加载成功！")
        except Exception as e:
            print(f"Qwen加载失败: {e}")
            model_container.chat_model = None
            model_container.processor = None
    else:
        print("跳过Qwen加载 (当前模式不使用Qwen)")

    cap = cv2.VideoCapture(CONFIG["video_source"])
    if not cap.isOpened():
        print(f"无法打开视频: {CONFIG['video_source']}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scale = CONFIG["max_display_height"] / orig_h
    display_h = CONFIG["max_display_height"]
    display_w = int(orig_w * scale)
    win_w = display_w + CONFIG["info_panel_width"]
    win_h = display_h

    video_name = Path(CONFIG["video_source"]).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(CONFIG["output_folder"],
                               f"{video_name}_{timestamp}_mode_{EXPERIMENT_MODE}.mp4")

    if CONFIG["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps / CONFIG["skip_frames"] if CONFIG["skip_frames"] > 0 else fps
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (win_w, win_h))

    scene_smoother = SceneSmoother(window_size=5, min_threshold=2)
    safety_system = DualMoESafetySystem(CONFIG, detector_weld, detector_cut, macro_router)

    if micro_gate_net is not None:
        safety_system.set_gate_network(micro_gate_net, device)

    frame_count = 0
    processed_count = 0
    fps_counter = deque(maxlen=30)
    last_time = time.time()

    print("开始处理视频...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % CONFIG["skip_frames"] != 0:
            continue

        processed_count += 1
        current_time = time.time()

        if current_time > last_time:
            instant_fps = 1.0 / (current_time - last_time) * CONFIG["skip_frames"]
            fps_counter.append(instant_fps)
        avg_fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
        last_time = current_time

        if processed_count % CONFIG["router_interval"] == 0:
            scene, confidence, has_smoke, smoke_score, routing_entropy = macro_router.detect_scene(frame, current_time)
            current_scene = scene_smoother.update(scene)
        else:
            current_scene = scene_smoother.get_scene()
            has_smoke = False
            smoke_score = 0.0
            routing_entropy = 0

        processed, status, is_alert, counts, mode, violation_reason = safety_system.process_frame(
            frame, current_scene, current_time, processed_count, has_smoke, smoke_score, routing_entropy
        )

        display = cv2.resize(processed, (display_w, display_h))
        full = np.zeros((display_h, win_w, 3), dtype=np.uint8)
        full[0:display_h, 0:display_w] = display

        panel_x = display_w
        panel_color = (40, 40, 60) if not is_alert else (60, 40, 40)
        cv2.rectangle(full, (panel_x, 0), (panel_x + CONFIG["info_panel_width"], display_h), panel_color, -1)

        y = 25

        full = put_chinese_text(full, f"安全监控系统 (模式{EXPERIMENT_MODE})", (panel_x + 20, y), 16, (0, 255, 255))
        y += 35

        scene_names = {"weld": "电焊", "cut": "切割", "idle": "空闲"}
        scene_text = scene_names.get(current_scene, "未知")
        full = put_chinese_text(full, f"场景：{scene_text}", (panel_x + 20, y), 14)
        y += 28

        mode_names = {"RULE": "规则", "CACHED": "缓存", "LIVE": "实时"}
        mode_text = mode_names.get(mode, mode)
        full = put_chinese_text(full, f"[{mode_text}]", (panel_x + 20, y), 12, (200, 200, 100))
        y += 28

        status_color = (0, 255, 0) if not is_alert else (0, 0, 255)
        full = put_chinese_text(full, status, (panel_x + 20, y), 22, status_color)
        y += 40

        if violation_reason and is_alert:
            full = put_chinese_text(full, f"原因：{violation_reason}", (panel_x + 20, y), 12, (255, 200, 200))
        elif is_alert and not violation_reason:
            full = put_chinese_text(full, "原因：违规", (panel_x + 20, y), 12, (255, 200, 200))
        y += 25

        full = put_chinese_text(full, f"冲突K: {safety_system.current_conflict_K:.3f}",
                                (panel_x + 20, y), 11, (200, 180, 100))
        y += 20

        full = put_chinese_text(full, f"平滑窗口: {safety_system.current_dynamic_window}帧",
                                (panel_x + 20, y), 11, (200, 180, 100))
        y += 20

        weights = safety_system.current_expert_weights
        full = put_chinese_text(full, "专家权重:", (panel_x + 20, y), 11, (180, 180, 180))
        y += 18
        full = put_chinese_text(full,
                                f"YOLO:{weights.get('yolo', 0.6):.2f} 时序:{weights.get('temporal', 0.25):.2f} 烟雾:{weights.get('smoke', 0.15):.2f}",
                                (panel_x + 20, y), 10, (200, 200, 200))
        y += 20

        full = put_chinese_text(full, "统计：", (panel_x + 20, y), 12, (200, 200, 200))
        y += 20

        if current_scene == "weld":
            full = put_chinese_text(full,
                                    f"工人:{counts.get('worker', 0)} 收集器:{counts.get('collector', 0)} 面罩:{counts.get('mask', 0)}",
                                    (panel_x + 20, y), 10, (200, 200, 200))
            y += 18
        elif current_scene == "cut":
            full = put_chinese_text(full, f"工人:{counts.get('worker', 0)} 灭火器:{counts.get('extinguisher', 0)}",
                                    (panel_x + 20, y), 10, (200, 200, 200))
            y += 18
        else:
            full = put_chinese_text(full, "监控中", (panel_x + 20, y), 10, (200, 200, 200))
            y += 18

        y += 10

        progress = 100 * frame_count / total_frames if total_frames > 0 else 0
        full = put_chinese_text(full, f"进度：{progress:.1f}%", (panel_x + 20, y), 11, (150, 150, 150))
        y += 20

        full = put_chinese_text(full, f"FPS：{avg_fps:.1f}", (panel_x + 20, y), 11, (150, 150, 150))
        y += 20

        total_qwen_triggers = safety_system.trigger_count
        full = put_chinese_text(full, f"Qwen触发：{total_qwen_triggers}", (panel_x + 20, y), 11, (255, 165, 0))

        if CONFIG["save_video"]:
            out.write(full)

        if processed_count % 100 == 0:
            print(f"进度: {frame_count}/{total_frames} ({100 * frame_count / total_frames:.1f}%) | "
                  f"场景: {current_scene} | Qwen触发: {total_qwen_triggers}")

    cap.release()
    if CONFIG["save_video"]:
        out.release()

    if model_container.executor:
        model_container.executor.shutdown(wait=True)

    print(f"\n处理完成!")
    print(f"输出视频: {output_path}")
    print(f"Qwen触发次数: {safety_system.trigger_count}")
    print(f"实验模式 {EXPERIMENT_MODE} 运行完毕")


if __name__ == "__main__":
    main()
