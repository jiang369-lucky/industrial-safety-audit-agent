"""
安全监控系统 - 基于双层MoE架构的智能焊接/切割作业安全监控系统

【系统简介】
本系统是一个面向工业焊接/切割场景的智能安全监控系统，通过双层MoE（混合专家）架构，
结合YOLO目标检测和Qwen视觉语言模型，实现对工人安全装备佩戴情况的实时检测与违规判断。

【应用场景】
- 焊接作业：检测工人是否正确佩戴电焊面罩、是否开启烟雾收集器
- 切割作业：检测工人是否配备灭火器
- 空闲状态：系统待机监控


【一、核心功能】

1. 场景识别（大MoE场景路由器）
   自动识别当前作业场景（焊接/切割/空闲），并调用对应的检测专家。
   - 焊接场景：启动完整的面罩+收集器检测流程
   - 切割场景：启动灭火器检测流程
   - 空闲场景：低功耗监控模式

2. 焊接场景检测
   采用双层MoE架构，分为以下层次：

   (1) 第一层：YOLO目标检测
       - 检测工人位置、面罩、烟雾收集器
       - 输出检测框坐标和置信度
       - 每帧实时检测，不使用缓存（保证实时性）
       - 检测平滑器：5帧中至少3帧检测到才判定为存在，防止单帧误检

   (2) 第二层：几何惰性触发器
       - 判断面罩是否在工人头部区域（基于几何位置）
       - 若面罩在头部且置信度高（>0.6）→ CLEAR_PASS，无需Qwen介入
       - 若面罩位置异常或置信度低 → AMBIGUOUS，触发Qwen复核

   (3) 第三层：不确定性D-S证据融合
       - 融合YOLO检测结果和Qwen判断结果
       - 处理光照不足、火花干扰、烟雾遮挡等复杂情况
       - 计算检测结果的可信度（熵权折扣）

   (4) 第四层：小MoE专家集群
       - YOLO专家：基于当前帧的目标检测
       - 时序专家：分析面罩置信度的变化趋势（判断是否在摘面罩）
       - 烟雾专家：评估烟雾遮挡对检测的影响
       - 加权融合三个专家的结果

3. 切割场景检测
   - 检测工人和灭火器
   - 若有工人但无灭火器 → 违规，原因：“缺少灭火器”
   - 每帧实时检测，不使用缓存

4. 异步Qwen推理
   - 仅在可疑时调用（几何规则判断为AMBIGUOUS或D-S融合判断需要复核）
   - 异步执行，不阻塞主检测流程
   - Qwen结果缓存2秒，避免重复调用（因为Qwen推理较慢）

5. 多级平滑机制
   - 检测平滑：5帧中至少3帧检测到才判定存在，防止单帧误检
   - 状态平滑：10帧中至少6帧违规才输出违规，防止状态跳变
   - 场景平滑：5帧中至少2帧为同一场景才切换，防止场景误判

6. 风险感知自适应缓存（仅针对Qwen结果）
   - 强合规状态：缓存2.0秒（面罩在头部且置信度>0.85）
   - 强违规状态：缓存1.0秒（无面罩或置信度<0.3）
   - 模糊状态：缓存0.3秒（面罩位置异常或中等置信度），高频检查
   - 安全守卫：缓存说合规但YOLO证据不足时，否决缓存，防止漏报
   - YOLO优先：YOLO检测到高置信度面罩时，强制清除违规缓存


【二、电焊违规原因组合规则】

系统根据YOLO检测结果和Qwen判断结果，动态组合违规原因：

| 情况                                        | 显示原因                                    |
|---------------------------------------------|---------------------------------------------|
| 只有缺少面罩                                | 缺少面罩                                    |
| 只有缺少收集器                              | 缺少收集器                                  |
| 缺少面罩和缺少收集器                        | 缺少面罩和缺少收集器                        |
| 两者都有但面罩未正确佩戴                    | 未正确佩戴面罩                              |
| 未正确佩戴面罩且缺少收集器                  | 未正确佩戴面罩和缺少收集器                  |


【三、技术架构】

- 大MoE：场景路由器，负责场景分类和负载均衡
- 小MoE：三个专家（YOLO专家、时序专家、烟雾专家）加权融合
- D-S证据理论：处理检测结果的不确定性
- 几何规则：判断面罩位置，减少无效VLM调用
- 风险感知缓存：动态TTL策略，平衡精度与效率


【四、输出说明】

- 实时标注视频：在视频画面上框出检测到的目标（工人、面罩、收集器、灭火器）
- 置信度显示：每个检测框旁显示置信度数值
- 违规判断：红色“违规”或绿色“合规”
- 违规原因：具体说明缺少什么装备或佩戴问题
- 统计信息：工人数量、面罩数量、收集器数量、灭火器数量
- 触发次数：Qwen调用次数、融合触发次数


【五、性能优化】

- YOLO检测：每帧实时检测，不缓存（速度快）
- Qwen推理：仅在可疑时异步调用，结果缓存2秒（平衡精度与效率）
- 检测平滑：5帧滑动窗口，防止单帧误检
- 状态平滑：10帧滑动窗口，防止状态跳变
- 场景平滑：5帧滑动窗口，防止场景误判
"""

import cv2
import torch
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

# ================= 实验模式配置 =================
EXPERIMENT_MODE = "D"

FEATURE_CONFIG = {
    "enable_qwen": EXPERIMENT_MODE in ["B", "D"],
    "qwen_every_frame": EXPERIMENT_MODE == "B",
    "enable_lazy_trigger": EXPERIMENT_MODE == "D",
    "enable_entropy_discount": EXPERIMENT_MODE in ["C", "D"],
    "enable_uncertainty_ds": EXPERIMENT_MODE == "D",
    "enable_geometric_filter": EXPERIMENT_MODE in ["A", "C", "D"],
    "enable_dual_moe": EXPERIMENT_MODE == "D",
    "enable_micro_experts": EXPERIMENT_MODE == "D",
}

# ================= 配置区域 =================
CONFIG = {
    "router_model": r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router\exp2_raw_yolo_aug\weights\best.pt",
    "weld_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_weld_expert\weights\best.pt",
    "cut_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_cut_expert\weights\best.pt",
    "base_model": r"D:\yolo\Spark-Gated MoE Framework\model\Qwen3-VL-4B-Instruct",
    "lora_path": r"D:\yolo\Spark-Gated MoE Framework\LlamaFactory-main\saves\qwen3-vl-weld-3epoch",

    "video_source": r"D:\yolo\Spark-Gated MoE Framework\test_video\RWL_VID_00014.mp4",
    "conf_thres": 0.25,
    "router_interval": 5,
    "save_dir": "violations",
    "skip_frames": 4,
    "max_detection_size": 640,
    "max_display_height": 900,
    "info_panel_width": 300,

    "qwen_timeout": 10,
    "qwen_image_size": 512,
    "use_8bit": True,
    "check_interval": 2.0,
    "check_interval_min": 1.0,
    "check_interval_max": 3.0,
    "cache_duration": 2.0,  # Qwen结果缓存时间
    "mask_conf_thresh": 0.6,

    "detection_smooth_window": 5,
    "detection_smooth_thresh": 3,
    "state_smooth_window": 10,
    "state_smooth_thresh": 6,
    "state_cooldown_frames": 15,
    "head_zone_top_ratio": 0.25,
    "head_zone_bottom_ratio": 0.65,

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
    "weld_threshold_low": 0.30,

    "output_folder": r"D:\yolo\Spark-Gated MoE Framework\detection_results\dual_moe",
    "save_video": True,
    "show_display": True,

    "font_paths": ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"],
}

WINDOW_TITLE = f"Dual_MoE_Mode_{EXPERIMENT_MODE}"


@dataclass
class ModelContainer:
    processor: Any = None
    chat_model: Any = None
    device: str = "cuda"


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


def is_mask_on_head(worker_box, mask_box, head_zone_top_ratio=0.25, head_zone_bottom_ratio=0.65):
    if worker_box is None or mask_box is None:
        return False
    worker_h = worker_box[3] - worker_box[1]
    head_zone_top = worker_box[1] + worker_h * head_zone_top_ratio
    head_zone_bottom = worker_box[1] + worker_h * head_zone_bottom_ratio
    mask_center_y = (mask_box[1] + mask_box[3]) / 2
    return head_zone_top <= mask_center_y <= head_zone_bottom


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


class MicroExpertPool:
    def __init__(self, config):
        self.config = config
        self.expert_weights = {"yolo_expert": 0.6, "temporal_expert": 0.25, "smoke_expert": 0.15}
        self.confidence_history = deque(maxlen=10)
        self.expert_calls = {k: 0 for k in self.expert_weights}

    def yolo_expert(self, mask_conf, has_mask, has_collector):
        self.expert_calls["yolo_expert"] += 1
        if not has_mask:
            return 0.15
        if not has_collector:
            return 0.35
        return min(0.95, max(0.2, mask_conf * 0.8 + 0.2))

    def temporal_expert(self, current_mask_conf):
        self.expert_calls["temporal_expert"] += 1
        self.confidence_history.append(current_mask_conf)
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

    def smoke_expert(self, has_smoke, smoke_score):
        self.expert_calls["smoke_expert"] += 1
        if not has_smoke:
            return 0.5
        if smoke_score > 0.5:
            return 0.4
        return 0.45

    def fuse_experts(self, yolo_score, temporal_score, smoke_score):
        if not FEATURE_CONFIG["enable_micro_experts"]:
            return yolo_score
        fused = (self.expert_weights["yolo_expert"] * yolo_score +
                 self.expert_weights["temporal_expert"] * temporal_score +
                 self.expert_weights["smoke_expert"] * smoke_score)
        return min(0.95, max(0.1, fused))

    def get_stats(self):
        total = sum(self.expert_calls.values())
        return {k: v / total if total > 0 else 0 for k, v in self.expert_calls.items()}


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


class GeometricLazyTrigger:
    def __init__(self, config):
        self.config = config
        self.head_zone_top_ratio = 0.25
        self.head_zone_bottom_ratio = 0.65
        self.mask_conf_thresh = 0.6
        self.trigger_count = 0
        self.clear_pass_count = 0
        self.clear_fail_count = 0

    def check_ambiguity(self, worker_box, mask_box, mask_conf):
        if mask_box is None:
            self.clear_fail_count += 1
            return "CLEAR_FAIL", "未检测到面罩"

        worker_h = worker_box[3] - worker_box[1]
        head_zone_top = worker_box[1] + worker_h * self.head_zone_top_ratio
        head_zone_bottom = worker_box[1] + worker_h * self.head_zone_bottom_ratio
        mask_center_y = (mask_box[1] + mask_box[3]) / 2

        if head_zone_top <= mask_center_y <= head_zone_bottom:
            if mask_conf >= self.mask_conf_thresh:
                self.clear_pass_count += 1
                return "CLEAR_PASS", "面罩在头部区域"
            else:
                self.trigger_count += 1
                return "AMBIGUOUS", f"面罩在头部但置信度低({mask_conf:.2f})"

        self.trigger_count += 1
        return "AMBIGUOUS", "面罩位置异常"

    def get_stats(self):
        total = self.clear_pass_count + self.clear_fail_count + self.trigger_count
        trigger_rate = self.trigger_count / total if total > 0 else 0
        return {"clear_pass": self.clear_pass_count, "clear_fail": self.clear_fail_count,
                "ambiguous": self.trigger_count, "trigger_rate": trigger_rate}


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
            return 0.5
        belief = (m1["Compliant"] * m2["Compliant"]) / (1 - K)
        return belief

    def fuse_uncertainty(self, yolo_conf, qwen_result, qwen_triggered, spark_intensity, brightness, has_smoke):
        self.mask_history.append(yolo_conf)
        reli = self.compute_dynamic_reliability(spark_intensity, brightness, has_smoke)

        h_yolo = self.compute_entropy(yolo_conf)
        alpha_yolo = self.compute_discount_factor(h_yolo)
        m_yolo_compliant = yolo_conf * alpha_yolo * reli["yolo_mask"]
        m_yolo = {"Compliant": m_yolo_compliant, "Omega": 1 - m_yolo_compliant}

        if qwen_triggered and qwen_result is not None:
            h_qwen = 0.1
            alpha_qwen = self.compute_discount_factor(h_qwen)
            qwen_compliant_prob = 0.0 if qwen_result else 1.0
            m_qwen_compliant = qwen_compliant_prob * alpha_qwen
        else:
            h_qwen = 1.0
            alpha_qwen = self.compute_discount_factor(h_qwen)
            m_qwen_compliant = 0.5 * alpha_qwen

        m_qwen = {"Compliant": m_qwen_compliant, "Omega": 1 - m_qwen_compliant}
        fused_belief = self.dempster_combination(m_yolo, m_qwen)

        self.fusion_history.append(fused_belief)
        return np.mean(list(self.fusion_history)[-3:]) if len(self.fusion_history) >= 3 else fused_belief

    def fuse_basic(self, yolo_conf, spark_intensity, brightness, has_smoke):
        reli = self.compute_dynamic_reliability(spark_intensity, brightness, has_smoke)
        return yolo_conf * reli["yolo_mask"]

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


class DetectionSmoother:
    def __init__(self, window_size=5, min_positive=3):
        self.window_size = window_size
        self.min_positive = min_positive
        self.mask_history = deque(maxlen=window_size)
        self.collector_history = deque(maxlen=window_size)
        for _ in range(window_size):
            self.mask_history.append(True)
            self.collector_history.append(True)

    def update(self, has_mask, has_collector):
        self.mask_history.append(has_mask)
        self.collector_history.append(has_collector)
        smoothed_mask = sum(self.mask_history) >= self.min_positive
        smoothed_collector = sum(self.collector_history) >= self.min_positive
        return smoothed_mask, smoothed_collector


class StateSmoother:
    def __init__(self, window_size=10, min_positive=6, cooldown_frames=15):
        self.window_size = window_size
        self.min_positive = min_positive
        self.cooldown_frames = cooldown_frames
        self.violation_history = deque(maxlen=window_size)
        self.current_state = False
        self.cooldown_counter = 0
        self.consecutive_violations = 0
        self.pending_violations = []
        for _ in range(window_size):
            self.violation_history.append(False)

    def update(self, raw_is_violation, reason=""):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.violation_history.append(raw_is_violation)
            if raw_is_violation:
                self.pending_violations.append((time.time(), reason))
            return self.current_state, False
        self.violation_history.append(raw_is_violation)
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
        return self.current_state, state_changed


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


class QwenAsyncInference:
    def __init__(self, timeout=10):
        self.result_queue = queue.Queue()
        self.timeout = timeout
        self.is_running = False
        self.lock = threading.Lock()
        self.last_result = None

    def call_async(self, frame):
        with self.lock:
            if self.is_running:
                return
            self.is_running = True

        def _inference():
            try:
                result = self._run_inference(frame)
                self.result_queue.put(result)
                self.last_result = result
            except Exception as e:
                print(f"Qwen推理错误: {e}")
                self.result_queue.put({"violation": False, "reason": f"推理失败: {e}"})
            finally:
                with self.lock:
                    self.is_running = False

        thread = threading.Thread(target=_inference, daemon=True)
        thread.start()

    def _run_inference(self, frame):
        if model_container.chat_model is None or model_container.processor is None:
            return {"violation": False, "reason": "Qwen模型未加载"}

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
- 合规：工人正确佩戴面罩(你只要看见面罩的框基本包裹住面部即可，不需要观察他们佩戴是否十分正确，只要不是拿在手上或者顶在头顶上这种非常错误的场景都算合规)，并且有烟雾收集器
- 违规：缺少面罩 或 缺少烟雾收集器
请直接回答"合规"或"违规" ，如果违规请给出缺少面罩或者缺失烟雾收集器"""

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

            violation = "违规" in response
            return {"violation": violation, "reason": response.strip()}

        except Exception as e:
            print(f"Qwen推理异常: {e}")
            return {"violation": False, "reason": f"推理异常: {e}"}

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


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


class DualMoESafetySystem:
    def __init__(self, config, detector_weld, detector_cut, macro_router):
        self.config = config
        self.detector_weld = detector_weld
        self.detector_cut = detector_cut
        self.macro_router = macro_router

        self.micro_experts = MicroExpertPool(config)

        # Qwen结果缓存（只缓存Qwen结果）
        self.qwen_cache = {"valid": False, "expire_time": 0, "is_violation": False, "reason": ""}
        self.last_qwen_call_time = 0
        self.qwen_async = QwenAsyncInference(timeout=config["qwen_timeout"])
        self.trigger_count = 0
        self.last_trigger_reason = ""
        self._last_trigger_frame = 0

        self.spark_detector = SparkIntensityDetector(config)
        self.current_spark_intensity = 0.0

        # 平滑器
        self.detection_smoother = DetectionSmoother(
            window_size=config.get("detection_smooth_window", 5),
            min_positive=config.get("detection_smooth_thresh", 3)
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

    # ================= 原因生成器函数 =================
    def _get_violation_reason(self, has_mask, mask_is_proper, has_collector):
        """根据当前帧的检测状态，生成标准化的违规原因字符串"""
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
            return self._process_welding(frame, current_time, frame_count,
                                         has_smoke, smoke_score, routing_entropy)
        else:
            return self._process_idle(frame)

    def _process_idle(self, frame):
        return frame, "监控中", False, {}, "IDLE", ""

    def _process_cutting(self, frame):
        # 切割场景：每帧实时检测，不使用缓存
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

    def _process_welding(self, frame, current_time, frame_count, has_smoke, smoke_score, routing_entropy):
        # 电焊场景：每帧实时YOLO检测，不使用缓存
        results = detect_single_scale(frame, self.detector_weld, self.config["conf_thres"])

        display_frame = frame.copy()
        counts = {"worker": 0, "collector": 0, "mask": 0}
        max_mask_conf = 0.0
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
            elif cls == 1:
                counts["worker"] += 1
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

        # 使用检测平滑器平滑面罩和收集器的检测结果（防止单帧误检）
        smoothed_mask, smoothed_collector = self.detection_smoother.update(raw_has_mask, raw_has_collector)
        has_mask = smoothed_mask
        has_collector = smoothed_collector

        brightness = get_brightness(frame)
        self.current_spark_intensity = self.spark_detector.detect_spark_intensity(frame)

        yolo_score = self.micro_experts.yolo_expert(max_mask_conf, has_mask, has_collector)
        temporal_score = self.micro_experts.temporal_expert(max_mask_conf)
        smoke_score_val = self.micro_experts.smoke_expert(has_smoke, smoke_score)
        micro_fused_score = self.micro_experts.fuse_experts(yolo_score, temporal_score, smoke_score_val)

        if FEATURE_CONFIG["enable_uncertainty_ds"]:
            self.fused_mask_conf = self.uncertainty_fusion.fuse_uncertainty(
                yolo_conf=micro_fused_score,
                qwen_result=self.last_qwen_result,
                qwen_triggered=self.last_qwen_triggered,
                spark_intensity=self.current_spark_intensity,
                brightness=brightness,
                has_smoke=has_smoke
            )
        elif FEATURE_CONFIG["enable_entropy_discount"]:
            self.fused_mask_conf = self.uncertainty_fusion.fuse_basic(
                yolo_conf=micro_fused_score,
                spark_intensity=self.current_spark_intensity,
                brightness=brightness,
                has_smoke=has_smoke
            )
        else:
            self.fused_mask_conf = micro_fused_score

        # 判断面罩是否在头部（用于几何规则）
        mask_on_head = is_mask_on_head(worker_box, mask_box,
                                       CONFIG["head_zone_top_ratio"],
                                       CONFIG["head_zone_bottom_ratio"])

        # 获取几何状态
        geo_status, geo_reason = self.geometric_trigger.check_ambiguity(
            worker_box=worker_box, mask_box=mask_box, mask_conf=micro_fused_score
        )

        # ========== 核心电焊违规判断逻辑 ==========

        # 1. 获取基础检测状态
        has_mask_bool = has_mask
        has_collector_bool = has_collector

        # 2. 初步判断面罩是否"看起来"正常（用于生成初始原因）
        mask_looks_proper = False
        if has_mask_bool and mask_box is not None and worker_box is not None:
            if max_mask_conf > 0.6 and mask_on_head:
                mask_looks_proper = True

        # 3. 生成当前帧的原始原因
        current_reason = self._get_violation_reason(has_mask_bool, mask_looks_proper, has_collector_bool)
        is_violation_now = bool(current_reason)

        # 4. 处理Qwen逻辑
        final_reason = current_reason
        final_is_violation = is_violation_now

        should_trigger_qwen = False
        trigger_reason = ""

        if FEATURE_CONFIG["qwen_every_frame"]:
            should_trigger_qwen = True
            trigger_reason = "每帧调用(B模式)"
        elif FEATURE_CONFIG["enable_lazy_trigger"] and worker_box is not None:
            if geo_status == "AMBIGUOUS":
                should_fusion_trigger, fusion_reason = self.uncertainty_fusion.should_trigger_qwen(
                    yolo_conf=micro_fused_score, fused_conf=self.fused_mask_conf,
                    spark_intensity=self.current_spark_intensity, has_smoke=has_smoke, brightness=brightness
                )
                if should_fusion_trigger:
                    should_trigger_qwen = True
                    trigger_reason = f"几何:{geo_reason}+融合:{fusion_reason}"
                    self.fusion_trigger_count += 1

        self.last_qwen_triggered = False

        if should_trigger_qwen and FEATURE_CONFIG["enable_qwen"]:
            cache_valid = current_time < self.qwen_cache.get("expire_time", 0)
            if not cache_valid:
                self.trigger_count += 1
                self.last_trigger_reason = trigger_reason
                self._last_trigger_frame = frame_count
                self.qwen_async.call_async(frame)
                self.last_qwen_triggered = True

                qwen_result = self.qwen_async.get_result()
                if qwen_result:
                    qwen_is_violation = qwen_result.get("violation", False)
                    qwen_reason_text = qwen_result.get("reason", "")

                    if qwen_is_violation:
                        if "缺少面罩" in qwen_reason_text:
                            mask_looks_proper = False
                            has_mask_bool = False
                        elif "未正确佩戴" in qwen_reason_text or "佩戴" in qwen_reason_text:
                            mask_looks_proper = False
                            has_mask_bool = True
                    else:
                        if has_mask_bool:
                            mask_looks_proper = True

                    final_reason = self._get_violation_reason(has_mask_bool, mask_looks_proper, has_collector_bool)
                    final_is_violation = bool(final_reason)

                    # 缓存Qwen结果
                    self.qwen_cache = {
                        "valid": True,
                        "expire_time": current_time + self.config["cache_duration"],
                        "is_violation": final_is_violation,
                        "reason": final_reason
                    }
            else:
                # 使用缓存的Qwen结果
                final_reason = self.qwen_cache.get("reason", current_reason)
                final_is_violation = self.qwen_cache.get("is_violation", is_violation_now)
        elif self.qwen_cache.get("valid", False):
            # 使用缓存的Qwen结果
            final_reason = self.qwen_cache.get("reason", current_reason)
            final_is_violation = self.qwen_cache.get("is_violation", is_violation_now)

        # 5. 安全守卫：YOLO检测到面罩时，强制清除错误的违规缓存
        if final_is_violation and has_mask_bool and max_mask_conf > 0.6:
            # YOLO明确检测到面罩，但缓存说违规，可能是Qwen误判，强制清除
            if "缺少面罩" in final_reason:
                final_reason = self._get_violation_reason(has_mask_bool, mask_looks_proper, has_collector_bool)
                final_is_violation = bool(final_reason)
                self.qwen_cache["valid"] = False

        # 6. 平滑处理
        smoothed_is_violation, state_changed = self.state_smoother.update(final_is_violation, final_reason)

        # 7. 确定模式
        if should_trigger_qwen:
            mode = "LIVE"
        elif self.qwen_cache.get("valid", False):
            mode = "CACHED"
        else:
            mode = "RULE"

        status = "违规" if smoothed_is_violation else "合规"
        return display_frame, status, smoothed_is_violation, counts, mode, final_reason


def main():
    print("\n" + "=" * 70)
    print("双层MoE安全监控系统 - 实时检测版（仅缓存Qwen结果）")
    print(f"实验模式: {EXPERIMENT_MODE}")
    print("=" * 70)

    os.makedirs(CONFIG["output_folder"], exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    try:
        router = YOLO(CONFIG["router_model"]).to(device)
        detector_weld = YOLO(CONFIG["weld_detector"]).to(device)
        detector_cut = YOLO(CONFIG["cut_detector"]).to(device)
        print("YOLO模型加载成功")
    except Exception as e:
        print(f"YOLO加载失败: {e}")
        return

    macro_router = MacroMoERouter(router, CONFIG)

    if FEATURE_CONFIG["enable_qwen"]:
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
            print("8-bit基础模型加载成功")

            chat_model = PeftModel.from_pretrained(base_model, CONFIG["lora_path"])
            print("LoRA加载成功")

            chat_model.eval()
            model_container.chat_model = chat_model
            model_container.device = device
            print("Qwen-VL加载成功！")
        except Exception as e:
            print(f"Qwen加载失败: {e}")
            model_container.chat_model = None
            model_container.processor = None
    else:
        print("跳过Qwen加载（当前模式不需要）")

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
                               f"{video_name}_Mode_{EXPERIMENT_MODE}_{timestamp}.mp4")

    if CONFIG["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps / CONFIG["skip_frames"] if CONFIG["skip_frames"] > 0 else fps
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (win_w, win_h))

    scene_smoother = SceneSmoother(window_size=5, min_threshold=2)
    safety_system = DualMoESafetySystem(CONFIG, detector_weld, detector_cut, macro_router)

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

        y = 30

        full = put_chinese_text(full, "安全监控", (panel_x + 20, y), 20, (0, 255, 255))
        y += 40

        scene_names = {"weld": "电焊", "cut": "切割", "idle": "空闲"}
        scene_text = scene_names.get(current_scene, "未知")
        full = put_chinese_text(full, f"场景：{scene_text}", (panel_x + 20, y), 16)
        y += 35

        mode_names = {"RULE": "规则", "CACHED": "缓存", "LIVE": "实时"}
        mode_text = mode_names.get(mode, mode)
        full = put_chinese_text(full, f"[{mode_text}]", (panel_x + 20, y), 14, (200, 200, 100))
        y += 35

        status_color = (0, 255, 0) if not is_alert else (0, 0, 255)
        full = put_chinese_text(full, status, (panel_x + 20, y), 24, status_color)
        y += 45

        if violation_reason and is_alert:
            full = put_chinese_text(full, f"原因：{violation_reason}", (panel_x + 20, y), 14, (255, 200, 200))
            y += 30

        full = put_chinese_text(full, "统计：", (panel_x + 20, y), 14, (200, 200, 200))
        y += 25

        if current_scene == "weld":
            full = put_chinese_text(full, f"  worker: {counts.get('worker', 0)}", (panel_x + 25, y), 12,
                                    (200, 200, 200))
            y += 20
            full = put_chinese_text(full, f"  collector: {counts.get('collector', 0)}", (panel_x + 25, y), 12,
                                    (200, 200, 200))
            y += 20
            full = put_chinese_text(full, f"  mask: {counts.get('mask', 0)}", (panel_x + 25, y), 12, (200, 200, 200))
            y += 20
        elif current_scene == "cut":
            full = put_chinese_text(full, f"  worker: {counts.get('worker', 0)}", (panel_x + 25, y), 12,
                                    (200, 200, 200))
            y += 20
            full = put_chinese_text(full, f"  extinguisher: {counts.get('extinguisher', 0)}", (panel_x + 25, y), 12,
                                    (200, 200, 200))
            y += 20
        else:
            full = put_chinese_text(full, "  监控中", (panel_x + 25, y), 12, (200, 200, 200))
            y += 20

        y += 15

        progress = 100 * frame_count / total_frames if total_frames > 0 else 0
        full = put_chinese_text(full, f"进度：{progress:.1f}%", (panel_x + 20, y), 12, (150, 150, 150))
        y += 25

        full = put_chinese_text(full, f"FPS：{avg_fps:.1f}", (panel_x + 20, y), 12, (150, 150, 150))
        y += 25

        total_qwen_triggers = safety_system.trigger_count
        full = put_chinese_text(full, f"触发：{total_qwen_triggers}", (panel_x + 20, y), 12, (255, 165, 0))

        if CONFIG["save_video"]:
            out.write(full)

        if processed_count % 100 == 0:
            print(
                f"进度: {frame_count}/{total_frames} ({100 * frame_count / total_frames:.1f}%) | 场景: {current_scene}")

    cap.release()
    if CONFIG["save_video"]:
        out.release()

    print(f"\n处理完成!")
    print(f"输出视频: {output_path}")
    print(f"Qwen触发次数: {safety_system.trigger_count}")
    print(f"融合触发次数: {safety_system.fusion_trigger_count}")


if __name__ == "__main__":
    main()