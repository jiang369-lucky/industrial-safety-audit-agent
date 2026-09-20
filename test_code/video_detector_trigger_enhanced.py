# 对于9这种远距离、火花小多烟雾且光线差的视频来说，识别不准确
import cv2
import torch
import os
import time
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 在加载 YOLO 模型前，强制关闭其内部弹窗机制
os.environ["YOLO_VERBOSE"] = "False"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import threading
import queue
import gc
import sys
from pathlib import Path

# ================= 配置区域 =================
CONFIG = {
    # ==================== 模型路径配置 ====================
    "router_model": r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router\exp2_raw_yolo_aug\weights\best.pt",
    # 场景路由模型路径，用于识别当前是电焊/切割/空闲场景
    "weld_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_weld_expert\weights\best.pt",
    # 电焊专家检测模型路径，用于检测电焊场景下的工人、面罩、集尘器
    "cut_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_cut_expert\weights\best.pt",
    # 切割专家检测模型路径，用于检测切割场景下的工人、灭火器
    "base_model": r"D:\yolo\Spark-Gated MoE Framework\model\Qwen3-VL-4B-Instruct",
    # Qwen-VL基础模型路径，4B参数量的多模态大模型
    "lora_path": r"D:\yolo\Spark-Gated MoE Framework\LlamaFactory-main\saves\qwen3-vl-weld-3epoch",
    # LoRA微调权重路径，使Qwen具备电焊安全判断能力


    # ==================== 输入输出配置 ====================
    "video_source": r"D:\yolo\Spark-Gated MoE Framework\test_video\RWL_VID_00014.mp4",
    # 输入视频源路径，支持.mp4、.avi等常见格式

    "conf_thres": 0.25,
    # YOLO检测置信度阈值，低于此值的检测框将被过滤，值越高误报越少但漏检越多

    "router_interval": 5,
    # 场景路由执行间隔（帧数），每5帧执行一次场景分类，减少计算开销

    "save_dir": "violations",
    # 违规截图保存目录，当检测到违规时自动保存当前帧

    "skip_frames": 2,
    # 跳帧处理，每2帧处理1帧，提高处理速度，值为1表示处理所有帧

    "max_detection_size": 640,
    # YOLO检测时的最大图像尺寸，超过此尺寸会等比缩放，平衡检测精度和速度

    "max_display_height": 900,
    # 显示窗口的最大高度，超过此高度会等比缩放，适应不同分辨率屏幕

    "info_panel_width": 400,
    # 右侧信息面板宽度（像素），显示场景、状态、统计等信息

    # ==================== Qwen调用配置 ====================
    "check_interval": 2.0,
    # Qwen定时巡检间隔（秒），在电焊场景下每隔2秒触发一次Qwen推理
    "cache_duration": 2.0,
    # Qwen结果缓存有效期（秒），2秒内重复使用上次结果，避免频繁调用
    "mask_conf_thresh": 0.6,
    # 面罩检测置信度阈值，低于此值认为面罩检测不可靠，可能触发Qwen
    "qwen_timeout": 10,
    # Qwen推理超时时间（秒），超过此时间未返回结果则视为失败
    "qwen_image_size": 512,
    # 传给Qwen的图像尺寸，会等比缩放到此大小，减少推理计算量
    "use_8bit": True,
    # 是否使用8-bit量化加载Qwen模型，大幅降低显存占用，True为启用



    # ==================== 烟雾检测参数 ====================
    "enable_smoke_detection": True,
    # 是否启用烟雾检测功能，True启用，False禁用
    "smoke_threshold": 0.06,
    # 烟雾检测阈值，运动区域占比超过此值判定为可能有烟雾，值越小越敏感
    "smoke_min_frames": 3,
    # 烟雾连续确认帧数，连续3帧都检测到烟雾才确认有烟雾，防止误报
    "smoke_cooldown": 3.0,
    # 烟雾触发冷却时间（秒），同一烟雾事件3秒内不重复触发Qwen
    "silent_frames": 60,
    # MOG2背景建模静默帧数，前60帧只学习背景不检测烟雾，让模型适应场景


    # ==================== 低光增强参数 ====================
    "enable_lowlight_enhance": True,
    # 是否启用低光图像增强，True启用，False禁用
    "brightness_threshold": 45,
    # 亮度阈值，图像平均亮度低于45时触发CLAHE增强，值越大越容易触发
    "clahe_clip_limit": 2.0,
    # CLAHE对比度限制参数，限制对比度放大倍数，值越大增强效果越强


    # ==================== 场景分类阈值 ====================
    "cut_threshold": 0.35,
    # 切割场景判定阈值，切割置信度超过0.35时判定为切割场景（优先级最高）
    "weld_threshold_high": 0.65,
    # 电焊高置信度阈值，电焊置信度超过0.65时直接判定为电焊场景
    "weld_threshold_low": 0.30,
    # 电焊低置信度阈值，配合烟雾检测使用，有烟雾时置信度超过0.30即判定为电焊


    # ==================== 检测结果平滑参数 ====================
    "detection_smooth_window": 5,
    # 检测结果平滑窗口大小（帧数），维护最近5帧的检测结果用于平滑
    "detection_smooth_thresh": 3,
    # 检测结果平滑阈值，窗口内至少3帧检测到才算有（面罩/集尘器），防止单帧漏检


    # ==================== 状态平滑参数 ====================
    "state_smooth_window": 10,
    # 状态平滑窗口大小（帧数），维护最近10帧的违规判定结果
    "state_smooth_thresh": 8,
    # 状态平滑阈值，窗口内至少8帧判定为违规才真正切换到违规状态，防止状态跳变
    "state_cooldown_frames": 15,
    # 状态冷却帧数，从违规恢复到合规后15帧内不会立即跳回违规，防止反复报警


    # ==================== 输出配置 ====================
    "output_folder": r"D:\yolo\Spark-Gated MoE Framework\detection_results\video\trigger_enhanced",
    # 输出文件夹路径，保存处理后的视频文件

    "save_video": True,
    # 是否保存处理后的视频，True保存，False不保存

    "show_display": True,
    # 是否显示实时预览窗口，True显示，False不显示（无头模式运行）

    # ==================== 字体配置 ====================
    "font_paths": [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体字体路径（Windows）
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑字体路径（Windows）
        "C:/Windows/Fonts/simsun.ttc",  # 宋体字体路径（Windows）
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux备用字体
        "./simhei.ttf"  # 当前目录备用字体
    ]
    # 中文字体路径列表，按顺序查找第一个存在的字体用于中文显示
}

CLASS_NAMES = {
    "weld": {0: "collector", 1: "worker", 2: "mask"},
    "cut": {0: "worker", 1: "extinguisher"}
}

WINDOW_TITLE = "Safety_Monitor_System"

# ================= 字体管理 =================
_font_cache = {}


def get_font(font_size=30):
    if font_size in _font_cache:
        return _font_cache[font_size]
    for path in CONFIG["font_paths"]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size, encoding='utf-8')
                _font_cache[font_size] = font
                return font
            except:
                continue
    try:
        font = ImageFont.load_default()
        _font_cache[font_size] = font
        return font
    except:
        return None


def put_chinese_text(img, text, position, font_size=30, color=(255, 255, 255), max_width=None):
    if not text:
        return img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = get_font(font_size)
    if font is None:
        cv2.putText(img, text, (position[0], position[1] + font_size),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img
    lines = []
    if max_width and font_size > 0:
        avg_char_width = font_size * 0.6
        chars_per_line = max(1, int(max_width / avg_char_width))
        current_line = ""
        for char in text:
            if len(current_line) < chars_per_line:
                current_line += char
            else:
                lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)
    else:
        lines = [text]
    y_offset = 0
    for line in lines:
        draw.text((position[0], position[1] + y_offset), line, font=font, fill=color)
        y_offset += int(font_size * 1.2)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ================= 检测结果平滑器 =================
class DetectionSmoother:
    """检测结果平滑器，避免单帧抖动导致的状态跳变"""

    def __init__(self, window_size=5, min_positive=3):
        """
        window_size: 滑动窗口大小（帧数）
        min_positive: 判定为"有"所需的最小正样本数
        """
        self.window_size = window_size
        self.min_positive = min_positive

        # 分别维护面罩和集尘器的历史
        self.mask_history = deque(maxlen=window_size)
        self.collector_history = deque(maxlen=window_size)

        # 初始化历史（默认认为有，避免启动时误报）
        for _ in range(window_size):
            self.mask_history.append(True)
            self.collector_history.append(True)

    def update(self, has_mask, has_collector):
        """更新检测结果，返回平滑后的结果"""
        self.mask_history.append(has_mask)
        self.collector_history.append(has_collector)

        # 计算平滑后的结果：至少 min_positive 帧检测到才认为有
        smoothed_mask = sum(self.mask_history) >= self.min_positive
        smoothed_collector = sum(self.collector_history) >= self.min_positive

        return smoothed_mask, smoothed_collector

    def reset(self):
        """重置平滑器（场景切换时调用）"""
        self.mask_history.clear()
        self.collector_history.clear()
        for _ in range(self.window_size):
            self.mask_history.append(True)
            self.collector_history.append(True)


# ================= 状态平滑器（解决合规/违规跳变）=================
class StateSmoother:
    """判定结果平滑器，避免合规/违规状态频繁跳变"""

    def __init__(self, window_size=10, min_positive=6, cooldown_frames=15):
        """
        window_size: 平滑窗口大小
        min_positive: 判定为违规所需的最小帧数
        cooldown_frames: 从违规恢复合规后的冷却帧数（防止立即跳回违规）
        """
        self.window_size = window_size
        self.min_positive = min_positive
        self.cooldown_frames = cooldown_frames
        self.violation_history = deque(maxlen=window_size)
        self.current_state = False  # False=合规, True=违规
        self.cooldown_counter = 0
        self.consecutive_violations = 0
        self.last_raw_violation = False

        # 初始化历史（默认合规）
        for _ in range(window_size):
            self.violation_history.append(False)

    def update(self, raw_is_violation, reason=""):
        """更新判定结果，返回平滑后的状态和是否发生了状态变化"""
        self.last_raw_violation = raw_is_violation

        # 冷却期：如果处于冷却期，强制保持合规
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            # 但仍然记录历史，为冷却结束后做准备
            self.violation_history.append(raw_is_violation)
            # 冷却期内如果有违规记录，延长冷却
            if raw_is_violation:
                self.cooldown_counter = min(self.cooldown_counter + 3, self.cooldown_frames)
            return False, False

        # 正常平滑
        self.violation_history.append(raw_is_violation)

        # 统计窗口内违规帧数
        violation_count = sum(self.violation_history)
        old_state = self.current_state

        # 判定逻辑：需要连续多帧违规才切换到违规
        if violation_count >= self.min_positive:
            self.consecutive_violations += 1
            # 需要连续两轮窗口都判定违规，才真正切换到违规
            if self.consecutive_violations >= 2:
                if not self.current_state:
                    print(
                        f"[状态平滑] 合规 -> 违规 (窗口违规帧数: {violation_count}/{self.window_size}, 原因: {reason})")
                self.current_state = True
                self.consecutive_violations = min(self.consecutive_violations, 3)
        else:
            # 如果窗口内违规帧数不足，且当前是违规状态，准备恢复合规
            if self.current_state:
                self.consecutive_violations -= 1
                if self.consecutive_violations <= 0:
                    print(f"[状态平滑] 违规 -> 合规 (窗口违规帧数: {violation_count}/{self.window_size})")
                    self.current_state = False
                    # 进入冷却期，防止立即跳回违规
                    self.cooldown_counter = self.cooldown_frames
            else:
                # 当前已是合规，保持
                self.consecutive_violations = max(0, self.consecutive_violations - 1)

        state_changed = (old_state != self.current_state)
        return self.current_state, state_changed

    def get_current_state(self):
        """获取当前平滑后的状态"""
        return self.current_state

    def get_raw_stats(self):
        """获取原始统计信息"""
        return {
            "violation_count": sum(self.violation_history),
            "window_size": self.window_size,
            "cooldown": self.cooldown_counter,
            "consecutive": self.consecutive_violations,
            "raw": self.last_raw_violation
        }

    def reset(self):
        """重置状态平滑器（场景切换时调用）"""
        self.violation_history.clear()
        self.current_state = False
        self.cooldown_counter = 0
        self.consecutive_violations = 0
        for _ in range(self.window_size):
            self.violation_history.append(False)
        print("[状态平滑] 已重置")


# ================= 模型容器 =================
class ModelContainer:
    def __init__(self):
        self.chat_model = None
        self.processor = None
        self.lock = threading.Lock()


model_container = ModelContainer()


# ================= 低光增强 =================
def enhance_low_light(frame):
    """CLAHE低光增强"""
    if not CONFIG["enable_lowlight_enhance"]:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


def get_brightness(frame):
    """计算图像平均亮度"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# ================= MOG2烟雾检测器 =================
class SmokeMotionDetector:
    """使用MOG2背景减除检测烟雾运动"""

    def __init__(self, config):
        self.config = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=16, detectShadows=True
        )
        self.frame_count = 0
        self.smoke_frame_count = 0
        self.smoke_history = deque(maxlen=10)
        self.last_smoke_time = 0

    def detect(self, frame, current_time):
        if not self.config["enable_smoke_detection"]:
            return 0.0, False

        self.frame_count += 1

        # 静默期：前60帧只学习背景，不检测
        if self.frame_count < self.config["silent_frames"]:
            self.bg_subtractor.apply(frame)
            return 0.0, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 应用背景减除
        fg_mask = self.bg_subtractor.apply(gray)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # 计算运动区域占比
        motion_ratio = cv2.countNonZero(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1])

        # 纹理分析（只在有运动时计算）
        texture_score = 0.0
        if motion_ratio > 0.001:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()
            texture_score = min(1.0, laplacian_var / 800.0)

        # 综合烟雾分数
        smoke_score = 0.8 * motion_ratio * 100 + 0.2 * texture_score
        smoke_score = min(1.0, smoke_score)

        self.smoke_history.append(smoke_score)
        smoothed_score = np.mean(self.smoke_history)

        # 连续帧确认
        if smoothed_score > self.config["smoke_threshold"]:
            self.smoke_frame_count += 1
        else:
            self.smoke_frame_count = max(0, self.smoke_frame_count - 1)

        has_smoke = self.smoke_frame_count >= self.config["smoke_min_frames"]
        if has_smoke:
            self.last_smoke_time = current_time

        return smoothed_score, has_smoke


# ================= 增强版场景识别器 =================
class EnhancedSceneRouter:
    def __init__(self, router_model, config):
        self.router = router_model
        self.config = config
        self.smoke_detector = SmokeMotionDetector(config)

    def detect_scene(self, frame, current_time):
        # 1. 亮度检测与低光增强
        brightness = get_brightness(frame)
        is_low_light = brightness < self.config["brightness_threshold"]

        # 2. 烟雾检测（独立于分类，用于辅助判断）
        smoke_score, has_smoke = self.smoke_detector.detect(frame, current_time)

        # 3. 基础视觉分类（使用增强后的图像）
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

        # 获取各类别概率
        weld_prob = 0.0
        cut_prob = 0.0

        for i, name in self.router.names.items():
            if 'weld' in name.lower():
                weld_prob = probs.data[i].item()
            elif 'cut' in name.lower():
                cut_prob = probs.data[i].item()

        # ===== 三级递进决策 =====
        # 第一优先级：切割场景
        if cut_prob > self.config["cut_threshold"]:
            return "cut", cut_prob, has_smoke, smoke_score

        # 第二优先级：高置信度电焊
        if weld_prob > self.config["weld_threshold_high"]:
            return "weld", weld_prob, has_smoke, smoke_score

        # 第三优先级：烟雾辅助判断
        if has_smoke and weld_prob > self.config["weld_threshold_low"]:
            return "weld", weld_prob + 0.1, has_smoke, smoke_score

        # 第四优先级：正常判断
        if weld_prob > 0.5:
            return "weld", weld_prob, has_smoke, smoke_score
        elif cut_prob > 0.3:
            return "cut", cut_prob, has_smoke, smoke_score

        return "idle", max(probs.data).item(), has_smoke, smoke_score


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


# ================= 单尺度检测 =================
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


# ================= Qwen推理 =================
def run_qwen_inference_safe(frame, timeout=10):
    with model_container.lock:
        if model_container.chat_model is None or model_container.processor is None:
            return {"violation": False, "reason": "Qwen未加载"}

    result_queue = queue.Queue()

    def inference_task():
        try:
            target_size = CONFIG["qwen_image_size"]
            h, w = frame.shape[:2]
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            new_h = (new_h // 28) * 28
            new_w = (new_w // 28) * 28
            if new_h == 0: new_h = 28
            if new_w == 0: new_w = 28

            small_frame = cv2.resize(frame, (new_w, new_h))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            prompt = """电焊作业安全判断规则：
- 合规：工人正确佩戴面罩(你只要看见面罩的框基本包裹住面部即可，不需要观察他们佩戴是否十分正确，只要不是拿在手上或者顶在头顶上这种非常错误的场景都算合规)，并且有烟雾收集器
- 违规：缺少面罩 或 缺少烟雾收集器
请直接回答"合规"或"违规" """

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

            with model_container.lock:
                processor = model_container.processor
                chat_model = model_container.chat_model

                text_inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=[text_inputs], images=[frame_rgb], return_tensors="pt", padding=True)

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = chat_model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]
                response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            violation = "违规" in response
            result_queue.put(("success", {"violation": violation, "reason": response.strip()}))

        except Exception as e:
            print(f"Qwen推理错误: {e}")
            result_queue.put(("error", str(e)))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    thread = threading.Thread(target=inference_task)
    thread.daemon = True
    thread.start()

    try:
        status, result = result_queue.get(timeout=timeout)
        return result if status == "success" else {"violation": False, "reason": "推理失败"}
    except queue.Empty:
        return {"violation": False, "reason": "超时"}


class QwenAsyncInference:
    def __init__(self, timeout=10):
        self.result_queue = queue.Queue()
        self.timeout = timeout
        self.is_running = False
        self.lock = threading.Lock()

    def call_async(self, frame):
        with self.lock:
            if self.is_running:
                return
            self.is_running = True

        def _inference():
            try:
                result = run_qwen_inference_safe(frame, self.timeout)
                self.result_queue.put(result)
            except Exception as e:
                self.result_queue.put({"violation": False, "reason": f"错误"})
            finally:
                with self.lock:
                    self.is_running = False

        thread = threading.Thread(target=_inference)
        thread.daemon = True
        thread.start()

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except:
            return None


# ================= 触发式安全检测系统（带双平滑器）=================
class SafetyTriggerSystem:
    def __init__(self, config, detector_weld, detector_cut):
        self.config = config
        self.detector_weld = detector_weld
        self.detector_cut = detector_cut
        self.prev_scene = 'idle'
        self.qwen_cache = {"valid": False, "expire_time": 0, "is_violation": False, "reason": ""}
        self.last_qwen_call_time = 0
        self.qwen_async = QwenAsyncInference(timeout=config["qwen_timeout"])
        self.trigger_count = 0
        self.last_trigger_reason = ""
        self._scene_triggered = False
        self._last_force_alarm_time = 0
        self._last_low_conf_time = 0
        self._last_trigger_frame = 0
        self.last_smoke_trigger_time = 0

        # 检测结果平滑器（平滑面罩和集尘器的检测结果）
        self.detection_smoother = DetectionSmoother(
            window_size=config.get("detection_smooth_window", 5),
            min_positive=config.get("detection_smooth_thresh", 3)
        )

        # 状态平滑器（平滑最终的合规/违规判定结果）
        self.state_smoother = StateSmoother(
            window_size=config.get("state_smooth_window", 10),
            min_positive=config.get("state_smooth_thresh", 6),
            cooldown_frames=config.get("state_cooldown_frames", 15)
        )

        # 记录上一帧的平滑结果，用于判断状态变化
        self.prev_smoothed_mask = True
        self.prev_smoothed_collector = True
        self.last_display_state = False
        self.last_display_status = "合规"

    def _reset_for_new_scene(self):
        """场景切换时重置所有平滑器"""
        self.detection_smoother.reset()
        self.state_smoother.reset()
        self.prev_smoothed_mask = True
        self.prev_smoothed_collector = True
        print("所有平滑器已重置")

    def process_frame(self, frame, current_scene, current_time, frame_count, has_smoke=False, smoke_score=0):
        if current_scene == 'cut':
            return self._process_cutting(frame)
        elif current_scene == 'weld':
            return self._process_welding(frame, current_time, frame_count, has_smoke, smoke_score)
        else:
            return self._process_idle(frame)

    def _process_idle(self, frame):
        self.prev_scene = 'idle'
        self.qwen_cache["valid"] = False
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
            if cls == 0:
                counts["worker"] += 1
            elif cls == 1:
                counts["extinguisher"] += 1

        has_worker = counts["worker"] > 0
        has_extinguisher = counts["extinguisher"] > 0
        violation_reason = ""

        if has_worker and not has_extinguisher:
            self.prev_scene = 'cut'
            violation_reason = "无灭火器"
            return display_frame, "违规", True, counts, "RULE", violation_reason
        elif has_worker:
            self.prev_scene = 'cut'
            return display_frame, "合规", False, counts, "RULE", ""
        else:
            self.prev_scene = 'cut'
            return display_frame, "无工人", False, counts, "RULE", ""

    def _process_welding(self, frame, current_time, frame_count, has_smoke=False, smoke_score=0):
        results = detect_single_scale(frame, self.detector_weld, self.config["conf_thres"])
        display_frame = frame.copy()
        counts = {"worker": 0, "collector": 0, "mask": 0}
        max_mask_conf = 0.0

        # 原始检测结果
        raw_has_mask = False
        raw_has_collector = False

        for det in results:
            cls, conf, box = det['cls'], det['conf'], det['box']
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 1 else (255, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            if cls == 0:
                counts["collector"] += 1
                raw_has_collector = True
            elif cls == 1:
                counts["worker"] += 1
            elif cls == 2:
                counts["mask"] += 1
                raw_has_mask = True
                max_mask_conf = max(max_mask_conf, conf)

        has_worker = counts["worker"] > 0

        # 如果没有工人，不进行平滑处理，直接返回
        if not has_worker:
            self.prev_scene = 'weld'
            return display_frame, "无工人", False, counts, "CACHED", ""

        # 应用检测结果平滑器
        smoothed_mask, smoothed_collector = self.detection_smoother.update(raw_has_mask, raw_has_collector)

        # 记录状态变化（用于调试）
        if smoothed_mask != self.prev_smoothed_mask:
            print(f"[检测平滑] 面罩: {self.prev_smoothed_mask} -> {smoothed_mask} (原始:{raw_has_mask})")
        if smoothed_collector != self.prev_smoothed_collector:
            print(
                f"[检测平滑] 集尘器: {self.prev_smoothed_collector} -> {smoothed_collector} (原始:{raw_has_collector})")

        self.prev_smoothed_mask = smoothed_mask
        self.prev_smoothed_collector = smoothed_collector

        # 使用平滑后的结果进行判定
        has_mask = smoothed_mask
        has_collector = smoothed_collector

        # ========== 触发决策 ==========
        should_trigger = False
        trigger_reason = ""
        cache_valid = current_time < self.qwen_cache["expire_time"]
        frames_since_last = frame_count - self._last_trigger_frame

        # P0: 安全熔断（使用平滑后的结果）
        if not should_trigger and self.prev_scene == 'weld' and not cache_valid and frames_since_last > 30:
            if not has_mask or not has_collector:
                if (current_time - self._last_force_alarm_time) > 10:
                    should_trigger = True
                    trigger_reason = "缺失面罩/收集器"
                    self._last_force_alarm_time = current_time

        # P1: 场景切换
        if not should_trigger and self.prev_scene == 'idle' and not self._scene_triggered:
            should_trigger = True
            trigger_reason = "场景切换"
            self._scene_triggered = True
            # 场景切换时重置所有平滑器
            self._reset_for_new_scene()

        # P2: 定时巡检
        if not should_trigger and self.prev_scene == 'weld' and not cache_valid and frames_since_last > 30:
            if (current_time - self.last_qwen_call_time) > self.config["check_interval"]:
                should_trigger = True
                trigger_reason = "定时巡检"

        # P3: 置信度波动
        if not should_trigger and self.prev_scene == 'weld' and not cache_valid and frames_since_last > 30:
            if has_mask and max_mask_conf < self.config["mask_conf_thresh"]:
                if (current_time - self._last_low_conf_time) > 5:
                    should_trigger = True
                    trigger_reason = f"置信度{max_mask_conf:.2f}"
                    self._last_low_conf_time = current_time

        # 烟雾触发（有冷却）
        if not should_trigger and has_smoke and (current_time - self.last_smoke_trigger_time) > self.config[
            "smoke_cooldown"]:
            if frames_since_last > 30:
                should_trigger = True
                trigger_reason = f"烟雾检测({smoke_score:.2f})"
                self.last_smoke_trigger_time = current_time

        if should_trigger:
            self.trigger_count += 1
            self.last_trigger_reason = trigger_reason
            self._last_trigger_frame = frame_count
            print(f"触发Qwen ({self.trigger_count}次): {trigger_reason}")
            self.qwen_async.call_async(frame)

        qwen_result = self.qwen_async.get_result()
        if qwen_result:
            self.qwen_cache = {
                "valid": True,
                "expire_time": current_time + self.config["cache_duration"],
                "is_violation": qwen_result["violation"],
                "reason": qwen_result["reason"]
            }
            self.last_qwen_call_time = current_time
            print(f"Qwen结果: {qwen_result['reason']}")

        mode = "LIVE" if should_trigger else ("CACHED" if self.qwen_cache["valid"] else "WAITING")

        # 获取原始违规判定（未经状态平滑）
        if not has_mask or not has_collector:
            raw_is_violation = True
            raw_violation_reason = "缺少面罩" if not has_mask else "缺少收集器"
        elif self.qwen_cache["valid"]:
            raw_is_violation = self.qwen_cache["is_violation"]
            raw_violation_reason = self.qwen_cache["reason"] if raw_is_violation else ""
        else:
            raw_is_violation = False
            raw_violation_reason = ""

        # 应用状态平滑器
        smoothed_is_violation, state_changed = self.state_smoother.update(raw_is_violation, raw_violation_reason)

        # 获取最终的违规原因（只有违规时才显示）
        if smoothed_is_violation:
            if not has_mask or not has_collector:
                final_reason = raw_violation_reason
            elif self.qwen_cache["valid"] and self.qwen_cache["is_violation"]:
                final_reason = self.qwen_cache["reason"]
            else:
                final_reason = "检测到违规行为"
        else:
            final_reason = ""

        status = "违规" if smoothed_is_violation else "合规"

        # 如果状态发生变化，打印日志
        if state_changed:
            print(f"[最终状态] {status} (原因: {final_reason})")

        self.prev_scene = 'weld'
        return display_frame, status, smoothed_is_violation, counts, mode, final_reason


# ================= 加载模型 =================
print("安全监控系统启动 - 双平滑版")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

try:
    router = YOLO(CONFIG["router_model"]).to(device)
    detector_weld = YOLO(CONFIG["weld_detector"]).to(device)
    detector_cut = YOLO(CONFIG["cut_detector"]).to(device)
    print("YOLO模型加载成功")
except Exception as e:
    print(f"YOLO加载失败: {e}")
    sys.exit(1)

enhanced_router = EnhancedSceneRouter(router, CONFIG)

print("正在加载 8-bit 量化的 Qwen-VL...")
try:
    processor = AutoProcessor.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    print("Processor加载成功")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CONFIG["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("8-bit基础模型加载成功")

    chat_model = PeftModel.from_pretrained(
        base_model,
        CONFIG["lora_path"]
    )
    print("LoRA加载成功")

    chat_model.eval()

    model_container.chat_model = chat_model
    model_container.processor = processor
    print("Qwen-VL加载成功！")

except Exception as e:
    print(f"Qwen-VL加载失败：{e}")
    model_container.chat_model = None
    model_container.processor = None


# ================= 视频检测主程序 =================
def video_detection():
    os.makedirs(CONFIG["output_folder"], exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    video_path = Path(CONFIG["video_source"])
    video_name = video_path.stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(CONFIG["output_folder"], f"{video_name}_result_{timestamp}.mp4")

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

    if CONFIG["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps / CONFIG["skip_frames"] if CONFIG["skip_frames"] > 0 else fps
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (win_w, win_h))

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    time.sleep(0.3)
    cv2.destroyAllWindows()

    if CONFIG["show_display"]:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, win_w, win_h)
        print(f"显示窗口已启动: {WINDOW_TITLE}")

    scene_smoother = SceneSmoother(window_size=5, min_threshold=2)
    trigger_system = SafetyTriggerSystem(CONFIG, detector_weld, detector_cut)

    # 烟雾显示平滑
    smoke_display_history = deque(maxlen=10)
    display_smoke = False

    frame_count = 0
    processed_count = 0
    last_raw_scene = "idle"
    fps_counter = deque(maxlen=30)
    last_time = time.time()

    print("\n开始处理...")

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

        # 初始化变量
        has_smoke = False
        smoke_score = 0.0

        if processed_count % CONFIG["router_interval"] == 0:
            raw_scene, confidence, has_smoke, smoke_score = enhanced_router.detect_scene(frame, current_time)

            if raw_scene != last_raw_scene:
                print(f"场景: {last_raw_scene} -> {raw_scene} ({confidence:.2f})")
            last_raw_scene = raw_scene
            current_scene = scene_smoother.update(raw_scene)

            # 烟雾显示平滑
            smoke_display_history.append(has_smoke)
            if len(smoke_display_history) >= 5:
                smoke_ratio = sum(smoke_display_history) / len(smoke_display_history)
                # 切割场景不显示烟雾
                if current_scene == "cut":
                    display_smoke = False
                else:
                    display_smoke = smoke_ratio > 0.6
        else:
            current_scene = scene_smoother.get_scene()

        processed, status, is_alert, counts, mode, violation_reason = trigger_system.process_frame(
            frame, current_scene, current_time, processed_count, has_smoke, smoke_score
        )

        display = cv2.resize(processed, (display_w, display_h))
        full = np.zeros((display_h, win_w, 3), dtype=np.uint8)
        full[0:display_h, 0:display_w] = display

        panel_x = display_w
        panel_color = (40, 40, 60) if not is_alert else (60, 40, 40)
        cv2.rectangle(full, (panel_x, 0), (panel_x + CONFIG["info_panel_width"], display_h), panel_color, -1)

        y = 30
        scene_names = {"weld": "电焊", "cut": "切割", "idle": "空闲"}
        scene_text = scene_names.get(current_scene, "未知")

        full = put_chinese_text(full, "安全监控", (panel_x + 20, y), 24)
        y += 40
        full = put_chinese_text(full, f"场景: {scene_text}", (panel_x + 20, y), 20)
        y += 35

        # 只在电焊场景且检测到烟雾时显示
        if display_smoke and current_scene == "weld":
            full = put_chinese_text(full, f"烟雾中", (panel_x + 20, y), 14, (0, 255, 255))
            y += 20

        mode_colors = {"LIVE": (0, 165, 255), "CACHED": (0, 255, 0), "RULE": (255, 255, 0), "WAITING": (255, 255, 255)}
        full = put_chinese_text(full, f"[{mode}]", (panel_x + 20, y), 16, mode_colors.get(mode, (255, 255, 255)))
        y += 30

        status_color = (0, 255, 0) if not is_alert else (0, 0, 255)
        full = put_chinese_text(full, status, (panel_x + 20, y), 28, status_color)
        y += 45

        # 显示违规原因
        if violation_reason and is_alert:
            full = put_chinese_text(full, f"原因: {violation_reason}", (panel_x + 20, y), 16, (255, 200, 200))
            y += 30

        full = put_chinese_text(full, "统计:", (panel_x + 20, y), 16, (200, 200, 200))
        y += 25
        for obj, cnt in counts.items():
            if cnt > 0:
                colors = {"worker": (255, 255, 0), "extinguisher": (0, 255, 0), "collector": (0, 255, 255),
                          "mask": (255, 0, 255)}
                full = put_chinese_text(full, f"  {obj}: {cnt}", (panel_x + 30, y), 14,
                                        colors.get(obj, (255, 255, 255)))
                y += 20

        progress = (frame_count / total_frames) * 100
        full = put_chinese_text(full, f"进度: {progress:.1f}%", (panel_x + 20, y + 10), 14, (150, 150, 150))
        y += 30
        full = put_chinese_text(full, f"FPS: {avg_fps:.1f}", (panel_x + 20, y), 12, (150, 150, 150))
        y += 18
        full = put_chinese_text(full, f"触发: {trigger_system.trigger_count}", (panel_x + 20, y), 12, (255, 165, 0))

        if CONFIG["save_video"]:
            out.write(full)
        if CONFIG["show_display"]:
            cv2.imshow(WINDOW_TITLE, full)

        if is_alert:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(CONFIG["save_dir"], f"{ts}_violation.jpg"), processed)

        if processed_count % 100 == 0:
            print(f"进度: {frame_count}/{total_frames} ({progress:.1f}%) | 触发: {trigger_system.trigger_count}")

        if CONFIG["show_display"]:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出")
                break

    cap.release()
    if CONFIG["save_video"]:
        out.release()
    if CONFIG["show_display"]:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print(f"\n处理完成! 触发次数: {trigger_system.trigger_count}")
    print(f"输出: {output_path}")


def main():
    print("\n")
    print("安全监控系统")
    print(f"输入视频: {CONFIG['video_source']}")
    print(f"输出文件夹: {CONFIG['output_folder']}")
    print("=" * 50)

    video_detection()

    print("\n程序结束")


if __name__ == "__main__":
    main()