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
    "router_model": r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router\exp2_raw_yolo_aug\weights\best.pt",
    "weld_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_weld_expert\weights\best.pt",
    "cut_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_cut_expert\weights\best.pt",
    "base_model": r"D:\yolo\Spark-Gated MoE Framework\model\Qwen3-VL-4B-Instruct",
    "lora_path": r"D:\yolo\Spark-Gated MoE Framework\LlamaFactory-main\saves\qwen3-vl-weld-3epoch",

    "video_source": r"D:\yolo\Spark-Gated MoE Framework\test_video\RWL_VID_00001.mp4",
    "conf_thres": 0.25,
    "router_interval": 5,
    "save_dir": "violations",

    "skip_frames": 2,
    "max_detection_size": 640,
    "max_display_height": 900,
    "info_panel_width": 400,

    "check_interval": 2.0,
    "cache_duration": 2.0,
    "mask_conf_thresh": 0.6,
    "qwen_timeout": 10,
    "qwen_image_size": 512,
    "use_8bit": True,

    "output_folder": r"D:\yolo\Spark-Gated MoE Framework\detection_results\video\trigger_test",
    "save_video": True,
    "show_display": True,

    "font_paths": [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "./simhei.ttf"
    ]
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
            except Exception:
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


# ================= 模型容器 =================
class ModelContainer:
    def __init__(self):
        self.chat_model = None
        self.processor = None
        self.lock = threading.Lock()


model_container = ModelContainer()


# ================= 场景识别器 =================
class EnhancedSceneRouter:
    def __init__(self, router_model):
        self.router = router_model

    def detect_scene(self, frame):
        if frame.shape[1] > 640:
            scale = 640 / frame.shape[1]
            new_w = 640
            new_h = int(frame.shape[0] * scale)
            small_frame = cv2.resize(frame, (new_w, new_h))
        else:
            small_frame = frame

        results = self.router(small_frame, verbose=False)
        probs = results[0].probs
        top1_idx = probs.top1
        class_name = self.router.names[top1_idx].lower()
        confidence = probs.top1conf.item()

        if 'cut' in class_name:
            return "cut", confidence
        elif 'weld' in class_name:
            return "weld", confidence
        else:
            return "idle", confidence


# ================= 场景平滑器 =================
class SceneSmoother:
    def __init__(self, window_size=5, min_threshold=3, confidence_threshold=0.6):
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.confidence_threshold = confidence_threshold
        self.scene_history = deque(maxlen=window_size)
        self.conf_history = deque(maxlen=window_size)
        self.current_scene = "idle"
        for _ in range(window_size):
            self.scene_history.append("idle")
            self.conf_history.append(0.0)

    def update(self, new_scene, confidence):
        if confidence >= self.confidence_threshold:
            self.scene_history.append(new_scene)
            self.conf_history.append(confidence)
        else:
            self.scene_history.append(self.current_scene)
            self.conf_history.append(confidence)

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


# ================= Qwen推理（修复8-bit加载）=================
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
- 合规：工人正确佩戴面罩(你只要看见面罩的框基本包裹住面部即可，不需要观察他们佩戴是否十分正确，看不到工人的脸就算合规，只要不是拿在手上或者顶在头顶上这种非常错误的场景都算合规)，并且有烟雾收集器
- 违规：缺少面罩 或 缺少烟雾收集器
请直接回答"合规"或"违规"""

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


# ================= 异步Qwen管理器 =================
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
        except queue.Empty:
            return None


# ================= 【修复】触发式安全检测系统 =================
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

    def process_frame(self, frame, current_scene, current_time, frame_count):
        if current_scene == 'cut':
            return self._process_cutting(frame)
        elif current_scene == 'weld':
            return self._process_welding(frame, current_time, frame_count)
        else:
            return self._process_idle(frame)

    def _process_idle(self, frame):
        self.prev_scene = 'idle'
        self.qwen_cache["valid"] = False
        return frame, "监控中", False, {}, "IDLE"

    def _process_cutting(self, frame):
        results = detect_single_scale(frame, self.detector_cut, self.config["conf_thres"])
        display_frame = frame.copy()

        # 【修复】使用字典记录实际数量
        counts = {"worker": 0, "extinguisher": 0}

        for det in results:
            cls, conf, box = det['cls'], det['conf'], det['box']
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # 统计实际数量
            if cls == 0:
                counts["worker"] += 1
            elif cls == 1:
                counts["extinguisher"] += 1

        has_worker = counts["worker"] > 0
        has_extinguisher = counts["extinguisher"] > 0

        if has_worker and not has_extinguisher:
            self.prev_scene = 'cut'
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, "无灭火器", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return display_frame, "违规", True, counts, "RULE"
        elif has_worker:
            self.prev_scene = 'cut'
            return display_frame, "合规", False, counts, "RULE"
        else:
            self.prev_scene = 'cut'
            return display_frame, "无工人", False, counts, "RULE"

    def _process_welding(self, frame, current_time, frame_count):
        results = detect_single_scale(frame, self.detector_weld, self.config["conf_thres"])
        display_frame = frame.copy()

        # 【修复】使用字典记录实际数量
        counts = {"worker": 0, "collector": 0, "mask": 0}
        max_mask_conf = 0.0

        for det in results:
            cls, conf, box = det['cls'], det['conf'], det['box']
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 1 else (255, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # 统计实际数量
            if cls == 0:
                counts["collector"] += 1
            elif cls == 1:
                counts["worker"] += 1
            elif cls == 2:
                counts["mask"] += 1
                max_mask_conf = max(max_mask_conf, conf)

        has_worker = counts["worker"] > 0
        has_collector = counts["collector"] > 0
        has_mask = counts["mask"] > 0

        if not has_worker:
            self.prev_scene = 'weld'
            return display_frame, "无工人", False, counts, "CACHED"

        # 触发决策
        should_trigger = False
        trigger_reason = ""
        cache_valid = current_time < self.qwen_cache["expire_time"]

        # P1: 场景切换 - 从idle切换到weld时触发一次
        if self.prev_scene == 'idle' and not self._scene_triggered:
            should_trigger = True
            trigger_reason = "场景切换"
            self._scene_triggered = True

        # 只有当前是weld且之前也是weld时才考虑其他触发
        elif self.prev_scene == 'weld' and not cache_valid:
            frames_since_last = frame_count - self._last_trigger_frame
            if frames_since_last > 30:
                # P0: 安全熔断
                if not has_mask or not has_collector:
                    if (current_time - self._last_force_alarm_time) > 10:
                        should_trigger = True
                        trigger_reason = "缺失面罩/收集器"
                        self._last_force_alarm_time = current_time

                # P2: 定时巡检
                elif (current_time - self.last_qwen_call_time) > self.config["check_interval"]:
                    should_trigger = True
                    trigger_reason = "定时巡检"

                # P3: 低置信度
                elif has_mask and max_mask_conf < self.config["mask_conf_thresh"]:
                    if (current_time - self._last_low_conf_time) > 5:
                        should_trigger = True
                        trigger_reason = f"置信度{max_mask_conf:.2f}"
                        self._last_low_conf_time = current_time

        # 执行触发
        if should_trigger:
            self.trigger_count += 1
            self.last_trigger_reason = trigger_reason
            self._last_trigger_frame = frame_count
            print(f"触发Qwen ({self.trigger_count}次): {trigger_reason}")
            self.qwen_async.call_async(frame)

        # 获取Qwen结果
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

        # 确定最终状态
        mode = "LIVE" if should_trigger else ("CACHED" if self.qwen_cache["valid"] else "WAITING")

        # 安全熔断优先
        if not has_mask or not has_collector:
            is_alert = True
            status = "违规"
        elif self.qwen_cache["valid"]:
            is_alert = self.qwen_cache["is_violation"]
            status = "违规" if is_alert else "合规"
        else:
            is_alert = False
            status = "分析中"

        self.prev_scene = 'weld'
        return display_frame, status, is_alert, counts, mode


# ================= 加载模型 =================
print("安全监控系统启动")

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

enhanced_router = EnhancedSceneRouter(router)

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
    output_path = os.path.join(CONFIG["output_folder"], f"{video_name}_trigger_{timestamp}.mp4")

    cap = cv2.VideoCapture(CONFIG["video_source"])
    if not cap.isOpened():
        print(f"无法打开视频: {CONFIG['video_source']}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n视频: {orig_w}x{orig_h}, {fps:.1f}fps, {total_frames}帧")

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

    scene_smoother = SceneSmoother(
        window_size=5,
        min_threshold=3,
        confidence_threshold=0.6
    )
    trigger_system = SafetyTriggerSystem(CONFIG, detector_weld, detector_cut)

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

        if processed_count % CONFIG["router_interval"] == 0:
            raw_scene, confidence = enhanced_router.detect_scene(frame)
            if raw_scene != last_raw_scene:
                print(f"原始场景: {last_raw_scene} -> {raw_scene} ({confidence:.2f})")
            last_raw_scene = raw_scene
            current_scene = scene_smoother.update(raw_scene, confidence)
        else:
            current_scene = scene_smoother.get_scene()

        processed, status, is_alert, counts, mode = trigger_system.process_frame(
            frame, current_scene, current_time, processed_count
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

        mode_colors = {"LIVE": (0, 165, 255), "CACHED": (0, 255, 0), "RULE": (255, 255, 0), "WAITING": (255, 255, 255)}
        full = put_chinese_text(full, f"[{mode}]", (panel_x + 20, y), 16, mode_colors.get(mode, (255, 255, 255)))
        y += 30

        status_color = (0, 255, 0) if not is_alert else (0, 0, 255)
        full = put_chinese_text(full, status, (panel_x + 20, y), 28, status_color)
        y += 45

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
    print("安全监控系统 - 智能触发版")
    print(f"输入视频: {CONFIG['video_source']}")
    print(f"输出文件夹: {CONFIG['output_folder']}")


    video_detection()

    print("\n程序结束")


if __name__ == "__main__":
    main()