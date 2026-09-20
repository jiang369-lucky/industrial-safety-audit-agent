import cv2
import torch
import os
import time
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
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
    # 模型路径
    "router_model": r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router\exp2_raw_yolo_aug\weights\best.pt",
    "weld_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_weld_expert\weights\best.pt",
    "cut_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_cut_expert\weights\best.pt",
    "base_model": r"D:\yolo\Spark-Gated MoE Framework\model\Qwen3-VL-4B-Instruct",
    "lora_path": r"D:\yolo\Spark-Gated MoE Framework\LlamaFactory-main\saves\qwen3-vl-weld-3epoch",

    # 运行参数
    "video_source": r"D:\yolo\Spark-Gated MoE Framework\test_video\RWL_VID_00011.mp4",
    "conf_thres": 0.25,
    "router_interval": 5,
    "scene_smooth_window": 3,
    "min_detection_threshold": 2,
    "qwen_interval": 30,             # 每30帧调用一次Qwen
    "save_dir": "violations",

    # 显示参数 - 适应竖屏
    "max_display_height": 900,  # 最大显示高度
    "info_panel_width": 400,  # 信息面板宽度

    # 增强检测参数
    "enable_spark_detection": True,
    "spark_threshold": 0.02,
    "spark_boost_factor": 1.3,
    "min_cut_confidence": 0.35,
    "spark_min_area": 200,
    "spark_consecutive_frames": 3,

    # Qwen优化参数
    "qwen_max_new_tokens": 32,
    "qwen_use_cache": True,
    "qwen_timeout": 8,
    "qwen_image_size": 224,
    "use_8bit": True,

    # 输出视频参数
    "output_folder": r"D:\yolo\Spark-Gated MoE Framework\detection_results\video",
    "save_video": True,  # 是否保存视频
    "show_display": True,  # 是否实时显示
}

# 类别名称映射
CLASS_NAMES = {
    "weld": {0: "collector", 1: "worker", 2: "mask"},
    "cut": {0: "worker", 1: "extinguisher"}
}


# ================= 中文文本绘制函数 =================
def put_chinese_text(img, text, position, font_size=30, color=(255, 255, 255), max_width=None):
    """使用 PIL 绘制中文文本，支持自动换行"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size, encoding='utf-8')
                break
            except:
                continue

    if font is None:
        cv2.putText(img, text, (position[0], position[1] + font_size),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img

    if max_width:
        chars_per_line = max_width // (font_size // 2)
        lines = []
        current_line = ""
        for char in text:
            if len(current_line) < chars_per_line:
                current_line += char
            else:
                lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)

        y_offset = 0
        for line in lines:
            draw.text((position[0], position[1] + y_offset), line, font=font, fill=color)
            y_offset += font_size + 5
    else:
        draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ================= 增强的场景识别器 =================
class EnhancedSceneRouter:
    def __init__(self, router_model, config):
        self.router = router_model
        self.config = config
        self.scene_history = deque(maxlen=10)
        self.spark_frames = deque(maxlen=config["spark_consecutive_frames"] * 2)

    def detect_scene(self, frame):
        results = self.router(frame)
        probs = results[0].probs
        top1_idx = probs.top1
        class_name = self.router.names[top1_idx]
        yolo_confidence = probs.top1conf.item()

        has_spark, spark_ratio, spark_count, _ = self.detect_sparks_strict(frame)
        self.spark_frames.append(has_spark)
        consecutive_spark = self.check_consecutive_sparks()

        raw_scene = "idle"
        final_confidence = yolo_confidence
        detection_reason = f"YOLO: {class_name}({yolo_confidence:.2f})"

        if 'cut' in class_name.lower():
            if yolo_confidence > self.config["min_cut_confidence"]:
                if has_spark or consecutive_spark:
                    final_confidence = min(1.0, yolo_confidence * self.config["spark_boost_factor"])
                    raw_scene = "cut"
                    detection_reason += f" + 火花确认"
                elif yolo_confidence > 0.5:
                    raw_scene = "cut"
                    detection_reason += f" (高置信度)"

        elif 'weld' in class_name.lower():
            if yolo_confidence > 0.4:
                raw_scene = "weld"

        if raw_scene == "idle" and self.config["enable_spark_detection"]:
            if consecutive_spark and spark_ratio > self.config["spark_threshold"] * 3:
                raw_scene = "cut"
                final_confidence = 0.6
                detection_reason = f"强火花强制切割 (比例: {spark_ratio:.3f})"

        print(f"场景识别: {raw_scene} (置信度: {final_confidence:.2f}) | {detection_reason}")
        return raw_scene, final_confidence

    def detect_sparks_strict(self, frame):
        visual_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([5, 150, 220])
        upper1 = np.array([25, 255, 255])
        lower2 = np.array([0, 0, 230])
        upper2 = np.array([180, 30, 255])
        lower3 = np.array([20, 150, 200])
        upper3 = np.array([35, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask3 = cv2.inRange(hsv, lower3, upper3)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        spark_area = 0
        spark_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.config["spark_min_area"]:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.5:
                        spark_area += area
                        spark_count += 1

        frame_area = frame.shape[0] * frame.shape[1]
        spark_ratio = spark_area / frame_area
        has_spark = spark_ratio > self.config["spark_threshold"] and spark_count > 1

        return has_spark, spark_ratio, spark_count, visual_frame

    def check_consecutive_sparks(self):
        if len(self.spark_frames) < self.config["spark_consecutive_frames"]:
            return False
        recent_frames = list(self.spark_frames)[-self.config["spark_consecutive_frames"]:]
        return all(recent_frames)


# ================= 场景平滑管理器 =================
class SceneSmoother:
    def __init__(self, window_size=3, min_threshold=2):
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.scene_history = deque(maxlen=window_size)
        self.current_scene = "idle"

    def update(self, new_scene, confidence=1.0):
        self.scene_history.append(new_scene)

        counts = {}
        for scene in self.scene_history:
            counts[scene] = counts.get(scene, 0) + 1

        for scene, count in counts.items():
            if count >= self.min_threshold:
                self.current_scene = scene
                break

        return self.current_scene

    def get_scene(self):
        return self.current_scene


# ================= 多尺度检测 =================
def detect_with_multiscale(frame, detector, conf_thres=0.2):
    h, w = frame.shape[:2]
    scales = [1.0, 1.5, 2.0]
    all_results = []

    for scale in scales:
        if scale > 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w > 1920 or new_h > 1080:
                continue
            resized_frame = cv2.resize(frame, (new_w, new_h))
        else:
            resized_frame = frame

        results = detector(resized_frame, conf=conf_thres)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                if scale > 1.0:
                    box = box / scale
                all_results.append({
                    'box': box.cpu().numpy(),
                    'conf': conf.item(),
                    'cls': int(cls.item())
                })

    return all_results


# ================= 预处理图像 =================
def preprocess_for_qwen(frame, max_size=224):
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(frame, (new_w, new_h))


# ================= Qwen推理 =================
def run_qwen_inference_safe(frame_annotated, timeout=8):
    global chat_model, processor

    if chat_model is None or processor is None:
        return "模型未加载"

    result_queue = queue.Queue()

    def inference_task():
        try:
            small_frame = preprocess_for_qwen(frame_annotated, CONFIG["qwen_image_size"])
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            prompt = """电焊作业安全判断规则：
- 合规：工人正确佩戴面罩(你只要看见面罩的框基本包裹住面部即可，不需要观察他们佩戴是否十分正确，只要不是拿在手上或者顶在头顶上这种非常错误的场景都算合规)，并且有烟雾收集器
- 违规：缺少面罩 或 缺少烟雾收集器
请直接回答"合规"或"违规"""

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text_inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            inputs = processor(
                text=[text_inputs],
                images=[frame_rgb],
                return_tensors="pt",
                padding=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = chat_model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["qwen_max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                    use_cache=CONFIG["qwen_use_cache"],
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            if "合规" in response:
                result = "合规"
            elif "违规" in response:
                result = "违规"
            else:
                result = "合规"

            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result_queue.put(("success", result))

        except Exception as e:
            result_queue.put(("error", str(e)))

    thread = threading.Thread(target=inference_task)
    thread.daemon = True
    thread.start()

    try:
        status, result = result_queue.get(timeout=timeout)
        if status == "success":
            return result
        else:
            return "推理失败"
    except queue.Empty:
        return "超时"


# ================= 加载模型 =================
print("视频检测")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

try:
    router = YOLO(CONFIG["router_model"]).to(device)
    detector_weld = YOLO(CONFIG["weld_detector"]).to(device)
    detector_cut = YOLO(CONFIG["cut_detector"]).to(device)
    print(f"YOLO 模型加载完成")
except Exception as e:
    print(f"YOLO 模型加载失败：{e}")
    sys.exit(1)

enhanced_router = EnhancedSceneRouter(router, CONFIG)

chat_model = None
processor = None

print("正在加载 8-bit 量化的 Qwen-VL...")
try:
    processor = AutoProcessor.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    print("Processor 加载成功")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        CONFIG["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("8-bit基础模型加载成功")

    chat_model = PeftModel.from_pretrained(
        base_model,
        CONFIG["lora_path"],
        torch_dtype=torch.float16
    )
    print("LoRA 加载成功")

    chat_model.eval()
    print(f"Qwen-VL 加载成功！")

except Exception as e:
    print(f"Qwen-VL 加载失败：{e}")
    chat_model = None
    processor = None


# ================= 处理帧函数 =================
def process_frame(frame, current_scene, frame_count, last_qwen_result, qwen_queue):
    result_text = ""
    is_violation = False
    display_frame = frame.copy()
    new_qwen_result = last_qwen_result

    detection_counts = {"worker": 0, "extinguisher": 0, "collector": 0, "mask": 0}
    has_worker = False
    has_extinguisher = False

    if current_scene == "weld":
        results = detect_with_multiscale(frame, detector_weld, CONFIG["conf_thres"])

        for det in results:
            box = det['box']
            conf = det['conf']
            cls = det['cls']

            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 1 else (255, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            class_name = CLASS_NAMES["weld"].get(cls, f"class_{cls}")
            label = f"{class_name} {conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cls == 0:
                detection_counts["collector"] += 1
            elif cls == 1:
                detection_counts["worker"] += 1
                has_worker = True
            elif cls == 2:
                detection_counts["mask"] += 1

        if has_worker:
            try:
                while not qwen_queue.empty():
                    new_qwen_result = qwen_queue.get_nowait()
                    print(f"Qwen结果: {new_qwen_result}")
            except:
                pass

            if frame_count % CONFIG["qwen_interval"] == 0:
                if not hasattr(process_frame, "inference_running") or not process_frame.inference_running:
                    print(f"\n触发推理 (帧{frame_count})")
                    process_frame.inference_running = True

                    def callback():
                        result = run_qwen_inference_safe(display_frame, CONFIG["qwen_timeout"])
                        qwen_queue.put(result)
                        process_frame.inference_running = False

                    thread = threading.Thread(target=callback)
                    thread.daemon = True
                    thread.start()

            if new_qwen_result:
                if "合规" in new_qwen_result:
                    result_text = "合规"
                elif "违规" in new_qwen_result:
                    result_text = "违规"
                    is_violation = True
                else:
                    result_text = new_qwen_result
            else:
                result_text = "分析中..."
        else:
            result_text = "无工人"
            new_qwen_result = ""

    elif current_scene == "cut":
        results = detect_with_multiscale(frame, detector_cut, CONFIG["conf_thres"])

        for det in results:
            box = det['box']
            conf = det['conf']
            cls = det['cls']

            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            class_name = CLASS_NAMES["cut"].get(cls, f"class_{cls}")
            label = f"{class_name} {conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cls == 0:
                detection_counts["worker"] += 1
                has_worker = True
            elif cls == 1:
                detection_counts["extinguisher"] += 1
                has_extinguisher = True

        if has_worker:
            if has_extinguisher:
                result_text = f"合规"
            else:
                result_text = f"违规"
                is_violation = True
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "!!! 无灭火器 !!!", (w // 4, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            result_text = "无工人"

    else:
        result_text = "监控中"
        new_qwen_result = ""

    return display_frame, result_text, is_violation, detection_counts, new_qwen_result


process_frame.inference_running = False


# ================= 视频检测主程序 =================
def video_detection():
    # 创建输出目录
    os.makedirs(CONFIG["output_folder"], exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 获取输入视频文件名
    video_path = Path(CONFIG["video_source"])
    video_name = video_path.stem
    #output_video_path = os.path.join(CONFIG["output_folder"], f"{video_name}_result.mp4")
    output_video_path = os.path.join(CONFIG["output_folder"], f"{video_name}_exp2.mp4")

    # 打开输入视频
    cap = cv2.VideoCapture(CONFIG["video_source"])
    if not cap.isOpened():
        print(f"无法打开视频源：{CONFIG['video_source']}")
        return

    # 获取视频信息
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n视频信息:")
    print(f"  分辨率: {orig_w} x {orig_h}")
    print(f"  总帧数: {total_frames}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  输出视频: {output_video_path}")

    # 计算显示尺寸 - 适应竖屏
    scale = CONFIG["max_display_height"] / orig_h
    display_h = CONFIG["max_display_height"]
    display_w = int(orig_w * scale)

    # 窗口总宽度 = 视频宽度 + 信息面板宽度
    window_width = display_w + CONFIG["info_panel_width"]
    window_height = display_h

    # 初始化视频写入器
    if CONFIG["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps,
                              (window_width, window_height))

    # 创建显示窗口
    if CONFIG["show_display"]:
        window_name = '安全监控系统'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)

        # 将窗口移动到屏幕中央
        screen_width = 1920
        screen_height = 1080
        win_x = max(0, (screen_width - window_width) // 2)
        win_y = max(0, (screen_height - window_height) // 2)
        cv2.moveWindow(window_name, win_x, win_y)

    frame_count = 0
    scene_smoother = SceneSmoother(
        window_size=CONFIG["scene_smooth_window"],
        min_threshold=CONFIG["min_detection_threshold"]
    )

    last_raw_scene = "idle"
    qwen_queue = queue.Queue()
    last_qwen_result = ""

    fps_counter = deque(maxlen=30)
    last_time = time.time()

    print(f"\n开始处理视频...")
    print(f"  {'显示窗口: 开启' if CONFIG['show_display'] else '显示窗口: 关闭'}")
    print(f"  {'保存视频: 开启' if CONFIG['save_video'] else '保存视频: 关闭'}")
    print(f"  按 'q' 键提前退出\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

        # 计算FPS
        fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
        fps_counter.append(fps)
        avg_fps = sum(fps_counter) / len(fps_counter)
        last_time = current_time

        # 场景检测
        if frame_count % CONFIG["router_interval"] == 0:
            raw_scene, confidence = enhanced_router.detect_scene(frame)

            if raw_scene != last_raw_scene and raw_scene != "idle":
                print(f"场景: {last_raw_scene} -> {raw_scene}")
                if raw_scene != "weld":
                    last_qwen_result = ""

            last_raw_scene = raw_scene
            current_scene = scene_smoother.update(raw_scene, confidence)

        current_scene = scene_smoother.get_scene()

        # 处理帧
        processed_frame, status_text, is_alert, detection_counts, last_qwen_result = process_frame(
            frame, current_scene, frame_count, last_qwen_result, qwen_queue
        )

        # 创建显示图像
        display_frame = cv2.resize(processed_frame, (display_w, display_h))
        full_display = np.zeros((display_h, window_width, 3), dtype=np.uint8)

        # 左侧：视频帧
        full_display[0:display_h, 0:display_w] = display_frame

        # 右侧：信息面板
        panel_x = display_w
        panel_color = (40, 40, 60) if not is_alert else (60, 40, 40)
        cv2.rectangle(full_display, (panel_x, 0),
                      (panel_x + CONFIG["info_panel_width"], display_h), panel_color, -1)
        cv2.line(full_display, (panel_x, 0), (panel_x, display_h), (100, 100, 100), 2)

        # 显示信息
        scene_map = {"weld": "电焊", "cut": "切割", "idle": "空闲"}
        scene_label = scene_map.get(current_scene, "未知")

        y_pos = 30
        line_height = 30

        # 标题
        full_display = put_chinese_text(full_display, "安全监控", (panel_x + 20, y_pos), 24, (255, 255, 255))
        y_pos += 40

        # 场景
        full_display = put_chinese_text(full_display, f"场景: {scene_label}", (panel_x + 20, y_pos), 20,
                                        (255, 255, 255))
        y_pos += 35

        # 结果
        status_color = (0, 255, 0) if not is_alert else (0, 0, 255)
        full_display = put_chinese_text(full_display, f"{status_text}", (panel_x + 20, y_pos), 28, status_color)
        y_pos += 45

        # 检测统计
        full_display = put_chinese_text(full_display, "检测统计:", (panel_x + 20, y_pos), 16, (200, 200, 200))
        y_pos += 25

        for obj, count in detection_counts.items():
            if count > 0:
                color = {"worker": (255, 255, 0), "extinguisher": (0, 255, 0),
                         "collector": (0, 255, 255), "mask": (255, 0, 255)}.get(obj, (255, 255, 255))
                full_display = put_chinese_text(full_display, f"  {obj}: {count}", (panel_x + 30, y_pos), 14, color)
                y_pos += 20

        # 进度信息
        y_pos += 10
        progress = (frame_count / total_frames) * 100
        full_display = put_chinese_text(full_display, f"进度: {progress:.1f}%", (panel_x + 20, y_pos), 14,
                                        (150, 150, 150))
        y_pos += 20
        full_display = put_chinese_text(full_display, f"帧: {frame_count}/{total_frames}", (panel_x + 20, y_pos), 12,
                                        (150, 150, 150))
        y_pos += 18
        full_display = put_chinese_text(full_display, f"FPS: {avg_fps:.1f}", (panel_x + 20, y_pos), 12, (150, 150, 150))

        # 保存视频帧
        if CONFIG["save_video"]:
            out.write(full_display)

        # 显示
        if CONFIG["show_display"]:
            cv2.imshow(window_name, full_display)

        # 保存违规截图
        if is_alert:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(CONFIG["save_dir"], f"{timestamp}_{current_scene}_违规.jpg")
            cv2.imwrite(save_path, processed_frame)
            print(f"保存违规截图: {save_path}")

        # 每处理10%打印一次进度
        if frame_count % max(1, total_frames // 10) == 0:
            print(f"  进度: {frame_count}/{total_frames} ({progress:.1f}%)")

        # 按键处理
        if CONFIG["show_display"]:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户中断")
                break

    # 释放资源
    cap.release()
    if CONFIG["save_video"]:
        out.release()
    if CONFIG["show_display"]:
        cv2.destroyAllWindows()

    print(f"\n视频处理完成!")
    print(f"  总处理帧数: {frame_count}")
    print(f"  输出视频: {output_video_path}")
    print(f"  违规截图保存到: {CONFIG['save_dir']}")


# ================= 主程序 =================
def main():
    print("\n")
    print("视频检测系统")
    print(f"输入视频: {CONFIG['video_source']}")
    print(f"输出文件夹: {CONFIG['output_folder']}")

    # 确认开始
    response = input("\n开始处理视频? (y/n): ").strip().lower()
    if response == 'y':
        video_detection()
    else:
        print("已取消")

    print("\n程序结束")


if __name__ == "__main__":
    main()