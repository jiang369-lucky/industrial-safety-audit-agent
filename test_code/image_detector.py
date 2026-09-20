import cv2
import torch
import os
import time  # 添加这一行
import numpy as np
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
    "router_model": r"D:\yolo\Spark-Gated MoE Framework\runs\classify\runs\router\weld_cut_router\weights\best.pt",
    "weld_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_weld_expert\weights\best.pt",
    "cut_detector": r"D:\yolo\Spark-Gated MoE Framework\runs\detect\runs\detect\yolo_cut_expert\weights\best.pt",
    "base_model": r"D:\yolo\Spark-Gated MoE Framework\model\Qwen3-VL-4B-Instruct",
    "lora_path": r"D:\yolo\Spark-Gated MoE Framework\LlamaFactory-main\saves\qwen3-vl-weld-3epoch",

    # 运行参数
    "conf_thres": 0.25,
    "save_dir": "violations",

    # 输入输出路径
    "input_folder": r"D:\yolo\Spark-Gated MoE Framework\test_image",
    "output_folder": r"D:\yolo\Spark-Gated MoE Framework\detection_results\images",

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
}

# 类别名称映射
CLASS_NAMES = {
    "weld": {0: "collector", 1: "worker", 2: "mask"},
    "cut": {0: "worker", 1: "extinguisher"}
}

# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


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
        self.spark_frames = []

    def detect_scene(self, frame):
        results = self.router(frame)
        probs = results[0].probs
        top1_idx = probs.top1
        class_name = self.router.names[top1_idx]
        yolo_confidence = probs.top1conf.item()

        has_spark, spark_ratio, spark_count, _ = self.detect_sparks_strict(frame)

        raw_scene = "idle"
        final_confidence = yolo_confidence
        detection_reason = f"YOLO: {class_name}({yolo_confidence:.2f})"

        if 'cut' in class_name.lower():
            if yolo_confidence > self.config["min_cut_confidence"]:
                if has_spark:
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
            if has_spark and spark_ratio > self.config["spark_threshold"] * 2:
                raw_scene = "cut"
                final_confidence = 0.6
                detection_reason = f"强火花检测 (比例: {spark_ratio:.3f})"

        print(f"场景识别: {raw_scene} (置信度: {final_confidence:.2f}) | {detection_reason}")
        return raw_scene, final_confidence

    def detect_sparks_strict(self, frame):
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

        return has_spark, spark_ratio, spark_count, None


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
print("图片检测")

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
    print(" Processor 加载成功")

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


# ================= 处理单张图片 =================
def process_single_image(frame, current_scene):
    result_text = ""
    is_violation = False
    display_frame = frame.copy()

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
            qwen_result = run_qwen_inference_safe(display_frame, CONFIG["qwen_timeout"])
            print(f"Qwen结果: {qwen_result}")

            if qwen_result:
                if "合规" in qwen_result:
                    result_text = "合规"
                elif "违规" in qwen_result:
                    result_text = "违规"
                    is_violation = True
                else:
                    result_text = qwen_result
        else:
            result_text = "无工人"

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

    return display_frame, result_text, is_violation, detection_counts


# ================= 批量处理图片 =================
def batch_process_images():
    # 创建输出目录
    os.makedirs(CONFIG["output_folder"], exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 获取所有图片文件
    input_path = Path(CONFIG["input_folder"])
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    total_images = len(image_files)
    print(f"\n找到 {total_images} 张图片需要处理")
    print(f"输入文件夹: {CONFIG['input_folder']}")
    print(f"输出文件夹: {CONFIG['output_folder']}")
    print("\n" + "=" * 60)

    if total_images == 0:
        print("没有找到图片文件")
        return

    stats = {
        "total": total_images,
        "success": 0,
        "failed": 0,
        "合规": 0,
        "违规": 0,
        "无工人": 0,
        "未知": 0
    }

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{total_images}] 处理: {image_path.name}")

        # 读取图片
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"无法读取图片")
            stats["failed"] += 1
            continue

        # 场景识别
        raw_scene, confidence = enhanced_router.detect_scene(frame)

        # 处理图片
        processed_frame, result_text, is_alert, detection_counts = process_single_image(
            frame, raw_scene
        )

        # 在图片上添加结果
        h, w = processed_frame.shape[:2]

        scene_map = {"weld": "电焊", "cut": "切割", "idle": "空闲"}
        scene_label = scene_map.get(raw_scene, "未知")

        # 在图片左上角添加信息
        y_offset = 30
        processed_frame = put_chinese_text(processed_frame, f"场景: {scene_label}", (10, y_offset), 24, (255, 255, 255))
        y_offset += 40

        result_color = (0, 255, 0) if not is_alert else (0, 0, 255)
        processed_frame = put_chinese_text(processed_frame, f"结果: {result_text}", (10, y_offset), 32, result_color)
        y_offset += 50

        # 添加检测统计
        processed_frame = put_chinese_text(processed_frame, "检测统计:", (10, y_offset), 18, (200, 200, 200))
        y_offset += 25

        for obj, count in detection_counts.items():
            if count > 0:
                color = {"worker": (255, 255, 0), "extinguisher": (0, 255, 0),
                         "collector": (0, 255, 255), "mask": (255, 0, 255)}.get(obj, (255, 255, 255))
                processed_frame = put_chinese_text(processed_frame, f"  {obj}: {count}", (20, y_offset), 16, color)
                y_offset += 22

        # 保存结果
        output_filename = f"{image_path.stem}_result{image_path.suffix}"
        output_path = os.path.join(CONFIG["output_folder"], output_filename)
        cv2.imwrite(output_path, processed_frame)

        # 更新统计
        stats["success"] += 1
        if result_text == "合规":
            stats["合规"] += 1
        elif result_text == "违规":
            stats["违规"] += 1
        elif result_text == "无工人":
            stats["无工人"] += 1
        else:
            stats["未知"] += 1

        # 保存违规图片副本
        if is_alert:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(CONFIG["save_dir"], f"{timestamp}_{raw_scene}_违规.jpg")
            cv2.imwrite(save_path, processed_frame)
            print(f"保存违规截图: {save_path}")

        print(f"场景: {scene_label}, 结果: {result_text}" + (" ⚠️" if is_alert else ""))

        # 每处理10张图片打印一次进度
        if i % 10 == 0 or i == total_images:
            print(f"\n进度: {i}/{total_images} ({i / total_images * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("批量处理完成!")
    print("=" * 60)
    print(f"统计信息:")
    print(f"  总图片数: {stats['total']}")
    print(f"  成功处理: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"\n  检测结果:")
    print(f"    合规: {stats['合规']}")
    print(f"    违规: {stats['违规']}")
    print(f"    无工人: {stats['无工人']}")
    print(f"    未知: {stats['未知']}")
    print("=" * 60)


# ================= 主程序 =================
def main():
    print("\n" + "=" * 60)
    print("图片批量检测系统")
    print("=" * 60)
    print(f"输入文件夹: {CONFIG['input_folder']}")
    print(f"输出文件夹: {CONFIG['output_folder']}")
    print("=" * 60)

    # 确认开始
    response = input("\n开始批量处理? (y/n): ").strip().lower()
    if response == 'y':
        batch_process_images()
    else:
        print("已取消")

    print("\n程序结束")


if __name__ == "__main__":
    main()