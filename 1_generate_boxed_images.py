import os
import cv2
from ultralytics import YOLO

# ================= 配置区域 (路径已确认为真) =================
# 项目根目录
PROJECT_ROOT = r"D:\yolo\Spark-Gated MoE Framework"

# 【关键修正】模型路径：补全了重复的 runs/detect 层级
# 电焊模型
WELD_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs/detect/runs/detect/yolo_weld_expert/weights/best.pt")
# 切割模型
CUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs/detect/runs/detect/yolo_cut_expert/weights/best.pt")

# 数据集目录
COT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "dataset_cot")
WELD_SAMPLES_DIR = os.path.join(COT_DATASET_ROOT, "weld_samples")
CUT_SAMPLES_DIR = os.path.join(COT_DATASET_ROOT, "cut_samples")

# 支持的图片格式
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


# ===========================================

def process_folder(folder_path, model, model_name):
    """处理单个文件夹内的所有图片"""
    if not os.path.exists(folder_path):
        print(f"警告：文件夹不存在 -> {folder_path}")
        return 0

    count = 0
    print(f"\n开始处理 [{model_name}] 场景：{folder_path}")

    for filename in os.listdir(folder_path):
        # 跳过已经处理过的 boxed 图片，避免重复处理
        if filename.lower().endswith(IMG_EXTENSIONS) and not filename.endswith('_boxed.jpg'):
            img_path = os.path.join(folder_path, filename)

            # 构造输出文件名 (例如：img_001.jpg -> img_001_boxed.jpg)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_boxed{ext}"
            output_path = os.path.join(folder_path, output_filename)

            try:
                # 1. 加载图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图片：{filename}")
                    continue

                # 2. YOLO 推理
                results = model(img)

                # 3. 绘制检测框
                annotated_img = results[0].plot(line_width=2, font_size=0.6, labels=True)

                # 4. 保存结果
                cv2.imwrite(output_path, annotated_img)
                count += 1
                print(f"已处理：{filename} -> {output_filename}")

            except Exception as e:
                print(f"处理失败 {filename}: {str(e)}")

    print(f"[{model_name}] 场景处理完成！共生成 {count} 张带框图片。")
    return count


def main():
    print("=" * 60)
    print("YOLO 自动标注绘图工具 (CoT 数据准备)")
    print("=" * 60)

    # 检查模型文件是否存在
    if not os.path.exists(WELD_MODEL_PATH):
        print(f"错误：找不到电焊模型！请检查路径：\n{WELD_MODEL_PATH}")
        return
    if not os.path.exists(CUT_MODEL_PATH):
        print(f"错误：找不到切割模型！请检查路径：\n{CUT_MODEL_PATH}")
        return

    # 加载模型
    print("\n正在加载电焊模型...")
    weld_model = YOLO(WELD_MODEL_PATH)

    print("正在加载切割模型...")
    cut_model = YOLO(CUT_MODEL_PATH)

    print("模型加载完毕！开始绘图...\n")

    # 处理电焊样本
    total_weld = process_folder(WELD_SAMPLES_DIR, weld_model, "电焊 (Welding)")

    # 处理切割样本
    total_cut = process_folder(CUT_SAMPLES_DIR, cut_model, "切割 (Cutting)")

    print("\n" + "=" * 60)
    print(f"全部完成！")
    print(f"   - 电焊带框图：{total_weld} 张")
    print(f"   - 切割带框图：{total_cut} 张")
    print(f"   - 总计：{total_weld + total_cut} 张")
    print("=" * 60)
    print("\n下一步提示：")
    print(f"请前往以下目录检查生成的 *_boxed.jpg 图片：")
    print(f"   1. {WELD_SAMPLES_DIR}")
    print(f"   2. {CUT_SAMPLES_DIR}")
    print("确认框的位置正确后，就可以开始编写 JSONL 格式的 CoT 训练数据了！")


if __name__ == "__main__":
    main()
