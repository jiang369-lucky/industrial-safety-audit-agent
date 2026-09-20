import json
import os

# 配置区域
PROJECT_ROOT = r"D:\yolo\Spark-Gated MoE Framework"

# 输入文件 (混合数据)
INPUT_FILE = os.path.join(PROJECT_ROOT, "cot_train_cleaned.jsonl")

# 输出文件 (拆分后的目标路径)
OUTPUT_WELD = os.path.join(PROJECT_ROOT, "dataset", "dataset_cot", "cot_auto_weld.jsonl")
OUTPUT_CUT = os.path.join(PROJECT_ROOT, "dataset", "dataset_cot", "cot_auto_cut.jsonl")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件！\n路径：{INPUT_FILE}")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_WELD), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CUT), exist_ok=True)

    print(f"开始拆分数据集...")
    print(f"源文件：{INPUT_FILE}")
    print(f"电焊目标：{OUTPUT_WELD}")
    print(f"切割目标：{OUTPUT_CUT}")

    weld_count = 0
    cut_count = 0
    error_count = 0
    unknown_count = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_WELD, 'w', encoding='utf-8') as f_weld, \
            open(OUTPUT_CUT, 'w', encoding='utf-8') as f_cut:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                img_path = data.get('image', '')

                # 判断逻辑：根据图片路径中的文件夹名称
                if 'weld_samples_boxed' in img_path:
                    f_weld.write(json.dumps(data, ensure_ascii=False) + '\n')
                    weld_count += 1
                elif 'cut_samples_boxed' in img_path:
                    f_cut.write(json.dumps(data, ensure_ascii=False) + '\n')
                    cut_count += 1
                else:
                    # 如果路径里既没有 weld 也没有 cut，尝试通过文件名或其他关键词判断
                    if 'weld' in img_path.lower() or 'dht' in img_path.lower():
                        # 假设 dht 开头的是电焊 (根据你的之前提供的文件名推测)
                        f_weld.write(json.dumps(data, ensure_ascii=False) + '\n')
                        weld_count += 1
                    elif 'cut' in img_path.lower() or 'qgt' in img_path.lower():
                        # 假设 qgt 开头的是切割 (根据你的之前提供的文件名推测)
                        f_cut.write(json.dumps(data, ensure_ascii=False) + '\n')
                        cut_count += 1
                    else:
                        print(f"第 {line_num} 行无法分类，跳过：{img_path}")
                        unknown_count += 1

            except json.JSONDecodeError:
                print(f"第 {line_num} 行 JSON 格式错误，跳过。")
                error_count += 1
            except Exception as e:
                print(f"第 {line_num} 行处理出错：{e}")
                error_count += 1

    # 关闭文件
    f_in.close()
    f_weld.close()
    f_cut.close()

    print("\n" + "=" * 60)
    print("拆分完成！")
    print(f"电焊数据 ({os.path.basename(OUTPUT_WELD)}): {weld_count} 条")
    print(f"切割数据 ({os.path.basename(OUTPUT_CUT)}): {cut_count} 条")
    if unknown_count > 0:
        print(f"未分类/跳过：{unknown_count} 条 (请检查图片路径命名)")
    if error_count > 0:
        print(f"解析错误：{error_count} 条")

    print("=" * 60)


if __name__ == "__main__":
    main()
