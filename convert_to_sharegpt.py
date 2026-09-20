import json
import os

# 读取你的原始数据
input_file = r"D:\yolo\Spark-Gated MoE Framework\dataset\dataset_cot\cot_auto_weld_alpaca.jsonl"  # 你的原始文件
output_file = r"D:\yolo\Spark-Gated MoE Framework\dataset\dataset_cot\cot_auto_weld_sharegpt.json"  # 输出文件

converted_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)

        # 处理图片路径：将反斜杠替换为正斜杠
        # item["image"] 是列表，处理列表中的每个路径
        processed_images = []
        for img_path in item["image"]:  # 你的字段是 "image"
            # 替换反斜杠为正斜杠
            processed_path = img_path.replace('\\', '/')
            processed_images.append(processed_path)

        # 转换格式
        new_item = {
            "id": item.get("id", f"sample_{len(converted_data)}"),
            "images": processed_images,  # 使用处理后的路径
            "conversations": [
                {
                    "from": "human",
                    "value": item["instruction"]
                },
                {
                    "from": "gpt",
                    "value": item["output"]
                }
            ]
        }
        converted_data.append(new_item)

# 保存为 JSON 文件（注意是 .json 不是 .jsonl）
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"转换完成！共 {len(converted_data)} 条数据")
print(f"保存到: {output_file}")

# 可选：显示第一条数据作为预览
if converted_data:
    print("\n预览第一条数据的图片路径：")
    for path in converted_data[0]["images"]:
        print(f"  {path}")