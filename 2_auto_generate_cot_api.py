import os
import json
import dashscope
from dashscope import MultiModalConversation

# ================= 配置区域 =================
# 从环境变量读取阿里云 DashScope API Key，不把密钥写入代码
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    raise RuntimeError("请先设置 DASHSCOPE_API_KEY 环境变量")

PROJECT_ROOT = r"D:\yolo\Spark-Gated MoE Framework"

# 数据目录
WELD_BOXED_DIR = os.path.join(PROJECT_ROOT, "dataset", "dataset_cot", "weld_samples_boxed")
CUT_BOXED_DIR = os.path.join(PROJECT_ROOT, "dataset", "dataset_cot", "cut_samples_boxed")

# 输出文件
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "cot_train_auto_generated.jsonl")

# 提示词
PROMPT_WELD = """这是一张电焊作业现场的监控图片，图中已经用彩色框标出了关键物体（worker: 工人，mask: 面罩，collector: 烟雾收集器）。
背景中有明显的电焊弧光或火花，表明工人正在进行动火作业。

请根据视觉信息严格执行以下检查逻辑：
1. 判断面罩佩戴状态：
   - 检查 `mask` 框与 `worker` 头部/面部区域的空间关系。
   - 正面视角：`mask` 框必须完全覆盖面部（眼睛、鼻子、嘴巴）。如果在头顶、额头、手中、脖子上或地上，均视为“未佩戴”。
   - 侧面视角：由于角度原因，可能只能看到侧脸。只要 `mask` 框覆盖了工人的口鼻区域（即使只看到侧面轮廓），即视为“已佩戴”。如果侧面能看到脸部皮肤裸露且无面罩遮挡，或面罩框位于非面部位置，视为“未佩戴”。
2. 判断设备存在性：检查图中是否存在 `collector` 框（烟雾收集器）。
3. 最终判定：
   - 只有当“面罩已正确佩戴（无论正面还是侧面）” 且 “有收集器”时，才判定为【合规】。
   - 其他任何情况（未佩戴、佩戴位置错误、缺失收集器）均为【违规】。

请直接返回一段连贯的自然语言分析文本，包含：
- 推理过程：详细描述你看到的框的位置关系（特别是侧面时的判断依据）。
- 最终结论：明确写出“合规”或“违规”。
- 具体原因：一句话总结核心违规点（如：“侧面观察到面罩未覆盖口鼻”或“面罩拿在手中”）。
注意：不要使用 JSON 格式，直接用自然语言回答。"""

PROMPT_CUT = """这是一张切割作业现场的监控图片，图中已经用彩色框标出了关键物体（worker: 工人，extinguisher: 灭火器）。
背景中有明显的切割火花飞溅，表明工人正在进行动火作业。

请根据视觉信息严格执行以下检查逻辑：
1. 确认作业状态：图片中存在火花，确认为动火作业场景。
2. 判断设备存在性：检查图中是否存在 `extinguisher` 框（灭火器）。
3. 最终判定：
   - 规则：在动火作业期间，必须配备灭火器。
   - 如果检测到 `worker` 且有火花，但未发现 `extinguisher` 框，判定为【违规】。
   - 如果检测到 `extinguisher` 框，判定为【合规】。

请直接返回一段连贯的自然语言分析文本，包含：
- 推理过程：描述是否检测到火花（动火），以及是否找到了灭火器框。
- 最终结论：明确写出“合规”或“违规”。
- 具体原因：一句话总结（如：“动火作业现场未检测到灭火器”）。
注意：不要使用 JSON 格式，直接用自然语言回答。"""

def call_qwen_api(image_path, prompt):
    #调用 Qwen-VL-Max 或 Plus 进行推理
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": prompt}
            ]
        }
    ]

    try:
        # 使用 qwen-vl-max 或 qwen-vl-plus
        response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"Error: {response.code} - {response.message}"
    except Exception as e:
        return f"Exception: {str(e)}"


def process_directory(dir_path, scene_type, output_list):
    if not os.path.exists(dir_path):
        print(f"目录不存在：{dir_path}")
        return

    files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"\n开始处理 [{scene_type}] 场景：共 {len(files)} 张图片")

    prompt = PROMPT_WELD if scene_type == "welding" else PROMPT_CUT

    for i, filename in enumerate(files):
        img_path = os.path.join(dir_path, filename)
        # 构造相对路径
        rel_path = os.path.relpath(img_path, PROJECT_ROOT).replace("\\", "/")

        print(f"  [{i + 1}/{len(files)}] 处理：{filename}...", end=" ")

        # 调用 API
        response_text = call_qwen_api(img_path, prompt)

        # 构造 JSONL 条目
        # 格式：{"image": "...", "text": "User: ...\nAssistant: ..."}
        user_content = prompt
        assistant_content = response_text

        entry = {
            "image": rel_path,
            "text": f"User: {user_content}\nAssistant: {assistant_content}"
        }

        output_list.append(entry)
        print("正确")


def main():
    if dashscope.api_key == "YOUR_API_KEY_HERE":
        print("错误：请先在代码中填入你的 API KEY！")
        return

    print("=" * 60)
    print("Qwen-VL API 自动 CoT 数据生成器")
    print("=" * 60)

    data_entries = []

    # 处理电焊
    process_directory(WELD_BOXED_DIR, "welding", data_entries)

    # 处理切割
    process_directory(CUT_BOXED_DIR, "cutting", data_entries)

    # 保存
    if data_entries:
        with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
            for entry in data_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print("\n" + "=" * 60)
        print(f"完成！共生成 {len(data_entries)} 条数据。")
        print(f"预估消耗：约 {len(data_entries) * 0.005} 元人民币 (按 qwen-vl-plus 估算，非常便宜)")
        print(f"保存路径：{OUTPUT_JSONL}")
        print("下一步：打开文件抽查，修正少量错误，即可开始微调！")
        print("=" * 60)
    else:
        print("未生成数据。")


if __name__ == "__main__":
    main()
