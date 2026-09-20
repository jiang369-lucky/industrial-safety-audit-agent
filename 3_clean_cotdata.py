import json
import re

input_file = 'cot_train_auto_generated.jsonl'
output_file = 'cot_train_cleaned.jsonl'

print(f"🧹 正在清洗文件：{input_file} ...")

with open(input_file, 'r', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8') as f_out:
    count = 0
    for line in f_in:
        try:
            data = json.loads(line)
            text_content = data['text']

            # 【关键清洗步骤】
            # 如果 text 里面包含了 "[{'text': '...'}]" 这种结构，我们需要提取出里面的真实文本
            # 这种情况通常发生在 API 返回的对象被直接 str() 转换了

            if "Assistant: [{'text':" in text_content:
                # 使用正则提取 'text': '...' 中间的内容
                match = re.search(r"Assistant: \[\{'text': '(.*?)'\}\]", text_content, re.DOTALL)
                if match:
                    real_response = match.group(1)
                    # 替换掉转义字符 (如果需要)
                    real_response = real_response.replace("\\n", "\n").replace("\\'", "'")

                    # 重构标准的 User \n Assistant 格式
                    # 先找到 User 的部分 (假设 User 部分没问题)
                    user_part = text_content.split("Assistant:")[0].strip()
                    data['text'] = f"{user_part}\nAssistant: {real_response}"
                    count += 1
                    print(f"修复了一条数据")
                else:
                    print(f"正则匹配失败，跳过该行：{text_content[:50]}...")

            # 写入清洗后的数据
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"处理行出错：{e}")

print(f"清洗完成！共修复 {count} 条数据。")
print(f"干净的数据已保存至：{output_file}")

