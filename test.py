# 导入 requests 库，用于发送 HTTP 请求
import requests

# 设置请求头，告诉服务器发送的数据是 JSON 格式
headers = {"Content-Type": "application/json"}

# 准备要发送的数据（JSON 格式）
data = {
    "model": "deepseek-r1:7b",  # 指定要使用的模型名称(在ollama中下载的模型)
    "prompt": "你是谁？",         # 要问模型的问题
    "stream": False              # 是否流式输出，False 表示一次性返回完整结果
}

# 发送 POST 请求到 Ollama 的 API 接口
# url: Ollama 默认运行在本地 11434 端口（http://127.0.0.1:11434这是ollama的端口）
# json: 自动将 data 转为 JSON 格式并发送
# headers: 附带请求头
response = requests.post(
    url="http://127.0.0.1:11434/api/generate",
    json=data,
    headers=headers
)

# 打印服务器返回的完整 JSON 响应（包含各种元数据）
print(response.json())

# 从 JSON 响应中提取 "response" 字段，即模型实际生成的回答内容
# .get("response", "") 表示如果没有 "response" 字段，则返回空字符串
print(response.json().get("response", ""))