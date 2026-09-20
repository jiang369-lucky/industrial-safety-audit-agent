#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-VL-4B-Instruct')

print(f"模型路径位于：{model_dir}")
