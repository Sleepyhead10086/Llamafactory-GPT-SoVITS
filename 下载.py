# download_qwen.py
from modelscope.hub.snapshot_download import snapshot_download

# 下载Qwen3-0.6B到指定目录（替换为你想保存的路径）
model_dir = snapshot_download(
    model_id="qwen/Qwen3-VL-8B-Instruct",
    cache_dir="C:\Git Bash-x64_5.1.5\Qwen3-VL-8B-Instruct"
)
print(f"模型下载完成，路径：{model_dir}")
