import os
import urllib.request
import sys
import time


def reporthook(count, block_size, total_size):
    """显示下载进度的回调函数"""
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = int(count * block_size * 100 / total_size)

    sys.stdout.write(f"\r下载进度: {percent}% | 已下载: {progress_size / (1024 * 1024):.2f} MB | 速度: {speed} KB/s")
    sys.stdout.flush()


def download_file(url, save_path):
    print(f"正在开始下载: {save_path}")
    print(f"源地址: {url}")
    try:
        urllib.request.urlretrieve(url, save_path, reporthook)
        print("\n下载完成!")
    except Exception as e:
        print(f"\n下载失败: {e}")


if __name__ == "__main__":
    # 确保 models 目录存在
    if not os.path.exists('models'):
        os.makedirs('models')

    # YOLOv3 权重文件 (约 237MB)
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    weights_path = "models/yolov3.weights"

    # YOLOv3 配置文件
    cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    cfg_path = "models/yolov3.cfg"

    print("=== 开始自动下载模型文件 ===")

    # 下载 cfg
    if not os.path.exists(cfg_path):
        download_file(cfg_url, cfg_path)
    else:
        print(f"文件已存在，跳过: {cfg_path}")

    # 下载 weights
    if not os.path.exists(weights_path):
        download_file(weights_url, weights_path)
    else:
        print(f"文件已存在，跳过: {weights_path}")

    print("=== 所有文件准备就绪 ===")