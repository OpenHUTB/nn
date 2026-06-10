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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 8192
            with open(save_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        speed = int(downloaded / (1024 * (time.time() - start_time))) if time.time() - start_time > 0 else 0
                        sys.stdout.write(f"\r下载进度: {percent}% | 已下载: {downloaded / (1024 * 1024):.2f} MB | 速度: {speed} KB/s")
                        sys.stdout.flush()
        print("\n下载完成!")
    except Exception as e:
        print(f"\n下载失败: {e}")


if __name__ == "__main__":
    # 确保 models 目录存在
    if not os.path.exists('models'):
        os.makedirs('models')

    # 初始化下载计时器
    start_time = time.time()

    # YOLOv3-tiny 权重文件（24MB）
    weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
    weights_path = "models/yolov3-tiny.weights"

    cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
    cfg_path = "models/yolov3-tiny.cfg"

    # 新增：添加coco.names下载（代码中用到但原脚本未下载）
    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    names_path = "models/coco.names"

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
    
    # 新增：下载coco.names
    if not os.path.exists(names_path):
        download_file(names_url, names_path)
    else:
        print(f"文件已存在，跳过: {names_path}")

    print("=== 所有文件准备就绪 ===")