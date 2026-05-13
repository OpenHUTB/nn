import torch

def check_gpu():
    print("===== GPU & CUDA 环境检测 =====")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("当前使用 CPU 训练")

if __name__ == "__main__":
    check_gpu()