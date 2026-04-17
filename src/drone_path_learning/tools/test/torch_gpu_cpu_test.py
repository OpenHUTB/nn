import torch


def _yes_no(value: bool) -> str:
    return "是" if value else "否"


def main() -> None:
    print("=" * 56)
    print("PyTorch / CUDA 环境检查报告")
    print("=" * 56)

    torch_version = torch.__version__
    cuda_build_version = torch.version.cuda or "未包含 CUDA"
    cuda_available = torch.cuda.is_available()

    print(f"[基础信息] PyTorch 版本: {torch_version}")
    print(f"[基础信息] 此 PyTorch 编译使用的 CUDA 版本: {cuda_build_version}")
    print(f"[运行状态] 当前 CUDA 是否可用: {_yes_no(cuda_available)}")

    if not cuda_available:
        print("[结论] 当前环境无法使用 GPU，训练将回退到 CPU。")
        print(
            "[建议] 请检查 NVIDIA 驱动、CUDA 运行时、以及 PyTorch CUDA 版本是否匹配。"
        )
        return

    current_device = torch.cuda.current_device()
    cudnn_available = torch.backends.cudnn.is_available()

    print(f"[GPU信息] 当前使用的 GPU 编号: {current_device}")
    print(f"[GPU信息] 当前 GPU 名称: {torch.cuda.get_device_name(current_device)}")
    print(f"[GPU信息] cuDNN 是否可用: {_yes_no(cudnn_available)}")

    print("[结论] GPU 可正常被 PyTorch 识别，可用于模型训练。")


if __name__ == "__main__":
    main()
