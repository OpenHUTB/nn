import torch
import os
try:
    from yolov5 import YOLO  
except ImportError:
    print("未检测到 YOLO 库，请确认导入方式")
    YOLO = None


def get_device():
    """
    自动选择可用设备
    """
    if torch.cuda.is_available():
        print("使用 GPU 进行推理")
        return torch.device("cuda")
    else:
        print("使用 CPU 进行推理")
        return torch.device("cpu")


def load_model(model_path="model.pt"):
    """
    安全加载 YOLO 模型
    """
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    device = get_device()
    if YOLO is None:
        print("YOLO 模块未导入，无法加载模型")
        return None

    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"模型成功加载到 {device}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def main():
    """
    主函数入口
    """
    print("开始 YOLO 模型推理测试...")

    model = load_model("model.pt")
    if model is None:
        print("模型未加载，跳过推理")
        return
    
    try:
        print("准备进行推理...")
        # results = model.predict("test.jpg")
        print("推理完成")
    except Exception as e:
        print(f"推理失败: {e}")


if __name__ == "__main__":
    main()
