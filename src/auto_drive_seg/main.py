"""
自动驾驶车辆语义分割 —— 推理入口

加载预训练 U-Net 模型，对一张 CARLA 街景图做 8 类语义分割，
输出叠加图 (overlay) 和纯掩码图 (mask)。

用法：
    python main.py                                  # 用默认示例图 + 默认模型
    python main.py <输入图.png>                      # 指定输入图
    python main.py <输入图.png> <模型目录>           # 指定输入图和模型

模型说明：
    预训练模型为二进制大文件（每个约 17MB），未随仓库提交。
    请从原始项目 hlfshell/rbe549-project-segmentation 的 models/ 目录获取，
    例如 unet_model_256x256_50，放到本模块的 models/ 目录下。详见 README.md。
"""
import os
import sys

# 让 `import semantic.*` 不依赖当前工作目录
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import numpy as np
from PIL import Image

from semantic.unet.utils import infer, labels_to_image, overlay_labels_on_input

DEFAULT_INPUT = os.path.join(MODULE_DIR, "examples", "sample_input.png")
DEFAULT_MODEL = os.path.join(MODULE_DIR, "models", "unet_model_256x256_50")


def load_segmentation_model(model_dir):
    if not os.path.isdir(model_dir):
        print(
            f"[错误] 找不到模型目录: {model_dir}\n"
            f"预训练模型未随仓库提交（二进制大文件，每个约 17MB）。\n"
            f"请从原始项目获取模型并放到 models/ 目录：\n"
            f"  git clone https://github.com/hlfshell/rbe549-project-segmentation\n"
            f"  复制 rbe549-project-segmentation/models/unet_model_256x256_50 "
            f"到 {os.path.join(MODULE_DIR, 'models')}\\\n"
            f"详见本模块 README.md。",
            file=sys.stderr,
        )
        sys.exit(1)

    from keras.models import load_model

    return load_model(model_dir, compile=False)


def main():
    img_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_INPUT
    model_dir = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL

    if not os.path.isfile(img_path):
        print(f"[错误] 找不到输入图: {img_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] 加载模型: {model_dir}")
    model = load_segmentation_model(model_dir)
    print(f"      模型输入尺寸: {model.layers[0].get_output_at(0).get_shape()}")

    print(f"[2/4] 读取图片: {img_path}")
    img = Image.open(img_path).convert("RGB")
    print(f"      图片尺寸: {img.size}")

    print("[3/4] 运行语义分割推理 ...")
    labels = infer(model, img)
    print(f"      输出张量形状: {labels.shape}  (H, W, 类别数=8)")

    print("[4/4] 生成可视化结果 ...")
    overlay = overlay_labels_on_input(img, labels, alpha=0.45).convert("RGB")
    mask = labels_to_image(labels, output_size=img.size)

    base, _ = os.path.splitext(img_path)
    overlay_path = f"{base}_overlay.png"
    mask_path = f"{base}_mask.png"
    overlay.save(overlay_path)
    mask.save(mask_path)
    print(f"      已写出: {overlay_path}")
    print(f"      已写出: {mask_path}")
    print("完成。")


if __name__ == "__main__":
    main()
