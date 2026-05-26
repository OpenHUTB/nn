"""
自动驾驶车辆语义分割 —— 推理入口

加载预训练 U-Net 模型，对 CARLA 街景做 8 类语义分割。
支持三种模式，按扩展名/参数自动识别：
  - 图片 (.png/.jpg/...)：输出叠加图 (overlay) 和纯掩码图 (mask)
  - 视频 (.mp4/.avi/...)：逐帧分割，输出叠加视频，并生成采样帧拼图
  - --augment 图片：对同一张图分别施加 6 种训练时用到的数据增强并排展示（不需要模型）

用法：
    python main.py                                   # 用默认示例图 + 默认模型
    python main.py <输入>                             # 指定输入（图片或视频）
    python main.py <输入> <模型目录>                  # 指定输入和模型
    python main.py <输入> <模型目录> <最大帧数>       # 视频限制处理帧数
    python main.py --augment                         # 默认示例图，可视化 6 种数据增强
    python main.py --augment <输入图>                 # 指定输入图做增强可视化

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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from semantic.unet.utils import infer, labels_to_image, overlay_labels_on_input

DEFAULT_INPUT = os.path.join(MODULE_DIR, "examples", "sample_input.png")
DEFAULT_MODEL = os.path.join(MODULE_DIR, "models", "unet_model_256x256_50")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
DEFAULT_MAX_FRAMES = 150


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


def segment_image(model, img):
    """对单张 PIL 图片做语义分割，返回 (叠加图, 掩码图) 均为 PIL RGB。"""
    labels = infer(model, img)
    overlay = overlay_labels_on_input(img, labels, alpha=0.45).convert("RGB")
    mask = labels_to_image(labels, output_size=img.size)
    return overlay, mask


def make_montage(frames, cols=3):
    """把若干 PIL 图拼成网格图，用作入库效果图。"""
    if not frames:
        return None
    rows = (len(frames) + cols - 1) // cols
    w, h = frames[0].size
    montage = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, frame in enumerate(frames):
        montage.paste(frame, ((i % cols) * w, (i // cols) * h))
    return montage


def label_panel(panel, text):
    """在 PIL 图左上角绘制半透明黑底白字标签。"""
    out = panel.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", max(14, panel.size[0] // 28))
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 6
    box = (0, 0, bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad)
    draw.rectangle(box, fill=(0, 0, 0, 180))
    draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)
    return out


def visualize_augmentations(img):
    """对单张 PIL 图分别施加 6 种训练时的数据增强，返回 (标签, 图) 列表。

    增强逻辑取自 semantic/unet/dataset.py，但这里每种增强独立施加在原图上，
    便于直观对比每种变换的效果（训练时 6 种增强会被随机组合）。
    """
    panels = [("original", img.copy())]

    # 1. 水平翻转
    panels.append(("horizontal flip", ImageOps.mirror(img)))

    # 2. 亮度 +30%（训练时范围 ±40%）
    panels.append(("brightness +30%", ImageEnhance.Brightness(img).enhance(1.30)))

    # 3. 对比度 -30%（训练时范围 -40% 到 0%）
    panels.append(("contrast -30%", ImageEnhance.Contrast(img).enhance(0.70)))

    # 4. 高斯模糊（半径 3，训练时范围 0-5）
    panels.append(("gaussian blur r=3", img.filter(ImageFilter.GaussianBlur(3.0))))

    # 5. 椒盐噪声 5%（训练时范围 0-7%）
    from skimage.util import random_noise

    arr = np.asarray(img.convert("RGB"), dtype="uint8")
    noisy = (255 * random_noise(arr, mode="salt", amount=0.05)).astype("uint8")
    panels.append(("salt noise 5%", Image.fromarray(noisy)))

    # 6. 中心裁剪缩放（训练时 50% 概率走该分支，从图像中间横向滑窗选 256x256 子区域）
    w, h = img.size
    crop_size = 256
    cx = w // 2
    cy = h // 2
    cropped = img.crop(
        (cx - crop_size // 2, cy - crop_size // 2, cx + crop_size // 2, cy + crop_size // 2)
    )
    panels.append(("center crop 256", cropped.resize((w, h))))

    return panels


def run_augment(input_path):
    """加载图片，可视化 6 种数据增强，输出并排对比图（无需模型）。"""
    img = Image.open(input_path).convert("RGB")
    print(f"      输入图片: {input_path}  尺寸: {img.size}")
    print("[1/2] 施加 6 种训练时使用的数据增强 ...")
    panels = visualize_augmentations(img)

    print("[2/2] 拼接对比图 ...")
    # 每个面板缩到原图一半再拼，避免最终图过大
    half = (img.size[0] // 2, img.size[1] // 2)
    labeled = [label_panel(p.resize(half), name) for name, p in panels]
    # 7 个面板用 4 列布局 → 2 行（最后一格留空）
    grid = make_montage(labeled, cols=4)

    base, _ = os.path.splitext(input_path)
    out_path = f"{base}_augment.png"
    grid.save(out_path)
    print(f"      已写出数据增强可视化: {out_path}")


def run_image(model, img_path):
    print(f"[2/3] 读取图片: {img_path}")
    img = Image.open(img_path).convert("RGB")
    print(f"      图片尺寸: {img.size}")

    print("[3/3] 运行语义分割并生成可视化 ...")
    overlay, mask = segment_image(model, img)

    base, _ = os.path.splitext(img_path)
    overlay.save(f"{base}_overlay.png")
    mask.save(f"{base}_mask.png")
    print(f"      已写出: {base}_overlay.png")
    print(f"      已写出: {base}_mask.png")


def run_video(model, video_path, max_frames):
    print(f"[2/3] 打开视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_process = min(total, max_frames) if total > 0 else max_frames
    print(f"      总帧数: {total}, FPS: {fps:.1f}, 尺寸: {width}x{height}")
    print(f"      本次处理: {n_process} 帧")

    base, _ = os.path.splitext(video_path)
    out_path = f"{base}_seg.mp4"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print("[3/3] 逐帧语义分割 ...")
    montage_idx = {int(i * n_process / 6) for i in range(6)}
    montage_frames = []
    done = 0
    while done < n_process:
        ok, frame = cap.read()
        if not ok:
            break
        # cv2 是 BGR，模型/PIL 用 RGB
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        overlay, _ = segment_image(model, img)
        writer.write(cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR))
        if done in montage_idx:
            montage_frames.append(overlay.resize((width // 2, height // 2)))
        done += 1
        if done % 20 == 0 or done == n_process:
            print(f"      已处理 {done}/{n_process} 帧")

    cap.release()
    writer.release()
    print(f"      已写出叠加视频: {out_path}")

    montage = make_montage(montage_frames)
    if montage is not None:
        montage_path = f"{base}_seg_frames.png"
        montage.save(montage_path)
        print(f"      已写出采样帧拼图: {montage_path}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--augment":
        rest = args[1:]
        input_path = rest[0] if rest else DEFAULT_INPUT
        if not os.path.isfile(input_path):
            print(f"[错误] 找不到输入文件: {input_path}", file=sys.stderr)
            sys.exit(1)
        run_augment(input_path)
        print("完成。")
        return

    input_path = args[0] if len(args) >= 1 else DEFAULT_INPUT
    model_dir = args[1] if len(args) >= 2 else DEFAULT_MODEL
    max_frames = int(args[2]) if len(args) >= 3 else DEFAULT_MAX_FRAMES

    if not os.path.isfile(input_path):
        print(f"[错误] 找不到输入文件: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] 加载模型: {model_dir}")
    model = load_segmentation_model(model_dir)
    print(f"      模型输入尺寸: {model.layers[0].get_output_at(0).get_shape()}")

    is_video = input_path.lower().endswith(VIDEO_EXTS)
    if is_video:
        run_video(model, input_path, max_frames)
    else:
        run_image(model, input_path)
    print("完成。")


if __name__ == "__main__":
    main()
