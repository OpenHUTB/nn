"""
车道线检测模块唯一入口。

用法:
  python main.py                  # 默认：步骤1 基础 Canny+霍夫
  python main.py --mode hsv       # 步骤2 HSV 多车道检测
  python main.py --mode advanced  # 步骤3 透视变换 + 滑动窗口 + 多项式拟合
  python main.py --save-docs      # 将效果图写入 docs/lane_detection/images
"""
import argparse
  python main.py              # 默认：步骤1 基础 Canny+霍夫
  python main.py --mode hsv   # 步骤2 HSV 多车道检测
  python main.py --save-docs  # 将效果图写入 docs/lane_detection/images
"""
import argparse
from pathlib import Path

import cv2

from config import CONFIG, DEFAULT_IMAGE, MODULE_DIR
from lane_advanced import run_advanced_pipeline
from lane_detect import run_hsv_pipeline
from lane_preprocess import run_basic_pipeline

DOCS_IMAGE_DIR = MODULE_DIR.parent.parent / "docs" / "lane_detection" / "images"


def parse_args():
    parser = argparse.ArgumentParser(description="车道线检测（Carla 场景）")
    parser.add_argument(
        "--mode",
        choices=["basic", "hsv", "advanced"],
        default="basic",
        help="basic=灰度+Canny+霍夫；hsv=黄白线提取+多车道拟合；"
             "advanced=透视变换+滑动窗口+多项式拟合",
        choices=["basic", "hsv"],
        default="basic",
        help="basic=灰度+Canny+霍夫；hsv=黄白线提取+多车道拟合",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="输入图像路径，默认使用模块内 carla_test.jpg",
    )
    parser.add_argument(
        "--save-docs",
        action="store_true",
        help="保存效果图到 docs/lane_detection/images（用于 mkdocs 文档）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出 OpenCV 窗口（CI 或无图形界面时使用）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG["img_path"] = args.image
    save_dir = DOCS_IMAGE_DIR if args.save_docs else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "basic":
        outputs = run_basic_pipeline(args.image, save_dir=save_dir)
        if outputs is None:
            return 1
        display = outputs["result"]
        window = "Lane Detection Step1 (Canny + Hough)"
    elif args.mode == "hsv":
    else:
        display = run_hsv_pipeline(args.image, save_dir=save_dir)
        if display is None:
            return 1
        window = "Lane Detection Step2 (HSV Multi-lane)"
    else:
        display = run_advanced_pipeline(args.image, save_dir=save_dir)
        if display is None:
            return 1
        window = "Lane Detection Step3 (Advanced: Perspective + Polynomial)"

    if not args.no_show:
        cv2.imshow(window, display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.save_docs:
        print(f"效果图已保存至: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
