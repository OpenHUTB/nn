"""
车道线检测模块唯一入口。

用法:
  python main.py                     # 默认：步骤1 基础 Canny+霍夫
  python main.py --mode hsv          # 步骤2 HSV 多车道检测
  python main.py --mode advanced     # 步骤3 透视变换 + 滑动窗口 + 多项式拟合
  python main.py --mode video --video path/to/video.mp4  # 步骤4 视频模式
  python main.py --no-metrics              # 隐藏曲率与偏移信息
  python main.py --no-warning              # 隐藏预警状态（关掉车道区域颜色变化）
  python main.py --save-docs               # 将效果图写入 docs/lane_detection/images
  python main.py --save-docs               # 将效果图写入 docs/lane_detection/images
"""
import argparse
  python main.py --save-docs         # 将效果图写入 docs/lane_detection/images
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

import cv2

from config import CONFIG, DEFAULT_IMAGE, MODULE_DIR
from lane_advanced import run_advanced_pipeline
from lane_detect import run_hsv_pipeline
from lane_preprocess import run_basic_pipeline
from lane_video import run_video_pipeline

DOCS_IMAGE_DIR = MODULE_DIR.parent.parent / "docs" / "lane_detection" / "images"


def parse_args():
    parser = argparse.ArgumentParser(description="车道线检测（Carla 场景）")
    parser.add_argument(
        "--mode",
        choices=["basic", "hsv", "advanced", "video"],
        default="basic",
        help="basic=灰度+Canny+霍夫；hsv=黄白线提取+多车道拟合；"
        choices=["basic", "hsv", "advanced"],
        default="basic",
        help="basic=灰度+Canny+霍夫；hsv=黄白线提取+多车道拟合；"
             "advanced=透视变换+滑动窗口+多项式拟合",
        choices=["basic", "hsv"],
        default="basic",
        help="basic=灰度+Canny+霍夫；hsv=黄白线提取+多车道拟合；"
             "advanced=透视变换+滑动窗口+多项式拟合；"
             "video=视频模式（逐帧检测+EMA平滑）",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="输入图像路径，默认使用模块内 carla_test.jpg",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="输入视频路径（仅 --mode video 时有效）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=CONFIG["ema_alpha"],
        help=f"EMA 平滑系数，0~1，越小越平滑（默认: {CONFIG['ema_alpha']}）",
    )
    parser.add_argument(
        "--save-docs",
        action="store_true",
        help="保存效果图/视频到 docs/lane_detection/images",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出 OpenCV 窗口（CI 或无图形界面时使用）",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="隐藏曲率半径与偏移量信息（仅 advanced / video 模式）",
    )
    parser.add_argument(
        "--no-warning",
        action="store_true",
        help="隐藏车道偏离预警信息（仅 advanced / video 模式）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = DOCS_IMAGE_DIR if args.save_docs else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 处理 --no-metrics 标志
    if args.no_metrics:
        CONFIG["show_metrics"] = False

    # 处理 --no-warning 标志
    if args.no_warning:
        CONFIG["show_warning"] = False

    if args.mode == "video":
        video_path = args.video
        if not video_path:
            print("错误：视频模式需要指定 --video 参数")
            return 1
        ok = run_video_pipeline(video_path, save_dir=save_dir,
                                alpha=args.alpha, show=not args.no_show)
        return 0 if ok else 1

    CONFIG["img_path"] = args.image

    if args.mode == "basic":
        outputs = run_basic_pipeline(args.image, save_dir=save_dir)
        if outputs is None:
            return 1
        display = outputs["result"]
        window = "Lane Detection Step1 (Canny + Hough)"
    elif args.mode == "hsv":
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