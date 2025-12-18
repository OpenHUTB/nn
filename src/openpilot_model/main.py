#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Line Detection Program (Enhanced Version)
优化点：
1. 新增实际车道线检测（边缘检测+霍夫变换），替代硬编码坐标；
2. 日志同时输出到终端+文件，进度条可视化；
3. 支持批量处理、自定义参数、视频编码自适应；
4. 清理冗余代码，函数解耦，鲁棒性增强；
5. 保留原有中文乱码修复、可视化核心逻辑。
适用场景：Ubuntu系统下的车道线检测/可视化，支持单视频/批量视频处理
"""


import sys
import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # 导入image模块，修复类型注解
from matplotlib.font_manager import FontProperties
from tqdm import tqdm  # 关键修改：从tqdm模块导入tqdm函数（而非导入整个模块）


# 基础路径配置
PROJECT_ROOT = Path(os.path.expanduser("~/nn"))
VIDEO_PATH = PROJECT_ROOT / "sample.mp4"
RESULT_DIR = PROJECT_ROOT / "lane_detection_results"
RESULT_DIR.mkdir(exist_ok=True)  # 自动创建结果目录
LOG_FILE = PROJECT_ROOT / "lane_detection.log"

# 中文字体配置
CHINESE_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

# 视频参数
DEFAULT_MAX_FRAMES = 10
FPS = 25
DEFAULT_DETECT_MODE = "dynamic"  # dynamic=动态检测，fixed=固定坐标（兼容原有逻辑）

# 车道线检测参数（可调整）
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LEN = 40
HOUGH_MAX_LINE_GAP = 20


def setup_logger() -> logging.Logger:
    """初始化日志：同时输出到终端和文件"""
    logger = logging.getLogger("LaneDetection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 避免重复添加处理器

    # 终端处理器
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(stream_formatter)

    # 文件处理器
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()


def init_environment() -> FontProperties:
    """
    初始化Matplotlib环境，修复中文乱码
    返回：加载好的中文字体对象
    """
    # 适配GUI/无GUI后端
    if os.environ.get('DISPLAY') is None:
        matplotlib.use('Agg')
        logger.info("Matplotlib后端：Agg (无GUI模式)")
    else:
        matplotlib.use('TkAgg')
        logger.info("Matplotlib后端：TkAgg (GUI模式)")

    # 校验字体文件
    if not os.path.exists(CHINESE_FONT_PATH):
        logger.error(f"中文字体文件不存在：{CHINESE_FONT_PATH}")
        logger.error("安装命令：sudo apt install fonts-wqy-microhei")
        sys.exit(1)

    # 加载字体
    try:
        chinese_font = FontProperties(fname=CHINESE_FONT_PATH, size=12)
        logger.info("中文字体加载成功")
        return chinese_font
    except Exception as e:
        logger.error(f"字体加载失败：{str(e)}")
        sys.exit(1)


def read_video(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES
) -> Tuple[List[np.ndarray], Optional[cv2.VideoWriter], Tuple[int, int]]:
    """
    读取视频帧，初始化视频写入器
    返回：帧列表、视频写入器、视频分辨率
    """
    # 校验视频文件
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在：{video_path}")
        return [], None, (0, 0)

    # 打开视频（用with语句自动释放资源）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"视频打开失败：{video_path} (检查FFmpeg安装)")
        cap.release()
        return [], None, (0, 0)

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS  # 自适应原视频帧率
    logger.info(f"视频信息：{video_path} | 分辨率 {width}x{height} | 帧率 {fps}")

    # 初始化视频写入器（动态生成输出路径）
    video_name = Path(video_path).stem
    result_video_path = RESULT_DIR / f"{video_name}_lane_detected.mp4"
    # 自适应编码（解决不同系统编码兼容问题）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(result_video_path), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        logger.error(f"视频写入器创建失败：{result_video_path}")
        cap.release()
        return [], None, (width, height)

    # 读取帧（带进度条）
    frames = []
    count = 0
    logger.info(f"开始读取视频帧（最大{max_frames}帧）...")
    # 修复tqdm调用方式（已正确导入tqdm函数）
    with tqdm(total=max_frames, desc="读取帧") as pbar:
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
            pbar.update(1)

    cap.release()
    logger.info(f"读取完成：共{len(frames)}帧")
    return frames, writer, (width, height)


def detect_lane_lines(
    frame: np.ndarray,
    mode: str = "dynamic"
) -> np.ndarray:
    """
    检测/绘制车道线
    mode: dynamic=动态检测（边缘+霍夫），fixed=固定坐标（兼容原有逻辑）
    返回：绘制了车道线的帧
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]

    if mode == "fixed":
        # 原有固定坐标逻辑
        cv2.line(frame_copy, (w//3, h), (w//3, h//2), (255, 0, 0), 5)
        cv2.line(frame_copy, (2*w//3, h), (2*w//3, h//2), (0, 0, 255), 5)
        cv2.line(frame_copy, (w//2, h), (w//2, h//2), (0, 255, 0), 3)
        return frame_copy

    # 新增：动态车道线检测（边缘检测+霍夫变换）
    try:
        # 1. 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. 高斯模糊去噪
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 3. 边缘检测
        edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
        # 4. 区域掩码（只检测下半部分，聚焦车道）
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, h),
            (w//2, h//2),
            (w, h)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        # 5. 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            masked_edges,
            HOUGH_RHO,
            HOUGH_THETA,
            HOUGH_THRESHOLD,
            np.array([]),
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        # 6. 绘制检测到的车道线
        line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                line_count += 1

        # 7. 标注检测到的车道线数量
        cv2.putText(
            frame_copy,
            f"检测到车道线：{line_count}条",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )
        return frame_copy

    except Exception as e:
        logger.warning(f"车道线检测失败：{str(e)}，使用固定坐标替代")
        # 降级到固定坐标
        cv2.line(frame_copy, (w//3, h), (w//3, h//2), (255, 0, 0), 5)
        cv2.line(frame_copy, (2*w//3, h), (2*w//3, h//2), (0, 0, 255), 5)
        return frame_copy


def init_visualization(
    chinese_font: FontProperties,
    width: int,
    height: int
) -> Tuple[plt.Figure, plt.Axes, mpimg.AxesImage]:  # 正确的类型注解
    """初始化可视化窗口，添加增强标注"""
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(
        "车道线检测结果（叠加可视化）",
        fontproperties=chinese_font,
        fontweight='bold',
        fontsize=16
    )

    # 新增：详细标注说明
    ax.text(
        0.02, 0.95,
        "动态检测车道线（绿色）| 固定左车道（蓝色）| 固定右车道（红色）| 按Q退出",
        transform=ax.transAxes,
        color='white',
        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'),
        fontproperties=chinese_font
    )
    ax.axis('off')

    # 初始化图像显示
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    img_display = ax.imshow(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))
    return fig, ax, img_display


def batch_process(
    input_dir: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    detect_mode: str = "dynamic"
):
    """批量处理指定目录下的所有视频文件"""
    if not os.path.isdir(input_dir):
        logger.error(f"输入目录不存在：{input_dir}")
        return

    # 获取所有视频文件（支持mp4、avi、mov）
    video_extensions = (".mp4", ".avi", ".mov")
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(video_extensions)
    ]

    if not video_files:
        logger.info(f"目录{input_dir}下未找到视频文件")
        return

    logger.info(f"批量处理：共{len(video_files)}个视频")
    chinese_font = init_environment()

    for video_file in video_files:
        logger.info(f"\n开始处理：{video_file}")
        try:
            # 单视频处理逻辑
            frames, writer, (width, height) = read_video(video_file, max_frames)
            if not frames:
                continue

            fig, ax, img_display = init_visualization(chinese_font, width, height)

            # 逐帧处理（带进度条）
            with tqdm(total=len(frames), desc=f"处理{Path(video_file).stem}") as pbar:
                for i, frame in enumerate(frames):
                    start_time = time.time()
                    # 检测车道线
                    frame_with_lane = detect_lane_lines(frame, detect_mode)
                    # 计算处理耗时
                    process_time = time.time() - start_time
                    # 标注帧号和耗时
                    cv2.putText(
                        frame_with_lane,
                        f"帧：{i+1}/{len(frames)} | 耗时：{process_time:.3f}s",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
                    # 更新可视化
                    img_display.set_data(cv2.cvtColor(frame_with_lane, cv2.COLOR_BGR2RGB))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    # 保存帧到视频
                    if writer:
                        writer.write(frame_with_lane)
                    # 按Q退出
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        logger.info("用户按Q键退出")
                        break
                    pbar.update(1)

            # 释放资源
            plt.ioff()
            plt.close(fig)
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info(f"处理完成：{video_file} → 结果保存到{RESULT_DIR}")

        except Exception as e:
            logger.error(f"处理{video_file}失败：{str(e)}")
            continue


def main():
    """主函数：解析参数+执行检测"""
    # 解析命令行参数（新增批量处理、检测模式等参数）
    parser = argparse.ArgumentParser(description="车道线检测程序（增强版）")
    parser.add_argument("input", type=str, help="输入视频文件路径或文件夹路径（批量处理）")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="最大处理帧数")
    parser.add_argument("--detect-mode", type=str, default=DEFAULT_DETECT_MODE,
                        choices=["dynamic", "fixed"], help="车道线检测模式：dynamic=动态检测，fixed=固定坐标")
    parser.add_argument("--batch", action="store_true", help="批量处理模式（输入为文件夹时启用）")
    args = parser.parse_args()

    # 初始化环境
    chinese_font = init_environment()

    # 批量处理
    if args.batch or os.path.isdir(args.input):
        batch_process(args.input, args.max_frames, args.detect_mode)
        return

    # 单视频处理
    if os.path.isfile(args.input):
        frames, writer, (width, height) = read_video(args.input, args.max_frames)
        if not frames:
            return

        fig, ax, img_display = init_visualization(chinese_font, width, height)

        # 逐帧处理（带进度条）
        with tqdm(total=len(frames), desc="处理帧") as pbar:
            for i, frame in enumerate(frames):
                start_time = time.time()
                # 检测车道线
                frame_with_lane = detect_lane_lines(frame, args.detect_mode)
                # 标注帧号和耗时
                process_time = time.time() - start_time
                cv2.putText(
                    frame_with_lane,
                    f"帧：{i+1}/{len(frames)} | 耗时：{process_time:.3f}s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )
                # 更新可视化
                img_display.set_data(cv2.cvtColor(frame_with_lane, cv2.COLOR_BGR2RGB))
                fig.canvas.draw()
                fig.canvas.flush_events()
                # 保存帧
                if writer:
                    writer.write(frame_with_lane)
                # 按Q退出
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    logger.info("用户按Q键退出")
                    break
                pbar.update(1)

        # 释放资源
        logger.info("释放资源...")
        plt.ioff()
        plt.close(fig)
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # 结果提示
        result_video = RESULT_DIR / f"{Path(args.input).stem}_lane_detected.mp4"
        if os.path.exists(result_video):
            logger.info(f"\n处理完成！结果视频：{result_video}")
            logger.info(f"播放命令：totem {result_video}")
            logger.info(f"日志文件：{LOG_FILE}")
        return

    logger.error(f"输入无效：{args.input}（请输入视频文件路径或文件夹路径）")


if __name__ == "__main__":
    try:
        # 校验tqdm是否正确安装并导入
        try:
            from tqdm import tqdm  # 再次确认导入方式
        except ImportError:
            logger.error("缺少依赖tqdm，请安装：pip install tqdm")
            sys.exit(1)
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断（Ctrl+C）")
    except Exception as e:
        logger.error(f"程序异常终止：{str(e)}", exc_info=True)  # 输出完整异常栈
        sys.exit(1)

