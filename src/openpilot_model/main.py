#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Line Detection Program (Enhanced Version)

核心功能：
1. 基于边缘检测+霍夫变换实现动态车道线检测，兼容固定坐标绘制降级方案；
2. 支持单视频/批量视频处理，进度条可视化处理状态；
3. 日志双端输出（终端+文件），视频编码自适应，资源自动释放；
4. 修复Matplotlib中文乱码问题，增强程序鲁棒性。

适用环境：Ubuntu (Python 3.10 + OpenCV + Matplotlib + tqdm)
"""

# ===================== 模块导入 =====================
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
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

# ===================== 全局配置 =====================
# 项目根路径
PROJECT_ROOT = Path(os.path.expanduser("~/nn"))
# 默认视频路径
VIDEO_PATH = PROJECT_ROOT / "sample.mp4"
# 结果输出目录（自动创建）
RESULT_DIR = PROJECT_ROOT / "lane_detection_results"
RESULT_DIR.mkdir(exist_ok=True)
# 日志文件路径
LOG_FILE = PROJECT_ROOT / "lane_detection.log"

# 中文字体路径（微黑字体）
CHINESE_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

# 视频处理参数
DEFAULT_MAX_FRAMES = 10  # 默认最大处理帧数
FPS = 25  # 默认帧率（自适应原视频帧率）
DEFAULT_DETECT_MODE = "dynamic"  # 默认检测模式：dynamic-动态检测，fixed-固定坐标

# 车道线检测参数（霍夫变换+边缘检测）
CANNY_LOW_THRESH = 50       # Canny边缘检测低阈值
CANNY_HIGH_THRESH = 150     # Canny边缘检测高阈值
HOUGH_RHO = 1               # 霍夫变换极坐标rho步长
HOUGH_THETA = np.pi / 180   # 霍夫变换极坐标theta步长
HOUGH_THRESHOLD = 20        # 霍夫变换检测阈值
HOUGH_MIN_LINE_LEN = 40     # 最小线段长度
HOUGH_MAX_LINE_GAP = 20     # 最大线段间隙

# ===================== 日志初始化 =====================
def setup_logger() -> logging.Logger:
    """初始化日志处理器：同时输出到终端和文件
    
    Returns:
        logging.Logger: 配置完成的日志实例
    """
    logger = logging.getLogger("LaneDetection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除重复处理器

    # 终端日志处理器
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(stream_formatter)

    # 文件日志处理器（UTF-8编码）
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

# ===================== 环境初始化 =====================
def init_environment() -> FontProperties:
    """初始化Matplotlib环境，修复中文显示乱码
    
    Returns:
        FontProperties: 加载完成的中文字体实例
    
    Raises:
        SystemExit: 字体文件不存在或加载失败时退出程序
    """
    # 适配GUI/无GUI环境的Matplotlib后端
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

    # 加载中文字体
    try:
        chinese_font = FontProperties(fname=CHINESE_FONT_PATH, size=12)
        logger.info("中文字体加载成功")
        return chinese_font
    except Exception as e:
        logger.error(f"字体加载失败：{str(e)}")
        sys.exit(1)

# ===================== 视频读取 =====================
def read_video(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES
) -> Tuple[List[np.ndarray], Optional[cv2.VideoWriter], Tuple[int, int]]:
    """读取视频帧并初始化视频写入器
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大读取帧数
    
    Returns:
        Tuple[List[np.ndarray], Optional[cv2.VideoWriter], Tuple[int, int]]:
            - 读取到的视频帧列表
            - 初始化后的视频写入器（失败则为None）
            - 视频分辨率(width, height)
    """
    # 校验视频文件存在性
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在：{video_path}")
        return [], None, (0, 0)

    # 打开视频流
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"视频打开失败：{video_path} (请检查FFmpeg是否安装：sudo apt install ffmpeg)")
        cap.release()
        return [], None, (0, 0)

    # 获取视频基础信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS  # 自适应原视频帧率
    logger.info(f"视频信息：{video_path} | 分辨率 {width}x{height} | 帧率 {fps}")

    # 初始化视频写入器（输出带车道线的视频）
    video_name = Path(video_path).stem
    result_video_path = RESULT_DIR / f"{video_name}_lane_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码格式
    writer = cv2.VideoWriter(
        str(result_video_path), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        logger.error(f"视频写入器初始化失败：{result_video_path}")
        cap.release()
        return [], None, (width, height)

    # 读取视频帧（带进度条）
    frames = []
    count = 0
    logger.info(f"开始读取视频帧（最大{max_frames}帧）...")
    with tqdm(total=max_frames, desc="读取帧") as pbar:
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
            pbar.update(1)

    cap.release()
    logger.info(f"视频帧读取完成：共读取 {len(frames)} 帧")
    return frames, writer, (width, height)

# ===================== 车道线检测 =====================
def detect_lane_lines(
    frame: np.ndarray,
    mode: str = "dynamic"
) -> np.ndarray:
    """检测并绘制车道线
    
    Args:
        frame: 单帧视频图像（BGR格式）
        mode: 检测模式，可选值：dynamic/fixed
    
    Returns:
        np.ndarray: 绘制了车道线的帧图像
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]

    # 固定坐标绘制车道线（降级方案）
    if mode == "fixed":
        cv2.line(frame_copy, (w//3, h), (w//3, h//2), (255, 0, 0), 5)   # 左车道线（蓝）
        cv2.line(frame_copy, (2*w//3, h), (2*w//3, h//2), (0, 0, 255), 5) # 右车道线（红）
        cv2.line(frame_copy, (w//2, h), (w//2, h//2), (0, 255, 0), 3)   # 中心线（绿）
        return frame_copy

    # 动态检测车道线（Canny边缘检测 + 霍夫变换）
    try:
        # 1. 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. 高斯模糊去噪
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 3. 边缘检测
        edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
        # 4. 区域掩码（仅检测图像下半部分，聚焦车道区域）
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
                cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 车道线（绿）
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
        logger.warning(f"动态车道线检测失败：{str(e)}，自动切换为固定坐标绘制")
        # 降级到固定坐标绘制
        cv2.line(frame_copy, (w//3, h), (w//3, h//2), (255, 0, 0), 5)
        cv2.line(frame_copy, (2*w//3, h), (2*w//3, h//2), (0, 0, 255), 5)
        return frame_copy

# ===================== 可视化初始化 =====================
def init_visualization(
    chinese_font: FontProperties,
    width: int,
    height: int
) -> Tuple[plt.Figure, plt.Axes, mpimg.AxesImage]:
    """初始化Matplotlib可视化窗口
    
    Args:
        chinese_font: 中文字体实例
        width: 视频宽度
        height: 视频高度
    
    Returns:
        Tuple[plt.Figure, plt.Axes, mpimg.AxesImage]:
            - 可视化画布实例
            - 坐标轴实例
            - 图像显示对象
    """
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(
        "车道线检测结果（叠加可视化）",
        fontproperties=chinese_font,
        fontweight='bold',
        fontsize=16
    )

    # 标注说明文字
    ax.text(
        0.02, 0.95,
        "动态检测车道线（绿色）| 固定左车道（蓝色）| 固定右车道（红色）| 按Q退出",
        transform=ax.transAxes,
        color='white',
        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'),
        fontproperties=chinese_font
    )
    ax.axis('off')  # 隐藏坐标轴

    # 初始化空图像
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    img_display = ax.imshow(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))
    return fig, ax, img_display

# ===================== 批量处理 =====================
def batch_process(
    input_dir: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    detect_mode: str = "dynamic"
):
    """批量处理指定目录下的所有视频文件
    
    Args:
        input_dir: 视频目录路径
        max_frames: 单视频最大处理帧数
        detect_mode: 检测模式（dynamic/fixed）
    """
    if not os.path.isdir(input_dir):
        logger.error(f"输入目录不存在：{input_dir}")
        return

    # 筛选视频文件（支持mp4/avi/mov）
    video_extensions = (".mp4", ".avi", ".mov")
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(video_extensions)
    ]

    if not video_files:
        logger.info(f"目录{input_dir}下未检测到视频文件（支持格式：{video_extensions}）")
        return

    logger.info(f"批量处理启动：共检测到 {len(video_files)} 个视频文件")
    chinese_font = init_environment()

    # 逐视频处理
    for video_file in video_files:
        logger.info(f"\n开始处理视频：{video_file}")
        try:
            # 读取视频帧
            frames, writer, (width, height) = read_video(video_file, max_frames)
            if not frames:
                continue

            # 初始化可视化窗口
            fig, ax, img_display = init_visualization(chinese_font, width, height)

            # 逐帧处理（带进度条）
            with tqdm(total=len(frames), desc=f"处理{Path(video_file).stem}") as pbar:
                for i, frame in enumerate(frames):
                    start_time = time.time()
                    # 检测并绘制车道线
                    frame_with_lane = detect_lane_lines(frame, detect_mode)
                    # 计算单帧处理耗时
                    process_time = time.time() - start_time
                    # 标注帧信息
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
                    # 写入视频文件
                    if writer:
                        writer.write(frame_with_lane)
                    # 按Q键退出
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        logger.info("用户按下Q键，终止处理")
                        break
                    pbar.update(1)

            # 释放资源
            plt.ioff()
            plt.close(fig)
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info(f"视频处理完成：{video_file} → 结果保存至 {RESULT_DIR}")

        except Exception as e:
            logger.error(f"视频处理失败：{video_file} | 错误信息：{str(e)}")
            continue

# ===================== 主函数 =====================
def main():
    """程序主入口：解析命令行参数，执行视频处理"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="车道线检测程序（增强版）")
    parser.add_argument("input", type=str, help="输入路径：视频文件路径或文件夹路径（批量处理）")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="最大处理帧数（默认：10）")
    parser.add_argument("--detect-mode", type=str, default=DEFAULT_DETECT_MODE,
                        choices=["dynamic", "fixed"], help="检测模式（dynamic-动态检测，fixed-固定坐标）")
    parser.add_argument("--batch", action="store_true", help="批量处理模式（输入为文件夹时启用）")
    args = parser.parse_args()

    # 初始化环境（加载中文字体）
    chinese_font = init_environment()

    # 批量处理模式
    if args.batch or os.path.isdir(args.input):
        batch_process(args.input, args.max_frames, args.detect_mode)
        return

    # 单视频处理模式
    if os.path.isfile(args.input):
        frames, writer, (width, height) = read_video(args.input, args.max_frames)
        if not frames:
            return

        # 初始化可视化窗口
        fig, ax, img_display = init_visualization(chinese_font, width, height)

        # 逐帧处理（带进度条）
        with tqdm(total=len(frames), desc="处理帧") as pbar:
            for i, frame in enumerate(frames):
                start_time = time.time()
                # 检测并绘制车道线
                frame_with_lane = detect_lane_lines(frame, args.detect_mode)
                # 计算单帧处理耗时
                process_time = time.time() - start_time
                # 标注帧信息
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
                # 写入视频文件
                if writer:
                    writer.write(frame_with_lane)
                # 按Q键退出
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    logger.info("用户按下Q键，终止处理")
                    break
                pbar.update(1)

        # 释放资源
        logger.info("开始释放资源...")
        plt.ioff()
        plt.close(fig)
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # 输出结果信息
        result_video = RESULT_DIR / f"{Path(args.input).stem}_lane_detected.mp4"
        if os.path.exists(result_video):
            logger.info(f"\n处理完成！")
            logger.info(f"结果视频：{result_video}")
            logger.info(f"播放命令：totem {result_video}")
            logger.info(f"日志文件：{LOG_FILE}")
        return

    # 输入路径无效
    logger.error(f"输入路径无效：{args.input}（请输入视频文件路径或文件夹路径）")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    try:
        # 校验tqdm依赖
        from tqdm import tqdm
    except ImportError:
        logger.error("缺少依赖库tqdm，请执行安装：pip install tqdm")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断（Ctrl+C）")
    except Exception as e:
        logger.error(f"程序异常终止：{str(e)}", exc_info=True)
        sys.exit(1)
