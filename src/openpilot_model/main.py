#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Line Detection Program (Final Version)
Fix Chinese garbled text in annotations & optimize visualization
适用场景：Ubuntu系统下的车道线检测可视化，解决中文标注乱码+修复语法缩进错误
核心功能：读取视频帧→绘制车道线→可视化展示→保存结果视频
"""


# 系统/日志模块：用于路径处理、日志输出、程序退出
import sys
import os
import logging
# 命令行参数解析：支持自定义视频路径和处理帧数
import argparse
# 计算机视觉核心库：视频读取、帧处理、绘图、视频写入
import cv2
# 数值计算库：处理图像像素数组
import numpy as np
# 可视化库：解决中文乱码、动态展示视频帧
import matplotlib
import matplotlib.pyplot as plt
# 字体管理：显式加载中文字体，修复标注乱码
from matplotlib.font_manager import FontProperties


# 基础路径配置：使用expanduser兼容用户主目录（~）的路径解析
PROJECT_ROOT = os.path.expanduser("~/nn")  # 项目根目录，适配不同用户的主目录路径
VIDEO_PATH = os.path.join(PROJECT_ROOT, "sample.mp4")  # 默认输入视频路径
RESULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "lane_pred_result.mp4")  # 结果视频保存路径
# 中文字体路径：Ubuntu系统默认的文泉驿微米黑字体，解决Matplotlib中文乱码
CHINESE_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
# 视频参数配置：限制最大处理帧数避免内存溢出，FPS与原视频保持一致
DEFAULT_MAX_FRAMES = 10  # 默认最大处理帧数（测试用，可根据需求调整）
FPS = 25  # 视频帧率（需与输入视频匹配，否则播放速度异常）

# 日志配置：标准化日志输出格式，便于调试和运行状态追踪
logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO(普通信息)、WARNING(警告)、ERROR(错误)
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式：时间-级别-内容
    datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
)
logger = logging.getLogger(__name__)  # 创建日志实例，绑定当前模块


def init_environment():
    """
    Initialize Matplotlib environment to fix Chinese garbled text
    Set backend and load Chinese font explicitly
    核心作用：
    1. 适配不同运行环境的Matplotlib后端（GUI/无GUI）
    2. 校验并加载中文字体，彻底解决可视化中文乱码
    """
    # Set Matplotlib backend (TkAgg for GUI, Agg for non-GUI)
    # 后端选择逻辑：无DISPLAY环境（如服务器/无桌面虚拟机）用Agg，有桌面用TkAgg
    if os.environ.get('DISPLAY') is None:
        matplotlib.use('Agg')
        logger.info("Matplotlib backend set to: Agg (non-GUI mode)")
    else:
        matplotlib.use('TkAgg')
        logger.info("Matplotlib backend set to: TkAgg (GUI mode)")

    # Check font file existence
    # 校验字体文件是否存在，不存在则提示安装并退出
    if not os.path.exists(CHINESE_FONT_PATH):
        logger.error(f"Chinese font file not found: {CHINESE_FONT_PATH}")
        logger.error("Install font: sudo apt install fonts-wqy-microhei")
        sys.exit(1)  # 字体缺失会导致中文乱码，强制退出
    
    # Load Chinese font (explicit path to avoid garbled text)
    try:
        global chinese_font  # 定义全局字体变量，供后续可视化函数调用
        chinese_font = FontProperties(fname=CHINESE_FONT_PATH, size=12)  # 指定字体文件和大小
        logger.info("Chinese font loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Chinese font: {e}")
        sys.exit(1)


def read_video(video_path, max_frames=DEFAULT_MAX_FRAMES):
    """
    Read video frames and initialize video writer
    Args:
        video_path: Path to input video file
        max_frames: Maximum number of frames to process
    Returns:
        frames: List of original video frames (BGR格式，OpenCV默认)
        writer: VideoWriter object for saving result
    核心步骤：
    1. 校验视频文件有效性
    2. 打开视频流并获取分辨率
    3. 初始化视频写入器（用于保存结果）
    4. 读取指定帧数的视频帧并返回
    """
    # Validate video file
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return [], None
    
    # Open video capture：创建视频捕获对象，读取视频流
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  # 视频打开失败（如格式不支持/文件损坏）
        logger.error(f"Failed to open video: {video_path} (Check FFmpeg installation)")
        return [], None
    
    # Get video resolution：获取视频宽高，用于视频写入器初始化
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video resolution: {width}x{height}")

    # Initialize video writer：创建视频写入器，保存处理后的视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式（mp4v对应mp4格式）
    # 参数说明：保存路径、编码格式、帧率、分辨率（必须与帧一致）
    writer = cv2.VideoWriter(RESULT_VIDEO_PATH, fourcc, FPS, (width, height))
    if not writer.isOpened():  # 写入器创建失败（如路径无权限/编码不支持）
        logger.error(f"Failed to create video writer: {RESULT_VIDEO_PATH}")
        cap.release()  # 释放视频捕获资源，避免内存泄漏
        return [], None

    # Read frames (limit max frames to avoid memory overflow)
    frames = []  # 存储读取的视频帧
    count = 0  # 已读取帧数计数器
    # 循环读取帧：直到视频结束或达到最大帧数
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()  # ret=是否读取成功，frame=帧数据（BGR数组）
        if not ret:  # 帧读取失败（如视频结束）
            break
        frames.append(frame)  # 保存有效帧
        count += 1  # 计数器+1
    
    cap.release()  # 释放视频捕获资源
    logger.info(f"Successfully read {len(frames)} frames")
    return frames, writer


    # 初始化模型状态
    state = np.zeros((1, 512))
    desire = np.zeros((1, 8))

    plt.ion()  # 开启交互模式（简化版）
    fig, ax = plt.subplots(figsize=(8, 6))  # 单个窗口，避免子图渲染压力
    ax.set_title("车道线预测（蓝=左车道，红=右车道，绿=路径）")
    ax.set_ylim(0, 191)  # 固定Y轴，减少重绘计算
    ax.invert_xaxis()     # 匹配驾驶视角
    ax.grid(alpha=0.3)    # 简单网格，不占资源

    # 初始化三条线（提前创建，避免每次重绘新建）
    lll_line, = ax.plot([], [], "b-", linewidth=3, label="左车道线")
    rll_line, = ax.plot([], [], "r-", linewidth=3, label="右车道线")
    path_line, = ax.plot([], [], "g-", linewidth=2, label="预测路径")
    ax.legend()


    
    print(f"\n开始推理+可视化（共{len(frame_tensors)-1}帧，按Q键退出）...")
    for i in range(len(frame_tensors) - 1):
        try:
            # 模型推理
            inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
            outs = supercombo.predict(inputs, verbose=0)
            parsed = parser(outs)
            state = outs[-1]

        
            lll_line.set_data(parsed["lll"][0], range(192))  # 只更新左车道线数据
            rll_line.set_data(parsed["rll"][0], range(192))  # 只更新右车道线数据
            path_line.set_data(parsed["path"][0], range(192))# 只更新路径数据
            fig.canvas.draw()  # 轻量重绘（只更改造变的部分）
            fig.canvas.flush_events()  # 强制刷新窗口，避免卡住
      

            # 显示原始帧（简化版，用Matplotlib显示，避免OpenCV额外窗口）
            if i < len(raw_frames):
                # 新建一个小窗口显示原始帧，减少渲染压力
                cv2.imshow("原始帧", cv2.resize(raw_frames[i], (480, 270)))  # 缩小尺寸
                if cv2.waitKey(100) & 0xFF == ord('q'):  # 延长等待时间，给CPU喘息
                    print("用户按Q键退出")
                    break

            print(f"✅ 帧 {i+1}/{len(frame_tensors)-1} 完成")

        except Exception as e:
            print(f"⚠️  帧 {i+1} 失败：{str(e)}")
            continue


    print("\n🎉 处理完成！")
    plt.ioff()
    plt.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

def draw_lane_lines(frame):
    """
    Draw lane lines on video frame (fixed coordinates for stable visualization)
    Args:
        frame: Original video frame (BGR format)
    Returns:
        frame_with_lane: Frame with lane lines drawn
    绘图规则：
    - 左车道线：蓝色（BGR=(255,0,0)），位置w//3（宽度1/3处）
    - 右车道线：红色（BGR=(0,0,255)），位置2*w//3（宽度2/3处）
    - 预测路径：绿色（BGR=(0,255,0)），位置w//2（宽度中间）
    注意：OpenCV中颜色通道为BGR，与Matplotlib的RGB相反
    """
    h, w = frame.shape[:2]  # 获取帧的高度和宽度（shape=[高,宽,通道数]）
    frame_copy = frame.copy()  # 复制原帧，避免修改原数据

    # Left lane line (blue)：绘制左车道线，参数：帧、起点、终点、颜色、线宽
    cv2.line(frame_copy, (w//3, h), (w//3, h//2), (255, 0, 0), 5)
    # Right lane line (red)：绘制右车道线
    cv2.line(frame_copy, (2*w//3, h), (2*w//3, h//2), (0, 0, 255), 5)
    # Predicted path (green)：绘制预测路径
    cv2.line(frame_copy, (w//2, h), (w//2, h//2), (0, 255, 0), 3)

    return frame_copy


def main():
    """Main function for lane line detection and visualization
    主流程：
    1. 解析命令行参数（自定义视频路径/处理帧数）
    2. 初始化环境（修复中文乱码）
    3. 读取视频帧并初始化写入器
    4. 初始化Matplotlib可视化窗口
    5. 逐帧处理：绘制车道线→更新可视化→保存帧到结果视频
    6. 释放资源并输出结果提示
    """
    # Parse command line arguments：解析命令行参数，支持自定义输入
    parser = argparse.ArgumentParser(description="Lane Line Detection (Chinese Annotation Fix)")
    # 可选参数：视频路径（默认使用全局配置的VIDEO_PATH）
    parser.add_argument("video_path", type=str, nargs='?', default=VIDEO_PATH,
                        help=f"Video file path (default: {VIDEO_PATH})")
    # 可选参数：最大处理帧数（默认DEFAULT_MAX_FRAMES=10）
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"Max frames to process (default: {DEFAULT_MAX_FRAMES})")
    args = parser.parse_args()  # 解析参数并存储到args对象

    # Initialize environment (fix Chinese garbled text)
    init_environment()  # 初始化Matplotlib和中文字体

    # Read video frames
    frames, writer = read_video(args.video_path, args.max_frames)
    if not frames:  # 无有效帧则直接返回，避免后续报错
        return

    # Initialize visualization window
    plt.ion()  # 开启Matplotlib交互模式，支持动态更新图像（关键：逐帧展示）
    fig, ax = plt.subplots(figsize=(12, 8))  # 创建画布和轴对象，设置窗口大小
    # 设置画布标题，指定中文字体避免乱码
    fig.suptitle("车道线预测结果（叠加可视化）", fontproperties=chinese_font, fontweight='bold', fontsize=16)
    
    # Chinese annotation (explicit font to fix garbled text)
    # 添加中文标注说明，参数：位置、内容、坐标变换、颜色、背景框、字体
    ax.text(
        0.02, 0.95,  # 相对坐标（0-1），避免随帧大小变化偏移
        "左车道线(蓝色) | 右车道线(红色) | 预测路径(绿色)",
        transform=ax.transAxes,  # 使用轴的相对坐标，适配不同画布大小
        color='white',
        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'),  # 黑色半透明背景，提升可读性
        fontproperties=chinese_font
    )
    ax.axis('off')  # 隐藏坐标轴，专注展示视频帧
    # 初始化图像展示：将第一帧（BGR转RGB）显示到画布（Matplotlib默认RGB，OpenCV默认BGR）
    img_display = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))

    # Process frames one by one：逐帧处理并更新可视化
    for i, frame in enumerate(frames):
        try:
            # Draw lane lines：绘制车道线（返回带车道线的帧）
            frame_with_lane = draw_lane_lines(frame)
            
            # Update visualization window：更新画布显示内容
            img_display.set_data(cv2.cvtColor(frame_with_lane, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()  # 重绘画布
            fig.canvas.flush_events()  # 刷新事件，确保实时更新
            
            # Save frame to result video：将处理后的帧写入结果视频
            if writer:
                writer.write(frame_with_lane)
            
            # Exit with Q key：按下Q键退出（需聚焦Matplotlib窗口）
            if cv2.waitKey(20) & 0xFF == ord('q'):
                logger.info("Exit by Q key")
                break  # 退出循环
            
            logger.info(f"Processed frame {i+1}/{len(frames)}")  # 输出帧处理进度

        except Exception as e:  # 单帧处理失败不终止程序，仅警告并跳过
            logger.warning(f"Failed to process frame {i+1}: {e}, skip")
            continue

    # Release resources：释放所有资源，避免内存泄漏
    logger.info("Releasing resources...")
    plt.ioff()  # 关闭交互模式
    plt.close(fig)  # 关闭Matplotlib窗口
    if writer:
        writer.release()  # 释放视频写入器
    cv2.destroyAllWindows()  # 关闭OpenCV窗口

    # Result prompt：输出程序完成提示和结果路径
    logger.info("\nProgram completed!")
    if os.path.exists(RESULT_VIDEO_PATH):
        logger.info(f"Result video saved to: {RESULT_VIDEO_PATH}")
        logger.info(f"Play video: totem {RESULT_VIDEO_PATH}")  # totem是Ubuntu默认视频播放器


if __name__ == "__main__":  # 程序入口（仅直接运行时执行）
    try:
        main()  # 调用主函数
    except KeyboardInterrupt:  # 捕获Ctrl+C中断，友好退出
        logger.info("Program interrupted by user")
    except Exception as e:  # 捕获其他未预期错误，输出日志并终止
        logger.error(f"Program terminated with error: {e}")

