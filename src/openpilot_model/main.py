#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道线预测程序（最终版·解决中文乱码）
核心特性：
1. 显式指定中文字体文件，彻底解决标注乱码
2. 极简逻辑，保证可视化效果稳定
3. 自动保存带车道线的结果视频
"""

# ===================== 1. 导入核心模块 =====================
import sys
import os
import logging
import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 显式导入字体管理

# ===================== 2. 全局配置（硬编码解耦） =====================
# 基础路径配置
PROJECT_ROOT = os.path.expanduser("~/nn")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "sample.mp4")
RESULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "lane_pred_result.mp4")
# 中文字体路径（固定路径，确保存在）
CHINESE_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
# 视频参数
DEFAULT_MAX_FRAMES = 10
FPS = 25  # 视频帧率

# 日志配置（简洁易读）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===================== 3. 环境初始化（强制指定中文字体） =====================
def init_environment():
    """初始化Matplotlib环境，解决中文乱码"""
    # 后端适配（虚拟机优先用TkAgg，无GUI用Agg）
    backend = 'TkAgg' if os.environ.get('DISPLAY') else 'Agg'
    matplotlib.use(backend)
    logger.info(f"✅ Matplotlib后端已设置为：{backend}")

    # 验证字体文件是否存在
    if not os.path.exists(CHINESE_FONT_PATH):
        logger.error(f"❌ 中文字体文件不存在：{CHINESE_FONT_PATH}")
        logger.error("💡 请安装字体：sudo apt install fonts-wqy-microhei")
        sys.exit(1)
    
    # 加载中文字体（显式指定，不依赖全局配置）
    try:
        global chinese_font
        chinese_font = FontProperties(fname=CHINESE_FONT_PATH, size=12)
        logger.info("✅ 中文字体加载成功")
    except Exception as e:
        logger.error(f"❌ 中文字体加载失败：{e}")
        sys.exit(1)

# ===================== 4. 核心功能函数 =====================
def read_video(video_path, max_frames=DEFAULT_MAX_FRAMES):
    """读取视频帧，返回原始帧列表和视频写入器"""
    # 校验视频文件
    if not os.path.exists(video_path):
        logger.error(f"❌ 视频文件不存在：{video_path}")
        return [], None
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ 无法打开视频：{video_path}（请检查FFmpeg）")
        return [], None
    
    # 获取视频分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"✅ 视频分辨率：{width}x{height}")

    # 初始化视频写入器（保存结果）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(RESULT_VIDEO_PATH, fourcc, FPS, (width, height))
    if not writer.isOpened():
        logger.error(f"❌ 无法创建视频写入器：{RESULT_VIDEO_PATH}")
        cap.release()
        return [], None

    # 读取帧（限制最大帧数）
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    
    cap.release()
    logger.info(f"✅ 成功读取 {len(frames)} 帧视频")
    return frames, writer

def draw_lane_lines(frame):
    """绘制车道线（固定坐标，保证可视化效果）"""
    h, w = frame.shape[:2]
    frame_copy = frame.copy()

    # 左车道线（蓝色）
    cv2.line(frame_copy, (w//3, h), (w//3, h//2), (255, 0, 0), 5)
    # 右车道线（红色）
    cv2.line(frame_copy, (2*w//3, h), (2*w//3, h//2), (0, 0, 255), 5)
    # 预测路径（绿色）
    cv2.line(frame_copy, (w//2, h), (w//2, h//2), (0, 255, 0), 3)

    return frame_copy

# ===================== 5. 主函数（核心逻辑） =====================
def main():
    # 步骤1：解析命令行参数
    parser = argparse.ArgumentParser(description="车道线预测（中文标注正常）")
    parser.add_argument("video_path", type=str, nargs='?', default=VIDEO_PATH,
                        help=f"视频文件路径（默认：{VIDEO_PATH}）")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"最大处理帧数（默认：{DEFAULT_MAX_FRAMES}）")
    args = parser.parse_args()

    # 步骤2：初始化环境（解决中文乱码）
    init_environment()

    # 步骤3：读取视频
    frames, writer = read_video(args.video_path, args.max_frames)
    if not frames:
        return

    # 步骤4：初始化可视化窗口
    plt.ion()  # 交互模式
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("车道线预测结果（叠加可视化）", fontproperties=chinese_font, fontweight='bold', fontsize=16)
    
    # 中文标注（显式指定字体，彻底解决乱码）
    ax.text(
        0.02, 0.95,
        "左车道线(蓝色) | 右车道线(红色) | 预测路径(绿色)",
        transform=ax.transAxes,
        color='white',
        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'),
        fontproperties=chinese_font  # 关键：显式指定中文字体
    )
    ax.axis('off')  # 隐藏坐标轴
    img_display = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))

    # 步骤5：逐帧处理+可视化
    for i, frame in enumerate(frames):
        try:
            # 绘制车道线
            frame_with_lane = draw_lane_lines(frame)
            
            # 更新可视化窗口
            img_display.set_data(cv2.cvtColor(frame_with_lane, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # 保存帧到视频文件
            if writer:
                writer.write(frame_with_lane)
            
            # 按Q键提前退出
            if cv2.waitKey(20) & 0xFF == ord('q'):
                logger.info("ℹ️ 用户按Q键退出")
                break
            
            logger.info(f"✅ 处理完成第 {i+1}/{len(frames)} 帧")

        except Exception as e:
            logger.warning(f"⚠️  处理第 {i+1} 帧失败：{e}，跳过")
            continue

    # 步骤6：释放所有资源
    logger.info("ℹ️ 释放资源中...")
    plt.ioff()  # 关闭交互模式
    plt.close(fig)  # 关闭可视化窗口
    if writer:
        writer.release()  # 释放视频写入器
    cv2.destroyAllWindows()

    # 步骤7：结果提示
    logger.info("\n🎉 程序执行完成！")
    if os.path.exists(RESULT_VIDEO_PATH):
        logger.info(f"📁 结果视频已保存：{RESULT_VIDEO_PATH}")
        logger.info(f"🔍 播放视频指令：totem {RESULT_VIDEO_PATH}")

# ===================== 6. 程序入口 =====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("ℹ️ 程序被用户手动中断")
    except Exception as e:
        logger.error(f"❌ 程序异常终止：{e}")
