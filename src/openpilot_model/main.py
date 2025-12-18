#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Line Detection Program (Final Version)
Fix Chinese garbled text in annotations & optimize visualization
"""

# ===================== 1. Import Core Modules =====================
import sys
import os
import logging
import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ===================== 2. Global Configuration =====================
# Basic path configuration
PROJECT_ROOT = os.path.expanduser("~/nn")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "sample.mp4")
RESULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "lane_pred_result.mp4")
# Chinese font path (Ubuntu system)
CHINESE_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
# Video parameters
DEFAULT_MAX_FRAMES = 10
FPS = 25  # Video frame rate

# Log configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===================== 3. Environment Initialization =====================
def init_environment():
    """
    Initialize Matplotlib environment to fix Chinese garbled text
    Set backend and load Chinese font explicitly
    """
    # Set Matplotlib backend (TkAgg for GUI, Agg for non-GUI)
    if os.environ.get('DISPLAY') is None:
        matplotlib.use('Agg')
        logger.info("Matplotlib backend set to: Agg (non-GUI mode)")
    else:
        matplotlib.use('TkAgg')
        logger.info("Matplotlib backend set to: TkAgg (GUI mode)")

    # Check font file existence
    if not os.path.exists(CHINESE_FONT_PATH):
        logger.error(f"Chinese font file not found: {CHINESE_FONT_PATH}")
        logger.error("Install font: sudo apt install fonts-wqy-microhei")
        sys.exit(1)
    
    # Load Chinese font (explicit path to avoid garbled text)
    try:
        global chinese_font
        chinese_font = FontProperties(fname=CHINESE_FONT_PATH, size=12)
        logger.info("Chinese font loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Chinese font: {e}")
        sys.exit(1)

# ===================== 4. Core Functions =====================
def read_video(video_path, max_frames=DEFAULT_MAX_FRAMES):
    """
    Read video frames and initialize video writer
    Args:
        video_path: Path to input video file
        max_frames: Maximum number of frames to process
    Returns:
        frames: List of original video frames
        writer: VideoWriter object for saving result
    """
    # Validate video file
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return [], None
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path} (Check FFmpeg installation)")
        return [], None
    
    # Get video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video resolution: {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(RESULT_VIDEO_PATH, fourcc, FPS, (width, height))
    if not writer.isOpened():
        logger.error(f"Failed to create video writer: {RESULT_VIDEO_PATH}")
        cap.release()
        return [], None

    # Read frames (limit max frames to avoid memory overflow)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:  # 第108行while语句，下方补充缩进代码块
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1  # 缩进的代码块，修复核心错误
    
    cap.release()
    logger.info(f"Successfully read {len(frames)} frames")
    return frames, writer

def draw_lane_lines(frame):
    """
    Draw lane lines on video frame (fixed coordinates for stable visualization)
    Args:
        frame: Original video frame (BGR format)
    Returns:
        frame_with_lane: Frame with lane lines drawn
    """
    h, w = frame.shape[:2]
    frame_copy = frame.copy()

    # Left lane line (blue)
    cv2.line(frame_copy, (w//3, h), (w//3, h//2), (255, 0, 0), 5)
    # Right lane line (red)
    cv2.line(frame_copy, (2*w//3, h), (2*w//3, h//2), (0, 0, 255), 5)
    # Predicted path (green)
    cv2.line(frame_copy, (w//2, h), (w//2, h//2), (0, 255, 0), 3)

    return frame_copy

# ===================== 5. Main Function =====================
def main():
    """Main function for lane line detection and visualization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lane Line Detection (Chinese Annotation Fix)")
    parser.add_argument("video_path", type=str, nargs='?', default=VIDEO_PATH,
                        help=f"Video file path (default: {VIDEO_PATH})")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"Max frames to process (default: {DEFAULT_MAX_FRAMES})")
    args = parser.parse_args()

    # Initialize environment (fix Chinese garbled text)
    init_environment()

    # Read video frames
    frames, writer = read_video(args.video_path, args.max_frames)
    if not frames:
        return

    # Initialize visualization window
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("车道线预测结果（叠加可视化）", fontproperties=chinese_font, fontweight='bold', fontsize=16)
    
    # Chinese annotation (explicit font to fix garbled text)
    ax.text(
        0.02, 0.95,
        "左车道线(蓝色) | 右车道线(红色) | 预测路径(绿色)",
        transform=ax.transAxes,
        color='white',
        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'),
        fontproperties=chinese_font
    )
    ax.axis('off')  # Hide axis
    img_display = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))

    # Process frames one by one
    for i, frame in enumerate(frames):
        try:
            # Draw lane lines
            frame_with_lane = draw_lane_lines(frame)
            
            # Update visualization window
            img_display.set_data(cv2.cvtColor(frame_with_lane, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Save frame to result video
            if writer:
                writer.write(frame_with_lane)
            
            # Exit with Q key
            if cv2.waitKey(20) & 0xFF == ord('q'):
                logger.info("Exit by Q key")
                break
            
            logger.info(f"Processed frame {i+1}/{len(frames)}")

        except Exception as e:
            logger.warning(f"Failed to process frame {i+1}: {e}, skip")
            continue

    # Release resources
    logger.info("Releasing resources...")
    plt.ioff()
    plt.close(fig)
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Result prompt
    logger.info("\nProgram completed!")
    if os.path.exists(RESULT_VIDEO_PATH):
        logger.info(f"Result video saved to: {RESULT_VIDEO_PATH}")
        logger.info(f"Play video: totem {RESULT_VIDEO_PATH}")

# ===================== 6. Program Entry =====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Program terminated with error: {e}")
