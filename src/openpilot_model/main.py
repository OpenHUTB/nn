#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道线预测程序（最终版·标注中文正常显示）
核心：用Matplotlib绘制中文标注（替代OpenCV的putText）
"""
import sys
import os
import logging
import argparse
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ===================== 环境初始化（核心解决中文显示） =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 设置Matplotlib后端
matplotlib.use('Agg') if os.environ.get('DISPLAY') is None else matplotlib.use('TkAgg')

# 加载中文字体（仅给Matplotlib用）
def setup_chinese_font():
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    if os.path.exists(font_path):
        font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        logger.info(f"✅ 中文字体加载成功：{font_path}")
    else:
        logger.warning("⚠️  未找到wqy-microhei字体，使用默认英文字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 96
    plt.rcParams['savefig.dpi'] = 100

setup_chinese_font()

# 项目根目录
PROJECT_ROOT = "/home/dacun/nn"
sys.path.append(PROJECT_ROOT)

# 依赖检测
def check_dependencies():
    required_libs = {
        'numpy': np.__version__,
        'cv2': cv2.__version__,
        'matplotlib': matplotlib.__version__,
        'tensorflow': '2.x'
    }
    for lib, ver in required_libs.items():
        try:
            if lib == 'tensorflow':
                import tensorflow as tf
                assert tf.__version__.startswith('2.'), f"TensorFlow版本需≥2.0，当前：{tf.__version__}"
            logger.info(f"✅ 依赖检测通过：{lib} (版本：{ver})")
        except (ImportError, AssertionError) as e:
            logger.error(f"❌ 依赖缺失/版本错误：{lib} - {e}")
            logger.error(f"💡 修复命令：pip install {lib}>={ver.split('.')[0]}")
            sys.exit(1)

# 导入项目模块
try:
    from common.transformations.camera import transform_img, eon_intrinsics
    from common.transformations.model import medmodel_intrinsics
    from common.tools.lib.parser import parser
    logger.info("✅ 项目模块导入成功")
except ImportError as e:
    logger.error(f"❌ 项目模块导入失败：{e}")
    logger.error("💡 确认common文件夹路径：/home/dacun/nn/common/")
    sys.exit(1)

# ===================== 核心函数 =====================
def frames_to_tensor(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        logger.warning("输入帧为空，返回空张量")
        return np.array([])
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    tensor = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.float32)
    tensor[:, 0] = frames[:, 0:H:2, 0::2]
    tensor[:, 1] = frames[:, 1:H:2, 0::2]
    tensor[:, 2] = frames[:, 0:H:2, 1::2]
    tensor[:, 3] = frames[:, 1:H:2, 1::2]
    tensor[:, 4] = frames[:, H:H+H//4].reshape(-1, H//2, W//2)
    tensor[:, 5] = frames[:, H+H//4:H+H//2].reshape(-1, H//2, W//2)
    return tensor / 128.0 - 1.0

def preprocess_frame(img: np.ndarray) -> np.ndarray:
    try:
        return transform_img(
            img,
            from_intr=eon_intrinsics,
            to_intr=medmodel_intrinsics,
            yuv=True,
            output_size=(512, 256)
        )
    except Exception as e:
        logger.warning(f"单帧预处理失败：{e}，返回空帧")
        return np.zeros((384, 512), dtype=np.uint8)

def preprocess_frames(imgs: list) -> np.ndarray:
    if not imgs:
        return np.array([])
    processed_frames = [preprocess_frame(img) for img in imgs]
    processed_frames = np.array(processed_frames)
    empty_frames = np.sum(np.all(processed_frames == 0, axis=(1, 2)))
    if empty_frames > 0:
        logger.warning(f"共{empty_frames}帧预处理失败，已填充空帧")
    return frames_to_tensor(processed_frames)

def read_video(video_path: str, max_frames: int = 10) -> tuple:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")
    if not video_path.lower().endswith('.mp4'):
        raise ValueError("仅支持MP4格式视频")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频（请安装FFmpeg）：{video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_path = os.path.join(PROJECT_ROOT, "lane_pred_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    logger.info(f"✅ 结果视频保存路径：{save_path}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    yuv_frames = []
    raw_frames = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            logger.info(f"视频读取完毕，共读取{i}帧（目标：{max_frames}帧）")
            break
        raw_frames.append(frame)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        yuv_resized = cv2.resize(yuv, (512, 384), interpolation=cv2.INTER_LINEAR)
        yuv_frames.append(yuv_resized)
    cap.release()
    return yuv_frames, raw_frames, video_writer

def draw_lane_lines(frame: np.ndarray, pred: dict) -> np.ndarray:
    """仅绘制车道线（文字标注交给Matplotlib）"""
    h, w = frame.shape[:2]
    pred_lll = np.interp(pred["lll"][0], (0, 191), (0, h))
    pred_rll = np.interp(pred["rll"][0], (0, 191), (0, h))
    pred_path = np.interp(pred["path"][0], (0, 191), (0, h))
    x_coords = np.linspace(0, w, len(pred_lll))
    frame_copy = frame.copy()
    # 左车道线（蓝）
    for i in range(len(x_coords)-1):
        cv2.line(
            frame_copy,
            (int(x_coords[i]), int(pred_lll[i])),
            (int(x_coords[i+1]), int(pred_lll[i+1])),
            (255, 0, 0), 3, cv2.LINE_AA
        )
    # 右车道线（红）
    for i in range(len(x_coords)-1):
        cv2.line(
            frame_copy,
            (int(x_coords[i]), int(pred_rll[i])),
            (int(x_coords[i+1]), int(pred_rll[i+1])),
            (0, 0, 255), 3, cv2.LINE_AA
        )
    # 预测路径（绿）
    for i in range(len(x_coords)-1):
        cv2.line(
            frame_copy,
            (int(x_coords[i]), int(pred_path[i])),
            (int(x_coords[i+1]), int(pred_path[i+1])),
            (0, 255, 0), 2, cv2.LINE_AA
        )
    return cv2.addWeighted(frame_copy, 0.7, frame, 0.3, 0)

# ===================== 主函数 =====================
def main():
    parser_arg = argparse.ArgumentParser(description="车道线预测程序（标注中文正常）")
    parser_arg.add_argument("video_path", type=str, help="视频文件绝对路径")
    parser_arg.add_argument("--max-frames", type=int, default=10, help="最大读取帧数")
    parser_arg.add_argument("--save-result", action="store_true", default=True, help="保存结果视频")
    args = parser_arg.parse_args()
    
    check_dependencies()
    
    # 加载模型
    model_path = os.path.join(PROJECT_ROOT, "models/supercombo.h5")
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在：{model_path}")
        sys.exit(1)
    try:
        logger.info(f"开始加载模型：{model_path}")
        start_time = time.time()
        model = load_model(model_path, compile=False)
        logger.info(f"✅ 模型加载完成，耗时{round(time.time()-start_time,2)}秒")
    except Exception as e:
        logger.error(f"❌ 模型加载失败：{e}")
        sys.exit(1)
    
    # 读取视频
    try:
        yuv_frames, raw_frames, video_writer = read_video(args.video_path, args.max_frames)
        if not raw_frames:
            logger.error("未读取到有效视频帧")
            sys.exit(1)
        logger.info(f"✅ 视频读取完成，共{len(raw_frames)}帧")
    except Exception as e:
        logger.error(f"❌ 视频读取失败：{e}")
        sys.exit(1)
    
    # 预处理
    frame_tensor = preprocess_frames(yuv_frames)
    if frame_tensor.size == 0:
        logger.error("帧预处理后无有效数据")
        sys.exit(1)
    
    # 推理初始化
    state = np.zeros((1, 512))
    desire = np.zeros((1, 8))
    total_frames = len(frame_tensor) - 1
    logger.info(f"开始推理，共{total_frames}帧（按Q键退出）")
    
    # 可视化（用Matplotlib添加中文标注）
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("车道线预测结果（叠加可视化）", fontsize=14, fontweight='bold')
    # 添加中文图例标注（Matplotlib支持中文）
    ax.text(
        0.02, 0.95, 
        "左车道线(蓝) | 右车道线(红) | 预测路径(绿)",
        transform=ax.transAxes,
        fontsize=10,
        color='white',
        bbox=dict(facecolor='black', alpha=0.5)
    )
    ax.set_axis_off()
    img_display = ax.imshow(cv2.cvtColor(raw_frames[0], cv2.COLOR_BGR2RGB))
    
    for i in range(total_frames):
        try:
            # 推理
            input_tensor = np.vstack(frame_tensor[i:i+2])[None]
            outputs = model.predict([input_tensor, desire, state], verbose=0)
            pred_result = parser(outputs)
            state = outputs[-1]
            
            # 绘制车道线
            result_frame = draw_lane_lines(raw_frames[i], pred_result)
            
            # 更新显示
            img_display.set_data(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # 保存结果
            if args.save_result:
                video_writer.write(result_frame)
            
            # 退出
            if cv2.waitKey(30) & 0xFF == ord('q'):
                logger.info("用户按Q键退出")
                break
            
            logger.info(f"✅ 完成第{i+1}/{total_frames}帧推理")
        
        except Exception as e:
            logger.warning(f"⚠️  第{i+1}帧推理失败：{e}，跳过")
            continue
    
    # 资源释放
    plt.ioff()
    plt.close(fig)
    video_writer.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    # 成果提示
    logger.info("\n🎉 程序执行完成！")
    logger.info(f"📁 结果视频：{os.path.join(PROJECT_ROOT, 'lane_pred_result.mp4')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ 程序异常终止：{e}")
        sys.exit(1)
