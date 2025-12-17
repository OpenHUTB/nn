#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # 声明编码，解决中文注释/输出乱码
"""
车道线预测程序（优化版）
核心功能：读取MP4视频帧 → 预处理 → 模型推理 → 车道线/路径可视化
适配环境：Linux虚拟机（Python3 + TensorFlow + OpenCV + Matplotlib）
"""
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

# ===================== 基础配置（核心！解决中文乱码+路径问题） =====================
# 项目根目录（绝对路径，适配虚拟机）
PROJECT_ROOT = "/home/dacun/nn"
sys.path.append(PROJECT_ROOT)

# 解决Matplotlib中文显示乱码（关键优化）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入项目本地模块
try:
    from common.transformations.camera import transform_img, eon_intrinsics
    from common.transformations.model import medmodel_intrinsics
    from common.tools.lib.parser import parser
except ImportError as e:
    print(f"❌ 导入common模块失败：{e}")
    print("💡 请确保common文件夹在项目根目录：/home/dacun/nn/common/")
    sys.exit(1)

# ===================== 核心函数（规范化+精简冗余） =====================
def frames_to_tensor(frames):
    """
    将视频帧转换为模型输入张量
    :param frames: 原始视频帧数组 (N, H, W, C)
    :return: 归一化后的张量 (N, 6, H//2, W//2)
    """
    if len(frames) == 0:
        return np.array([])
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    tensor = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.float32)
    # 张量维度映射（模型输入要求）
    tensor[:, 0] = frames[:, 0:H:2, 0::2]
    tensor[:, 1] = frames[:, 1:H:2, 0::2]
    tensor[:, 2] = frames[:, 0:H:2, 1::2]
    tensor[:, 3] = frames[:, 1:H:2, 1::2]
    tensor[:, 4] = frames[:, H:H+H//4].reshape(-1, H//2, W//2)
    tensor[:, 5] = frames[:, H+H//4:H+H//2].reshape(-1, H//2, W//2)
    return tensor / 128.0 - 1.0  # 归一化到[-1, 1]

def preprocess_frames(imgs):
    """
    视频帧预处理（适配模型输入格式）
    :param imgs: 原始YUV帧列表
    :return: 预处理后的张量
    """
    if not imgs:
        return np.array([])
    processed = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
    # 精准捕获异常，避免通捕导致问题隐藏
    for i, img in enumerate(imgs):
        try:
            processed[i] = transform_img(
                img, 
                from_intr=eon_intrinsics, 
                to_intr=medmodel_intrinsics, 
                yuv=True, 
                output_size=(512, 256)
            )
        except (TypeError, ValueError) as e:
            print(f"⚠️  第{i+1}帧预处理失败：{str(e)}，填充空帧")
            processed[i] = np.zeros((384, 512), dtype=np.uint8)
    return frames_to_tensor(processed)

def read_video_frames(video_path, max_frames=10):
    """
    读取视频帧（仅支持MP4），简化函数名更直观
    :param video_path: 视频文件路径
    :param max_frames: 最大读取帧数
    :return: 预处理用YUV帧 + 原始BGR帧
    """
    # 精准校验视频格式
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")
    if not video_path.lower().endswith('.mp4'):
        raise ValueError("仅支持MP4格式视频，请更换文件格式")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频（请安装FFmpeg）：{video_path}")
    
    # 降低缓存，减少虚拟机内存占用
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    yuv_frames = []
    raw_frames = []
    
    # 进度条可视化（核心保留优化）
    for i in tqdm(range(max_frames), desc="读取视频帧", ncols=80):
        ret, frame = cap.read()
        if not ret:
            tqdm.write(f"⚠️  视频读取完毕，共读取{i}帧（不足{max_frames}帧）")
            break
        raw_frames.append(frame)
        # BGR转YUV_I420（模型输入要求）
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        yuv_resized = cv2.resize(yuv, (512, 384), interpolation=cv2.INTER_LINEAR)
        yuv_frames.append(yuv_resized)
    
    cap.release()
    return yuv_frames, raw_frames

# ===================== 主函数（精简+可视化升级） =====================
def main():
    # 1. 参数校验（精简且专业）
    if len(sys.argv) != 2:
        print("🚨 使用错误：缺少视频文件路径")
        print("✅ 正确用法：python main.py <视频文件绝对路径>")
        print("💡 示例：python main.py /home/dacun/nn/test.mp4")
        sys.exit(1)
    video_path = sys.argv[1]

    # 2. 加载模型（精准路径+异常捕获）
    model_path = os.path.join(PROJECT_ROOT, "models/supercombo.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}（请放入models目录）")
    
    try:
        print(f"📌 加载模型：{model_path}")
        model = load_model(model_path, compile=False)
    except (IOError, ValueError) as e:
        print(f"❌ 模型加载失败：{str(e)}")
        sys.exit(1)

    # 3. 读取+预处理视频帧
    try:
        yuv_frames, raw_frames = read_video_frames(video_path)
        if not yuv_frames:
            raise RuntimeError("未读取到有效视频帧")
        frame_tensor = preprocess_frames(yuv_frames)
        if frame_tensor.size == 0:
            raise RuntimeError("帧预处理后无有效数据")
    except Exception as e:
        print(f"❌ 视频处理失败：{str(e)}")
        sys.exit(1)

    # 4. 模型推理初始化
    state = np.zeros((1, 512))  # 模型状态初始化
    desire = np.zeros((1, 8))   # 行驶意图初始化
    total_frames = len(frame_tensor) - 1

    # 5. 可视化升级（解决乱码+效果优化）
    plt.ion()  # 交互式模式
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 分屏显示：原始帧+预测结果
    fig.suptitle("车道线预测结果", fontsize=14, fontweight='bold')

    # 子图1：原始视频帧
    ax1.set_title("原始视频帧", fontsize=12)
    ax1.axis('off')  # 关闭坐标轴，更清晰
    img_display = ax1.imshow(cv2.cvtColor(raw_frames[0], cv2.COLOR_BGR2RGB))

    # 子图2：车道线预测（优化线条+标注）
    ax2.set_title("车道线/路径预测", fontsize=12)
    ax2.set_xlabel("横向像素", fontsize=10)
    ax2.set_ylabel("纵向像素", fontsize=10)
    ax2.set_ylim(0, 191)
    ax2.invert_xaxis()  # 匹配驾驶视角（左/右对齐）
    ax2.grid(alpha=0.2, linestyle='--')  # 轻量化网格

    # 初始化预测线条（颜色标准化+标签清晰）
    left_line, = ax2.plot([], [], 'b-', linewidth=2.5, label='左车道线')
    right_line, = ax2.plot([], [], 'r-', linewidth=2.5, label='右车道线')
    path_line, = ax2.plot([], [], 'g-', linewidth=2, label='预测路径')
    ax2.legend(loc='lower left', fontsize=9)  # 图例位置优化

    # 6. 逐帧推理+可视化更新
    print(f"\n🚀 开始推理（共{total_frames}帧，按Q键退出）")
    try:
        for i in range(total_frames):
            # 模型推理（核心逻辑无改动）
            input_tensor = np.vstack(frame_tensor[i:i+2])[None]
            outputs = model.predict([input_tensor, desire, state], verbose=0)
            pred_result = parser(outputs)
            state = outputs[-1]

            # 更新预测线条（对齐维度）
            left_line.set_data(pred_result["lll"][0], range(192))
            right_line.set_data(pred_result["rll"][0], range(192))
            path_line.set_data(pred_result["path"][0], range(192))

            # 更新原始帧显示
            if i < len(raw_frames):
                img_display.set_data(cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2RGB))

            # 刷新画布
            fig.canvas.draw()
            fig.canvas.flush_events()

            # 键盘退出（仅保留Q键，删除冗余自动退出）
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print("🛑 用户按Q键退出推理")
                break

            print(f"✅ 完成第{i+1}/{total_frames}帧推理")

    finally:
        # 资源释放（彻底+规范）
        print("\n🧹 释放资源中...")
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()
        # 强制清除CV2残留
        cv2.waitKey(1)
        print("🎉 程序正常结束")

if __name__ == "__main__":
    # 全局异常捕获（更专业）
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序异常终止：{str(e)}")
        sys.exit(1)
