#!/usr/bin/env python3
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser

def frames_to_tensor(frames):
    if len(frames) == 0:
        return np.array([])
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    tensor = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.float32)
    tensor[:, 0] = frames[:, 0:H:2, 0::2]
    tensor[:, 1] = frames[:, 1:H:2, 0::2]
    tensor[:, 2] = frames[:, 0:H:2, 1::2]
    tensor[:, 3] = frames[:, 1:H:2, 1::2]
    tensor[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    tensor[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return tensor / 128.0 - 1.0

def preprocess_frames(imgs):
    if not imgs:
        return np.array([])
    processed = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
    for i, img in enumerate(imgs):
        try:
            processed[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True, output_size=(512, 256))
        except:
            processed[i] = np.zeros((384, 512), dtype=np.uint8)
    return frames_to_tensor(processed)

def read_video_with_opencv(video_path, max_frames=10):  # 关键：帧数从20减到10，进一步降低压力
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频：{video_path}，请安装FFmpeg（sudo apt install ffmpeg）")
    imgs = []
    raw_frames = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        yuv_resized = cv2.resize(yuv, (512, 384), interpolation=cv2.INTER_AREA)
        imgs.append(yuv_resized)
    cap.release()
    return imgs, raw_frames

def main():
    if len(sys.argv) != 2:
        print("用法: python main.py <视频文件路径>")
        sys.exit(1)
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 - {video_path}")
        sys.exit(1)

    model_path = "models/supercombo.h5"
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        sys.exit(1)

    # 加载模型
    try:
        print(f"加载模型：{model_path}")
        supercombo = load_model(model_path, compile=False)
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        sys.exit(1)

    # 读取视频（仅10帧）
    try:
        print(f"读取视频：{video_path}（仅10帧，轻量化模式）")
        imgs, raw_frames = read_video_with_opencv(video_path)
        if not imgs:
            print("错误：未读取到帧")
            sys.exit(1)
    except Exception as e:
        print(f"视频读取失败：{str(e)}")
        sys.exit(1)

    # 预处理帧
    print("预处理帧数据...")
    frame_tensors = preprocess_frames(imgs)
    if frame_tensors.size == 0:
        print("错误：预处理无有效数据")
        sys.exit(1)

    # 初始化模型状态
    state = np.zeros((1, 512))
    desire = np.zeros((1, 8))

    # -------------------------- 轻量化可视化（仅1个窗口，只画车道线） --------------------------
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
    # -------------------------------------------------------------------

    # 逐帧推理+轻量化可视化
    print(f"\n开始推理+可视化（共{len(frame_tensors)-1}帧，按Q键退出）...")
    for i in range(len(frame_tensors) - 1):
        try:
            # 模型推理
            inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
            outs = supercombo.predict(inputs, verbose=0)
            parsed = parser(outs)
            state = outs[-1]

            # -------------------------- 仅更新线的数据，不重绘整个窗口 --------------------------
            lll_line.set_data(parsed["lll"][0], range(192))  # 只更新左车道线数据
            rll_line.set_data(parsed["rll"][0], range(192))  # 只更新右车道线数据
            path_line.set_data(parsed["path"][0], range(192))# 只更新路径数据
            fig.canvas.draw()  # 轻量重绘（只更改造变的部分）
            fig.canvas.flush_events()  # 强制刷新窗口，避免卡住
            # -------------------------------------------------------------------

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

    # 释放资源（简化版）
    print("\n🎉 处理完成！")
    plt.ioff()
    plt.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()