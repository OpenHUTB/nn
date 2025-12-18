#!/usr/bin/env python3
# 第一步：优先配置Python搜索路径，确保能找到common模块
import sys
import os
import time  # 新增：用于计时/超时提示

# 项目根目录（绝对路径，适配你的虚拟机路径）
PROJECT_ROOT = "/home/dacun/nn"
# 将根目录加入Python搜索路径
sys.path.append(PROJECT_ROOT)
# 验证路径是否添加成功（可选，可删除）
print(f"✅ 项目根目录已添加到Python搜索路径：{PROJECT_ROOT}")

# 第二步：导入依赖库（包括common模块）
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于进度提示
from tensorflow.keras.models import load_model

# 优化点1：设置matplotlib非交互式后端（适配虚拟机无GUI场景，避免plt崩溃）
plt.switch_backend('Agg') if os.environ.get('DISPLAY') is None else plt.switch_backend('TkAgg')

# 现在能正常导入common模块
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser

def frames_to_tensor(frames):
    if len(frames) == 0:
        return np.array([])
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    # 优化点2：numpy向量化重构，替换部分循环，提升张量计算效率（减少虚拟机CPU占用）
    tensor = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.float32)
    tensor[:, 0] = frames[:, 0:H:2, 0::2]
    tensor[:, 1] = frames[:, 1:H:2, 0::2]
    tensor[:, 2] = frames[:, 0:H:2, 1::2]
    tensor[:, 3] = frames[:, 1:H:2, 1::2]
    # 向量化reshape，避免逐元素操作
    tensor[:, 4] = frames[:, H:H+H//4].reshape(-1, H//2, W//2)
    tensor[:, 5] = frames[:, H+H//4:H+H//2].reshape(-1, H//2, W//2)
    return tensor / 128.0 - 1.0

def preprocess_frames(imgs):
    if not imgs:
        return np.array([])
    processed = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
    # 优化点3：批量处理+异常捕获细化，避免单帧错误导致整批失效
    valid_imgs = np.array(imgs, dtype=object)
    mask = np.ones(len(valid_imgs), dtype=bool)
    for i, img in enumerate(valid_imgs):
        try:
            processed[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True, output_size=(512, 256))
        except Exception as e:
            mask[i] = False
            processed[i] = np.zeros((384, 512), dtype=np.uint8)
    if np.sum(~mask) > 0:
        print(f"⚠️  有 {np.sum(~mask)} 帧预处理失败，已填充空帧")
    return frames_to_tensor(processed)

def read_video_with_opencv(video_path, max_frames=10):
    # 优化点4：校验视频格式（仅支持MP4），提前拦截错误
    if not video_path.lower().endswith('.mp4'):
        raise ValueError(f"❌ 仅支持MP4格式视频，当前文件：{video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频：{video_path}，请安装FFmpeg（sudo apt install ffmpeg）")
    # 设置缓存大小，降低虚拟机内存占用
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    imgs = []
    raw_frames = []
    for i in tqdm(range(max_frames), desc="读取视频帧"):
        ret, frame = cap.read()
        if not ret:
            tqdm.write(f"⚠️  视频仅读取到 {i} 帧，已达末尾")
            break
        raw_frames.append(frame)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        yuv_resized = cv2.resize(yuv, (512, 384), interpolation=cv2.INTER_AREA)
        imgs.append(yuv_resized)
    cap.release()
    return imgs, raw_frames

def main():
    # 优化点5：支持默认视频路径，不传参数时优先用默认（更友好）
    default_video_path = os.path.join(PROJECT_ROOT, "sample.mp4")
    if len(sys.argv) == 1:
        if os.path.exists(default_video_path):
            video_path = default_video_path
            print(f"ℹ️  未传入视频路径，使用默认路径：{video_path}")
        else:
            print("❌ 缺少视频文件路径参数！")
            print("✅ 正确用法: python main.py <视频文件路径>")
            print("💡 示例: python main.py /home/dacun/nn/sample.mp4")
            sys.exit(1)
    else:
        video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ 错误：视频文件不存在 - {video_path}")
        sys.exit(1)

    model_path = "models/supercombo.h5"
    model_abs_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(model_abs_path):
        print(f"❌ 错误：模型文件不存在 - {model_abs_path}")
        sys.exit(1)

    # 加载模型（添加计时+超时提示）
    try:
        print(f"📌 加载模型：{model_abs_path}")
        start_time = time.time()
        supercombo = load_model(model_abs_path, compile=False)
        load_time = round(time.time() - start_time, 2)
        print(f"✅ 模型加载完成，耗时 {load_time} 秒")
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        sys.exit(1)

    # 读取视频
    try:
        print(f"📌 读取视频：{video_path}（仅10帧，轻量化模式）")
        imgs, raw_frames = read_video_with_opencv(video_path)
        if not imgs:
            print("❌ 错误：未读取到帧")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 视频读取失败：{str(e)}")
        sys.exit(1)

    # 预处理帧
    print("📌 预处理帧数据...")
    frame_tensors = preprocess_frames(imgs)
    if frame_tensors.size == 0:
        print("❌ 错误：预处理无有效数据")
        sys.exit(1)

    # 初始化模型状态
    state = np.zeros((1, 512))
    desire = np.zeros((1, 8))

    # 轻量化可视化
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("车道线预测（蓝=左车道，红=右车道，绿=路径）")
    ax.set_ylim(0, 191)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    lll_line, = ax.plot([], [], "b-", linewidth=3, label="左车道线")
    rll_line, = ax.plot([], [], "r-", linewidth=3, label="右车道线")
    path_line, = ax.plot([], [], "g-", linewidth=2, label="预测路径")
    ax.legend()

    # 逐帧推理+可视化
    total_frames = len(frame_tensors) - 1
    print(f"\n🚀 开始推理+可视化（共{total_frames}帧，按Q键/5秒无操作自动退出）...")
    try:
        for i in range(total_frames):
            try:
                inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
                outs = supercombo.predict(inputs, verbose=0)
                parsed = parser(outs)
                state = outs[-1]

                # 更新线条数据
                lll_line.set_data(parsed["lll"][0], range(192))
                rll_line.set_data(parsed["rll"][0], range(192))
                path_line.set_data(parsed["path"][0], range(192))
                fig.canvas.draw()
                fig.canvas.flush_events()

                # 优化点6：CV2窗口5秒无操作自动关闭，避免虚拟机卡死
                if i < len(raw_frames):
                    cv2.imshow("原始帧", cv2.resize(raw_frames[i], (480, 270)))
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        print("🛑 用户按Q键退出")
                        break
                    # 5秒无操作自动退出（100ms*50=5秒）
                    if i % 50 == 0 and i != 0:
                        print("⚠️  5秒无操作，自动退出...")
                        break

                print(f"✅ 帧 {i+1}/{total_frames} 完成")

            except Exception as e:
                print(f"⚠️  帧 {i+1} 失败：{str(e)}")
                continue
    finally:
        # 彻底释放资源（适配虚拟机）
        print("\n🧹 释放资源中...")
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()
        
        for _ in range(2):
            cv2.waitKey(1)
        print("🎉 处理完成！")

if __name__ == "__main__":
    main()
