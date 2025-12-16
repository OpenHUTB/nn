#!/usr/bin/env python3
"""
openpilot-model 车道线与路径预测主程序
核心功能：
1. 读取视频文件并预处理帧数据（BGR→YUV_I420、尺寸归一、张量转换）
2. 加载supercombo预训练模型，逐帧推理左/右车道线、行驶路径
3. 轻量化可视化：双窗口展示原始帧+预测结果，支持Q键退出
适配环境：Python 3.7~3.10，TensorFlow 2.x，资源受限的虚拟机/低配CPU
"""
import sys
import os
import numpy as np          # 数值计算，处理张量/数组
import cv2                  # 视频读取、帧格式转换、尺寸调整
import matplotlib.pyplot as plt  # 预测结果可视化
from tqdm import tqdm       # （潜在依赖）进度条展示（本版未启用，保留兼容）
from tensorflow.keras.models import load_model  # 加载预训练Keras模型

# 从openpilot的common模块导入核心依赖
from common.transformations.camera import transform_img, eon_intrinsics  # 相机内参转换、图像变换
from common.transformations.model import medmodel_intrinsics              # 模型输入内参配置
from common.tools.lib.parser import parser                                # 模型输出解析工具

def frames_to_tensor(frames):
    """
    将预处理后的YUV帧转换为模型输入的6通道张量
    参数：
        frames: 预处理后的YUV帧数组，形状为 (帧数量, 384, 512)
    返回：
        tensor: 6通道张量，形状为 (帧数量, 6, 192, 256)，归一化到 [-1, 1]
    """
    if len(frames) == 0:
        return np.array([])
    H = (frames.shape[1] * 2) // 3  # YUV_I420格式的Y通道高度
    W = frames.shape[2]             # 帧宽度
    # 初始化6通道张量（适配supercombo模型输入规格）
    tensor = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.float32)
    # 填充前4通道（Y通道的4个子采样）
    tensor[:, 0] = frames[:, 0:H:2, 0::2]
    tensor[:, 1] = frames[:, 1:H:2, 0::2]
    tensor[:, 2] = frames[:, 0:H:2, 1::2]
    tensor[:, 3] = frames[:, 1:H:2, 1::2]
    # 填充后2通道（U/V通道下采样）
    tensor[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    tensor[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    # 归一化到[-1, 1]（匹配模型训练时的输入范围）
    return tensor / 128.0 - 1.0

def preprocess_frames(imgs):
    """
    帧数据预处理：转换相机内参、调整尺寸、适配模型输入
    参数：
        imgs: 原始YUV帧数组，形状为 (帧数量, 384, 512)
    返回：
        转换后的张量（调用frames_to_tensor）
    """
    if not imgs:
        return np.array([])
    processed = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
    for i, img in enumerate(imgs):
        try:
            # 转换图像内参（从eon相机内参→模型输入内参），输出YUV格式、256x256尺寸
            processed[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True, output_size=(512, 256))
        except:
            # 预处理失败时填充空帧，避免程序中断
            processed[i] = np.zeros((384, 512), dtype=np.uint8)
    # 转换为模型输入张量
    return frames_to_tensor(processed)

def read_video_with_opencv(video_path, max_frames=10):
    """
    轻量化视频读取：仅读取指定帧数，转换为YUV_I420格式并调整尺寸
    参数：
        video_path: 视频文件路径
        max_frames: 最大读取帧数（默认10帧，降低CPU/内存压力）
    返回：
        imgs: 预处理后的YUV帧数组
        raw_frames: 原始BGR帧数组（用于可视化）
    异常：
        视频无法打开时抛出异常，提示安装FFmpeg
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频：{video_path}，请安装FFmpeg（sudo apt install ffmpeg）")
    imgs = []          # 存储预处理后的YUV帧
    raw_frames = []    # 存储原始BGR帧（用于可视化）
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break  # 视频帧读取完毕，提前退出
        raw_frames.append(frame)
        # BGR→YUV_I420（匹配模型输入格式）
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        # 调整尺寸到512x384（统一输入规格），使用INTER_AREA插值（低分辨率下更清晰）
        yuv_resized = cv2.resize(yuv, (512, 384), interpolation=cv2.INTER_AREA)
        imgs.append(yuv_resized)
    cap.release()  # 释放视频流资源
    return imgs, raw_frames

def main():
    """
    主程序入口：
    1. 解析命令行参数，校验视频/模型文件存在性
    2. 加载模型、读取视频、预处理帧数据
    3. 逐帧推理，轻量化可视化预测结果
    4. 异常捕获+资源释放，保证程序健壮性
    """
    # 1. 命令行参数校验（仅接受1个参数：视频文件路径）
    if len(sys.argv) != 2:
        print("用法: python main.py <视频文件路径>")
        sys.exit(1)
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 - {video_path}")
        sys.exit(1)

    # 2. 模型文件路径校验
    model_path = "models/supercombo.h5"
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        sys.exit(1)

    # 3. 加载supercombo预训练模型（compile=False：仅推理，不编译训练流程）
    try:
        print(f"加载模型：{model_path}")
        supercombo = load_model(model_path, compile=False)
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        sys.exit(1)

    # 4. 读取视频帧（轻量化模式，仅10帧）
    try:
        print(f"读取视频：{video_path}（仅10帧，轻量化模式）")
        imgs, raw_frames = read_video_with_opencv(video_path)
        if not imgs:
            print("错误：未读取到帧")
            sys.exit(1)
    except Exception as e:
        print(f"视频读取失败：{str(e)}")
        sys.exit(1)

    # 5. 帧数据预处理（转换为模型输入张量）
    print("预处理帧数据...")
    frame_tensors = preprocess_frames(imgs)
    if frame_tensors.size == 0:
        print("错误：预处理无有效数据")
        sys.exit(1)

    # 6. 初始化模型推理状态
    state = np.zeros((1, 512))  # 模型状态张量（保持帧间推理连续性）
    desire = np.zeros((1, 8))   # 行驶意图张量（默认无特定意图）

    # 7. 轻量化可视化配置（单窗口+预创建线条，降低渲染压力）
    plt.ion()  # 开启Matplotlib交互模式（支持实时刷新）
    fig, ax = plt.subplots(figsize=(8, 6))  # 单个可视化窗口（避免多窗口资源占用）
    ax.set_title("车道线预测（蓝=左车道，红=右车道，绿=路径）")
    ax.set_ylim(0, 191)  # 固定Y轴范围（减少重绘计算）
    ax.invert_xaxis()     # 反转X轴（匹配车辆前视视角）
    ax.grid(alpha=0.3)    # 浅灰色网格（辅助观察，不占资源）

    # 预创建三条线条（避免每次重绘新建对象，降低渲染耗时）
    lll_line, = ax.plot([], [], "b-", linewidth=3, label="左车道线")
    rll_line, = ax.plot([], [], "r-", linewidth=3, label="右车道线")
    path_line, = ax.plot([], [], "g-", linewidth=2, label="预测路径")
    ax.legend()  # 显示图例

    # 8. 逐帧推理 + 轻量化可视化
    print(f"\n开始推理+可视化（共{len(frame_tensors)-1}帧，按Q键退出）...")
    for i in range(len(frame_tensors) - 1):
        try:
            # 8.1 模型推理（输入：连续2帧张量+意图+状态）
            inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
            outs = supercombo.predict(inputs, verbose=0)  # verbose=0：关闭推理进度条
            parsed = parser(outs)  # 解析模型输出（提取车道线/路径坐标）
            state = outs[-1]       # 更新模型状态（保持帧间连续性）

            # 8.2 轻量化更新可视化（仅更新线条数据，不重绘整个窗口）
            lll_line.set_data(parsed["lll"][0], range(192))  # 更新左车道线坐标
            rll_line.set_data(parsed["rll"][0], range(192))  # 更新右车道线坐标
            path_line.set_data(parsed["path"][0], range(192))# 更新行驶路径坐标
            fig.canvas.draw()      # 轻量重绘（仅更新变化的线条）
            fig.canvas.flush_events()  # 强制刷新窗口（避免卡顿）

            # 8.3 显示原始帧（缩小尺寸+延长等待时间，降低CPU压力）
            if i < len(raw_frames):
                # 原始帧缩小到480x270（降低渲染压力）
                cv2.imshow("原始帧", cv2.resize(raw_frames[i], (480, 270)))
                # 等待100ms（延长等待时间，给CPU喘息，支持Q键退出）
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    print("用户按Q键退出")
                    break

            print(f"✅ 帧 {i+1}/{len(frame_tensors)-1} 完成")

        except Exception as e:
            # 单帧推理失败不中断整体流程，仅打印错误提示
            print(f"⚠️  帧 {i+1} 失败：{str(e)}")
            continue

    # 9. 释放资源（避免内存泄漏/窗口残留）
    print("\n🎉 处理完成！")
    plt.ioff()           # 关闭Matplotlib交互模式
    plt.close()          # 关闭可视化窗口
    cv2.destroyAllWindows()  # 关闭OpenCV原始帧窗口

# 程序入口（避免模块导入时执行）
if __name__ == "__main__":
    main()
# 注：已修复路径问题，common模块需放置在项目根目录或通过sys.path添加路径
