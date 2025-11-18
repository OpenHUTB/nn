#!/usr/bin/env python3
import sys
import os

# 直接写死项目根目录（你的nn文件夹路径），无需计算，100%生效
# 修改此行以适配你的项目结构（示例假设 common 在 ~/nn/src/common）
sys.path.append('/home/dacun/nn/src')


# 以下导入顺序不变
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

# 现在肯定能找到common模块了
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser
# ... 其余代码不变 ...


def frames_to_tensor(frames):
    """
    将预处理后的帧转换为模型输入所需的张量格式
    Args:
        frames: 预处理后的帧数组（shape: [帧数, 384, 512]，YUV格式）
    Returns:
        tensor: 模型输入张量（shape: [帧数, 6, 192, 256]），值归一化到[-1, 1]
    """
    if len(frames) == 0:
        return np.array([])
    # 计算帧的高度（H）和宽度（W），H为原高度的2/3（YUV_I420格式中Y通道占2/3空间）
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    # 初始化张量：6个通道（YUV的4个亚采样通道+两个额外特征通道），尺寸下采样为(H//2, W//2)
    tensor = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.float32)
    # 填充Y通道的4个亚采样部分（奇偶行+奇偶列组合）
    tensor[:, 0] = frames[:, 0:H:2, 0::2]  # Y通道：偶数行、偶数列
    tensor[:, 1] = frames[:, 1:H:2, 0::2]  # Y通道：奇数行、偶数列
    tensor[:, 2] = frames[:, 0:H:2, 1::2]  # Y通道：偶数行、奇数列
    tensor[:, 3] = frames[:, 1:H:2, 1::2]  # Y通道：奇数行、奇数列
    # 填充U、V通道（reshape为(H//2, W//2)）
    tensor[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))  # U通道
    tensor[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))  # V通道
    # 归一化：像素值从[0, 255]映射到[-1, 1]
    return tensor / 128.0 - 1.0

def preprocess_frames(imgs):
    """
    对读取的YUV图像帧进行预处理，适配模型输入要求
    Args:
        imgs: 读取的YUV图像帧列表（每个帧shape: [384, 512]）
    Returns:
        预处理后的帧数组，可直接传入frames_to_tensor转换为模型输入
    """
    if not imgs:
        return np.array([])
    # 初始化预处理后帧的数组（shape: [帧数, 384, 512]）
    processed = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
    for i, img in enumerate(imgs):
        try:
            # 图像变换：从相机内参（eon_intrinsics）转换到模型内参（medmodel_intrinsics）
            # 输出YUV格式，尺寸为(512, 256)，适配模型输入要求
            processed[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True, output_size=(512, 256))
        except:
            # 异常处理：变换失败时填充全零帧
            processed[i] = np.zeros((384, 512), dtype=np.uint8)
    return frames_to_tensor(processed)

def read_video_with_opencv(video_path, max_frames=10):  # 关键：帧数从20减到10，进一步降低压力
    """
    使用OpenCV读取视频文件，提取指定最大帧数的帧并转换为YUV格式
    Args:
        video_path: 视频文件路径
        max_frames: 最大读取帧数（默认10帧，轻量化设计）
    Returns:
        imgs: 转换后的YUV格式帧列表（每个帧shape: [384, 512]）
        raw_frames: 原始BGR格式帧列表（用于后续显示原始画面）
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 打开失败时抛出异常，提示安装FFmpeg依赖
        raise Exception(f"无法打开视频：{video_path}，请安装FFmpeg（sudo apt install ffmpeg）")
    imgs = []  # 存储处理后的YUV帧
    raw_frames = []  # 存储原始BGR帧
    # 读取指定最大帧数的帧
    for i in range(max_frames):
        ret, frame = cap.read()  # 读取一帧（ret: 读取成功标识，frame: 帧数据）
        if not ret:
            break  # 无更多帧时退出循环
        raw_frames.append(frame)  # 保存原始BGR帧
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)  # BGR格式转换为YUV_I420格式
        yuv_resized = cv2.resize(yuv, (512, 384), interpolation=cv2.INTER_AREA)  # 调整尺寸为(512, 384)
        imgs.append(yuv_resized)  # 保存处理后的YUV帧
    cap.release()  # 释放视频资源
    return imgs, raw_frames

def main():
    # 命令行参数校验：确保输入格式为 "python main.py <视频文件路径>"
    if len(sys.argv) != 2:
        print("用法: python main.py <视频文件路径>")
        sys.exit(1)
    video_path = sys.argv[1]  # 获取视频文件路径参数
    # 校验视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 - {video_path}")
        sys.exit(1)

    # 模型文件路径配置
    model_path = "models/supercombo.h5"
    # 校验模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        sys.exit(1)

    # 加载预训练模型
    try:
        print(f"加载模型：{model_path}")
        # 加载模型（compile=False：不重新编译，加快加载速度）
        supercombo = load_model(model_path, compile=False)
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        sys.exit(1)

    # 读取视频帧（轻量化：仅读取10帧）
    try:
        print(f"读取视频：{video_path}（仅10帧，轻量化模式）")
        imgs, raw_frames = read_video_with_opencv(video_path)
        if not imgs:
            print("错误：未读取到帧")
            sys.exit(1)
    except Exception as e:
        print(f"视频读取失败：{str(e)}")
        sys.exit(1)

    # 帧数据预处理（转换为模型输入格式）
    print("预处理帧数据...")
    frame_tensors = preprocess_frames(imgs)
    if frame_tensors.size == 0:
        print("错误：预处理无有效数据")
        sys.exit(1)

    # 初始化模型状态和期望向量（模型输入的必要参数）
    state = np.zeros((1, 512))  # 模型状态向量（shape: [1, 512]）
    desire = np.zeros((1, 8))  # 期望行为向量（shape: [1, 8]，如直行、左转、右转等）

    # -------------------------- 轻量化可视化（仅1个窗口，只画车道线） --------------------------
    plt.ion()  # 开启Matplotlib交互模式（支持实时更新图像）
    fig, ax = plt.subplots(figsize=(8, 6))  # 创建单个绘图窗口（减少渲染压力）
    ax.set_title("车道线预测（蓝=左车道，红=右车道，绿=路径）")  # 窗口标题
    ax.set_ylim(0, 191)  # 固定Y轴范围（0-191），减少重绘计算量
    ax.invert_xaxis()     # 反转X轴，匹配驾驶视角（左/右方向与实际一致）
    ax.grid(alpha=0.3)    # 显示透明度为0.3的网格（不占用过多资源）

    # 初始化三条线对象（提前创建，避免每次重绘新建，优化性能）
    lll_line, = ax.plot([], [], "b-", linewidth=3, label="左车道线")  # 蓝色：左车道线
    rll_line, = ax.plot([], [], "r-", linewidth=3, label="右车道线")  # 红色：右车道线
    path_line, = ax.plot([], [], "g-", linewidth=2, label="预测路径")  # 绿色：模型预测行驶路径
    ax.legend()  # 显示图例
    # -------------------------------------------------------------------

    # 逐帧推理+轻量化可视化（核心流程）
    print(f"\n开始推理+可视化（共{len(frame_tensors)-1}帧，按Q键退出）...")
    for i in range(len(frame_tensors) - 1):
        try:
            # 构建模型输入：连续两帧图像张量 + 期望向量 + 状态向量
            inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
            # 模型推理（verbose=0：不输出推理进度，减少冗余）
            outs = supercombo.predict(inputs, verbose=0)
            # 解析模型输出：提取车道线、路径等关键信息
            parsed = parser(outs)
            # 更新模型状态（当前帧输出作为下一帧输入状态）
            state = outs[-1]

            # -------------------------- 仅更新线的数据，不重绘整个窗口 --------------------------
            lll_line.set_data(parsed["lll"][0], range(192))  # 更新左车道线数据（x: 车道线位置，y: 0-191）
            rll_line.set_data(parsed["rll"][0], range(192))  # 更新右车道线数据
            path_line.set_data(parsed["path"][0], range(192))# 更新预测路径数据
            fig.canvas.draw()  # 轻量级重绘（仅更新变化的线，不重绘整个窗口）
            fig.canvas.flush_events()  # 强制刷新窗口，避免卡顿
            # -------------------------------------------------------------------

            # 显示原始帧（简化版：缩小尺寸，减少渲染压力）
            if i < len(raw_frames):
                # 缩小原始帧尺寸为(480, 270)后显示
                cv2.imshow("原始帧", cv2.resize(raw_frames[i], (480, 270)))
                # 等待100ms，支持按Q键退出（延长等待时间，给CPU喘息空间）
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    print("用户按Q键退出")
                    break

            print(f"✅ 帧 {i+1}/{len(frame_tensors)-1} 完成")

        except Exception as e:
            # 异常处理：单帧处理失败时打印日志，继续处理下一帧
            print(f"⚠️  帧 {i+1} 失败：{str(e)}")
            continue

    # 释放资源（简化版：关闭所有窗口，释放内存）
    print("\n🎉 处理完成！")
    plt.ioff()  # 关闭Matplotlib交互模式
    plt.close()  # 关闭Matplotlib绘图窗口
    cv2.destroyAllWindows()  # 关闭OpenCV显示窗口

# 程序入口：当脚本直接运行时执行main函数
if __name__ == "__main__":
    main()