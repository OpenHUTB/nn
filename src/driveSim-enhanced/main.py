#!/usr/bin/env python
import signal
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
import argparse
import time

# 自定义模块导入
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client

# ------------------------- 常量定义 -------------------------
MAX_DISTANCE = 140.      # 最大距离（米）
LANE_OFFSET = 1.8        # 车道偏移量
MAX_REL_V = 10.          # 最大相对速度
LEAD_X_SCALE = 10        # 前车X轴缩放
LEAD_Y_SCALE = 10        # 前车Y轴缩放
NBFRAME = 10000          # 处理的帧数
INPUT_WIDTH = 1164       # 输入图像宽度
INPUT_HEIGHT = 874       # 输入图像高度

# ------------------------- 信号处理函数 -------------------------
def signal_handler(sig, frame):
    """处理Ctrl+C中断信号，优雅关闭客户端"""
    print("\n正在关闭客户端...")
    client.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ------------------------- DeepGTAV客户端连接 -------------------------
client = Client(ip="localhost", port=8000)

# ------------------------- 加载模型 -------------------------
print("正在加载Supercombo模型...")
supercombo = load_model('/home/idir/Bureau/modeld-master/models/supercombo.keras')
print("模型加载完成！")

# ------------------------- 图像处理函数 -------------------------
def frame_to_tensorframe(frame):
    """
    将帧转换为张量格式，为模型输入做准备
    将YUV格式图像转换为6通道的张量格式
    """
    H = (frame.shape[0] * 2) // 3  # 计算高度
    W = frame.shape[1]              # 计算宽度
    
    # 初始化6通道张量
    in_img1 = np.zeros((6, H // 2, W // 2), dtype=np.uint8)
    
    # 提取Y通道的4个子采样区域
    in_img1[0] = frame[0:H:2, 0::2]      # Y通道子区域1
    in_img1[1] = frame[1:H:2, 0::2]      # Y通道子区域2
    in_img1[2] = frame[0:H:2, 1::2]      # Y通道子区域3
    in_img1[3] = frame[1:H:2, 1::2]      # Y通道子区域4
    in_img1[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))  # U通道
    in_img1[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))  # V通道
    
    return in_img1

def vidframe2img_yuv_reshaped():
    """
    从DeepGTAV客户端接收消息并将帧转换为YUV格式
    返回：BGR图像, YUV420格式图像
    """
    message = client.recvMessage()
    
    # 将接收到的帧转换为numpy数组
    frame = frame2numpy(message['frame'], (INPUT_WIDTH, INPUT_HEIGHT))
    
    # 转换为YUV色彩空间
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    
    return frame, img_yuv.reshape((INPUT_HEIGHT * 3 // 2, INPUT_WIDTH))

def vidframe2frame_tensors():
    """
    从视频帧获取模型输入张量
    返回：原始BGR图像, 处理后的模型输入张量
    """
    # 获取YUV图像
    frame, img = vidframe2img_yuv_reshaped()
    
    # 使用模型内参进行图像变换
    imgs_med_model = transform_img(
        img, 
        from_intr=eon_intrinsics,      # 源相机内参
        to_intr=medmodel_intrinsics,   # 目标模型内参
        yuv=True,
        output_size=(512, 256)          # 输出尺寸
    )
    
    # 转换为张量并归一化到[-1, 1]范围
    f2t = frame_to_tensorframe(np.array(imgs_med_model)).astype(np.float32) / 128.0 - 1.0
    
    return frame, f2t

# ------------------------- 初始化状态 -------------------------
state = np.zeros((1, 512))   # 模型状态向量
desire = np.zeros((1, 8))    # 驾驶意图向量

# ------------------------- 配置Vpilot场景 -------------------------
# 创建驾驶场景：设置驾驶模式、天气、车辆、时间、位置
scenario = Scenario(
    drivingMode=[786603, 100.0],  # 驾驶模式配置
    weather='EXTRASUNNY',          # 晴朗天气
    vehicle='blista',              # 使用blista车型
    time=[12, 0],                  # 中午12点
    location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]  # 初始位置
)

# 配置数据集参数
dataset = Dataset(rate=20, frame=[INPUT_WIDTH, INPUT_HEIGHT])  # 20Hz采样率

# 发送启动命令
client.sendMessage(Start(scenario=scenario, dataset=dataset))
print("DeepGTAV客户端已启动，开始处理视频流...")

# ------------------------- 主循环 -------------------------
print(f"开始处理 {NBFRAME} 帧图像...")

# 初始化第一帧
frame, frame_tensors1 = vidframe2frame_tensors()

# 主处理循环
for i in tqdm(range(NBFRAME - 1), desc="处理进度"):
    # 获取下一帧
    frame, frame_tensors2 = vidframe2frame_tensors()
    
    # 准备模型输入：堆叠前后两帧
    inputs = [np.vstack([frame_tensors1, frame_tensors2])[None], desire, state]
    
    # 执行模型推理
    outs = supercombo.predict(inputs, verbose=0)  # verbose=0 减少输出
    
    # 解析模型输出结果
    parsed = parser(outs)
    
    # 更新状态：状态向量和姿态信息用于下一帧
    state = outs[-1]      # 保存状态用于下一帧
    pose = outs[-2]       # 保存姿态信息
    
    # 更新帧缓存
    frame_tensors1 = frame_tensors2
    
    # ------------------------- 可视化 -------------------------
    # 只在特定帧或间隔显示，避免过慢
    if i % 10 == 0:  # 每10帧显示一次
        plt.clf()
        
        # 子图1：显示原始摄像头画面
        plt.subplot(1, 2, 1)
        plt.title("原始视频画面")
        plt.imshow(frame, aspect="auto")
        plt.axis('off')  # 隐藏坐标轴
        
        # 子图2：显示车道线预测结果
        plt.subplot(1, 2, 2)
        plt.title("车道线预测结果")
        
        # 绘制左车道线（蓝色）
        plt.plot(parsed["lll"][0], range(0, 192), "b-", linewidth=2, label="左车道线")
        
        # 绘制右车道线（红色）
        plt.plot(parsed["rll"][0], range(0, 192), "r-", linewidth=2, label="右车道线")
        
        # 绘制预测路径（绿色）
        plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=2, label="预测路径")
        
        # 反转X轴以匹配标准坐标系（左正右负）
        plt.gca().invert_xaxis()
        plt.xlabel("横向偏移（像素）")
        plt.ylabel("纵向距离（像素）")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)  # 添加半透明网格
        
        plt.tight_layout()  # 自动调整布局
        plt.pause(0.001)   # 短暂暂停以更新显示

# ------------------------- 清理和退出 -------------------------
print("\n处理完成！正在关闭...")
client.close()
plt.close('all')  # 关闭所有图形窗口
print("程序正常退出")
