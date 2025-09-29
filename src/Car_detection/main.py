#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser
from common.numpy_fast import interp

# 配置参数
MODEL_PATH = 'models/supercombo.keras'
FRAME_SIZE = (512, 256)  # 模型输入尺寸
MAX_FRAMES = 500  # 最大处理帧数
LANE_CONFIDENCE_THRESHOLD = 0.5  # 车道线置信度阈值

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.model = None
        self.frame_buffer = []
        self.processed_frames = []
        self.state = np.zeros((1, 512))  # 模型状态初始化
        self.desire = np.zeros((1, 8))   # 期望状态初始化

    def load_video(self):
        """加载视频并验证有效性"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_path}")
        
        frame_count = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)
        print(f"视频加载成功，共处理 {frame_count} 帧")
        return frame_count

    def preprocess_frame(self, frame):
        """预处理单帧图像用于模型输入"""
        # 转换为YUV格式
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        # 调整 intrinsics
        transformed = transform_img(
            yuv_frame.reshape((874*3//2, 1164)),
            from_intr=eon_intrinsics,
            to_intr=medmodel_intrinsics,
            yuv=True,
            output_size=FRAME_SIZE
        )
        return transformed

    def prepare_tensor(self, frames):
        """将预处理后的帧转换为模型输入张量"""
        H, W = FRAME_SIZE
        tensor = np.zeros((len(frames), 6, H//2, W//2), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            # 提取不同通道
            tensor[i, 0] = frame[0:H:2, 0::2]  # 偶数行偶数列
            tensor[i, 1] = frame[1:H:2, 0::2]  # 奇数行偶数列
            tensor[i, 2] = frame[0:H:2, 1::2]  # 偶数行奇数列
            tensor[i, 3] = frame[1:H:2, 1::2]  # 奇数行奇数列
            tensor[i, 4] = frame[H:H+H//4].reshape((H//2, W//2))  # 下采样通道1
            tensor[i, 5] = frame[H+H//4:H+H//2].reshape((H//2, W//2))  # 下采样通道2
        
        # 归一化
        return tensor / 128.0 - 1.0

    def load_model(self):
        """加载模型并验证"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        
        try:
            self.model = load_model(MODEL_PATH)
            print("模型加载成功")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def process_sequence(self):
        """处理视频序列并进行推理"""
        frame_count = self.load_video()
        self.load_model()

        # 读取并预处理帧
        for _ in tqdm(range(frame_count), desc="预处理帧"):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_buffer.append(frame)
            processed = self.preprocess_frame(frame)
            self.processed_frames.append(processed)
        
        self.cap.release()

        # 转换为模型输入张量
        input_tensor = self.prepare_tensor(self.processed_frames)
        if len(input_tensor) < 2:
            raise ValueError("有效帧数量不足，无法进行推理")

        # 创建可视化窗口
        cv2.namedWindow("原始图像与预测", cv2.WINDOW_NORMAL)
        plt.figure(figsize=(10, 6))

        # 逐帧推理
        for i in tqdm(range(len(input_tensor) - 1), desc="模型推理"):
            # 准备输入序列（连续两帧）
            inputs = [
                np.vstack(input_tensor[i:i+2])[None],
                self.desire,
                self.state
            ]

            # 模型预测
            outputs = self.model.predict(inputs)
            parsed = parser(outputs)
            self.state = outputs[-1]  # 更新状态

            # 可视化结果
            self.visualize_result(i, parsed)

            # 按'q'退出
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # 清理资源
        cv2.destroyAllWindows()
        plt.close()

    def visualize_result(self, frame_idx, parsed_data):
        """可视化原始图像和预测结果"""
        # 获取原始图像并调整大小
        orig_frame = self.frame_buffer[frame_idx].copy()
        orig_frame = cv2.resize(orig_frame, (800, 450))

        # 绘制车道线和路径预测
        plt.clf()
        
        # 绘制左车道线
        if 'lll' in parsed_data and parsed_data['lll_prob'] > LANE_CONFIDENCE_THRESHOLD:
            lll = parsed_data['lll'][0]
            plt.plot(lll, range(0, 192), 'b-', linewidth=2, label='左车道线')
        
        # 绘制右车道线
        if 'rll' in parsed_data and parsed_data['rll_prob'] > LANE_CONFIDENCE_THRESHOLD:
            rll = parsed_data['rll'][0]
            plt.plot(rll, range(0, 192), 'r-', linewidth=2, label='右车道线')
        
        # 绘制预测路径
        if 'path' in parsed_data:
            path = parsed_data['path'][0]
            plt.plot(path, range(0, 192), 'g-', linewidth=2, label='预测路径')

        plt.title(f"帧 {frame_idx + 1}/{len(self.frame_buffer)}")
        plt.xlabel("横向距离 (m)")
        plt.ylabel("纵向距离 (m)")
        plt.gca().invert_xaxis()  # 翻转X轴，使前方在上方
        plt.legend()
        plt.tight_layout()
        plt.pause(0.001)

        # 在原始图像上叠加文本信息
        cv2.putText(orig_frame, f"帧: {frame_idx + 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("原始图像与预测", orig_frame)

def main():
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <视频文件路径>")
        print("示例: python3 main.py sample_video.hevc")
        sys.exit(1)

    video_path = sys.argv[1]
    try:
        processor = VideoProcessor(video_path)
        processor.process_sequence()
    except Exception as e:
        print(f"处理失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
