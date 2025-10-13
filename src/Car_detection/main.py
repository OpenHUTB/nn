#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

# 确保导入openpilot的common模块
# 请将下面的路径替换为你实际的openpilot文件夹路径
OPENPILOT_PATH = "D:/nn/openpilot"
if OPENPILOT_PATH not in sys.path:
    sys.path.append(OPENPILOT_PATH)

from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.tools.lib.parser import parser
from common.numpy_fast import interp

# 配置参数 - 集中管理便于修改
class Config:
    MODEL_PATH = 'models/supercombo.keras'
    FRAME_SIZE = (512, 256)
    MAX_FRAMES = 500
    LANE_CONFIDENCE_THRESHOLD = 0.5
    VISUALIZATION_SCALE = (800, 450)  # 可视化窗口大小
    PLOT_DPI = 100
    WAIT_KEY_DELAY = 20  # 可视化延迟(毫秒)


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.model = None
        self.frame_buffer = []          # 存储原始帧
        self.processed_frames = []      # 存储预处理后的帧
        self.state = np.zeros((1, 512)) # 模型状态
        self.desire = np.zeros((1, 8))  # 期望状态
        
        # 初始化matplotlib，避免显示问题
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.ion()  # 开启交互模式

    def load_video(self):
        """加载视频文件并返回帧数"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_path}")
        
        frame_count = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), Config.MAX_FRAMES)
        print(f"视频加载成功，共处理 {frame_count} 帧")
        return frame_count

    def preprocess_frame(self, frame):
        """预处理帧用于模型输入"""
        try:
            # 转换为YUV格式
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            # 应用变换
            transformed = transform_img(
                yuv_frame.reshape((874*3//2, 1164)),
                from_intr=eon_intrinsics,
                to_intr=medmodel_intrinsics,
                yuv=True,
                output_size=Config.FRAME_SIZE
            )
            return transformed
        except Exception as e:
            raise RuntimeError(f"帧预处理失败: {str(e)}")

    def prepare_tensor(self, frames):
        """准备模型输入的张量"""
        H, W = Config.FRAME_SIZE
        tensor = np.zeros((len(frames), 6, H//2, W//2), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            # 提取不同通道
            tensor[i, 0] = frame[0:H:2, 0::2]
            tensor[i, 1] = frame[1:H:2, 0::2]
            tensor[i, 2] = frame[0:H:2, 1::2]
            tensor[i, 3] = frame[1:H:2, 1::2]
            tensor[i, 4] = frame[H:H+H//4].reshape((H//2, W//2))
            tensor[i, 5] = frame[H+H//4:H+H//2].reshape((H//2, W//2))
        
        # 归一化
        return tensor / 128.0 - 1.0

    def load_model(self):
        """加载预训练模型"""
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {Config.MODEL_PATH}")
        
        try:
            self.model = load_model(Config.MODEL_PATH, compile=False)
            print("模型加载成功")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def process_sequence(self):
        """处理视频序列并进行模型推理"""
        try:
            frame_count = self.load_video()
            self.load_model()

            # 预处理所有帧
            print("正在预处理帧...")
            for _ in tqdm(range(frame_count), desc="预处理进度"):
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_buffer.append(frame)
                processed = self.preprocess_frame(frame)
                self.processed_frames.append(processed)
            
            self.cap.release()

            # 检查有效帧数量
            if len(self.processed_frames) < 2:
                raise ValueError("有效帧数量不足，无法进行推理 (至少需要2帧)")

            # 准备输入张量
            input_tensor = self.prepare_tensor(self.processed_frames)
            
            # 创建可视化窗口
            cv2.namedWindow("原始图像与预测", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("原始图像与预测", *Config.VISUALIZATION_SCALE)

            # 模型推理与可视化
            print("开始模型推理与可视化...")
            for i in tqdm(range(len(input_tensor) - 1), desc="推理进度"):
                # 准备输入
                inputs = [
                    np.vstack(input_tensor[i:i+2])[None],
                    self.desire,
                    self.state
                ]

                # 模型预测
                outputs = self.model.predict(inputs, verbose=0)
                parsed = parser(outputs)
                self.state = outputs[-1]  # 更新状态

                # 可视化结果
                self.visualize_result(i, parsed)

                # 检查退出条件
                key = cv2.waitKey(Config.WAIT_KEY_DELAY) & 0xFF
                if key == ord('q'):
                    print("用户中断处理")
                    break
                elif key == ord('p'):
                    print("暂停，按任意键继续")
                    cv2.waitKey(0)

        except Exception as e:
            print(f"处理过程出错: {str(e)}")
        finally:
            # 清理资源
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            plt.close('all')
            plt.ioff()  # 关闭交互模式

    def visualize_result(self, frame_idx, parsed_data):
        """可视化原始图像和预测结果"""
        # 显示原始图像
        orig_frame = self.frame_buffer[frame_idx].copy()
        orig_frame = cv2.resize(orig_frame, Config.VISUALIZATION_SCALE)
        
        # 添加帧信息
        cv2.putText(orig_frame, f"帧: {frame_idx + 1}/{len(self.frame_buffer)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 绘制预测结果
        plt.clf()
        
        # 绘制左车道线
        if 'lll' in parsed_data and parsed_data.get('lll_prob', 0) > Config.LANE_CONFIDENCE_THRESHOLD:
            lll = parsed_data['lll'][0]
            plt.plot(lll, range(0, 192), 'b-', linewidth=2, label='左车道线')
        
        # 绘制右车道线
        if 'rll' in parsed_data and parsed_data.get('rll_prob', 0) > Config.LANE_CONFIDENCE_THRESHOLD:
            rll = parsed_data['rll'][0]
            plt.plot(rll, range(0, 192), 'r-', linewidth=2, label='右车道线')
        
        # 绘制预测路径
        if 'path' in parsed_data:
            path = parsed_data['path'][0]
            plt.plot(path, range(0, 192), 'g-', linewidth=2, label='预测路径')

        plt.title(f"帧 {frame_idx + 1} 预测结果")
        plt.xlabel("横向距离 (m)")
        plt.ylabel("纵向距离 (m)")
        plt.gca().invert_xaxis()  # 反转x轴，使前方在图像上方
        plt.legend()
        plt.tight_layout()
        plt.pause(0.001)  # 刷新图像

        # 显示原始图像
        cv2.imshow("原始图像与预测", orig_frame)


def validate_openpilot_path():
    """验证openpilot路径是否有效"""
    required_modules = [
        "common.transformations.camera",
        "common.transformations.model",
        "common.tools.lib.parser"
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            return False, f"无法导入模块 {module}，请检查openpilot路径是否正确"
    
    return True, "openpilot模块验证成功"


def main():
    # 验证openpilot路径
    valid, msg = validate_openpilot_path()
    if not valid:
        print(f"错误: {msg}")
        sys.exit(1)
    
    # 检查命令行参数
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <视频文件路径>")
        print("示例: python video_processor.py sample_video.mp4")
        print("按键说明: q-退出, p-暂停")
        sys.exit(1)

    video_path = sys.argv[1]
    try:
        processor = VideoProcessor(video_path)
        processor.process_sequence()
        print("处理完成")
    except Exception as e:
        print(f"处理失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
