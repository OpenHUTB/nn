"""
配置模块：支持环境变量加载+精细化事故判断配置
"""
import os
from dotenv import load_dotenv

# 加载.env环境文件（优先读取环境变量，无则用默认值）
load_dotenv()

# ====================== YOLO模型配置 ======================
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

# ====================== 检测源配置 ======================
# 支持环境变量指定：0=摄像头，或视频路径（如"test.mp4"）
DETECTION_SOURCE = os.getenv("DETECTION_SOURCE", 0)
# 转换为整数（摄像头）或字符串（视频路径）
if DETECTION_SOURCE.isdigit():
    DETECTION_SOURCE = int(DETECTION_SOURCE)

# ====================== 事故识别配置 ======================
ACCIDENT_CLASSES = [0, 2, 7]  # 0=人，2=汽车，7=卡车
MIN_VEHICLE_COUNT = int(os.getenv("MIN_VEHICLE_COUNT", 2))
PERSON_VEHICLE_CONTACT = os.getenv("PERSON_VEHICLE_CONTACT", "True").lower() == "true"
# 新增：行人和车辆的距离阈值（像素），小于该值才判定为接触
PERSON_VEHICLE_DISTANCE_THRESHOLD = int(os.getenv("PERSON_VEHICLE_DISTANCE_THRESHOLD", 50))

# ====================== 帧处理配置 ======================
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", 640))
RESIZE_HEIGHT = int(os.getenv("RESIZE_HEIGHT", 480))

# ====================== 依赖配置 ======================
REQUIRED_PACKAGES = [
    "ultralytics>=8.0.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "torch>=2.0.0",
    "python-dotenv>=1.0.0"  # 新增：支持.env文件
]
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# ====================== 输出配置 ======================
# 新增：是否保存检测结果视频
SAVE_RESULT_VIDEO = os.getenv("SAVE_RESULT_VIDEO", "False").lower() == "true"
RESULT_VIDEO_PATH = os.getenv("RESULT_VIDEO_PATH", "detection_result.mp4")
