"""
全局配置文件：集中管理参数，新增多语言支持配置
"""
# YOLOv8模型配置
YOLO_MODEL_PATH = "yolov8n.pt"  # 轻量化模型（自动下载）
CONFIDENCE_THRESHOLD = 0.5      # 目标检测置信度阈值（0-1）

# 检测源配置（默认值，可被命令行参数覆盖）
DEFAULT_DETECTION_SOURCE = 0  # 0=电脑摄像头，可改为视频路径如"test_accident.mp4"

# 事故识别配置
ACCIDENT_CLASSES = [0, 2, 7]    # YOLOv8类别：0=人，2=汽车，7=卡车
MIN_VEHICLE_COUNT = 2           # 至少2辆车判定为事故
PERSON_VEHICLE_CONTACT = True   # 行人和车辆同时出现判定为事故

# 帧处理优化配置（低配电脑推荐）
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# 新增：多语言配置
SUPPORTED_LANGUAGES = ["zh", "en"]  # 支持的语言列表：中文、英文

# 依赖包配置
REQUIRED_PACKAGES = [
    "ultralytics>=8.0.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "torch>=2.0.0"
]

# 清华镜像源（加速依赖下载）
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
