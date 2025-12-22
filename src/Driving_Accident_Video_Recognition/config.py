"""
配置模块：支持环境变量/`.env`加载 + 精细化事故判断配置
优化点：健壮性提升+代码简洁性+可读性增强
"""
import os
from typing import Any
from dotenv import load_dotenv

# 加载.env环境文件（优先读取环境变量，无则用代码默认值）
load_dotenv()


def get_env_config(key: str, default: Any, config_type: type) -> Any:
    """
    统一处理环境变量读取+类型转换（减少重复代码，提升健壮性）
    :param key: 环境变量键名
    :param default: 类型匹配的默认值
    :param config_type: 目标类型（int/float/bool/str等）
    :return: 转换后的配置值（转换失败自动回退到默认值）
    """
    env_value = os.getenv(key)
    if env_value is None:
        return default
    try:
        if config_type == bool:
            # 布尔类型特殊处理："True"/"true"转True，其余转False
            return env_value.strip().lower() == "true"
        return config_type(env_value)
    except (ValueError, TypeError):
        # 类型转换失败时，回退到默认值
        return default


# ====================== YOLO模型配置 ======================
# YOLO预训练模型路径（默认轻量型yolov8n.pt，适合实时检测）
YOLO_MODEL_PATH = get_env_config("YOLO_MODEL_PATH", "yolov8n.pt", str)
# 检测置信度阈值（0-1，值越高检测越严格，默认0.5平衡精度与召回）
CONFIDENCE_THRESHOLD = get_env_config("CONFIDENCE_THRESHOLD", 0.5, float)


# ====================== 检测源配置 ======================
# 检测源：支持摄像头（整数设备号）或视频文件路径（字符串）
DETECTION_SOURCE = os.getenv("DETECTION_SOURCE", "0")
try:
    # 尝试转换为整数（对应摄像头设备号，如0=默认摄像头）
    DETECTION_SOURCE = int(DETECTION_SOURCE)
except ValueError:
    # 转换失败则视为视频文件路径（保持字符串）
    pass


# ====================== 事故识别核心配置 ======================
# 事故检测关注的YOLO类别（0=person/行人、2=car/汽车、7=truck/卡车）
ACCIDENT_CLASSES = [0, 2, 7]
# 多车事故判定阈值：至少检测到N辆车辆才判定为多车事故
MIN_VEHICLE_COUNT = get_env_config("MIN_VEHICLE_COUNT", 2, int)
# 是否开启“人车接触”事故判定（True=开启，检测到行人+车辆即判定）
PERSON_VEHICLE_CONTACT = get_env_config("PERSON_VEHICLE_CONTACT", True, bool)
# 人车接触距离阈值（像素）：行人和车辆框中心距离＜该值时，判定为接触
PERSON_VEHICLE_DISTANCE_THRESHOLD = get_env_config(
    "PERSON_VEHICLE_DISTANCE_THRESHOLD", 50, int
)


# ====================== 帧处理配置（平衡速度与精度） ======================
# 检测帧缩放宽度（默认640，YOLO推荐输入尺寸，兼顾速度）
RESIZE_WIDTH = get_env_config("RESIZE_WIDTH", 640, int)
# 检测帧缩放高度（默认480，与宽度配合保持合理比例）
RESIZE_HEIGHT = get_env_config("RESIZE_HEIGHT", 480, int)


# ====================== 依赖包配置（自动安装时使用） ======================
REQUIRED_PACKAGES = [
    "ultralytics>=8.0.0",  # YOLOv8核心依赖
    "opencv-python>=4.8.0",  # 视频/图像读取、绘制标注
    "numpy>=1.24.0",  # 数值计算（坐标/距离运算）
    "torch>=2.0.0",  # YOLO模型推理（PyTorch后端）
    "python-dotenv>=1.0.0"  # 加载.env环境变量
]
# PyPI镜像源（加速国内环境的依赖安装）
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


# ====================== 检测结果输出配置 ======================
# 是否保存检测结果视频（True=保存，False=不保存）
SAVE_RESULT_VIDEO = get_env_config("SAVE_RESULT_VIDEO", False, bool)
# 检测结果视频保存路径（默认输出到项目根目录）
RESULT_VIDEO_PATH = get_env_config("RESULT_VIDEO_PATH", "detection_result.mp4", str)
