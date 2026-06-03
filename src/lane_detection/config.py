"""车道检测模块配置。"""
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = MODULE_DIR / "carla_test.jpg"

CONFIG = {
    "img_path": str(DEFAULT_IMAGE),
    "canny_low": 50,
    "canny_high": 150,
    "gaussian_kernel": (5, 5),
    "roi_scale": [0.05, 0.45, 0.55, 0.95, 0.6],
    "hough_threshold": 15,
    "min_line_length": 20,
    "max_line_gap": 80,
}
