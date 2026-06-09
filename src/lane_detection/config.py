"""车道检测模块配置。"""
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = MODULE_DIR / "carla_test.jpg"

CONFIG = {
    # 通用
    "img_path": str(DEFAULT_IMAGE),

    # 步骤1：基础版（Canny + 霍夫）
    "img_path": str(DEFAULT_IMAGE),
    "canny_low": 50,
    "canny_high": 150,
    "gaussian_kernel": (5, 5),
    "roi_scale": [0.05, 0.45, 0.55, 0.95, 0.6],
    "hough_threshold": 15,
    "min_line_length": 20,
    "max_line_gap": 80,

    # 步骤3：高级版（透视变换 + 多项式拟合）
    "sobel_thresh_min": 20,
    "sobel_thresh_max": 100,
    "white_thresh": 200,
    "yellow_h_low": 15,
    "yellow_h_high": 40,
    "yellow_s_low": 80,
    "perspective_src": [0.15, 0.65, 0.43, 0.65, 0.90, 0.95, 0.05, 0.95],
    "perspective_dst": [0.20, 0.0, 0.80, 0.0, 0.80, 1.0, 0.20, 1.0],
    "sliding_windows": 9,
    "sliding_margin": 100,
    "sliding_minpix": 50,
}
