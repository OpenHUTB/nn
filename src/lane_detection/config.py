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

    # 步骤4：视频模式（帧间平滑）
    "ema_alpha": 0.3,
    "video_fourcc": "XVID",

    # 步骤5：曲率半径与车辆偏移计算
    # 像素 → 米转换因子（鸟瞰图视角下）
    "ym_per_pix": 30 / 720,        # 纵向：约 30 米对应 720 像素
    "xm_per_pix": 3.7 / 700,       # 横向：单车道 3.7 米对应约 700 像素
    # 曲率计算位置（图像高度的比例，0.0=顶部 1.0=底部，车辆在底部）
    "curvature_eval_ratio": 1.0,
    # 是否默认显示曲率与偏移信息（advanced / video 模式）
    "show_metrics": True,

    # 步骤6：车道偏离预警
    # 偏移阈值（米）
    "warning_offset_caution": 0.15,   # 超过此值触发 注意
    "warning_offset_danger": 0.40,    # 超过此值触发 危险
    # 曲率半径阈值（米）
    "warning_curve_caution": 500,     # 低于此值触发 注意
    "warning_curve_danger": 200,      # 低于此值触发 危险
    # 是否默认启用预警（advanced / video 模式）
    "show_warning": True,
}
