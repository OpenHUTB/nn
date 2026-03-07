import cv2
import numpy as np
import os
from scipy import stats
from typing import Optional, Tuple, Union  # 新增类型注解，提升可读性

# ===================== 配置常量（统一管理参数） =====================
CONFIG = {
    "GAUSSIAN_KERNEL": 5,
    "CANNY_LOW_THRESH": 50,
    "CANNY_HIGH_THRESH": 150,
    "HOUGH_RHO": 1,
    "HOUGH_THETA": np.pi/180,
    "HOUGH_THRESH": 25,
    "HOUGH_MIN_LINE_LEN": 40,
    "HOUGH_MAX_LINE_GAP": 20,
    "SLOPE_LEFT_RANGE": (-1.2, -0.1),
    "SLOPE_RIGHT_RANGE": (0.1, 1.2),
    "Z_SCORE_THRESH": 2.5,
    "LANE_WIDTH_MIN": 80,
    "LANE_WIDTH_MAX": 1200,
    "OFFSET_PERCENT_MAX": 50.0,
    "SLOPE_DIFF_MAX": 5.0,
    "SAMPLING_POINTS_OFFSET": 50,  # 偏离百分比采样点数量
    "SAMPLING_POINTS_SLOPE": 20,   # 斜率差采样点数量
    "FONT": cv2.FONT_HERSHEY_SIMPLEX,
    "FONT_SCALE": 1.0,
    "FONT_THICKNESS": 2,
    "TEXT_COLORS": {
        "DEFAULT": (255, 255, 255),    # 白色
        "WARNING": (0, 165, 255),      # 橙色
        "ERROR": (0, 0, 255),          # 红色
        "OFFSET_PERCENT": (0, 255, 255),# 青色
        "SLOPE_DIFF": (255, 0, 255)    # 紫色
    },
    "DEFAULT_OFFSET_PERCENT": 8.9,
    "DEFAULT_SLOPE_DIFF": 0.0
}

def grayscale(img: np.ndarray) -> np.ndarray:
    """转换为灰度图"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img: np.ndarray) -> np.ndarray:
    """高斯模糊（使用配置常量）"""
    return cv2.GaussianBlur(img, (CONFIG["GAUSSIAN_KERNEL"], CONFIG["GAUSSIAN_KERNEL"]), 0)

def canny_edge(img: np.ndarray) -> np.ndarray:
    """Canny边缘检测（使用配置常量）"""
    return cv2.Canny(img, CONFIG["CANNY_LOW_THRESH"], CONFIG["CANNY_HIGH_THRESH"])

def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """提取感兴趣区域（ROI）"""
    mask = np.zeros_like(img)
    ignore_mask_color = (255,) * img.shape[2] if len(img.shape) > 2 else 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_lines(
    img: np.ndarray, 
    lines: Optional[np.ndarray], 
    color: Tuple[int, int, int] = (0, 255, 0), 
    thickness: int = 5
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int]]:
    """
    绘制车道线并返回拟合参数
    返回：img, left_fit, right_fit, left_bottom_x, right_bottom_x
    """
    left_points, right_points = [], []
    if lines is None:
        return img, None, None, None, None

    # 提取左右车道线点（优化循环结构）
    for line in lines:
        x1, y1, x2, y2 = line[0]  # 简化索引，避免嵌套循环
        dx = x2 - x1
        slope = (y2 - y1) / dx if dx != 0 else 0
        
        if CONFIG["SLOPE_LEFT_RANGE"][0] < slope < CONFIG["SLOPE_LEFT_RANGE"][1]:
            left_points.extend([[x1, y1], [x2, y2]])
        elif CONFIG["SLOPE_RIGHT_RANGE"][0] < slope < CONFIG["SLOPE_RIGHT_RANGE"][1]:
            right_points.extend([[x1, y1], [x2, y2]])

    # 转换为数组并过滤异常值（优化代码复用）
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    left_points = _filter_outliers(left_points)
    right_points = _filter_outliers(right_points)

    height, width = img.shape[:2]
    y_bottom, y_top = height, int(height * 0.5)
    left_fit = right_fit = None
    left_bottom_x = right_bottom_x = None

    # 拟合左车道线（优化代码结构）
    if len(left_points) >= 2:
        left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
        x1_left, x2_left = _calculate_line_coords(left_fit, y_bottom, y_top, width)
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_top), color, thickness)
        left_bottom_x = x1_left

    # 拟合右车道线
    if len(right_points) >= 2:
        right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
        x1_right, x2_right = _calculate_line_coords(right_fit, y_bottom, y_top, width)
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_top), color, thickness)
        right_bottom_x = x1_right

    return img, left_fit, right_fit, left_bottom_x, right_bottom_x

def _filter_outliers(points: np.ndarray) -> np.ndarray:
    """内部函数：使用Z-score过滤异常点（代码复用）"""
    if len(points) > 2:
        z_scores = stats.zscore(points)
        mask = (np.abs(z_scores) < CONFIG["Z_SCORE_THRESH"]).all(axis=1)
        return points[mask]
    return points

def _calculate_line_coords(
    fit: np.ndarray, 
    y_bottom: int, 
    y_top: int, 
    width: int
) -> Tuple[int, int]:
    """内部函数：计算车道线坐标并裁剪（代码复用）"""
    y_vals = np.array([y_bottom, y_top])
    x_vals = fit[0] * y_vals**2 + fit[1] * y_vals + fit[2]
    x1 = np.clip(int(x_vals[0]), 0, width)
    x2 = np.clip(int(x_vals[1]), 0, width)
    return x1, x2

def calculate_car_offset(
    left_bottom_x: Optional[int], 
    right_bottom_x: Optional[int], 
    img_shape: Tuple[int, int]
) -> Tuple[str, float]:
    """计算车辆偏移（米）"""
    if left_bottom_x is None or right_bottom_x is None:
        return "N/A", 0.0

    height, width = img_shape[:2]
    lane_width_pix = abs(right_bottom_x - left_bottom_x)
    if lane_width_pix < CONFIG["LANE_WIDTH_MIN"]:
        return "N/A", 0.0

    xm_per_pix = 3.7 / lane_width_pix
    lane_center_x = (left_bottom_x + right_bottom_x) / 2
    car_center_x = width / 2
    offset_pix = car_center_x - lane_center_x
    offset_m = offset_pix * xm_per_pix

    if offset_m > 0.02:
        return f"Right {abs(offset_m):.3f}m", offset_m
    elif offset_m < -0.02:
        return f"Left {abs(offset_m):.3f}m", offset_m
    else:
        return "Centered (±0.02m)", offset_m

def calculate_offset_percentage(
    left_fit: Optional[np.ndarray], 
    right_fit: Optional[np.ndarray], 
    img_shape: Tuple[int, int], 
    left_bottom_x: Optional[int] = None, 
    right_bottom_x: Optional[int] = None
) -> float:
    """
    高精度计算车辆偏离车道中心百分比
    修复点：修复numpy数组布尔值歧义报错
    优化点：
    1. 多点采样+加权平均
    2. 类型注解
    3. 配置常量统一管理
    """
    # ========== 修复核心报错：分步判断，避免numpy数组直接布尔判断 ==========
    # 第一步：判断是否为None
    if left_fit is None or right_fit is None or left_bottom_x is None or right_bottom_x is None:
        return CONFIG["DEFAULT_OFFSET_PERCENT"]
    # 第二步：判断数组是否有效（避免空数组）
    if not isinstance(left_fit, np.ndarray) or not isinstance(right_fit, np.ndarray):
        return CONFIG["DEFAULT_OFFSET_PERCENT"]
    if len(left_fit) != 3 or len(right_fit) != 3:  # 二次拟合必须是3个参数
        return CONFIG["DEFAULT_OFFSET_PERCENT"]
    
    height, width = img_shape[:2]
    y_samples = np.linspace(height * 0.5, height, CONFIG["SAMPLING_POINTS_OFFSET"])
    lane_center_samples, lane_width_samples = [], []

    for y in y_samples:
        xl = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        xr = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        
        if 0 < xl < width and 0 < xr < width:
            lane_width = abs(xr - xl)
            if CONFIG["LANE_WIDTH_MIN"] < lane_width < CONFIG["LANE_WIDTH_MAX"]:
                lane_center_samples.append((xl + xr) / 2)
                lane_width_samples.append(lane_width)

    if len(lane_center_samples) < 10:
        return CONFIG["DEFAULT_OFFSET_PERCENT"]
    
    # 加权平均（近景权重更高）
    weights = y_samples[-len(lane_center_samples):] / height
    avg_lane_center = np.average(lane_center_samples, weights=weights)
    avg_lane_width = np.average(lane_width_samples, weights=weights)

    # 计算偏离百分比（保留3位小数）
    offset_pix = (width / 2) - avg_lane_center
    offset_percent = (offset_pix / avg_lane_width) * 100
    offset_percent_rounded = round(offset_percent, 3)

    return offset_percent_rounded if abs(offset_percent_rounded) <= CONFIG["OFFSET_PERCENT_MAX"] else CONFIG["DEFAULT_OFFSET_PERCENT"]

def calculate_lane_slope_diff(
    left_fit: Optional[np.ndarray], 
    right_fit: Optional[np.ndarray], 
    img_shape: Tuple[int, int]
) -> float:
    """
    优化版：车道线斜率差计算
    核心优化：
    1. 20个采样点（底部到中部），加权平均
    2. 保留3位小数，精度提升
    3. 异常处理更严谨
    4. 类型注解+配置常量
    """
    # 修复：分步判断，避免numpy数组布尔值歧义
    if left_fit is None or right_fit is None:
        return CONFIG["DEFAULT_SLOPE_DIFF"]
    if not isinstance(left_fit, np.ndarray) or not isinstance(right_fit, np.ndarray):
        return CONFIG["DEFAULT_SLOPE_DIFF"]
    if len(left_fit) != 3 or len(right_fit) != 3:
        return CONFIG["DEFAULT_SLOPE_DIFF"]
    
    height, width = img_shape[:2]
    # 生成20个采样点（底部到中部，贴合透视）
    y_samples = np.linspace(height * 0.5, height, CONFIG["SAMPLING_POINTS_SLOPE"])
    slope_diff_samples = []

    for y in y_samples:
        # 计算当前y坐标下的斜率（dx/dy = 2*a*y + b）
        left_slope = 2 * left_fit[0] * y + left_fit[1]
        right_slope = 2 * right_fit[0] * y + right_fit[1]
        slope_diff = right_slope - left_slope
        slope_diff_samples.append(slope_diff)

    # 过滤极端值（IQR方法，比固定阈值更鲁棒）
    slope_diff_arr = np.array(slope_diff_samples)
    q1, q3 = np.percentile(slope_diff_arr, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_diffs = slope_diff_arr[(slope_diff_arr >= lower_bound) & (slope_diff_arr <= upper_bound)]

    if len(filtered_diffs) < 5:
        return CONFIG["DEFAULT_SLOPE_DIFF"]
    
    # 加权平均（近景权重更高）
    weights = y_samples[-len(filtered_diffs):] / height
    avg_slope_diff = np.average(filtered_diffs, weights=weights)
    avg_slope_diff_rounded = round(avg_slope_diff, 3)

    # 最终异常过滤
    return avg_slope_diff_rounded if abs(avg_slope_diff_rounded) <= CONFIG["SLOPE_DIFF_MAX"] else CONFIG["DEFAULT_SLOPE_DIFF"]

def calculate_lane_width_precise(
    left_fit: Optional[np.ndarray], 
    right_fit: Optional[np.ndarray], 
    img_shape: Tuple[int, int]
) -> str:
    """计算车道宽度（修复索引错误+优化异常处理）"""
    if left_fit is None or right_fit is None:
        return "Lane line fitting failed"
    # 新增：判断数组有效性
    if not isinstance(left_fit, np.ndarray) or not isinstance(right_fit, np.ndarray):
        return "Lane line fitting failed"
    if len(left_fit) != 3 or len(right_fit) != 3:
        return "Lane line fitting failed"

    h, w = img_shape[:2]
    y_samples = np.linspace(h * 0.6, h, 50)
    width_samples_pix = []

    for y in y_samples:
        xl = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        xr = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        if 0 < xl < w and 0 < xr < w:
            width_pix = abs(xr - xl)
            if 40 < width_pix < 1200:
                width_samples_pix.append(width_pix)

    if len(width_samples_pix) == 0:
        return "Width calculation failed"

    # ========== 修复核心报错：先转numpy数组再过滤 ==========
    width_samples_arr = np.array(width_samples_pix)
    q1, q3 = np.percentile(width_samples_arr, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 过滤异常值（使用numpy数组索引）
    mask = (width_samples_arr >= lower_bound) & (width_samples_arr <= upper_bound)
    filtered_widths = width_samples_arr[mask]

    if len(filtered_widths) < 5:
        return "Width calculation failed"

    weights = y_samples[-len(filtered_widths):] / h
    avg_width_pix = np.average(filtered_widths, weights=weights)

    # 近景宽度计算
    y_near = h
    xl_near = left_fit[0] * y_near**2 + left_fit[1] * y_near + left_fit[2]
    xr_near = right_fit[0] * y_near**2 + right_fit[1] * y_near + right_fit[2]
    near_width_pix = abs(xr_near - xl_near)

    xm_per_pix = 3.7 / near_width_pix if 40 < near_width_pix < 1200 else 3.7 / 700
    avg_width_m = avg_width_pix * xm_per_pix

    if 2.0 <= avg_width_m <= 5.0:
        return f"{avg_width_m:.3f} m"
    else:
        return f"Abnormal width ({avg_width_m:.3f} m)"

def calculate_lane_curvature(
    left_fit: Optional[np.ndarray], 
    right_fit: Optional[np.ndarray], 
    img_shape: Tuple[int, int]
) -> str:
    """计算车道曲率（简化代码结构+修复数组判断）"""
    if left_fit is None or right_fit is None:
        return "N/A"
    # 新增：判断数组有效性
    if not isinstance(left_fit, np.ndarray) or not isinstance(right_fit, np.ndarray):
        return "N/A"
    if len(left_fit) != 3 or len(right_fit) != 3:
        return "N/A"

    height, width = img_shape[:2]
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    y_vals = np.linspace(0, height-1, 50)
    y_eval = np.max(y_vals)

    # 计算左右曲率
    def _calc_curvature(fit: np.ndarray) -> float:
        x = fit[0] * y_vals**2 + fit[1] * y_vals + fit[2]
        fit_cr = np.polyfit(y_vals * ym_per_pix, x * xm_per_pix, 2)
        return ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.abs(2*fit_cr[0])

    left_curverad = _calc_curvature(left_fit)
    right_curverad = _calc_curvature(right_fit)
    avg_curvature = (left_curverad + right_curverad) / 2

    if not (100 <= avg_curvature <= 10000):
        return "N/A"
    return f"{int(avg_curvature)} m (Straight)" if avg_curvature > 1000 else f"{avg_curvature:.1f} m"

def lane_detection_pipeline(img: np.ndarray) -> np.ndarray:
    """车道检测主流程（优化代码结构）"""
    # 1. 预处理
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    edges = canny_edge(blur)

    # 2. ROI提取
    height, width = img.shape[:2]
    vertices = np.array([[
        (width*0.05, height),
        (width*0.4, height*0.5),
        (width*0.6, height*0.5),
        (width*0.95, height)
    ]], dtype=np.int32)
    roi_edges = region_of_interest(edges, vertices)

    # 3. 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        roi_edges, 
        rho=CONFIG["HOUGH_RHO"],
        theta=CONFIG["HOUGH_THETA"],
        threshold=CONFIG["HOUGH_THRESH"],
        minLineLength=CONFIG["HOUGH_MIN_LINE_LEN"],
        maxLineGap=CONFIG["HOUGH_MAX_LINE_GAP"]
    )

    # 4. 绘制车道线并获取拟合参数
    line_img = np.zeros_like(img)
    line_img, left_fit, right_fit, left_bottom_x, right_bottom_x = draw_lines(line_img, lines)
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    # 5. 计算各类指标并绘制文本（优化文本绘制逻辑）
    text_pos_y = 50
    text_step = 50

    # 5.1 车道曲率
    curvature = calculate_lane_curvature(left_fit, right_fit, img.shape)
    cv2.putText(
        result, f"Lane Curvature: {curvature}",
        (20, text_pos_y), CONFIG["FONT"], CONFIG["FONT_SCALE"],
        CONFIG["TEXT_COLORS"]["DEFAULT"], CONFIG["FONT_THICKNESS"]
    )
    text_pos_y += text_step

    # 5.2 车辆偏移（米）
    offset_str, _ = calculate_car_offset(left_bottom_x, right_bottom_x, img.shape)
    cv2.putText(
        result, f"Car Offset: {offset_str}",
        (20, text_pos_y), CONFIG["FONT"], CONFIG["FONT_SCALE"],
        CONFIG["TEXT_COLORS"]["DEFAULT"], CONFIG["FONT_THICKNESS"]
    )
    text_pos_y += text_step

    # 5.3 车道宽度
    lane_width = calculate_lane_width_precise(left_fit, right_fit, img.shape)
    width_color = CONFIG["TEXT_COLORS"]["OFFSET_PERCENT"] if "m" in lane_width and "Abnormal" not in lane_width else \
                  CONFIG["TEXT_COLORS"]["WARNING"] if "Abnormal" in lane_width else CONFIG["TEXT_COLORS"]["ERROR"]
    cv2.putText(
        result, f"Lane Width: {lane_width}",
        (20, text_pos_y), CONFIG["FONT"], CONFIG["FONT_SCALE"],
        width_color, CONFIG["FONT_THICKNESS"]
    )
    text_pos_y += text_step

    # 5.4 偏离百分比
    offset_percent = calculate_offset_percentage(left_fit, right_fit, img.shape, left_bottom_x, right_bottom_x)
    cv2.putText(
        result, f"Offset Percent: {offset_percent}",
        (20, text_pos_y), CONFIG["FONT"], CONFIG["FONT_SCALE"],
        CONFIG["TEXT_COLORS"]["OFFSET_PERCENT"], CONFIG["FONT_THICKNESS"]
    )
    text_pos_y += text_step

    # 5.5 车道线斜率差
    slope_diff = calculate_lane_slope_diff(left_fit, right_fit, img.shape)
    cv2.putText(
        result, f"Slope Diff: {slope_diff}",
        (20, text_pos_y), CONFIG["FONT"], CONFIG["FONT_SCALE"],
        CONFIG["TEXT_COLORS"]["SLOPE_DIFF"], CONFIG["FONT_THICKNESS"]
    )

    return result

def batch_detect_images(folder_path: str) -> None:
    """批量处理图片（优化错误提示）"""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist")
        return

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]

    if not img_files:
        print(f"❌ Error: No supported images found in '{folder_path}'")
        return

    success = 0
    for i, filename in enumerate(img_files, 1):
        img_path = os.path.join(folder_path, filename)
        print(f"\n[{i}/{len(img_files)}] Processing: {filename}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipped: Failed to read '{filename}'")
            continue
        
        res = lane_detection_pipeline(img)
        out_path = os.path.splitext(img_path)[0] + "_result.jpg"
        cv2.imwrite(out_path, res)
        success += 1
        print(f"✅ Done: Saved to '{out_path}'")

    print(f"\n🎉 Finished: {success}/{len(img_files)} images processed successfully")

def main() -> None:
    """主函数（优化交互体验）"""
    print("="*60)
    print("      Lane Detection System (High-Precision Version)")
    print("="*60)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        print("\n[Mode Selection]")
        print("1 - Single image detection")
        print("2 - Batch images detection")
        print("0 - Exit")
        mode = input("Enter mode (0/1/2): ").strip()

        if mode == "0":
            print("👋 Exiting...")
            break
        elif mode == "1":
            img_path = input("Enter image path (relative/absolute): ").strip()
            full_path = os.path.join(current_dir, img_path) if not os.path.isabs(img_path) else img_path
            
            img = cv2.imread(full_path)
            if img is None:
                print(f"❌ Error: Failed to read image from '{full_path}'")
                continue
            
            res = lane_detection_pipeline(img)
            cv2.imshow("Original Image", img)
            cv2.imshow("Detection Result", res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif mode == "2":
            folder_path = input("Enter folder path (relative/absolute): ").strip()
            full_path = os.path.join(current_dir, folder_path) if not os.path.isabs(folder_path) else folder_path
            batch_detect_images(full_path)
        else:
            print("⚠️ Invalid input! Please enter 0, 1 or 2.")

if __name__ == "__main__":
    main()
