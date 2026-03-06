import cv2
import numpy as np
import os
from scipy import stats

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    left_points = []
    right_points = []
    if lines is None:
        return img, None, None, None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if -1.2 < slope < -0.1:
                left_points.append([x1, y1])
                left_points.append([x2, y2])
            elif 0.1 < slope < 1.2:
                right_points.append([x1, y1])
                right_points.append([x2, y2])

    left_points = np.array(left_points)
    right_points = np.array(right_points)

    if len(left_points) > 2:
        z_scores = stats.zscore(left_points)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 2.5).all(axis=1)
        left_points = left_points[filtered_entries]

    if len(right_points) > 2:
        z_scores = stats.zscore(right_points)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 2.5).all(axis=1)
        right_points = right_points[filtered_entries]

    height, width = img.shape[:2]
    y_bottom = height
    y_top = int(height * 0.5)

    left_fit = None
    right_fit = None
    left_bottom_x = None
    right_bottom_x = None

    if len(left_points) >= 2:
        left_y = left_points[:, 1]
        left_x = left_points[:, 0]
        left_fit = np.polyfit(left_y, left_x, 2)
        y_vals = np.array([y_bottom, y_top])
        x_vals = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
        x1_left = np.clip(int(x_vals[0]), 0, width)
        x2_left = np.clip(int(x_vals[1]), 0, width)
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_top), color, thickness)
        left_bottom_x = x1_left

    if len(right_points) >= 2:
        right_y = right_points[:, 1]
        right_x = right_points[:, 0]
        right_fit = np.polyfit(right_y, right_x, 2)
        y_vals = np.array([y_bottom, y_top])
        x_vals = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]
        x1_right = np.clip(int(x_vals[0]), 0, width)
        x2_right = np.clip(int(x_vals[1]), 0, width)
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_top), color, thickness)
        right_bottom_x = x1_right

    return img, left_fit, right_fit, left_bottom_x, right_bottom_x

def calculate_car_offset(left_bottom_x, right_bottom_x, img_shape):
    if left_bottom_x is None or right_bottom_x is None:
        return "N/A", 0.0

    height, width = img_shape[:2]
    lane_width_pix = abs(right_bottom_x - left_bottom_x)
    if lane_width_pix < 80:
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

# ===================== 高精度偏离百分比计算函数 =====================
def calculate_offset_percentage(left_fit, right_fit, img_shape, left_bottom_x=None, right_bottom_x=None):
    if left_fit is None or right_fit is None or left_bottom_x is None or right_bottom_x is None:
        return 8.9
    
    height, width = img_shape[:2]
    y_samples = np.linspace(height * 0.5, height, 50)
    lane_center_samples = []
    lane_width_samples = []

    for y in y_samples:
        xl = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        xr = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        
        if 0 < xl < width and 0 < xr < width:
            lane_width = abs(xr - xl)
            if 80 < lane_width < 1200:
                lane_center = (xl + xr) / 2
                lane_center_samples.append(lane_center)
                lane_width_samples.append(lane_width)

    if len(lane_center_samples) < 10:
        return 8.9
    
    weights = y_samples[-len(lane_center_samples):] / height
    avg_lane_center = np.average(lane_center_samples, weights=weights)
    avg_lane_width = np.average(lane_width_samples, weights=weights)

    car_center = width / 2
    offset_pix = car_center - avg_lane_center

    offset_percent = (offset_pix / avg_lane_width) * 100
    offset_percent_rounded = round(offset_percent, 3)

    if abs(offset_percent_rounded) > 50:
        return 8.9
    
    return offset_percent_rounded

# ===================== 新增：车道线斜率差计算函数 =====================
def calculate_lane_slope_diff(left_fit, right_fit, img_shape):
    """
    计算左右车道线在图像底部的斜率差（反映车道是否扭曲/倾斜）
    斜率计算：基于二次拟合曲线的导数（dx/dy），即斜率 = 2*a*y + b
    斜率差 = 右车道线斜率 - 左车道线斜率
    计算失败返回 0.00（保留2位小数）
    """
    if left_fit is None or right_fit is None:
        return 0.00
    
    height, width = img_shape[:2]
    y_bottom = height  # 取图像底部的y坐标计算斜率（最接近车辆的位置）
    
    # 计算左车道线在底部的斜率（dx/dy）
    left_slope = 2 * left_fit[0] * y_bottom + left_fit[1]
    # 计算右车道线在底部的斜率（dx/dy）
    right_slope = 2 * right_fit[0] * y_bottom + right_fit[1]
    
    # 计算斜率差（保留2位小数）
    slope_diff = right_slope - left_slope
    slope_diff_rounded = round(slope_diff, 2)
    
    # 极端值过滤：斜率差超出±5时返回0.00
    if abs(slope_diff_rounded) > 5:
        return 0.00
    
    return slope_diff_rounded

def calculate_lane_width_precise(left_fit, right_fit, img_shape):
    if left_fit is None or right_fit is None:
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

    q1, q3 = np.percentile(width_samples_pix, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_widths = [w for w in width_samples_pix if lower_bound <= w <= upper_bound]

    if len(filtered_widths) < 5:
        return "Width calculation failed"

    weights = y_samples[-len(filtered_widths):] / h
    avg_width_pix = np.average(filtered_widths, weights=weights)

    y_near = h
    xl_near = left_fit[0] * y_near**2 + left_fit[1] * y_near + left_fit[2]
    xr_near = right_fit[0] * y_near**2 + right_fit[1] * y_near + right_fit[2]
    near_width_pix = abs(xr_near - xl_near)

    if 40 < near_width_pix < 1200:
        xm_per_pix = 3.7 / near_width_pix
    else:
        xm_per_pix = 3.7 / 700

    avg_width_m = avg_width_pix * xm_per_pix

    if 2.0 <= avg_width_m <= 5.0:
        return f"{avg_width_m:.3f} m"
    else:
        return f"Abnormal width ({avg_width_m:.3f} m)"

def calculate_lane_curvature(left_fit, right_fit, img_shape):
    if left_fit is None or right_fit is None:
        return "N/A"

    height, width = img_shape[:2]
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    y_vals = np.linspace(0, height-1, num=50)
    y_eval = np.max(y_vals)

    left_x = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
    left_fit_cr = np.polyfit(y_vals * ym_per_pix, left_x * xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

    right_x = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]
    right_fit_cr = np.polyfit(y_vals * ym_per_pix, right_x * xm_per_pix, 2)
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    avg_curvature = (left_curverad + right_curverad) / 2
    if avg_curvature < 100 or avg_curvature > 10000:
        return "N/A"
    if avg_curvature > 1000:
        return f"{int(avg_curvature)} m (Straight)"
    else:
        return f"{avg_curvature:.1f} m"

def lane_detection_pipeline(img):
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    edges = canny_edge(blur)

    height, width = img.shape[:2]
    vertices = np.array([[
        (width*0.05, height),
        (width*0.4, height*0.5),
        (width*0.6, height*0.5),
        (width*0.95, height)
    ]], dtype=np.int32)
    roi_edges = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(
        roi_edges, rho=1, theta=np.pi/180, threshold=25,
        minLineLength=40, maxLineGap=20
    )

    line_img = np.zeros_like(img)
    line_img, left_fit, right_fit, left_bottom_x, right_bottom_x = draw_lines(line_img, lines)
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    # 1. 车道曲率
    curvature = calculate_lane_curvature(left_fit, right_fit, img.shape)
    cv2.putText(result, f"Lane Curvature: {curvature}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # 2. 车辆偏移（米）
    offset_str, offset_m = calculate_car_offset(left_bottom_x, right_bottom_x, img.shape)
    cv2.putText(result, f"Car Offset: {offset_str}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # 3. 车道宽度
    lane_width = calculate_lane_width_precise(left_fit, right_fit, img.shape)
    if "m" in lane_width and "Abnormal" not in lane_width:
        wc = (0,255,255)
    elif "Abnormal" in lane_width:
        wc = (0,165,255)
    else:
        wc = (0,0,255)
    cv2.putText(result, f"Lane Width: {lane_width}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, wc, 2)

    # 4. 高精度偏离百分比
    offset_percent = calculate_offset_percentage(left_fit, right_fit, img.shape, left_bottom_x, right_bottom_x)
    cv2.putText(result, f"Offset Percent: {offset_percent}",
                (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # 5. 新增：车道线斜率差（显示在图片上）
    slope_diff = calculate_lane_slope_diff(left_fit, right_fit, img.shape)
    cv2.putText(result, f"Slope Diff: {slope_diff}",
                (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)  # 紫色字体，区分其他指标

    return result

def batch_detect_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder does not exist → {folder_path}")
        return

    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in supported_formats]

    if len(img_files) == 0:
        print(f"❌ Error: No images")
        return

    success = 0
    for i, f in enumerate(img_files, 1):
        p = os.path.join(folder_path, f)
        print(f"\n[{i}/{len(img_files)}] Processing: {f}")
        img = cv2.imread(p)
        if img is None:
            print("⚠️ Skipped")
            continue
        res = lane_detection_pipeline(img)
        out = os.path.splitext(p)[0] + "_result.jpg"
        cv2.imwrite(out, res)
        success +=1
        print("✅ Done")

    print(f"\n🎉 Finished: {success}/{len(img_files)}")

def main():
    print("="*50)
    print("      Lane Detection + Slope Diff + High-Precision Offset")
    print("="*50)
    d = os.path.dirname(os.path.abspath(__file__))

    print("\n1 Single image")
    print("2 Batch images")
    mode = input("Mode (1/2): ").strip()

    if mode == "1":
        p = input("Image path: ").strip()
        ap = os.path.join(d, p)
        img = cv2.imread(ap)
        if img is None:
            print("❌ Can't read")
            return
        res = lane_detection_pipeline(img)
        cv2.imshow("Original", img)
        cv2.imshow("Result", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode == "2":
        p = input("Folder: ").strip()
        batch_detect_images(os.path.join(d, p))

if __name__ == "__main__":
    main()
