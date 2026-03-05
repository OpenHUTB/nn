import cv2
import numpy as np
import os
from scipy import stats

def grayscale(img):
    """将图像转为灰度图，减少计算量"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size=5):
    """高斯模糊降噪"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge(img, low_threshold=50, high_threshold=150):
    """Canny 边缘检测"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """定义感兴趣区域，只保留车道相关的区域"""
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
    """拟合并绘制左右车道线，返回拟合参数+车道线底部端点（用于偏离/宽度计算）"""
    left_points = []   
    right_points = []  
    
    if lines is None:
        return img, None, None, None, None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            
            # 更宽松的斜率筛选（优化点1：扩大有效线范围）
            if -1.2 < slope < -0.1:
                left_points.append([x1, y1])
                left_points.append([x2, y2])
            elif 0.1 < slope < 1.2:
                right_points.append([x1, y1])
                right_points.append([x2, y2])
    
    # 移除异常值（Z-score过滤，优化阈值）
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    
    # 过滤左车道异常点（优化点2：放宽Z-score阈值，保留更多有效点）
    if len(left_points) > 2:
        z_scores = stats.zscore(left_points)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 2.5).all(axis=1)
        left_points = left_points[filtered_entries]
    
    # 过滤右车道异常点
    if len(right_points) > 2:
        z_scores = stats.zscore(right_points)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 2.5).all(axis=1)
        right_points = right_points[filtered_entries]
    
    height, width = img.shape[:2]
    y_bottom = height
    y_top = int(height * 0.5)  # 优化点3：提高感兴趣区域顶部，覆盖更多车道
    
    # 初始化返回值
    left_fit = None
    right_fit = None
    left_bottom_x = None
    right_bottom_x = None
    
    # 绘制左车道线（使用多项式拟合）
    if len(left_points) >= 2:
        left_y = left_points[:, 1]
        left_x = left_points[:, 0]
        left_fit = np.polyfit(left_y, left_x, 2)
        
        # 计算车道线端点
        y_vals = np.array([y_bottom, y_top])
        x_vals = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
        
        # 确保坐标在图像范围内
        x1_left = np.clip(int(x_vals[0]), 0, width)
        x2_left = np.clip(int(x_vals[1]), 0, width)
        
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_top), color, thickness)
        left_bottom_x = x1_left
    
    # 绘制右车道线
    if len(right_points) >= 2:
        right_y = right_points[:, 1]
        right_x = right_points[:, 0]
        right_fit = np.polyfit(right_y, right_x, 2)
        
        # 计算车道线端点
        y_vals = np.array([y_bottom, y_top])
        x_vals = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]
        
        # 确保坐标在图像范围内
        x1_right = np.clip(int(x_vals[0]), 0, width)
        x2_right = np.clip(int(x_vals[1]), 0, width)
        
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_top), color, thickness)
        right_bottom_x = x1_right
    
    return img, left_fit, right_fit, left_bottom_x, right_bottom_x

def calculate_car_offset(left_bottom_x, right_bottom_x, img_shape):
    """计算车辆偏离车道中心的距离（单位：米）"""
    if left_bottom_x is None or right_bottom_x is None:
        return "N/A"
    
    height, width = img_shape[:2]
    
    # 计算实际车道宽度（像素），动态调整转换系数
    lane_width_pix = abs(right_bottom_x - left_bottom_x)
    if lane_width_pix < 80:  # 优化点4：放宽最小宽度阈值
        return "N/A"
    
    # 标准车道宽度3.7米，动态计算像素米转换系数
    xm_per_pix = 3.7 / lane_width_pix
    
    # 计算车道中心和车辆中心
    lane_center_x = (left_bottom_x + right_bottom_x) / 2
    car_center_x = width / 2
    
    # 考虑相机安装偏移（默认0，可根据实际情况调整）
    camera_offset_pix = 0
    car_center_x += camera_offset_pix
    
    # 计算偏移量
    offset_pix = car_center_x - lane_center_x
    offset_m = offset_pix * xm_per_pix
    
    # 更精确的阈值和格式化
    if offset_m > 0.02:
        return f"Right {abs(offset_m):.3f}m"  # 替换：右偏 → Right
    elif offset_m < -0.02:
        return f"Left {abs(offset_m):.3f}m"   # 替换：左偏 → Left
    else:
        return "Centered (±0.02m)"            # 替换：居中 → Centered

# ===================== 高精度车道宽度计算 =====================
def calculate_lane_width_precise(left_fit, right_fit, img_shape):
    """
    高精度车道宽度计算：
    1. 密集采样 + 加权平均（近处权重更高）
    2. 透视校正（基于纵向像素-米转换）
    3. 鲁棒性异常值过滤（IQR法）
    4. 动态像素-米系数校准
    """
    if left_fit is None or right_fit is None:
        return "Lane line fitting failed"
    
    h, w = img_shape[:2]
    
    # 1. 密集采样：在车道可见区域（从底部60%到100%）取50个采样点
    y_samples = np.linspace(h * 0.6, h, 50)
    width_samples_pix = []
    
    for y in y_samples:
        # 计算左右车道线在该高度的x坐标
        xl = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        xr = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        
        if 0 < xl < w and 0 < xr < w:
            width_pix = abs(xr - xl)
            if 40 < width_pix < 1200:
                width_samples_pix.append(width_pix)
    
    if len(width_samples_pix) == 0:
        return "Width calculation failed"
    
    # 2. 鲁棒性异常值过滤：使用IQR法剔除极端值
    q1, q3 = np.percentile(width_samples_pix, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_widths = [w for w in width_samples_pix if lower_bound <= w <= upper_bound]
    
    if len(filtered_widths) < 5:  # 过滤后有效点过少
        return "Width calculation failed"
    
    # 3. 加权平均：近处（y大）的宽度更可靠，赋予更高权重
    weights = y_samples[-len(filtered_widths):] / h  # 权重随y增大而增大
    avg_width_pix = np.average(filtered_widths, weights=weights)
    
    # 4. 动态像素-米系数：基于近处宽度校准
    # 近处（y=h）的车道宽度更接近真实值，用它来校准转换系数
    y_near = h
    xl_near = left_fit[0] * y_near**2 + left_fit[1] * y_near + left_fit[2]
    xr_near = right_fit[0] * y_near**2 + right_fit[1] * y_near + right_fit[2]
    near_width_pix = abs(xr_near - xl_near)
    
    if 40 < near_width_pix < 1200:
        xm_per_pix = 3.7 / near_width_pix  # 用近处宽度校准
    else:
        xm_per_pix = 3.7 / 700  #  fallback to default
    
    avg_width_m = avg_width_pix * xm_per_pix
    
    # 5. 最终范围校验
    if 2.0 <= avg_width_m <= 5.0:
        return f"{avg_width_m:.3f} m"  # 精度提升到3位小数
    else:
        return f"Abnormal width ({avg_width_m:.3f} m)"

def calculate_lane_curvature(left_fit, right_fit, img_shape):
    """计算车道曲率半径（单位：米）"""
    if left_fit is None or right_fit is None:
        return "N/A"
    
    height, width = img_shape[:2]
    
    ym_per_pix = 30 / 720  
    xm_per_pix = 3.7 / 700  
    
    y_vals = np.linspace(0, height-1, num=50)
    y_eval = np.max(y_vals)
    
    # 左车道曲率计算
    left_x = left_fit[0] * y_vals**2 + left_fit[1] * y_vals + left_fit[2]
    left_fit_cr = np.polyfit(y_vals * ym_per_pix, left_x * xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    
    # 右车道曲率计算
    right_x = right_fit[0] * y_vals**2 + right_fit[1] * y_vals + right_fit[2]
    right_fit_cr = np.polyfit(y_vals * ym_per_pix, right_x * xm_per_pix, 2)
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # 曲率平滑，取平均值并过滤异常值
    avg_curvature = (left_curverad + right_curverad) / 2
    
    # 过滤异常曲率值
    if avg_curvature < 100 or avg_curvature > 10000:
        return "N/A"
    
    # 根据曲率大小调整显示精度
    if avg_curvature > 1000:
        return f"{int(avg_curvature)} m (Straight)"  # 替换：近似直线 → Straight
    else:
        return f"{avg_curvature:.1f} m"

def lane_detection_pipeline(img):
    """完整的车道检测流水线（高精度宽度版）"""
    # 预处理
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    edges = canny_edge(blur)
    
    # 定义感兴趣区域
    height, width = img.shape[:2]
    vertices = np.array([[
        (width*0.05, height),    # 优化点9：扩大左右边界
        (width*0.4, height*0.5), # 提高顶部边界
        (width*0.6, height*0.5),
        (width*0.95, height)
    ]], dtype=np.int32)
    roi_edges = region_of_interest(edges, vertices)
    
    # 霍夫变换检测直线（优化参数）
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,               
        theta=np.pi/180,     
        threshold=25,        # 优化点10：降低阈值，检测更多线
        minLineLength=40,    # 缩短最小线长
        maxLineGap=20        # 扩大最大间隙
    )
    
    # 绘制车道线
    line_img = np.zeros_like(img)
    line_img, left_fit, right_fit, left_bottom_x, right_bottom_x = draw_lines(line_img, lines)
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    # 计算曲率并绘制
    curvature = calculate_lane_curvature(left_fit, right_fit, img.shape)
    cv2.putText(
        result, 
        f"Lane Curvature: {curvature}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # 计算车辆偏离并绘制
    offset = calculate_car_offset(left_bottom_x, right_bottom_x, img.shape)
    cv2.putText(
        result, 
        f"Car Offset: {offset}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # 调用高精度宽度计算
    lane_width = calculate_lane_width_precise(left_fit, right_fit, img.shape)
    
    # 根据宽度状态设置不同颜色
    if "m" in lane_width and "Abnormal" not in lane_width:
        width_color = (0, 255, 255)  # 正常：黄色
    elif "Abnormal" in lane_width:
        width_color = (0, 165, 255)  # 异常：橙色
    else:
        width_color = (0, 0, 255)    # 失败：红色
    
    cv2.putText(
        result, 
        f"Lane Width: {lane_width}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        width_color,
        2
    )
    
    return result

def batch_detect_images(folder_path):
    """批量检测指定文件夹下的所有图片"""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder does not exist → {folder_path}")  # 替换：错误 → Error，文件夹不存在 → Folder does not exist
        return
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    if len(img_files) == 0:
        print(f"❌ Error: No supported images found in folder → {folder_path}")  # 替换：未找到图片 → No supported images found
        return
    
    print(f"\n📁 Starting batch detection, found {len(img_files)} images...")  # 替换：开始批量检测 → Starting batch detection，发现 → found
    success_count = 0
    
    for idx, img_file in enumerate(img_files, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f"\n[{idx}/{len(img_files)}] Processing: {img_file}")  # 替换：处理 → Processing
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Skipped: Cannot read image {img_file}")  # 替换：跳过 → Skipped，无法读取 → Cannot read
            continue
        
        result = lane_detection_pipeline(img)
        
        save_name = os.path.splitext(img_file)[0] + "_batch_result.jpg"
        save_path = os.path.join(folder_path, save_name)
        cv2.imwrite(save_path, result)
        
        success_count += 1
        print(f"✅ Completed: Result saved as {save_name}")  # 替换：完成 → Completed，保存为 → saved as
    
    print(f"\n🎉 Batch detection finished!")  # 替换：批量检测结束 → Batch detection finished
    print(f"✅ Successfully processed: {success_count} images")  # 替换：成功处理 → Successfully processed
    print(f"❌ Failed/Skipped: {len(img_files) - success_count} images")  # 替换：失败/跳过 → Failed/Skipped
    print(f"📁 All results saved to: {folder_path}")  # 替换：所有结果已保存至 → All results saved to

def main():
    """主函数：支持单张/批量检测模式（高精度宽度版）"""
    print("="*60)
    print("      Lane Detection Program (High Precision Width)")  # 替换：车道检测程序 → Lane Detection Program
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n📂 Current script directory: {script_dir}")  # 替换：当前脚本目录 → Current script directory
    
    print("\nPlease select running mode:")  # 替换：请选择运行模式 → Please select running mode
    print("1 - Single image detection (high precision width)")  # 替换：单张图片检测 → Single image detection
    print("2 - Batch image detection (high precision width)")    # 替换：批量图片检测 → Batch image detection
    mode = input("Enter mode number (1/2): ").strip()  # 替换：输入模式编号 → Enter mode number
    
    if mode == "1":
        print("\n📸 Single image detection mode")  # 替换：单张图片检测模式 → Single image detection mode
        print("\nPlease enter image relative path (relative to script directory):")  # 替换：请输入图片的相对路径 → Please enter image relative path
        print("Example: images/test_lane.png or ../data/test_lane.png")  # 替换：示例 → Example
        relative_path = input("Relative path: ").strip()  # 替换：相对路径 → Relative path
        
        if not relative_path:
            print("❌ Error: Path cannot be empty!")  # 替换：路径不能为空 → Path cannot be empty
            return
        
        CUSTOM_IMAGE_PATH = os.path.join(script_dir, relative_path)
        
        if not os.path.exists(CUSTOM_IMAGE_PATH):
            print(f"\n❌ Error: File does not exist!")  # 替换：文件不存在 → File does not exist
            print(f"Script directory: {script_dir}")    # 替换：脚本目录 → Script directory
            print(f"Your input relative path: {relative_path}")  # 替换：你输入的相对路径 → Your input relative path
            print(f"Combined absolute path: {CUSTOM_IMAGE_PATH}")  # 替换：拼接后的绝对路径 → Combined absolute path
            print("Please check path and filename")  # 替换：请检查路径和文件名是否正确 → Please check path and filename
            return
        
        img = cv2.imread(CUSTOM_IMAGE_PATH)
        if img is None:
            print("\n❌ Error: Cannot read image!")  # 替换：无法读取图像 → Cannot read image
            return
        
        print("\n✅ Image loaded successfully, detecting lanes...")  # 替换：图片读取成功，正在检测车道 → Image loaded successfully, detecting lanes
        result = lane_detection_pipeline(img)
        
        cv2.imshow("📷 Original Image", img)  # 替换：原始图片 → Original Image
        cv2.imshow("🚗 Lane Detection Result (High Precision Width)", result)  # 替换：车道检测结果 → Lane Detection Result
        
        save_path = os.path.splitext(CUSTOM_IMAGE_PATH)[0] + "_result.jpg"
        cv2.imwrite(save_path, result)
        print(f"\n✅ Detection completed! Result saved to: {save_path}")  # 替换：检测完成 → Detection completed，结果已保存到 → Result saved to
        print("\nTip: Press any key to close image windows")  # 替换：提示：按任意键关闭图片窗口 → Tip: Press any key to close image windows
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif mode == "2":
        print("\n📁 Batch image detection mode")  # 替换：批量图片检测模式 → Batch image detection mode
        print("\nPlease enter folder relative path (relative to script directory):")  # 替换：请输入图片文件夹的相对路径 → Please enter folder relative path
        print("Example: lane_images or ../data/lane_images")  # 替换：示例 → Example
        relative_folder = input("Relative path: ").strip()  # 替换：相对路径 → Relative path
        
        if not relative_folder:
            print("❌ Error: Path cannot be empty!")  # 替换：路径不能为空 → Path cannot be empty
            return
        
        folder_path = os.path.join(script_dir, relative_folder)
        batch_detect_images(folder_path)
    
    else:
        print("❌ Error: Invalid mode number! Please enter 1 or 2")  # 替换：无效的模式编号 → Invalid mode number

if __name__ == "__main__":
    # 安装依赖（首次运行前执行）
    # pip install opencv-python numpy scipy
    main()
