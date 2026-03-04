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
        filtered_entries = (abs_z_scores < 2.5).all(axis=1)  # 从2放宽到2.5
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
        return f"右偏 {abs(offset_m):.3f}m"
    elif offset_m < -0.02:
        return f"左偏 {abs(offset_m):.3f}m"
    else:
        return "居中 (±0.02m)"

def calculate_lane_width(left_bottom_x, right_bottom_x, left_fit, right_fit, img_shape):
    """
    优化版车道宽度计算：
    1. 放宽有效点阈值，保留更多采样点
    2. 扩大宽度有效范围，适配更多场景
    3. 增加降级策略（单采样点也能计算）
    4. 友好的失败提示
    """
    # 基础校验
    if left_bottom_x is None or right_bottom_x is None:
        return "未检测到车道线"
    if left_fit is None or right_fit is None:
        return "车道线拟合失败"
    
    height, width = img_shape[:2]
    
    # 优化点5：增加更多采样点，覆盖更广区域
    sample_heights = np.linspace(height * 0.5, height, num=15)  # 从10个增加到15个
    lane_widths_pix = []
    
    # 计算每个采样高度的车道宽度（像素）
    for y in sample_heights:
        # 计算左右车道线x坐标
        left_x = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        right_x = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        
        # 宽松的坐标过滤
        if 0 <= left_x <= width and 0 <= right_x <= width:
            lane_width_pix = abs(right_x - left_x)
            # 优化点6：放宽像素宽度范围（从50-1000改为40-1200）
            if 40 < lane_width_pix < 1200:
                lane_widths_pix.append(lane_width_pix)
    
    # 优化点7：降级策略 - 即使只有1个有效点也计算
    if len(lane_widths_pix) == 0:
        # 尝试用底部单点计算
        y = height
        left_x = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        right_x = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        if 0 <= left_x <= width and 0 <= right_x <= width:
            lane_width_pix = abs(right_x - left_x)
            if 40 < lane_width_pix < 1200:
                lane_widths_pix.append(lane_width_pix)
    
    # 仍无有效点
    if len(lane_widths_pix) == 0:
        return "宽度计算失败"
    
    # 计算平均宽度
    avg_width_pix = np.mean(lane_widths_pix)
    
    # 像素转米（动态校准）
    xm_per_pix = 3.7 / 700
    avg_width_m = avg_width_pix * xm_per_pix
    
    # 优化点8：放宽真实宽度范围（从2.5-4.5改为2.0-5.0）
    if 2.0 <= avg_width_m <= 5.0:
        return f"{avg_width_m:.2f} m"
    else:
        return f"异常宽度 ({avg_width_m:.2f} m)"

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
    
    if avg_curvature < 100 or avg_curvature > 10000:
        return "N/A"
    
    if avg_curvature > 1000:
        return f"{int(avg_curvature)} m (近似直线)"
    else:
        return f"{avg_curvature:.1f} m"

def lane_detection_pipeline(img):
    """完整的车道检测流水线（优化版）"""
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
    
    # 计算并绘制车道宽度（优化版）
    lane_width = calculate_lane_width(left_bottom_x, right_bottom_x, left_fit, right_fit, img.shape)
    # 根据宽度状态设置不同颜色
    if "m" in lane_width and "异常" not in lane_width:
        width_color = (0, 255, 255)  # 正常：黄色
    elif "异常" in lane_width:
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
        print(f"❌ 错误：文件夹不存在 → {folder_path}")
        return
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    if len(img_files) == 0:
        print(f"❌ 错误：文件夹内未找到支持的图片文件 → {folder_path}")
        return
    
    print(f"\n📁 开始批量检测，共发现 {len(img_files)} 张图片...")
    success_count = 0
    
    for idx, img_file in enumerate(img_files, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f"\n[{idx}/{len(img_files)}] 处理：{img_file}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  跳过：无法读取图片 {img_file}")
            continue
        
        result = lane_detection_pipeline(img)
        
        save_name = os.path.splitext(img_file)[0] + "_batch_result.jpg"
        save_path = os.path.join(folder_path, save_name)
        cv2.imwrite(save_path, result)
        
        success_count += 1
        print(f"✅ 完成：结果保存为 {save_name}")
    
    print(f"\n🎉 批量检测结束！")
    print(f"✅ 成功处理：{success_count} 张")
    print(f"❌ 失败/跳过：{len(img_files) - success_count} 张")
    print(f"📁 所有结果已保存至：{folder_path}")

def main():
    """主函数：支持单张/批量检测模式（优化版）"""
    print("="*60)
    print("      车道检测程序 - 单张/批量模式（宽度优化版）")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n📂 当前脚本目录：{script_dir}")
    
    print("\n请选择运行模式：")
    print("1 - 单张图片检测（宽度优化版）")
    print("2 - 批量图片检测（宽度优化版）")
    mode = input("输入模式编号（1/2）：").strip()
    
    if mode == "1":
        print("\n📸 单张图片检测模式")
        print("\n请输入图片的相对路径（相对于脚本目录）：")
        print("示例：images/test_lane.png 或 ../data/test_lane.png")
        relative_path = input("相对路径：").strip()
        
        if not relative_path:
            print("❌ 错误：路径不能为空！")
            return
        
        CUSTOM_IMAGE_PATH = os.path.join(script_dir, relative_path)
        
        if not os.path.exists(CUSTOM_IMAGE_PATH):
            print(f"\n❌ 错误：文件不存在！")
            print(f"脚本目录：{script_dir}")
            print(f"你输入的相对路径：{relative_path}")
            print(f"拼接后的绝对路径：{CUSTOM_IMAGE_PATH}")
            print("请检查路径和文件名是否正确")
            return
        
        img = cv2.imread(CUSTOM_IMAGE_PATH)
        if img is None:
            print("\n❌ 错误：无法读取图像！")
            return
        
        print("\n✅ 图片读取成功，正在检测车道...")
        result = lane_detection_pipeline(img)
        
        cv2.imshow("📷 原始图片", img)
        cv2.imshow("🚗 车道检测结果（宽度优化版）", result)
        
        save_path = os.path.splitext(CUSTOM_IMAGE_PATH)[0] + "_result.jpg"
        cv2.imwrite(save_path, result)
        print(f"\n✅ 检测完成！结果已保存到：{save_path}")
        print("\n提示：按任意键关闭图片窗口")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif mode == "2":
        print("\n📁 批量图片检测模式")
        print("\n请输入图片文件夹的相对路径（相对于脚本目录）：")
        print("示例：lane_images 或 ../data/lane_images")
        relative_folder = input("相对路径：").strip()
        
        if not relative_folder:
            print("❌ 错误：路径不能为空！")
            return
        
        folder_path = os.path.join(script_dir, relative_folder)
        batch_detect_images(folder_path)
    
    else:
        print("❌ 错误：无效的模式编号！请输入 1 或 2")

if __name__ == "__main__":
    # 安装依赖（首次运行前执行）
    # pip install opencv-python numpy scipy
    main()
