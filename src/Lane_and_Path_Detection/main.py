import cv2
import numpy as np
import os

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
    """拟合并绘制左右车道线，返回拟合参数+车道线底部端点（用于偏离计算）"""
    left_slope = []   
    left_intercept = [] 
    right_slope = []  
    right_intercept = [] 
    
    # 曲率计算相关：收集坐标点
    left_x, left_y = [], []
    right_x, right_y = [], []
    # ===================== 偏离计算新增部分开始 =====================
    # 保存车道线底部端点（图像底部的x坐标，用于计算中心）
    left_bottom_x = None
    right_bottom_x = None
    # ===================== 偏离计算新增部分结束 =====================
    
    if lines is None:
        # 曲率+偏离计算：返回空值
        return img, None, None, left_bottom_x, right_bottom_x
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            intercept = y1 - slope * x1
            
            # 区分左右车道线（左负右正）
            if -0.8 < slope < -0.3:
                left_slope.append(slope)
                left_intercept.append(intercept)
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            elif 0.3 < slope < 0.8:
                right_slope.append(slope)
                right_intercept.append(intercept)
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
    
    # 计算平均斜率和截距
    left_avg_slope = np.mean(left_slope) if left_slope else 0
    left_avg_intercept = np.mean(left_intercept) if left_intercept else 0
    right_avg_slope = np.mean(right_slope) if right_slope else 0
    right_avg_intercept = np.mean(right_intercept) if right_intercept else 0
    
    # 获取图像尺寸并计算车道线端点
    height, width = img.shape[:2]
    y_bottom = height
    y_top = int(height * 0.6)
    
    # 曲率计算相关：初始化拟合参数
    left_fit = None
    right_fit = None
    
    # 绘制左车道线
    if left_avg_slope != 0:
        x1_left = int((y_bottom - left_avg_intercept) / left_avg_slope)
        x2_left = int((y_top - left_avg_intercept) / left_avg_slope)
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_top), color, thickness)
        # 曲率计算相关：拟合
        if len(left_x) >= 2:
            left_fit = np.polyfit(left_y, left_x, 2)
        # ===================== 偏离计算新增部分开始 =====================
        # 保存左车道线底部x坐标
        left_bottom_x = x1_left
        # ===================== 偏离计算新增部分结束 =====================
    
    # 绘制右车道线
    if right_avg_slope != 0:
        x1_right = int((y_bottom - right_avg_intercept) / right_avg_slope)
        x2_right = int((y_top - right_avg_intercept) / right_avg_slope)
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_top), color, thickness)
        # 曲率计算相关：拟合
        if len(right_x) >= 2:
            right_fit = np.polyfit(right_y, right_x, 2)
        # ===================== 偏离计算新增部分开始 =====================
        # 保存右车道线底部x坐标
        right_bottom_x = x1_right
        # ===================== 偏离计算新增部分结束 =====================
    
    # 返回：原图、曲率拟合参数、车道线底部x坐标（用于偏离计算）
    return img, left_fit, right_fit, left_bottom_x, right_bottom_x

# ===================== 偏离计算新增函数开始 =====================
def calculate_car_offset(left_bottom_x, right_bottom_x, img_shape):
    """
    新增：计算车辆偏离车道中心的距离（单位：米）
    核心逻辑：
    1. 假设车辆中心在图像水平中心位置
    2. 计算车道中心与车辆中心的像素差，转换为真实世界距离
    """
    # 无有效车道线时返回N/A
    if left_bottom_x is None or right_bottom_x is None:
        return "N/A"
    
    height, width = img_shape[:2]
    # 1. 计算车道中心x坐标（左右车道线底部x的平均值）
    lane_center_x = (left_bottom_x + right_bottom_x) / 2
    # 2. 车辆中心x坐标（图像水平中心）
    car_center_x = width / 2
    # 3. 像素偏移量（正=右偏，负=左偏）
    offset_pix = car_center_x - lane_center_x
    # 4. 像素转米（横向：700像素=3.7米，标准车道宽度）
    xm_per_pix = 3.7 / 700
    offset_m = offset_pix * xm_per_pix
    
    # 格式化输出
    if offset_m > 0.05:  # 右偏（阈值避免微小误差）
        return f"右偏 {abs(offset_m):.2f}m"
    elif offset_m < -0.05:  # 左偏
        return f"左偏 {abs(offset_m):.2f}m"
    else:  # 居中
        return "居中"
# ===================== 偏离计算新增函数结束 =====================

def calculate_lane_curvature(left_fit, right_fit, img_shape):
    """计算车道曲率半径（单位：米）"""
    if left_fit is None or right_fit is None:
        return "N/A"
    
    height = img_shape[0]
    y_eval = np.max([height - 1, 0])
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    
    left_fit_cr = np.polyfit(
        np.array([y_eval]) * ym_per_pix,
        np.array([left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]]) * xm_per_pix,
        2
    )
    right_fit_cr = np.polyfit(
        np.array([y_eval]) * ym_per_pix,
        np.array([right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]]) * xm_per_pix,
        2
    )
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    avg_curvature = (left_curverad + right_curverad) / 2
    return f"{int(avg_curvature)} m"

def lane_detection_pipeline(img):
    """完整的车道检测流水线（含曲率+偏离计算，移除填充）"""
    # 预处理
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    edges = canny_edge(blur)
    
    # 定义感兴趣区域
    height, width = img.shape[:2]
    vertices = np.array([[
        (width*0.1, height),
        (width*0.45, height*0.6),
        (width*0.55, height*0.6),
        (width*0.9, height)
    ]], dtype=np.int32)
    roi_edges = region_of_interest(edges, vertices)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,               
        theta=np.pi/180,     
        threshold=20,        
        minLineLength=40,    
        maxLineGap=20        
    )
    
    # 绘制车道线（获取曲率参数+车道线底部x坐标）
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
    
    # ===================== 偏离计算新增部分开始 =====================
    # 计算车辆偏离并绘制
    offset = calculate_car_offset(left_bottom_x, right_bottom_x, img.shape)
    cv2.putText(
        result, 
        f"Car Offset: {offset}",
        (20, 100),  # 曲率文字下方，避免重叠
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    # ===================== 偏离计算新增部分结束 =====================
    
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
    """主函数：支持单张/批量检测模式（含曲率+偏离计算）"""
    print("="*60)
    print("      车道检测程序 - 单张/批量模式（曲率+偏离计算）")
    print("="*60)
    
    print("\n请选择运行模式：")
    print("1 - 单张图片检测（含曲率+偏离计算）")
    print("2 - 批量图片检测（含曲率+偏离计算）")
    mode = input("输入模式编号（1/2）：").strip()
    
    if mode == "1":
        print("\n📸 单张图片检测模式")
        print("\n请输入图片的完整路径（可直接复制粘贴）：")
        print("示例：C:\\Users\\apple\\Desktop\\zy\\test_lane.png")
        CUSTOM_IMAGE_PATH = input("图片路径：").strip()
        
        if not CUSTOM_IMAGE_PATH:
            print("❌ 错误：路径不能为空！")
            return
        
        if not os.path.exists(CUSTOM_IMAGE_PATH):
            print(f"\n❌ 错误：文件不存在！")
            print(f"当前输入的路径：{CUSTOM_IMAGE_PATH}")
            print("请检查：")
            print("  1. 路径是否正确（建议复制粘贴）")
            print("  2. 文件名和后缀（.jpg/.png）是否正确")
            print("  3. 文件是否真的存在于该目录下")
            return
        
        img = cv2.imread(CUSTOM_IMAGE_PATH)
        if img is None:
            print("\n❌ 错误：无法读取图像！")
            print("可能原因：")
            print("  1. 文件格式不支持（仅支持 jpg/png/bmp 等）")
            print("  2. 文件已损坏或不是有效的图片文件")
            return
        
        print("\n✅ 图片读取成功，正在检测车道...")
        result = lane_detection_pipeline(img)
        
        cv2.imshow("📷 原始图片", img)
        cv2.imshow("🚗 车道检测结果（曲率+偏离）", result)
        
        save_path = os.path.splitext(CUSTOM_IMAGE_PATH)[0] + "_result.jpg"
        cv2.imwrite(save_path, result)
        print(f"\n✅ 检测完成！结果已保存到：{save_path}")
        print("\n提示：按任意键关闭图片窗口")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif mode == "2":
        print("\n📁 批量图片检测模式")
        print("\n请输入图片文件夹的完整路径（可直接复制粘贴）：")
        print("示例：C:\\Users\\apple\\Desktop\\zy\\lane_images")
        folder_path = input("文件夹路径：").strip()
        batch_detect_images(folder_path)
    
    else:
        print("❌ 错误：无效的模式编号！请输入 1 或 2")

if __name__ == "__main__":
    # 安装依赖（首次运行前执行）
    # pip install opencv-python numpy
    main()
