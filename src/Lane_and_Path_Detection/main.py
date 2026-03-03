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
    """拟合并绘制左右车道线"""
    left_slope = []   
    left_intercept = [] 
    right_slope = []  
    right_intercept = [] 
    
    # ===================== 新增部分开始 =====================
    # 新增：收集车道线坐标点，用于多项式拟合计算曲率
    left_x, left_y = [], []
    right_x, right_y = [], []
    # ===================== 新增部分结束 =====================
    
    if lines is None:
        # ===================== 新增部分开始 =====================
        # 新增：无车道线时返回None（用于曲率计算判断）
        return img, None, None
        # ===================== 新增部分结束 =====================
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            intercept = y1 - slope * x1
            
            # 区分左右车道线（左负右正）
            if -0.8 < slope < -0.3:
                left_slope.append(slope)
                left_intercept.append(intercept)
                # ===================== 新增部分开始 =====================
                # 新增：收集左车道线坐标点
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
                # ===================== 新增部分结束 =====================
            elif 0.3 < slope < 0.8:
                right_slope.append(slope)
                right_intercept.append(intercept)
                # ===================== 新增部分开始 =====================
                # 新增：收集右车道线坐标点
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
                # ===================== 新增部分结束 =====================
    
    # 计算平均斜率和截距
    left_avg_slope = np.mean(left_slope) if left_slope else 0
    left_avg_intercept = np.mean(left_intercept) if left_intercept else 0
    right_avg_slope = np.mean(right_slope) if right_slope else 0
    right_avg_intercept = np.mean(right_intercept) if right_intercept else 0
    
    # 获取图像尺寸并计算车道线端点
    height, width = img.shape[:2]
    y_bottom = height
    y_top = int(height * 0.6)
    
    # ===================== 新增部分开始 =====================
    # 新增：初始化车道线拟合参数（用于曲率计算）
    left_fit = None
    right_fit = None
    # ===================== 新增部分结束 =====================
    
    # 绘制左车道线
    if left_avg_slope != 0:
        x1_left = int((y_bottom - left_avg_intercept) / left_avg_slope)
        x2_left = int((y_top - left_avg_intercept) / left_avg_slope)
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_top), color, thickness)
        # ===================== 新增部分开始 =====================
        # 新增：对左车道线做2次多项式拟合（曲率计算核心）
        if len(left_x) >= 2:
            left_fit = np.polyfit(left_y, left_x, 2)
        # ===================== 新增部分结束 =====================
    
    # 绘制右车道线
    if right_avg_slope != 0:
        x1_right = int((y_bottom - right_avg_intercept) / right_avg_slope)
        x2_right = int((y_top - right_avg_intercept) / right_avg_slope)
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_top), color, thickness)
        # ===================== 新增部分开始 =====================
        # 新增：对右车道线做2次多项式拟合（曲率计算核心）
        if len(right_x) >= 2:
            right_fit = np.polyfit(right_y, right_x, 2)
        # ===================== 新增部分结束 =====================
    
    # ===================== 新增部分开始 =====================
    # 新增：返回拟合参数（用于后续曲率计算）
    return img, left_fit, right_fit
    # ===================== 新增部分结束 =====================

# ===================== 新增部分开始 =====================
def calculate_lane_curvature(left_fit, right_fit, img_shape):
    """
    新增：计算车道曲率半径（单位：米）
    真实世界映射关系：
    - ym_per_pix: 纵向像素转米（720像素 ≈ 30米）
    - xm_per_pix: 横向像素转米（700像素 ≈ 3.7米，标准车道宽度）
    """
    # 无有效拟合参数时返回N/A
    if left_fit is None or right_fit is None:
        return "N/A"
    
    height = img_shape[0]
    # 取图像底部的点计算曲率（贴近车辆实际视角）
    y_eval = np.max([height - 1, 0])
    # 像素与真实世界的转换系数
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    
    # 转换左车道线拟合参数到真实世界坐标系
    left_fit_cr = np.polyfit(
        np.array([y_eval]) * ym_per_pix,
        np.array([left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]]) * xm_per_pix,
        2
    )
    # 转换右车道线拟合参数到真实世界坐标系
    right_fit_cr = np.polyfit(
        np.array([y_eval]) * ym_per_pix,
        np.array([right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]]) * xm_per_pix,
        2
    )
    
    # 曲率计算公式：R = (1 + (2Ay + B)^2)^1.5 / |2A|
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # 取左右车道线曲率的平均值
    avg_curvature = (left_curverad + right_curverad) / 2
    return f"{int(avg_curvature)} m"
# ===================== 新增部分结束 =====================

def lane_detection_pipeline(img):
    """完整的车道检测流水线"""
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
    
    # 绘制车道线并合并结果
    line_img = np.zeros_like(img)
    # ===================== 新增部分开始 =====================
    # 新增：接收draw_lines返回的拟合参数
    line_img, left_fit, right_fit = draw_lines(line_img, lines)
    # ===================== 新增部分结束 =====================
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    # ===================== 新增部分开始 =====================
    # 新增：计算曲率并绘制到图像左上角
    curvature = calculate_lane_curvature(left_fit, right_fit, img.shape)
    cv2.putText(
        result, 
        f"Lane Curvature: {curvature}",  # 曲率文字内容
        (20, 50),                       # 文字位置（左上角）
        cv2.FONT_HERSHEY_SIMPLEX,       # 字体
        1,                              # 字体大小
        (255, 255, 255),                # 文字颜色（白色）
        2                               # 文字粗细
    )
    # ===================== 新增部分结束 =====================
    
    return result

def main():
    """主函数：手动输入路径的车道检测"""
    print("="*50)
    # ===================== 新增部分开始 =====================
    print("      车道检测程序 - 手动输入图片路径版（含曲率计算）")
    # ===================== 新增部分结束 =====================
    print("="*50)
    
    # 1. 手动输入图片路径（支持复制粘贴）
    print("\n请输入图片的完整路径（可直接复制粘贴）：")
    print("示例：C:\\Users\\apple\\Desktop\\zy\\test_lane.png")
    CUSTOM_IMAGE_PATH = input("图片路径：").strip()
    
    # 2. 检查路径是否为空
    if not CUSTOM_IMAGE_PATH:
        print("❌ 错误：路径不能为空！")
        return
    
    # 3. 检查文件是否存在
    if not os.path.exists(CUSTOM_IMAGE_PATH):
        print(f"\n❌ 错误：文件不存在！")
        print(f"当前输入的路径：{CUSTOM_IMAGE_PATH}")
        print("请检查：")
        print("  1. 路径是否正确（建议复制粘贴）")
        print("  2. 文件名和后缀（.jpg/.png）是否正确")
        print("  3. 文件是否真的存在于该目录下")
        return
    
    # 4. 读取图片
    img = cv2.imread(CUSTOM_IMAGE_PATH)
    if img is None:
        print("\n❌ 错误：无法读取图像！")
        print("可能原因：")
        print("  1. 文件格式不支持（仅支持 jpg/png/bmp 等）")
        print("  2. 文件已损坏或不是有效的图片文件")
        return
    
    # 5. 执行车道检测
    print("\n✅ 图片读取成功，正在检测车道...")
    result = lane_detection_pipeline(img)
    
    # 6. 显示检测结果
    cv2.imshow("📷 原始图片", img)
    # ===================== 新增部分开始 =====================
    cv2.imshow("🚗 车道检测结果（含曲率）", result)
    # ===================== 新增部分结束 =====================
    
    # 7. 自动保存检测结果（和原图同目录，后缀加 _result）
    save_path = os.path.splitext(CUSTOM_IMAGE_PATH)[0] + "_result.jpg"
    cv2.imwrite(save_path, result)
    print(f"\n✅ 检测完成！结果已保存到：{save_path}")
    print("\n提示：按任意键关闭图片窗口")
    
    # 8. 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 安装依赖（首次运行前执行）
    # pip install opencv-python numpy
    main()
