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
    
    if lines is None:
        return img
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            intercept = y1 - slope * x1
            
            # 区分左右车道线（左负右正）
            if -0.8 < slope < -0.3:
                left_slope.append(slope)
                left_intercept.append(intercept)
            elif 0.3 < slope < 0.8:
                right_slope.append(slope)
                right_intercept.append(intercept)
    
    # 计算平均斜率和截距
    left_avg_slope = np.mean(left_slope) if left_slope else 0
    left_avg_intercept = np.mean(left_intercept) if left_intercept else 0
    right_avg_slope = np.mean(right_slope) if right_slope else 0
    right_avg_intercept = np.mean(right_intercept) if right_intercept else 0
    
    # 获取图像尺寸并计算车道线端点
    height, width = img.shape[:2]
    y_bottom = height
    y_top = int(height * 0.6)
    
    # 绘制左车道线
    if left_avg_slope != 0:
        x1_left = int((y_bottom - left_avg_intercept) / left_avg_slope)
        x2_left = int((y_top - left_avg_intercept) / left_avg_slope)
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_top), color, thickness)
    
    # 绘制右车道线
    if right_avg_slope != 0:
        x1_right = int((y_bottom - right_avg_intercept) / right_avg_slope)
        x2_right = int((y_top - right_avg_intercept) / right_avg_slope)
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_top), color, thickness)
    
    return img

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
    line_img = draw_lines(line_img, lines)
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    return result

def main():
    """主函数：手动输入路径的车道检测"""
    print("="*50)
    print("      车道检测程序 - 手动输入图片路径版")
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
    cv2.imshow("🚗 车道检测结果", result)
    
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