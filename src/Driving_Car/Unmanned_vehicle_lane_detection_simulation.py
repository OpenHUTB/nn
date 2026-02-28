import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale(img):
    """图像灰度化"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    """高斯模糊去噪"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge_detection(img, low_threshold, high_threshold):
    """Canny边缘检测"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """提取感兴趣区域（只保留道路区域，过滤背景）"""
    # 创建掩码
    mask = np.zeros_like(img)
    # 填充掩码的感兴趣区域
    cv2.fillPoly(mask, vertices, 255)
    # 与原图像进行按位与操作
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    """绘制检测到的车道线（拟合左右车道线后绘制）"""
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    if lines is None:
        return img

    for line in lines:
        for x1, y1, x2, y2 in line:
            # 计算斜率
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            # 过滤斜率过小的直线（非车道线）
            if abs(slope) < 0.5:
                continue
            # 区分左车道（斜率为负）和右车道（斜率为正）
            if slope < 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    # 拟合左车道线
    if left_x and left_y:
        left_fit = np.polyfit(left_y, left_x, 1)
        left_func = np.poly1d(left_fit)
        # 定义车道线的上下边界（y坐标）
        y_min = int(img.shape[0] * 0.6)  # 车道线上边界
        y_max = img.shape[0]  # 车道线下边界
        # 计算对应x坐标
        x_left_min = int(left_func(y_min))
        x_left_max = int(left_func(y_max))
        cv2.line(img, (x_left_min, y_min), (x_left_max, y_max), color, thickness)

    # 拟合右车道线
    if right_x and right_y:
        right_fit = np.polyfit(right_y, right_x, 1)
        right_func = np.poly1d(right_fit)
        y_min = int(img.shape[0] * 0.6)
        y_max = img.shape[0]
        x_right_min = int(right_func(y_min))
        x_right_max = int(right_func(y_max))
        cv2.line(img, (x_right_min, y_min), (x_right_max, y_max), color, thickness)

    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """霍夫变换检测直线"""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """将检测到的车道线与原图像融合"""
    return cv2.addWeighted(initial_img, α, img, β, γ)

def lane_detection_pipeline(image):
    """车道检测完整流水线"""
    # 1. 预处理：灰度化 + 高斯模糊 + Canny边缘检测
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size=5)
    edges = canny_edge_detection(blur_gray, low_threshold=50, high_threshold=150)

    # 2. 定义感兴趣区域（多边形顶点，根据图像尺寸调整）
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),
                          (imshape[1] / 2 - 20, imshape[0] / 2 + 60),
                          (imshape[1] / 2 + 20, imshape[0] / 2 + 60),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # 3. 霍夫变换检测直线
    rho = 1  # 霍夫空间的rho步长
    theta = np.pi / 180  # 霍夫空间的theta步长
    threshold = 15  # 检测直线的阈值
    min_line_len = 40  # 直线的最小长度
    max_line_gap = 20  # 直线之间的最大间隙
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # 4. 融合车道线与原图像
    result = weighted_img(line_img, image)

    return result, edges, masked_edges, line_img

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # 读取测试图像（可替换为自己的道路图像，建议使用车载摄像头视角的道路图）
    # 若没有测试图，可使用OpenCV生成模拟道路图像
    # 生成模拟道路图像
    def create_simulation_road_image(width=800, height=600):
        """生成模拟的道路图像（包含左右车道线）"""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
        # 绘制道路（灰色）
        cv2.rectangle(img, (100, 0), (700, height), (128, 128, 128), -1)
        # 绘制左车道线（白色）
        cv2.line(img, (250, height), (350, height//2), (255, 255, 255), 5)
        # 绘制右车道线（白色）
        cv2.line(img, (550, height), (450, height//2), (255, 255, 255), 5)
        return img

    # 生成模拟道路图像
    road_img = create_simulation_road_image(width=800, height=600)

    # 执行车道检测
    result, edges, masked_edges, line_img = lane_detection_pipeline(road_img)

    # 显示结果
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(road_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Road Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(masked_edges, cmap='gray')
    plt.title('Masked Edges (ROI)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Lane Detection Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # （可选）处理视频流（模拟车载摄像头实时检测）
    # 若需要处理视频，可替换为视频路径或摄像头编号（0为默认摄像头）
    # cap = cv2.VideoCapture('test_video.mp4')
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     result, _, _, _ = lane_detection_pipeline(frame)
    #     cv2.imshow('Lane Detection', result)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()