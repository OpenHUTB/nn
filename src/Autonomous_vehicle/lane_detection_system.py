import cv2
import numpy as np


def preprocess_image(image):
    """
    图像预处理：灰度化、高斯模糊、边缘检测
    :param image: 输入的原始图像
    :return: 预处理后的边缘图像
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    return edges


def region_of_interest(edges):
    """
    提取感兴趣区域（ROI）
    :param edges: 边缘检测后的图像
    :return: 提取后的ROI图像
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # 定义ROI的多边形顶点
    polygon = np.array([[
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]], np.int32)

    # 填充ROI区域
    cv2.fillPoly(mask, polygon, 255)

    # 与操作，只保留ROI区域的边缘
    masked_image = cv2.bitwise_and(edges, mask)

    return masked_image


def detect_lines(masked_image):
    """
    检测车道线
    :param masked_image: ROI图像
    :return: 检测到的车道线
    """
    # 霍夫线变换
    lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    return lines


def draw_lines(image, lines):
    """
    在图像上绘制车道线
    :param image: 原始图像
    :param lines: 检测到的车道线
    :return: 绘制了车道线的图像
    """
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # 将车道线图像与原始图像叠加
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return combined_image


def plan_path(image, lines):
    """
    规划路径
    :param image: 绘制了车道线的图像
    :param lines: 检测到的车道线
    :return: 规划了路径的图像
    """
    # 这里可以添加路径规划的逻辑
    # 例如，根据车道线的位置计算车辆的行驶路径
    # 目前只是简单返回绘制了车道线的图像
    return image


def main():
    # 读取图像
    image = cv2.imread('road.jpg')

    # 预处理
    edges = preprocess_image(image)

    # 提取ROI
    masked_image = region_of_interest(edges)

    # 检测车道线
    lines = detect_lines(masked_image)

    # 绘制车道线
    lane_image = draw_lines(image.copy(), lines)

    # 规划路径
    result = plan_path(lane_image, lines)

    # 显示结果
    cv2.imshow('Lane Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()