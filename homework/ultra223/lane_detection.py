"""
车道与路径检测项目 - 基础预处理模块
作者：ultra223
进度：图像预处理 + 边缘检测 + 感兴趣区域提取
"""
import cv2
import numpy as np

def image_process(img):
    """
    图像预处理：灰度化 + 高斯模糊 + Canny边缘检测
    """
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    canny = cv2.Canny(blur, 50, 150)
    return canny

def roi_extract(canny_img):
    """
    提取车道线感兴趣区域（去除天空、树木等干扰）
    """
    height = canny_img.shape[0]
    width = canny_img.shape[1]
    
    # 定义梯形区域（只保留路面）
    polygons = np.array([
        [(200, height), (width - 200, height), (width//2, height//2 + 50)]
    ])
    
    mask = np.zeros_like(canny_img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(canny_img, mask)
    return masked_img

def lane_detection(image_path):
    """
    主函数：完成功能
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法读取图片，请检查路径是否正确")
        return

    # 预处理
    canny_img = image_process(img)
    
    # 提取感兴趣区域
    roi_img = roi_extract(canny_img)

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Canny Edge", canny_img)
    cv2.imshow("ROI Result", roi_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=" * 50)
    print("车道与路径检测项目")
    print("作者：ultra223")
    print("进度：预处理 + 边缘检测 + ROI提取")
    print("=" * 50)
    lane_detection("test.jpg")