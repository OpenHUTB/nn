import cv2
import numpy as np
import os

# ===================== PR1：配置参数分离 =====================
# 全局配置（硬编码解耦，方便修改）
CONFIG = {
    "img_path": "carla_test.jpg",
    "canny_low": 50,
    "canny_high": 150,
    "gaussian_kernel": (5, 5),
    "roi_scale": [0.05, 0.45, 0.55, 0.95, 0.6],
    "hough_threshold": 15,
    "min_line_length": 20,
    "max_line_gap": 80
}

def load_image(img_path):
    """加载图片（标准化函数）"""
    if not os.path.exists(img_path):
        print(f"错误：找不到文件 {img_path}！")
        return None
    img = cv2.imread(img_path)
    return img if img is not None else None

def preprocess(img):
    """预处理：灰度+模糊+Canny边缘检测"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, CONFIG["gaussian_kernel"], 0)
    canny = cv2.Canny(blur, CONFIG["canny_low"], CONFIG["canny_high"])
    return canny

def roi_extract(canny, width, height):
    """ROI区域提取"""
    s = CONFIG["roi_scale"]
    vertices = np.array([[
        (int(width*s[0]), height),
        (int(width*s[1]), int(height*s[4])),
        (int(width*s[2]), int(height*s[4])),
        (int(width*s[3]), height)
    ]], dtype=np.int32)
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(canny, mask)

def main():
    img = load_image(CONFIG["img_path"])
    if img is None: return
    h, w = img.shape[:2]
    canny = preprocess(img)
    roi_img = roi_extract(canny, w, h)
    
    # 霍夫变换
    lines = cv2.HoughLinesP(roi_img, 1, np.pi/180, CONFIG["hough_threshold"],
                           minLineLength=CONFIG["min_line_length"], maxLineGap=CONFIG["max_line_gap"])
    
    # 绘制
    res = img.copy()
    if lines is not None:
        for line in lines:
            cv2.line(res, (line[0][0],line[0][1]),(line[0][2],line[0][3]), (0,255,0), 3)

    # 显示
    cv2.imshow("result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()