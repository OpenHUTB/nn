"""步骤1：灰度化、Canny 边缘检测、ROI 与霍夫直线检测。"""
import cv2
import numpy as np

from config import CONFIG


def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图片 {img_path}")
    return img


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, CONFIG["gaussian_kernel"], 0)
    return cv2.Canny(blur, CONFIG["canny_low"], CONFIG["canny_high"])


def roi_extract(canny, width, height):
    s = CONFIG["roi_scale"]
    vertices = np.array(
        [[
            (int(width * s[0]), height),
            (int(width * s[1]), int(height * s[4])),
            (int(width * s[2]), int(height * s[4])),
            (int(width * s[3]), height),
        ]],
        dtype=np.int32,
    )
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(canny, mask)


def detect_lines(roi_img):
    return cv2.HoughLinesP(
        roi_img,
        1,
        np.pi / 180,
        CONFIG["hough_threshold"],
        minLineLength=CONFIG["min_line_length"],
        maxLineGap=CONFIG["max_line_gap"],
    )


def draw_hough_lines(img, lines):
    result = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return result


def run_basic_pipeline(img_path=None, save_dir=None):
    """运行基础流水线，可选保存中间结果图。"""
    path = img_path or CONFIG["img_path"]
    img = load_image(path)
    if img is None:
        return None

    height, width = img.shape[:2]
    canny = preprocess(img)
    roi_img = roi_extract(canny, width, height)
    lines = detect_lines(roi_img)
    result = draw_hough_lines(img, lines)

    if save_dir:
        save_dir = str(save_dir)
        cv2.imwrite(f"{save_dir}/step01_input.jpg", img)
        cv2.imwrite(f"{save_dir}/step01_canny.jpg", canny)
        cv2.imwrite(f"{save_dir}/step01_roi.jpg", roi_img)
        cv2.imwrite(f"{save_dir}/step01_hough.jpg", result)

    return {
        "input": img,
        "canny": canny,
        "roi": roi_img,
        "result": result,
    }
