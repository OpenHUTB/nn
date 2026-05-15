import cv2
import numpy as np
import os

def main():
    # 1. 这里改成 CARLA 测试图片！！！
    img_path = 'carla_test.jpg'
    if not os.path.exists(img_path):
        print(f"错误：找不到文件 {img_path}！请把图片和代码放在同一个文件夹里。")
        return

    # 2. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取 {img_path}，文件可能损坏或格式不支持。")
        return

    height, width = img.shape[:2]
    print(f"图片读取成功，尺寸：{width}x{height}")

    # 3. 灰度化 + 高斯模糊 + Canny边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    # 4. 修复ROI区域（只保留下半部分道路，适配所有图片）
    roi_vertices = np.array([[
        (int(width*0.05), height),
        (int(width*0.45), int(height*0.6)),
        (int(width*0.55), int(height*0.6)),
        (int(width*0.95), height)
    ]], dtype=np.int32)

    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_img = cv2.bitwise_and(canny, mask)

    # 5. 霍夫变换检测车道线（降低参数，更容易识别）
    lines = cv2.HoughLinesP(
        roi_img,
        rho=1,
        theta=np.pi/180,
        threshold=15,
        minLineLength=20,
        maxLineGap=80
    )

    # 6. 绘制车道线
    result_img = img.copy()
    if lines is not None:
        print(f"检测到 {len(lines)} 条车道线")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    else:
        print("未检测到车道线，请检查图片或调整参数")

    # 7. 显示窗口（固定大小，避免显示不全）
    win_size = (640, 400)
    cv2.namedWindow("01-原图", cv2.WINDOW_NORMAL)
    cv2.namedWindow("02-边缘检测", cv2.WINDOW_NORMAL)
    cv2.namedWindow("03-ROI区域", cv2.WINDOW_NORMAL)
    cv2.namedWindow("04-车道线检测结果", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("01-原图", *win_size)
    cv2.resizeWindow("02-边缘检测", *win_size)
    cv2.resizeWindow("03-ROI区域", *win_size)
    cv2.resizeWindow("04-车道线检测结果", *win_size)

    cv2.imshow("01-原图", img)
    cv2.imshow("02-边缘检测", canny)
    cv2.imshow("03-ROI区域", roi_img)
    cv2.imshow("04-车道线检测结果", result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()