import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # 导入PIL库
import os

def calculate_distance(actual_height, focal_length, pixel_height):
    """
    计算物体到摄像头的距离
    
    参数:
        actual_height (float): 物体的实际高度（单位：米）
        focal_length (float): 相机的焦距（单位：像素）
        pixel_height (float): 物体在图像中的像素高度
    
    返回:
        float: 物体到摄像头的距离（单位：米）
    
    异常:
        ValueError: 如果参数为负数或零
    """
    if pixel_height <= 0 or actual_height <= 0 or focal_length <= 0:
        raise ValueError("参数必须为正数")
    return (actual_height * focal_length) / pixel_height

def select_roi_once(frame, window_name="框选物体（按Enter确认，Esc取消）"):
    """
    手动框选物体并返回像素高度
    
    参数:
        frame (numpy.ndarray): 输入图像帧
        window_name (str): 窗口名称
    
    返回:
        int: 物体的像素高度，如果用户取消框选则返回0
    """
    roi = cv2.selectROI(window_name, frame, False)
    if roi == (0, 0, 0, 0):  # 用户按Esc取消框选
        cv2.destroyWindow(window_name)
        return 0
    x, y, w, h = roi
    cv2.destroyWindow(window_name)
    return h  # 返回物体像素高度
def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """用PIL在OpenCV图像上绘制中文"""
    # 转换OpenCV图像（BGR）为PIL图像（RGB）
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 加载中文字体（优先查找系统字体）
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc"   # 宋体
    ]
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break
    if not font_path:
        print("警告：未找到中文字体，使用默认字体（可能显示异常）")
        font = ImageFont.load_default()
    else:
        # 加载字体
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    # 绘制文字
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    # 转换回OpenCV图像（BGR）
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    # ------------------- 核心参数（必须根据实际情况修改！） -------------------
    ACTUAL_HEIGHT = 0.15  # 物体实际高度（米），例如：一个易拉罐高15cm
    FOCAL_LENGTH = 500    # 相机焦距（像素），标定方法见备注
    # ----------------------------------------------------------------------

    # 打开摄像头（0为默认摄像头，多个摄像头可尝试1、2等）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！请检查设备连接")
        return

    pixel_height = 0  # 物体像素高度（初始为0）
    print("提示：\n- 首次运行请框选目标物体（拖动鼠标，按Enter确认）\n- 按 'r' 重新框选物体\n- 按 'q' 退出程序")

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取图像，退出...")
            break

        # 动态更新物体像素高度
        if cv2.waitKey(1) & 0xFF == ord('r') or pixel_height == 0:
            pixel_height = select_roi_once(frame)
            if pixel_height == 0:
                print("未框选有效区域，请重试！")
                continue
        # 计算距离并显示
        try:
            distance = calculate_distance(ACTUAL_HEIGHT, FOCAL_LENGTH, pixel_height)
            cv2.putText(
                frame,
                f"distance: {distance:.2f} m",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),  # 绿色文字
                3
            )
        except ValueError as e:
            cv2.putText(frame, f"错误: {str(e)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # 显示实时画面
        # 添加退出按钮
        button_pos = (frame.shape[1] - 100, frame.shape[0] - 30)  # 按钮位置（右下角）
        cv2.rectangle(
            frame,
            (button_pos[0], button_pos[1]),
            (button_pos[0] + 80, button_pos[1] + 30),
            (0, 0, 255),  # 红色背景
            -1
        )

        # 用PIL绘制按钮上的中文“退出”
        frame = cv2_add_chinese_text(
            frame,
            "退出",
            (button_pos[0] + 15, button_pos[1] + 5),  # 文字位置
            font_size=20,
            color=(255, 255, 255)  # 白色文字
        )

        # 在终端打印距离信息
        print(f"当前距离: {distance:.2f} 米")

        # 确保窗口已创建
        cv2.imshow("实时测距（按q退出，r重新框选）", frame)

        # 检测鼠标点击事件
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if button_pos[0] <= x <= button_pos[0] + 80 and button_pos[1] <= y <= button_pos[1] + 30:
                    print("正在退出程序...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)

        cv2.setMouseCallback("实时测距（按q退出，r重新框选）", mouse_callback)

        # 按'q'退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("正在退出程序...")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()