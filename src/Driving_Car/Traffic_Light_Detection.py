import cv2
import numpy as np
from PIL import Image, ImageDraw


# -------------------------- 第一步：生成模拟红绿灯图片 --------------------------
def generate_traffic_light(light_color="red"):
    """
    生成模拟红绿灯图片（红灯/黄灯/绿灯可选）
    :param light_color: 亮灯颜色，可选 "red", "yellow", "green"
    :return: 图片路径
    """
    # 图片尺寸：宽400px，高600px（模拟真实红绿灯比例）
    img_width, img_height = 400, 600
    background_color = (0, 0, 0)  # 背景黑色
    dark_color = (50, 50, 50)  # 未亮灯的暗灰色
    light_colors = {
        "red": (255, 0, 0),
        "yellow": (255, 255, 0),
        "green": (0, 255, 0)
    }

    # 创建空白图片（RGB模式）
    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)

    # 灯的位置：上（红）、中（黄）、下（绿），圆心坐标和半径
    light_radius = 80  # 灯的半径
    light_positions = [
        (img_width // 2, img_height // 4),  # 红灯位置（上）
        (img_width // 2, img_height // 2),  # 黄灯位置（中）
        (img_width // 2, 3 * img_height // 4)  # 绿灯位置（下）
    ]

    # 绘制三个灯（未亮灯为暗灰色，亮灯为对应颜色）
    for i, pos in enumerate(light_positions):
        color = dark_color
        if (i == 0 and light_color == "red") or \
                (i == 1 and light_color == "yellow") or \
                (i == 2 and light_color == "green"):
            color = light_colors[light_color]
        # 绘制圆形灯（填充+边框）
        draw.ellipse(
            [pos[0] - light_radius, pos[1] - light_radius,
             pos[0] + light_radius, pos[1] + light_radius],
            fill=color, outline=(200, 200, 200), width=5
        )

    # 保存图片
    img_path = f"traffic_light_{light_color}.jpg"
    img.save(img_path)
    print(f"已生成红绿灯图片：{img_path}（亮灯颜色：{light_color}）")
    return img_path


# -------------------------- 第二步：红绿灯识别核心逻辑 --------------------------
def detect_traffic_light(img_path):
    """
    基于颜色和形状识别红绿灯状态
    :param img_path: 红绿灯图片路径
    :return: 识别结果（"red", "yellow", "green", "unknown"）
    """
    # 1. 读取图片并转换为HSV颜色空间（对颜色分割更友好）
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图片 {img_path}")
        return "unknown"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 定义红、黄、绿三种颜色的HSV阈值（修复红色区间格式！）
    color_ranges = {
        "red": [
            [(0, 120, 70), (10, 255, 255)],  # 红色低区间（元组嵌套修复）
            [(170, 120, 70), (180, 255, 255)]  # 红色高区间（元组嵌套修复）
        ],
        "yellow": [(20, 120, 70), (30, 255, 255)],
        "green": [(35, 120, 70), (77, 255, 255)]
    }

    # 3. 对每种颜色进行掩码处理（筛选出对应颜色区域）
    light_detected = "unknown"
    max_light_area = 0  # 记录最大亮灯区域面积（避免误识别小色块）

    for color, ranges in color_ranges.items():
        # 生成颜色掩码（多个区间合并）
        mask = np.zeros_like(hsv[:, :, 0])
        # 处理红色的多区间（需循环每个子区间）
        if color == "red":
            for (lower, upper) in ranges:
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                mask += cv2.inRange(hsv, lower_np, upper_np)
        else:
            # 黄/绿单区间直接处理
            lower_np = np.array(ranges[0])
            upper_np = np.array(ranges[1])
            mask += cv2.inRange(hsv, lower_np, upper_np)

        # 4. 形态学处理（去除噪点，填充小缺口）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 5. 检测圆形轮廓（红绿灯的灯是圆形）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # 计算轮廓面积和圆形度（圆形度=4π面积/周长²，越接近1越圆）
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            # 筛选条件：面积足够大（排除小噪点）+ 圆形度高（排除非圆形）
            if area > 5000 and circularity > 0.7:
                # 记录最大面积的亮灯（避免多个颜色误检）
                if area > max_light_area:
                    max_light_area = area
                    light_detected = color

    # 6. 绘制识别结果并显示图片
    result_img = img.copy()
    cv2.putText(
        result_img, f"Detected: {light_detected.upper()}",
        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
    )
    cv2.imshow("Traffic Light Detection Result", result_img)
    cv2.waitKey(3000)  # 显示3秒
    cv2.destroyAllWindows()

    return light_detected


# -------------------------- 第三步：运行测试 --------------------------
if __name__ == "__main__":
    # 1. 生成红绿灯图片（可改为 "yellow" 或 "green" 测试）
    img_path = generate_traffic_light(light_color="red")

    # 2. 识别红绿灯
    result = detect_traffic_light(img_path)

    # 3. 输出结果
    print(f"\n最终识别结果：{result}灯")