import cv2
import numpy as np
from PIL import Image, ImageDraw

# -------------------------- 1. 快速生成模拟帧（精简绘制，缩小分辨率） --------------------------
def generate_traffic_light_frame(light_color="red"):
    """快速生成红绿灯帧（800x600分辨率，精简绘制逻辑）"""
    img_width, img_height = 800, 600  # 缩小分辨率，减少计算量
    background_color = (30, 30, 30)  # 简化背景
    dark_color = (60, 60, 60)
    light_colors = {
        "red": (255, 30, 30),
        "yellow": (255, 255, 30),
        "green": (30, 255, 30)
    }

    # 快速创建图片（减少冗余绘制）
    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)

    # 简化红绿灯绘制（只保留核心灯体，取消复杂装饰）
    light_radius = 40
    light_positions = [
        (img_width//2, img_height//3),
        (img_width//2, img_height//2),
        (img_width//2, 2*img_height//3)
    ]

    for i, pos in enumerate(light_positions):
        color = dark_color if not (
            (i==0 and light_color=="red") or
            (i==1 and light_color=="yellow") or
            (i==2 and light_color=="green")
        ) else light_colors[light_color]
        # 仅绘制核心灯体（取消光晕、简化边框）
        draw.ellipse(
            [pos[0]-light_radius, pos[1]-light_radius,
             pos[0]+light_radius, pos[1]+light_radius],
            fill=color, outline=(200,200,200), width=3
        )

    # 快速转换为OpenCV格式
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# -------------------------- 2. 优化识别逻辑（减少计算量） --------------------------
def detect_traffic_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 简化HSV阈值（减少判断耗时）
    color_ranges = {
        "red": [[(0, 120, 70), (10, 255, 255)], [(170, 120, 70), (180, 255, 255)]],
        "yellow": [(22, 120, 70), (32, 255, 255)],
        "green": [(45, 120, 70), (70, 255, 255)]
    }

    light_detected = "unknown"
    max_light_area = 0

    for color, ranges in color_ranges.items():
        mask = np.zeros_like(hsv[:, :, 0])
        # 简化循环逻辑
        if color == "red":
            mask = cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1])) + \
                   cv2.inRange(hsv, np.array(ranges[1][0]), np.array(ranges[1][1]))
        else:
            mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))

        # 缩小形态学核（减少运算量）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 快速轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            # 适配小分辨率的面积阈值
            if area > 3000 and circularity > 0.65:
                if area > max_light_area:
                    max_light_area = area
                    light_detected = color

    # 简化绘制标注
    cv2.putText(
        frame, f"TL: {light_detected.upper()}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
    )
    return light_detected, frame

# -------------------------- 3. 高速运行循环（无延迟切换） --------------------------
def run_fast_simulation():
    print("🚀 快速版模拟启动（按 'q' 退出）")
    light_sequence = ["red", "yellow", "green"]
    index = 0
    frame_count = 0  # 按帧切换，无强制休眠

    while True:
        # 每15帧切换一次灯态（约0.3秒切换，流畅无延迟）
        if frame_count % 15 == 0:
            current_light = light_sequence[index % len(light_sequence)]
            index += 1

        # 快速生成+识别
        frame = generate_traffic_light_frame(current_light)
        _, annotated_frame = detect_traffic_light(frame)

        # 高速显示（10ms刷新一次）
        cv2.imshow("Fast Traffic Light Detection", annotated_frame)
        frame_count += 1

        # 按q立即退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("✅ 模拟结束")

if __name__ == "__main__":
    run_fast_simulation()
