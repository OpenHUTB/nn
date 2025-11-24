import cv2
import numpy as np
from PIL import Image, ImageDraw
import time

# -------------------------- 1. 模拟Carla红绿灯场景（生成实时帧） --------------------------
def generate_traffic_light_frame(light_color="red"):
    """生成模拟Carla红绿灯的帧（替代Carla摄像头输入）"""
    img_width, img_height = 1920, 1080  # 匹配原Carla摄像头分辨率
    background_color = (30, 30, 30)  # 模拟道路暗背景
    dark_color = (60, 60, 60)        # 未亮灯暗灰色
    light_colors = {
        "red": (255, 30, 30),
        "yellow": (255, 255, 30),
        "green": (30, 255, 30)
    }

    # 创建图片
    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)

    # 红绿灯位置（模拟车辆前方远处）
    light_radius = 60
    light_positions = [
        (img_width//2, img_height//3),
        (img_width//2, img_height//2),
        (img_width//2, 2*img_height//3)
    ]

    # 绘制红绿灯
    for i, pos in enumerate(light_positions):
        color = dark_color
        if (i == 0 and light_color == "red") or \
           (i == 1 and light_color == "yellow") or \
           (i == 2 and light_color == "green"):
            color = light_colors[light_color]
        # 绘制灯体（带光晕效果）
        draw.ellipse(
            [pos[0]-light_radius, pos[1]-light_radius,
             pos[0]+light_radius, pos[1]+light_radius],
            fill=color, outline=(200, 200, 200), width=8
        )
        # 绘制灯座
        draw.rectangle(
            [img_width//2 - 80, img_height//4 - 40,
             img_width//2 + 80, 3*img_height//4 + 40],
            fill=(80, 80, 80), outline=(150, 150, 150), width=10
        )

    # 转换为OpenCV格式（BGR）
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

# -------------------------- 2. 红绿灯识别核心逻辑（与Carla版本一致） --------------------------
def detect_traffic_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 适配模拟场景的HSV阈值
    color_ranges = {
        "red": [
            [(0, 140, 90), (10, 255, 255)],
            [(170, 140, 90), (180, 255, 255)]
        ],
        "yellow": [(22, 140, 90), (32, 255, 255)],
        "green": [(45, 140, 90), (70, 255, 255)]
    }

    light_detected = "unknown"
    max_light_area = 0

    for color, ranges in color_ranges.items():
        mask = np.zeros_like(hsv[:, :, 0])
        if color == "red":
            for lower, upper in ranges:
                mask += cv2.inRange(hsv, np.array(lower), np.array(upper))
        else:
            lower, upper = ranges
            mask += cv2.inRange(hsv, np.array(lower), np.array(upper))

        # 形态学去噪
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 圆形轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if area > 5000 and circularity > 0.7:
                if area > max_light_area:
                    max_light_area = area
                    light_detected = color

    # 绘制识别结果
    result_frame = frame.copy()
    cv2.putText(
        result_frame, f"Traffic Light: {light_detected.upper()}",
        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4
    )
    return light_detected, result_frame

# -------------------------- 3. 模拟Carla实时检测（循环切换红绿灯） --------------------------
def run_simulation():
    print("📌 开始模拟Carla红绿灯识别（按 'q' 键退出）")
    print("模拟场景：自动切换红→黄→绿→红...")
    
    # 循环切换红绿灯状态（模拟车辆行驶中遇到的不同灯）
    light_sequence = ["red", "yellow", "green", "red", "yellow", "green"]
    index = 0

    while True:
        # 生成当前状态的红绿灯帧
        current_light = light_sequence[index % len(light_sequence)]
        frame = generate_traffic_light_frame(current_light)
        
        # 执行识别
        result, annotated_frame = detect_traffic_light(frame)
        
        # 显示结果
        cv2.imshow("Simulated Carla Traffic Light Detection", annotated_frame)
        
        # 切换灯状态（每3秒切换一次）
        time.sleep(3)
        index += 1

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("📌 模拟结束")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        cv2.destroyAllWindows()
