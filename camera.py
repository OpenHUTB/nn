import cv2
import numpy as np
from PIL import Image, ImageDraw
import random

# -------------------------- 1. 模拟无人车行驶场景（生成实时帧） --------------------------
def generate_driving_frame():
    """生成模拟无人车前方视角的帧（道路+随机障碍物）"""
    img_width, img_height = 1280, 720  # 适配识别分辨率
    frame = Image.new("RGB", (img_width, img_height), (100, 100, 100))  # 灰色天空背景
    draw = ImageDraw.Draw(frame)

    # 绘制道路（中间黑色路面，两侧白色标线）
    road_width = 800
    road_left = (img_width - road_width) // 2
    road_right = road_left + road_width
    # 黑色路面
    draw.rectangle([road_left, 0, road_right, img_height], fill=(50, 50, 50))
    # 白色边线
    draw.line([(road_left, 0), (road_left, img_height)], fill=(255, 255, 255), width=10)
    draw.line([(road_right, 0), (road_right, img_height)], fill=(255, 255, 255), width=10)
    # 中间虚线
    for y in range(0, img_height, 60):
        draw.rectangle([(img_width//2 - 10, y), (img_width//2 + 10, y + 30)], fill=(255, 255, 255))

    # 随机生成障碍物（车辆/行人，位置在道路中间）
    obstacle_type = random.choice(["car", "pedestrian", "car", "none"])  # 大概率生成车辆，偶尔行人/无障碍物
    obstacle_pos_y = random.randint(int(img_height * 0.4), int(img_height * 0.8))  # 前方不同距离
    obstacle_size = random.randint(80, 200)  # 大小=距离（越大越近）

    if obstacle_type == "car":
        # 绘制模拟车辆（矩形+车轮）
        car_x = random.randint(road_left + 50, road_right - 50 - obstacle_size)
        # 车身
        draw.rectangle([(car_x, obstacle_pos_y), (car_x + obstacle_size, obstacle_pos_y + obstacle_size//2)], fill=(255, 0, 0))
        # 车轮
        wheel_size = obstacle_size // 6
        draw.ellipse([(car_x + wheel_size, obstacle_pos_y + obstacle_size//2 - wheel_size), 
                      (car_x + 2*wheel_size, obstacle_pos_y + obstacle_size//2)], fill=(0,0,0))
        draw.ellipse([(car_x + obstacle_size - 2*wheel_size, obstacle_pos_y + obstacle_size//2 - wheel_size), 
                      (car_x + obstacle_size - wheel_size, obstacle_pos_y + obstacle_size//2)], fill=(0,0,0))
    elif obstacle_type == "pedestrian":
        # 绘制模拟行人（圆形+矩形）
        ped_x = random.randint(road_left + 50, road_right - 50 - obstacle_size//2)
        # 身体
        draw.rectangle([(ped_x + obstacle_size//4, obstacle_pos_y), (ped_x + 3*obstacle_size//4, obstacle_pos_y + obstacle_size)], fill=(0,0,255))
        # 头部
        draw.ellipse([(ped_x, obstacle_pos_y - obstacle_size//4), (ped_x + obstacle_size//2, obstacle_pos_y)], fill=(255, 255, 0))

    # 转换为OpenCV格式（BGR）
    return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR), obstacle_type  # 返回帧+真实障碍物类型（用于验证）

# -------------------------- 2. 障碍物识别核心逻辑（与Carla版本一致） --------------------------
def detect_obstacles(frame):
    # 预处理：灰度化+高斯模糊
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 形态学处理（连接断裂边缘）
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # 轮廓检测
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_obstacle = False
    danger_distance = False
    obstacle_area = 0

    # 感兴趣区域（前方路面）
    frame_height, frame_width = frame.shape[:2]
    roi_top = int(frame_height * 0.3)
    cv2.line(frame, (0, roi_top), (frame_width, roi_top), (255, 0, 0), 2)
    cv2.putText(frame, "Forward Area", (30, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:  # 过滤小噪点
            continue

        # 轮廓中心（只关注道路中间区域）
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 限制在道路中间60%区域
        if cx > frame_width * 0.2 and cx < frame_width * 0.8 and cy > roi_top:
            has_obstacle = True
            obstacle_area = area

            # 绘制标注
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # 危险距离判断（面积越大越近）
            if area > 15000:
                danger_distance = True

    # 显示识别结果
    if has_obstacle:
        if danger_distance:
            cv2.putText(frame, "⚠️ DANGER: OBSTACLE AHEAD!", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            cv2.putText(frame, "⚠️ OBSTACLE DETECTED", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
    else:
        cv2.putText(frame, "✅ No Obstacle", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    return has_obstacle, danger_distance, frame

# -------------------------- 3. 实时运行模拟（流畅无延迟） --------------------------
def run_obstacle_simulation():
    print("🚗 无人车障碍物识别模拟启动（按 'q' 键退出）")
    print("模拟场景：随机生成道路、车辆、行人，实时检测前方障碍物")

    while True:
        # 生成模拟行驶帧
        frame, _ = generate_driving_frame()
        # 执行障碍物识别
        _, _, annotated_frame = detect_obstacles(frame)
        # 实时显示（10ms刷新，无延迟）
        cv2.imshow("Obstacle Detection Simulation", annotated_frame)

        # 按q退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("✅ 模拟结束")

if __name__ == "__main__":
    try:
        run_obstacle_simulation()
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        cv2.destroyAllWindows()
