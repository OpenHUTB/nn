from djitellopy import Tello
import cv2
import numpy as np
import time

# 跟随参数
FOLLOW_DISTANCE = 100  # 目标在画面中的期望宽度（像素）
FOLLOW_HEIGHT = 50  # 目标在画面中的期望高度（像素）
MAX_SPEED = 30  # 最大飞行速度（cm/s）

# 初始化 Tello 无人机
tello = Tello()
tello.connect()
print(f"电池电量: {tello.get_battery()}%")

# 启动摄像头
tello.streamon()
cap = cv2.VideoCapture("udp://@0.0.0.0:11111")  # Tello 摄像头流地址


# 目标检测（简化：用颜色识别，可替换为 YOLO 识别行人/车辆）
def detect_target(frame):
    # 转换为 HSV 颜色空间（识别蓝色目标，可根据实际目标调整）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 取最大轮廓（假设最大的是目标）
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        return (x, y, w, h)  # 目标坐标和尺寸
    return None


# 跟随控制逻辑
def follow_target(target, frame):
    h, w, _ = frame.shape
    if target:
        x, y, tw, th = target
        # 计算目标中心
        target_center_x = x + tw // 2
        target_center_y = y + th // 2
        # 计算画面中心
        frame_center_x = w // 2
        frame_center_y = h // 2

        # 水平方向控制（左右）
        dx = target_center_x - frame_center_x
        left_right_speed = -dx * 0.3  # 比例控制
        left_right_speed = np.clip(left_right_speed, -MAX_SPEED, MAX_SPEED)

        # 垂直方向控制（上下）
        dy = target_center_y - frame_center_y
        up_down_speed = dy * 0.3
        up_down_speed = np.clip(up_down_speed, -MAX_SPEED, MAX_SPEED)

        # 前后方向控制（距离）
        distance_error = tw - FOLLOW_DISTANCE
        forward_backward_speed = -distance_error * 0.2
        forward_backward_speed = np.clip(forward_backward_speed, -MAX_SPEED, MAX_SPEED)

        # 发布速度指令
        tello.send_rc_control(
            int(left_right_speed),
            int(forward_backward_speed),
            int(up_down_speed),
            0  # 偏航速度（0 = 不旋转）
        )

        # 绘制目标框
        cv2.rectangle(frame, (x, y), (x + tw, y + th), (0, 255, 0), 2)
        cv2.circle(frame, (target_center_x, target_center_y), 5, (0, 0, 255), -1)
    else:
        # 未检测到目标，悬停
        tello.send_rc_control(0, 0, 0, 0)


# 主循环
tello.takeoff()  # 起飞
time.sleep(2)  # 悬停2秒
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 检测目标
        target = detect_target(frame)
        # 跟随控制
        follow_target(target, frame)
        # 显示画面
        cv2.imshow("Tello Follow", frame)
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    # 降落并清理资源
    tello.land()
    tello.streamoff()
    cap.release()
    cv2.destroyAllWindows()
    tello.end()