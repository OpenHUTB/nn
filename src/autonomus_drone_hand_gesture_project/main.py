# -*- coding: utf-8 -*-
"""
手势控制AirSim无人机 - 中文版
修复了中文乱码问题，支持中文显示
作者：xiaoshiyuan888
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
from PIL import Image, ImageDraw, ImageFont

print("=" * 50)
print("手势控制无人机系统 v1.0 - 中文版")
print("=" * 50)

# ========== 1. 检查并导入必要的库 ==========
try:
    import mediapipe as mp

    print("✓ MediaPipe 已安装")
except:
    print("✗ 需要安装 MediaPipe: pip install mediapipe")
    sys.exit(1)

# 检查AirSim
USE_AIRSIM = True  # 默认尝试使用AirSim
try:
    import airsim

    print("✓ AirSim 已安装")
except ImportError:
    print("⚠ AirSim 未安装，将使用模拟模式")
    print("  如果想使用真实AirSim，请安装: pip install airsim")
    USE_AIRSIM = False

# ========== 2. 初始化MediaPipe ==========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ========== 3. 初始化摄像头 ==========
cap = cv2.VideoCapture(0)  # 0表示第一个摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("错误：无法打开摄像头")
    sys.exit(1)

print("✓ 摄像头已打开")


# ========== 4. 加载中文字体 ==========
def load_chinese_font():
    """加载中文字体"""
    # 尝试多个可能的字体路径
    font_paths = [
        'simhei.ttf',  # 项目文件夹中的字体
        'C:/Windows/Fonts/simhei.ttf',  # Windows字体目录
        'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 20)
            print(f"✓ 加载字体成功: {font_path}")
            return font
        except Exception as e:
            continue

    print("⚠ 未找到中文字体，将使用默认字体")
    return ImageFont.load_default()


# 加载字体
chinese_font = load_chinese_font()


# ========== 5. 中文显示函数 ==========
def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在图像上绘制中文文本

    参数:
        img: OpenCV图像 (BGR格式)
        text: 要显示的中文文本
        position: (x, y) 位置
        font_size: 字体大小
        color: 字体颜色 (BGR格式)

    返回:
        添加了中文文本的图像
    """
    # 将OpenCV图像转换为PIL图像 (BGR -> RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)

    # 创建临时字体对象
    try:
        # 尝试加载指定大小的字体
        temp_font = ImageFont.truetype('simhei.ttf', font_size)
    except:
        # 如果失败，使用加载的字体
        temp_font = chinese_font

    # 绘制文本
    # PIL使用RGB颜色，OpenCV使用BGR，所以需要转换颜色顺序
    rgb_color = color[::-1]  # BGR -> RGB

    # 添加文本阴影效果（更清晰）
    shadow_color = (0, 0, 0)  # 黑色阴影
    shadow_position = (position[0] + 1, position[1] + 1)
    draw.text(shadow_position, text, font=temp_font, fill=shadow_color)

    # 绘制文本主体
    draw.text(position, text, font=temp_font, fill=rgb_color)

    # 将PIL图像转换回OpenCV图像 (RGB -> BGR)
    img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img_with_text


# ========== 6. 无人机状态和控制 ==========
class DroneController:
    """无人机控制器"""

    def __init__(self, use_airsim=True):
        self.use_airsim = use_airsim
        self.is_flying = False
        self.gesture_name = "等待手势"
        self.airsim_client = None

        # 模拟无人机位置（在屏幕上显示）
        self.drone_x = 320  # 屏幕中心
        self.drone_y = 240
        self.drone_size = 30

        if use_airsim:
            self.connect_airsim()

    def connect_airsim(self):
        """连接AirSim"""
        try:
            self.airsim_client = airsim.MultirotorClient()
            self.airsim_client.confirmConnection()
            self.airsim_client.enableApiControl(True)
            self.airsim_client.armDisarm(True)
            print("✓ AirSim连接成功！")
            return True
        except Exception as e:
            print(f"✗ AirSim连接失败: {e}")
            self.use_airsim = False
            return False

    def takeoff(self):
        """起飞"""
        if self.use_airsim and self.airsim_client:
            try:
                self.airsim_client.takeoffAsync().join()
                self.is_flying = True
                print("✓ 无人机已起飞")
            except Exception as e:
                print(f"起飞失败: {e}")
        else:
            self.is_flying = True
            print("✓ [模拟] 无人机已起飞")

    def land(self):
        """降落"""
        if self.use_airsim and self.airsim_client:
            try:
                self.airsim_client.landAsync().join()
                self.is_flying = False
                print("✓ 无人机已降落")
            except Exception as e:
                print(f"降落失败: {e}")
        else:
            self.is_flying = False
            print("✓ [模拟] 无人机已降落")

    def move_by_gesture(self, gesture_name):
        """根据手势移动无人机"""
        self.gesture_name = gesture_name

        if not self.is_flying:
            return

        # 模拟模式：在屏幕上移动无人机图标
        if not self.use_airsim:
            if gesture_name == "向上":
                self.drone_y -= 5
            elif gesture_name == "向下":
                self.drone_y += 5
            elif gesture_name == "向左":
                self.drone_x -= 5
            elif gesture_name == "向右":
                self.drone_x += 5
            elif gesture_name == "向前":
                self.drone_size = max(20, self.drone_size - 2)
            elif gesture_name == "向后":
                self.drone_size = min(50, self.drone_size + 2)

            # 限制在屏幕内
            self.drone_x = max(30, min(610, self.drone_x))
            self.drone_y = max(30, min(450, self.drone_y))

        # 真实AirSim模式
        elif self.airsim_client:
            try:
                if gesture_name == "向上":
                    self.airsim_client.moveByVelocityZAsync(0, 0, -2, 1).join()
                elif gesture_name == "向下":
                    self.airsim_client.moveByVelocityZAsync(0, 0, 2, 1).join()
                elif gesture_name == "向左":
                    self.airsim_client.moveByVelocityAsync(-2, 0, 0, 1).join()
                elif gesture_name == "向右":
                    self.airsim_client.moveByVelocityAsync(2, 0, 0, 1).join()
                elif gesture_name == "向前":
                    self.airsim_client.moveByVelocityAsync(0, -2, 0, 1).join()
                elif gesture_name == "向后":
                    self.airsim_client.moveByVelocityAsync(0, 2, 0, 1).join()
                elif gesture_name == "顺时针旋转":
                    self.airsim_client.rotateByYawRateAsync(30, 1).join()
                elif gesture_name == "逆时针旋转":
                    self.airsim_client.rotateByYawRateAsync(-30, 1).join()
                elif gesture_name == "停止":
                    self.airsim_client.hoverAsync().join()
            except Exception as e:
                print(f"控制失败: {e}")

    def draw_drone_on_frame(self, frame):
        """在画面上绘制无人机（模拟模式）"""
        if not self.use_airsim:
            # 绘制无人机图标
            color = (0, 255, 0) if self.is_flying else (100, 100, 100)

            # 无人机主体
            cv2.circle(frame, (self.drone_x, self.drone_y), self.drone_size, color, 2)

            # 十字表示螺旋桨
            cv2.line(frame, (self.drone_x - 20, self.drone_y),
                     (self.drone_x + 20, self.drone_y), color, 2)
            cv2.line(frame, (self.drone_x, self.drone_y - 20),
                     (self.drone_x, self.drone_y + 20), color, 2)

            # 无人机标签（使用中文显示函数）
            frame = put_chinese_text(frame, "无人机",
                                     (self.drone_x - 25, self.drone_y - 35),
                                     font_size=14, color=color)


# ========== 7. 手势识别函数 ==========
def recognize_gesture(hand_landmarks, frame_width, frame_height):
    """
    简单手势识别 - 基于食指和中指位置
    修复了 'int' object has no attribute 'y' 错误
    """
    try:
        # 确保手部关键点有效
        if not hand_landmarks or not hasattr(hand_landmarks, 'landmark'):
            return "等待手势"

        landmarks = hand_landmarks.landmark

        # 确保有足够的关键点
        if len(landmarks) < 13:  # 至少需要13个关键点
            return "手势不完整"

        # 获取关键点坐标
        wrist = landmarks[0]  # 手腕
        index_tip = landmarks[8]  # 食指尖
        middle_tip = landmarks[12]  # 中指尖

        # 判断手指是否伸直（指尖y坐标小于指关节y坐标）
        # 注意：摄像头坐标系中，y轴向下为正

        # 食指关节（第6个关键点）
        if len(landmarks) > 6:
            index_bent = index_tip.y > landmarks[6].y
        else:
            index_bent = True

        # 中指关节（第10个关键点）
        if len(landmarks) > 10:
            middle_bent = middle_tip.y > landmarks[10].y
        else:
            middle_bent = True

        # 简单手势判断
        if index_bent and middle_bent:  # 两个手指都弯曲
            return "停止"
        elif not index_bent and middle_bent:  # 食指伸直，中指弯曲
            # 判断方向
            if index_tip.y < wrist.y - 0.15:
                return "向上"
            elif index_tip.y > wrist.y + 0.15:
                return "向下"
            elif index_tip.x < wrist.x - 0.15:
                return "向左"
            elif index_tip.x > wrist.x + 0.15:
                return "向右"
            else:
                return "向前"
        elif not index_bent and not middle_bent:  # 两个手指都伸直
            return "剪刀手"
        else:
            return "未知手势"

    except Exception as e:
        # 调试时可以取消注释下面一行
        # print(f"手势识别错误: {e}")
        return "等待手势"


# ========== 8. 绘制中文界面函数 ==========
def draw_chinese_interface(frame, drone_controller, gesture_name, fps):
    """绘制中文用户界面"""
    h, w = frame.shape[:2]

    # 半透明背景区域
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 标题 - 使用中文
    frame = put_chinese_text(frame, "手势控制无人机", (10, 10),
                             font_size=24, color=(0, 255, 255))

    # 状态信息
    status = "飞行中" if drone_controller.is_flying else "已降落"
    status_color = (0, 255, 0) if drone_controller.is_flying else (0, 0, 255)

    frame = put_chinese_text(frame, f"状态: {status}", (10, 50),
                             font_size=20, color=status_color)

    frame = put_chinese_text(frame, f"手势: {gesture_name}", (10, 80),
                             font_size=20, color=(255, 255, 0))

    mode_text = "AirSim" if drone_controller.use_airsim else "模拟"
    frame = put_chinese_text(frame, f"模式: {mode_text}", (10, 110),
                             font_size=16, color=(200, 200, 200))

    # FPS显示（英文）
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 控制说明 - 使用中文
    frame = put_chinese_text(frame, "空格:起飞/降落  ESC:退出", (w - 300, h - 20),
                             font_size=14, color=(255, 255, 255))

    # 手势说明
    if not drone_controller.is_flying:
        frame = put_chinese_text(frame, "按空格键起飞无人机", (w // 2 - 100, h // 2),
                                 font_size=20, color=(0, 255, 255))

    # 绘制手势区域边界
    cv2.rectangle(frame, (50, 150), (w - 50, h - 50), (0, 255, 0), 2)
    frame = put_chinese_text(frame, "手势识别区域", (w // 2 - 50, 135),
                             font_size=14, color=(0, 255, 0))

    return frame


# ========== 9. 主程序 ==========
def main():
    """主函数"""
    print("\n系统初始化完成！")
    print("操作说明:")
    print("  1. 将手放在摄像头前")
    print("  2. 按空格键：起飞/降落无人机")
    print("  3. 使用手势控制无人机")
    print("  4. 按ESC键：退出程序")
    print("-" * 50)

    # 创建无人机控制器
    drone = DroneController(use_airsim=USE_AIRSIM)

    # FPS计算
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    print("程序启动成功！等待用户操作...")

    while True:
        # 读取摄像头
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头画面")
            break

        # 镜像翻转（看起来更自然）
        frame = cv2.flip(frame, 1)

        # 转换为RGB（MediaPipe需要）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手势检测
        results = hands.process(rgb_frame)

        current_gesture = "等待手势"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # 识别手势
                current_gesture = recognize_gesture(
                    hand_landmarks, frame.shape[1], frame.shape[0]
                )

                # 执行控制
                drone.move_by_gesture(current_gesture)

        # 绘制无人机（模拟模式）
        drone.draw_drone_on_frame(frame)

        # 计算FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()

        # 绘制中文用户界面
        frame = draw_chinese_interface(frame, drone, current_gesture, current_fps)

        # 显示画面（窗口标题用英文避免乱码）
        cv2.imshow('Gesture Controlled Drone - Press ESC to exit', frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC键
            print("用户按ESC键，退出程序...")
            break
        elif key == 32:  # 空格键
            if drone.is_flying:
                drone.land()
            else:
                drone.takeoff()
            time.sleep(0.3)  # 防抖动

    # 清理资源
    print("\n正在清理资源...")
    cap.release()
    cv2.destroyAllWindows()

    if drone.use_airsim and drone.airsim_client:
        try:
            if drone.is_flying:
                drone.land()
            drone.airsim_client.armDisarm(False)
            drone.airsim_client.enableApiControl(False)
            print("AirSim连接已关闭")
        except:
            pass

    print("程序已安全退出！")
    print("=" * 50)


# ========== 10. 程序入口 ==========
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback

        traceback.print_exc()