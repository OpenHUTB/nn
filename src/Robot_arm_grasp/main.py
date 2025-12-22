import cv2
import numpy as np
import pygame
import math
import threading
import time

# ====================== 全局配置参数（可根据需求微调）=====================
# ---------------------- 视觉识别参数（真实摄像头）----------------------
TARGET_COLOR_LOWER = np.array([0, 120, 70])    # 红色HSV下限（可适配自己的红色目标）
TARGET_COLOR_UPPER = np.array([10, 255, 255])  # 红色HSV上限
TARGET_COLOR_LOWER2 = np.array([170, 120, 70]) # 红色补集（解决红色跨0度的问题）
TARGET_COLOR_UPPER2 = np.array([180, 255, 255])
MIN_CONTOUR_AREA = 300  # 最小轮廓面积（过滤小噪声）
CAMERA_INDEX = 0        # 摄像头索引（0为默认，若没反应改1）
VISION_WIDTH = 640      # 摄像头画面宽度
VISION_HEIGHT = 480     # 摄像头画面高度

# ---------------------- 仿真窗口与机械臂参数 ----------------------
SIM_WIDTH = 800         # 仿真窗口宽度
SIM_HEIGHT = 600        # 仿真窗口高度
ARM_BASE_POS = (400, 500)# 机械臂基座在仿真窗口的位置
# 机械臂各段长度（4段，可调整）
ARM_SEGMENT_LENGTHS = [100, 80, 60, 40]
GRIPPER_SIZE = 20       # 夹爪大小
# 仿真目标物体初始位置（红色小球）
TARGET_OBJ_INIT_POS = (500, 300)
TARGET_OBJ_RADIUS = 15

# ---------------------- 全局变量（线程间通信）----------------------
target_pixel_pos = None  # 视觉识别的目标像素坐标
grasp_trigger = False    # 抓取触发标志
is_camera_available = True  # 摄像头是否可用

# ====================== 1. 视觉识别线程（独立运行，不阻塞仿真）=====================
def vision_recognition_thread():
    """处理真实摄像头的红色目标检测，输出目标像素坐标"""
    global target_pixel_pos, is_camera_available
    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VISION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VISION_HEIGHT)

    # 检查摄像头是否可用
    if not cap.isOpened():
        print(f"\n⚠️ 警告：无法打开摄像头（索引：{CAMERA_INDEX}）")
        print("💡 解决方案：1. 检查摄像头连接 2. 更换CAMERA_INDEX为1 3. 程序将使用模拟目标")
        is_camera_available = False
        # 模拟目标位置（摄像头画面中心）
        while True:
            target_pixel_pos = (VISION_WIDTH // 2, VISION_HEIGHT // 2)
            time.sleep(0.05)
            # 按q退出时终止线程
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return

    print("\n✅ 摄像头初始化成功！")
    print("🎯 正在检测红色目标...")

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ---------------------- 目标检测核心逻辑 ----------------------
        # 1. 转HSV颜色空间（便于颜色筛选）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 2. 生成红色掩码（合并两个区间，解决红色跨0度问题）
        mask1 = cv2.inRange(hsv, TARGET_COLOR_LOWER, TARGET_COLOR_UPPER)
        mask2 = cv2.inRange(hsv, TARGET_COLOR_LOWER2, TARGET_COLOR_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)
        # 3. 形态学操作（去噪声）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)  # 腐蚀
        mask = cv2.dilate(mask, kernel, iterations=1) # 膨胀
        # 4. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化目标位置
        target_pixel_pos = None
        if contours:
            # 取面积最大的轮廓（认为是目标）
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:
                # 计算轮廓中心
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    target_pixel_pos = (cX, cY)
                    # 绘制轮廓和中心标记
                    cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Target ({cX},{cY})", (cX-50, cY-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---------------------- 绘制提示信息 ----------------------
        if target_pixel_pos:
            cv2.putText(frame, "✅ Target Found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "❌ Target Lost", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示画面
        cv2.imshow("🤖 真实摄像头 - 红色目标检测", frame)

        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# ====================== 2. 机械臂仿真类（封装运动和绘制逻辑）=====================
class ArmSimulation:
    def __init__(self, base_pos, segment_lengths):
        """初始化机械臂"""
        self.base_pos = np.array(base_pos, dtype=float)  # 基座位置
        self.segment_lengths = segment_lengths          # 各段长度
        self.num_segments = len(segment_lengths)        # 段数（关节数）
        self.joint_angles = [0.0 for _ in range(self.num_segments)]  # 各关节角度（初始为0）
        self.end_pos = self.calculate_end_position()    # 末端执行器位置
        self.gripper_closed = False                     # 夹爪状态：False=张开，True=闭合

    def calculate_end_position(self):
        """正运动学：根据关节角度计算末端执行器位置"""
        pos = self.base_pos.copy()
        current_angle = 0.0  # 累计角度
        for i in range(self.num_segments):
            current_angle += self.joint_angles[i]
            # 计算当前段的位移（坐标系：y轴向下为正，需调整角度）
            dx = self.segment_lengths[i] * math.cos(current_angle - math.pi/2)
            dy = self.segment_lengths[i] * math.sin(current_angle - math.pi/2)
            pos += np.array([dx, dy])
        return pos

    def move_to_target(self, target_pos, step=0.01):
        """逆运动学简化版：移动末端执行器到目标位置"""
        current_pos = self.calculate_end_position()
        # 距离大于5像素时继续移动（精度阈值）
        while np.linalg.norm(current_pos - target_pos) > 5:
            # 计算方向向量（从当前位置到目标位置）
            dir_vec = target_pos - current_pos
            dir_vec = dir_vec / np.linalg.norm(dir_vec)  # 归一化
            # 调整各关节角度（比例控制，简单有效）
            for i in range(self.num_segments):
                self.joint_angles[i] += dir_vec[0] * step - dir_vec[1] * step
            # 更新末端位置
            current_pos = self.calculate_end_position()
            # 模拟运动速度（暂停10ms）
            pygame.time.wait(10)
        self.end_pos = current_pos

    def close_gripper(self):
        """闭合夹爪"""
        self.gripper_closed = True
        pygame.time.wait(300)  # 模拟夹爪闭合时间

    def open_gripper(self):
        """张开夹爪"""
        self.gripper_closed = False
        pygame.time.wait(300)

    def draw(self, screen):
        """在PyGame窗口中绘制机械臂"""
        # 1. 绘制基座
        pygame.draw.circle(screen, (50, 50, 150), tuple(map(int, self.base_pos)), 15)
        # 2. 绘制机械臂各段和关节
        pos = self.base_pos.copy()
        current_angle = 0.0
        for i in range(self.num_segments):
            current_angle += self.joint_angles[i]
            # 计算当前段的终点位置
            dx = self.segment_lengths[i] * math.cos(current_angle - math.pi/2)
            dy = self.segment_lengths[i] * math.sin(current_angle - math.pi/2)
            new_pos = pos + np.array([dx, dy])
            # 绘制段（线条）
            pygame.draw.line(screen, (150, 150, 150), tuple(map(int, pos)), tuple(map(int, new_pos)), 8)
            # 绘制关节（小圆）
            pygame.draw.circle(screen, (100, 100, 100), tuple(map(int, new_pos)), 8)
            pos = new_pos
        # 3. 绘制末端执行器（夹爪）
        if self.gripper_closed:
            # 闭合状态：三角形
            pygame.draw.polygon(screen, (200, 50, 50), [
                (pos[0]-GRIPPER_SIZE//2, pos[1]),
                (pos[0], pos[1]-GRIPPER_SIZE//2),
                (pos[0]+GRIPPER_SIZE//2, pos[1])
            ])
        else:
            # 张开状态：两条竖线+一条横线
            pygame.draw.line(screen, (200, 50, 50), (pos[0]-GRIPPER_SIZE, pos[1]), (pos[0]+GRIPPER_SIZE, pos[1]), 4)
            pygame.draw.line(screen, (200, 50, 50), (pos[0]-GRIPPER_SIZE//2, pos[1]-GRIPPER_SIZE//2),
                             (pos[0]-GRIPPER_SIZE//2, pos[1]+GRIPPER_SIZE//2), 4)
            pygame.draw.line(screen, (200, 50, 50), (pos[0]+GRIPPER_SIZE//2, pos[1]-GRIPPER_SIZE//2),
                             (pos[0]+GRIPPER_SIZE//2, pos[1]+GRIPPER_SIZE//2), 4)

# ====================== 3. 主程序（仿真窗口+抓取逻辑）=====================
def main():
    global grasp_trigger, target_pixel_pos
    # 1. 启动视觉识别线程（守护线程，主程序退出时自动终止）
    vision_thread = threading.Thread(target=vision_recognition_thread, daemon=True)
    vision_thread.start()
    time.sleep(1)  # 等待线程初始化完成

    # 2. 初始化PyGame
    pygame.init()
    screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
    pygame.display.set_caption("🤖 机械臂抓取仿真（开箱即用）")
    clock = pygame.time.Clock()

    # 3. 创建机械臂和仿真目标物体
    arm = ArmSimulation(ARM_BASE_POS, ARM_SEGMENT_LENGTHS)
    target_obj_pos = np.array(TARGET_OBJ_INIT_POS, dtype=float)
    target_obj_color = (255, 0, 0)  # 初始红色：未抓取
    target_obj_grabbed = False      # 目标是否被抓取

    # 4. 打印操作说明
    print("\n" + "="*50)
    print("操作说明：")
    print("  🎮 按【空格键】：执行抓取操作")
    print("  🎮 按【q键】：退出程序")
    print("="*50 + "\n")

    # 主循环
    running = True
    while running:
        # 填充背景色
        screen.fill((240, 240, 240))

        # ---------------------- 处理PyGame事件 ----------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    grasp_trigger = True  # 触发抓取

        # ---------------------- 绘制仿真元素 ----------------------
        # 1. 绘制仿真目标物体
        pygame.draw.circle(screen, target_obj_color, tuple(map(int, target_obj_pos)), TARGET_OBJ_RADIUS)
        # 2. 绘制机械臂
        arm.draw(screen)

        # ---------------------- 抓取逻辑处理 ----------------------
        if grasp_trigger:
            grasp_trigger = False
            if target_pixel_pos is not None:
                print("\n📢 开始执行抓取流程...")
                # 步骤1：将视觉像素坐标映射到仿真窗口坐标
                # 映射公式：将摄像头的像素坐标（0~640, 0~480）转换为仿真窗口的机械臂运动范围
                sim_x = ARM_BASE_POS[0] + (target_pixel_pos[0] - VISION_WIDTH//2) * 0.5
                sim_y = ARM_BASE_POS[1] - (target_pixel_pos[1] - VISION_HEIGHT//2) * 0.5
                target_sim_pos = np.array([sim_x, sim_y], dtype=float)
                print(f"🔍 视觉像素坐标：{target_pixel_pos} → 仿真坐标：({int(sim_x)}, {int(sim_y)})")

                # 步骤2：移动机械臂到目标位置
                print("🤖 机械臂正在移动到目标位置...")
                arm.move_to_target(target_sim_pos)

                # 步骤3：闭合夹爪
                print("🤖 夹爪闭合，抓取目标...")
                arm.close_gripper()

                # 步骤4：更新目标物体状态（模拟被抓取）
                target_obj_pos = arm.end_pos + np.array([0, -20])  # 目标随夹爪移动
                target_obj_color = (0, 255, 0)  # 绿色：已抓取
                target_obj_grabbed = True

                print("✅ 抓取流程完成！")
            else:
                print("❌ 未检测到目标，无法执行抓取！")

        # ---------------------- 实时更新目标物体位置（若被抓取）----------------------
        if target_obj_grabbed:
            target_obj_pos = arm.end_pos + np.array([0, -20])

        # 更新屏幕显示
        pygame.display.flip()
        # 控制帧率（60帧/秒）
        clock.tick(60)

    # 退出程序
    pygame.quit()
    cv2.destroyAllWindows()
    print("\n👋 程序已正常退出！")

# ====================== 程序入口 ======================
if __name__ == "__main__":
    main()