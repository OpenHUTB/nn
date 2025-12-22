"""
自动巡航小车 - 带前方障碍物检测与智能路径规划
- 巡航速度提高至0.003 m/s
- 前方0.5米内有障碍物时停止并规划新路径
- 避免摆头循环，寻找可行方向
- R 键可复位
"""
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
import math
import random
import time

# ------------------- 键盘监听 -------------------
KEYS = {keyboard.KeyCode.from_char('r'): False}

def on_press(k):
    if k in KEYS: KEYS[k] = True

def on_release(k):
    if k in KEYS: KEYS[k] = False

keyboard.Listener(on_press=on_press, on_release=on_release).start()

# ------------------- 加载模型 -------------------
model = mujoco.MjModel.from_xml_path("wheeled_car.xml")
data = mujoco.MjData(model)

# ------------------- 参数设置 -------------------
CRUISE_SPEED = 0.003
TURN_SPEED = CRUISE_SPEED * 0.4
OBSTACLE_DISTANCE_THRESHOLD = 0.7  # 增加检测距离
SAFE_DISTANCE = 0.3
TURN_ANGLE = 0.3
TURN_DURATION = 60
SCAN_RANGE = 1.0  # 扩大扫描范围

# 路径规划参数
MAX_TURN_ATTEMPTS = 3  # 最大转向尝试次数
PATH_SCAN_ANGLES = [-60, -30, 0, 30, 60]  # 扫描角度（度）
PATH_SCAN_DISTANCE = 1.5  # 路径扫描距离

# 障碍物名称列表
OBSTACLE_NAMES = [
    'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
    'obs_ball1', 'obs_ball2', 'obs_ball3',
    'wall1', 'wall2', 'front_dark_box'
]

# 提前获取所有障碍物的body id
obstacle_ids = {}
for obs_name in OBSTACLE_NAMES:
    obs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obs_name)
    if obs_id != -1:
        obstacle_ids[obs_name] = obs_id

# 小车底盘body id
chassis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

# ------------------- 复位函数 -------------------
def reset_car():
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.03
    print("\n>>> 已复位 <<<")

# ------------------- 障碍物检测函数 -------------------
def check_front_obstacle(direction_angle=0):
    """检测指定方向是否有障碍物"""

    # 获取小车当前位置
    chassis_pos = data.body(chassis_body_id).xpos

    # 获取小车当前速度方向
    velocity = data.qvel[:2]
    if np.linalg.norm(velocity) < 0.0001:
        forward_direction = np.array([1.0, 0.0])
    else:
        forward_direction = velocity / np.linalg.norm(velocity)

    # 旋转前向方向到指定角度
    if direction_angle != 0:
        cos_a = math.cos(direction_angle)
        sin_a = math.sin(direction_angle)
        rotated_direction = np.array([
            forward_direction[0] * cos_a - forward_direction[1] * sin_a,
            forward_direction[0] * sin_a + forward_direction[1] * cos_a
        ])
        forward_direction = rotated_direction

    # 扫描所有障碍物
    min_distance = float('inf')
    closest_obstacle = None

    for obs_name, obs_id in obstacle_ids.items():
        obs_pos = data.body(obs_id).xpos

        dx = obs_pos[0] - chassis_pos[0]
        dy = obs_pos[1] - chassis_pos[1]

        relative_pos = np.array([dx, dy])
        distance = np.linalg.norm(relative_pos)

        if distance > 0 and distance < SCAN_RANGE:
            obstacle_direction = relative_pos / distance

            dot_product = np.dot(obstacle_direction, forward_direction)
            dot_product = max(-1.0, min(1.0, dot_product))
            angle_diff = math.acos(dot_product)

            if angle_diff < math.radians(45) and dx > -0.2:
                cross_product = np.cross([forward_direction[0], forward_direction[1], 0],
                                         [obstacle_direction[0], obstacle_direction[1], 0])
                lateral_distance = abs(cross_product[2]) * distance

                if angle_diff < math.radians(25) and lateral_distance < 0.5:
                    if distance < min_distance:
                        min_distance = distance
                        closest_obstacle = obs_name

    if closest_obstacle is not None:
        if min_distance < SAFE_DISTANCE:
            return 2, min_distance, closest_obstacle
        else:
            return 1, min_distance, closest_obstacle

    return 0, 0, None

# ------------------- 路径扫描函数 -------------------
def scan_path_directions():
    """扫描多个方向的路径可行性"""

    path_scores = []

    for angle_deg in PATH_SCAN_ANGLES:
        angle_rad = math.radians(angle_deg)

        # 检查该方向是否有障碍物
        obstacle_status, obstacle_distance, _ = check_front_obstacle(angle_rad)

        # 计算分数：距离越远分数越高
        if obstacle_status == 0:
            # 无障碍物，给最高分
            score = PATH_SCAN_DISTANCE
        elif obstacle_status == 1:
            # 有障碍物但距离较远
            score = obstacle_distance
        else:
            # 有近距离障碍物
            score = 0

        path_scores.append((angle_rad, score, obstacle_status))

    # 按分数排序
    path_scores.sort(key=lambda x: x[1], reverse=True)

    return path_scores

# ------------------- 选择最佳路径 -------------------
def choose_best_path():
    """选择最佳路径方向"""

    # 扫描所有方向
    path_scores = scan_path_directions()

    # 选择最佳方向
    best_angle, best_score, best_status = path_scores[0]

    # 如果所有方向都有障碍物，选择障碍物最远的方向
    if best_score == 0:
        # 后退方案
        print("所有方向都有障碍物，尝试后退")
        return None, "BACKUP"

    # 如果最佳方向不是正前方，需要转向
    if abs(best_angle) > math.radians(10):
        if best_angle > 0:
            return best_angle, f"左转{abs(math.degrees(best_angle)):.0f}度"
        else:
            return best_angle, f"右转{abs(math.degrees(best_angle)):.0f}度"
    else:
        return 0, "直行"

# ------------------- 后退操作 -------------------
def perform_backup(backup_counter):
    """执行后退操作"""

    if backup_counter < 30:
        # 后退（负速度）
        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0
        data.ctrl[2] = -TURN_SPEED * 0.5
        data.ctrl[3] = -TURN_SPEED * 0.5
        data.ctrl[4] = -TURN_SPEED * 0.5
        data.ctrl[5] = -TURN_SPEED * 0.5
        return False, backup_counter + 1
    else:
        # 后退完成
        for i in range(len(data.ctrl)):
            data.ctrl[i] = 0.0
        return True, 0

# ------------------- 主循环 -------------------
mujoco.mj_resetData(model, data)

# 状态变量
CAR_STATE = "CRUISING"  # CRUISING, DECELERATING, STOPPED, PATH_PLANNING, TURNING, SCANNING, RESUME, BACKING_UP
turn_counter = 0
turn_angle = 0
turn_direction = ""
scan_counter = 0
deceleration_counter = 0
current_speed = CRUISE_SPEED
turn_attempts = 0  # 转向尝试次数
backup_counter = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -25

    while viewer.is_running():
        # 检查R键复位
        if KEYS.get(keyboard.KeyCode.from_char('r'), False):
            reset_car()
            CAR_STATE = "CRUISING"
            turn_counter = 0
            scan_counter = 0
            deceleration_counter = 0
            current_speed = CRUISE_SPEED
            turn_attempts = 0
            backup_counter = 0
            KEYS[keyboard.KeyCode.from_char('r')] = False

        # 根据当前状态执行不同操作
        if CAR_STATE == "CRUISING":
            # 巡航状态
            obstacle_status, obstacle_distance, obstacle_name = check_front_obstacle()

            if obstacle_status == 2:
                # 紧急停止
                CAR_STATE = "STOPPED"
                print(f"\n⚠️ 紧急停止！障碍物距离: {obstacle_distance:.2f}m")
                turn_counter = 0
                current_speed = 0

                for i in range(len(data.ctrl)):
                    data.ctrl[i] = 0.0

            elif obstacle_status == 1:
                # 检测到障碍物，减速
                CAR_STATE = "DECELERATING"
                deceleration_counter = 0
                print(f"\n⚠️ 检测到障碍物: {obstacle_name}({obstacle_distance:.2f}m)，开始减速...")
            else:
                # 无障碍物，高速巡航
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
                data.ctrl[2] = CRUISE_SPEED
                data.ctrl[3] = CRUISE_SPEED
                data.ctrl[4] = CRUISE_SPEED
                data.ctrl[5] = CRUISE_SPEED
                current_speed = CRUISE_SPEED

        elif CAR_STATE == "DECELERATING":
            # 减速状态
            deceleration_counter += 1

            decel_progress = min(1.0, deceleration_counter / 15.0)
            current_speed = CRUISE_SPEED * (1.0 - decel_progress)

            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
            data.ctrl[2] = current_speed
            data.ctrl[3] = current_speed
            data.ctrl[4] = current_speed
            data.ctrl[5] = current_speed

            if deceleration_counter > 20:
                CAR_STATE = "STOPPED"
                print("减速完成，准备规划路径")
                turn_counter = 0

        elif CAR_STATE == "STOPPED":
            # 停止状态，准备路径规划
            turn_counter += 1

            current_speed = 0
            for i in range(len(data.ctrl)):
                data.ctrl[i] = 0.0

            if turn_counter > 10:
                print("正在规划路径...")
                CAR_STATE = "PATH_PLANNING"
                turn_counter = 0

        elif CAR_STATE == "PATH_PLANNING":
            # 路径规划状态
            print("扫描可行路径...")

            # 选择最佳路径
            best_angle, path_direction = choose_best_path()

            if path_direction == "BACKUP":
                # 需要后退
                print("所有方向都有障碍物，执行后退")
                CAR_STATE = "BACKING_UP"
                backup_counter = 0
            else:
                # 找到可行路径
                turn_angle = best_angle
                turn_direction = path_direction

                # 检查是否需要大角度转向
                if abs(turn_angle) > math.radians(45):
                    turn_attempts += 1
                    if turn_attempts >= MAX_TURN_ATTEMPTS:
                        print(f"多次尝试后选择大角度转向: {turn_direction}")

                print(f"选择路径: {turn_direction}")
                CAR_STATE = "TURNING"
                turn_counter = 0

        elif CAR_STATE == "BACKING_UP":
            # 后退状态
            backup_complete, backup_counter = perform_backup(backup_counter)

            if backup_complete:
                print("后退完成，重新规划路径")
                CAR_STATE = "PATH_PLANNING"

        elif CAR_STATE == "TURNING":
            # 转向状态
            turn_counter += 1

            # 渐进式转向
            progress = min(1.0, turn_counter / 10.0)
            current_angle = turn_angle * progress

            data.ctrl[0] = current_angle
            data.ctrl[1] = current_angle

            if turn_counter > 5:
                speed_progress = min(1.0, (turn_counter - 5) / 15.0)
                current_speed = TURN_SPEED * speed_progress
                data.ctrl[2] = current_speed
                data.ctrl[3] = current_speed
                data.ctrl[4] = current_speed
                data.ctrl[5] = current_speed

            if turn_counter % 15 == 0:
                print(f"正在{turn_direction}，角度: {math.degrees(current_angle):.1f}度")

            if turn_counter > TURN_DURATION:
                print(f"{turn_direction}完成，开始扫描前方...")
                CAR_STATE = "SCANNING"
                turn_counter = 0
                scan_counter = 0

        elif CAR_STATE == "SCANNING":
            # 扫描状态
            scan_counter += 1

            data.ctrl[0] = turn_angle * 0.5
            data.ctrl[1] = turn_angle * 0.5
            data.ctrl[2] = TURN_SPEED * 0.6
            data.ctrl[3] = TURN_SPEED * 0.6
            data.ctrl[4] = TURN_SPEED * 0.6
            data.ctrl[5] = TURN_SPEED * 0.6
            current_speed = TURN_SPEED * 0.6

            if scan_counter % 10 == 0:
                obstacle_status, obstacle_distance, obstacle_name = check_front_obstacle()

                if obstacle_status == 0:
                    print("前方安全，准备恢复巡航")
                    CAR_STATE = "RESUME"
                    turn_counter = 0
                    turn_attempts = 0  # 重置尝试次数
                else:
                    print(f"转向后仍检测到障碍物: {obstacle_name}({obstacle_distance:.2f}m)，重新规划")
                    CAR_STATE = "STOPPED"
                    turn_counter = 0

            if scan_counter > 50:
                print("扫描超时，尝试恢复巡航")
                CAR_STATE = "RESUME"
                turn_counter = 0

        elif CAR_STATE == "RESUME":
            # 恢复巡航状态
            turn_counter += 1

            progress = min(1.0, turn_counter / 15.0)

            current_angle = turn_angle * (1.0 - progress)
            data.ctrl[0] = current_angle
            data.ctrl[1] = current_angle

            current_speed = TURN_SPEED + (CRUISE_SPEED - TURN_SPEED) * progress
            data.ctrl[2] = current_speed
            data.ctrl[3] = current_speed
            data.ctrl[4] = current_speed
            data.ctrl[5] = current_speed

            if turn_counter > 20:
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
                data.ctrl[2] = CRUISE_SPEED
                data.ctrl[3] = CRUISE_SPEED
                data.ctrl[4] = CRUISE_SPEED
                data.ctrl[5] = CRUISE_SPEED
                current_speed = CRUISE_SPEED

                obstacle_status, obstacle_distance, _ = check_front_obstacle()
                if obstacle_status == 0:
                    print("恢复巡航成功")
                    CAR_STATE = "CRUISING"
                    turn_counter = 0
                else:
                    print("恢复巡航时检测到障碍物，重新处理")
                    CAR_STATE = "STOPPED"
                    turn_counter = 0

        # 仿真步骤
        mujoco.mj_step(model, data)

        # 显示信息
        vel = np.linalg.norm(data.qvel[:3])
        current_steer = (data.ctrl[0] + data.ctrl[1]) / 2

        status_info = f"状态: {CAR_STATE}, 速度: {vel:7.5f} m/s"
        if abs(current_steer) > 0.01:
            status_info += f", 转向: {math.degrees(current_steer):.1f}度"

        status_info += f", 尝试次数: {turn_attempts}"

        if CAR_STATE == "CRUISING":
            obstacle_status, obstacle_distance, obstacle_name = check_front_obstacle()
            if obstacle_status > 0 and obstacle_name:
                status_info += f", 前方障碍: {obstacle_name}({obstacle_distance:.2f}m)"

        print(f"\r{status_info}", end='', flush=True)

        # 同步视图
        viewer.sync()

print("\n程序结束")