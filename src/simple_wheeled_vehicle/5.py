"""
自动巡航小车 - 带前方障碍物检测与90度转向避障
- 提高速度至0.0015 m/s（提高两倍）
- 前方0.5米内有障碍物时停止并进行90度转向
- 转向后恢复巡航速度
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
# 提高两倍的速度：0.0005 * 2 = 0.0015
CRUISE_SPEED = 0.0015  # 提高两倍的巡航速度
TURN_SPEED = CRUISE_SPEED * 0.5  # 转向时的速度（更低以确保安全）
OBSTACLE_DISTANCE_THRESHOLD = 0.5  # 障碍物检测距离
SAFE_DISTANCE = 0.2  # 安全距离
TURN_ANGLE = math.pi / 2  # 90度转向角度（π/2弧度）
TURN_DURATION = 150  # 90度转向需要更长时间

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
    data.qpos[2] = 0.03  # 设置初始高度
    print("\n>>> 已复位 <<<")

# ------------------- 障碍物检测函数 -------------------
def check_front_obstacle():
    """检测小车前方是否有障碍物"""

    # 获取小车当前位置
    chassis_pos = data.body(chassis_body_id).xpos  # 位置 (x, y, z)

    # 扫描所有障碍物
    min_distance = float('inf')
    closest_obstacle = None

    for obs_name, obs_id in obstacle_ids.items():
        # 获取障碍物位置
        obs_pos = data.body(obs_id).xpos

        # 计算相对于小车的位置
        dx = obs_pos[0] - chassis_pos[0]  # x方向距离
        dy = obs_pos[1] - chassis_pos[1]  # y方向距离

        # 计算水平距离
        distance = math.sqrt(dx**2 + dy**2)

        # 只考虑小车前方的障碍物（dx > 0 表示在小车前方）
        if dx > 0 and distance < OBSTACLE_DISTANCE_THRESHOLD:
            # 计算障碍物相对于小车前向方向的横向偏移
            lateral_distance = abs(dy)

            # 如果横向距离小于小车的宽度（约0.3米），则认为在正前方
            if lateral_distance < 0.3:
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = obs_name

    # 返回检测结果
    if closest_obstacle is not None:
        if min_distance < SAFE_DISTANCE:
            return 2, min_distance, closest_obstacle  # 紧急停止
        else:
            return 1, min_distance, closest_obstacle  # 检测到障碍物

    return 0, 0, None  # 无障碍物

# ------------------- 随机选择90度转向方向 -------------------
def choose_90_degree_turn():
    """随机选择向左或向右90度转向"""
    if random.random() > 0.5:
        return TURN_ANGLE  # 右转90度
    else:
        return -TURN_ANGLE  # 左转90度

# ------------------- 主循环 -------------------
mujoco.mj_resetData(model, data)

# 状态变量
CAR_STATE = "CRUISING"  # 状态: CRUISING, STOPPED, TURNING, RESUME
turn_counter = 0
turn_angle = 0
turn_direction = ""  # 记录转向方向

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -25

    while viewer.is_running():
        # 检查R键复位
        if KEYS.get(keyboard.KeyCode.from_char('r'), False):
            reset_car()
            CAR_STATE = "CRUISING"
            turn_counter = 0
            KEYS[keyboard.KeyCode.from_char('r')] = False

        # 根据当前状态执行不同操作
        if CAR_STATE == "CRUISING":
            # 巡航状态：检测前方障碍物
            obstacle_status, obstacle_distance, obstacle_name = check_front_obstacle()

            if obstacle_status == 2:
                # 紧急停止（距离过近）
                CAR_STATE = "STOPPED"
                print(f"\n紧急停止！障碍物距离: {obstacle_distance:.2f}m")
                turn_counter = 0

                # 停止所有驱动
                for i in range(len(data.ctrl)):
                    data.ctrl[i] = 0.0

            elif obstacle_status == 1:
                # 检测到前方障碍物，减速准备停止
                CAR_STATE = "STOPPED"
                print(f"\n检测到障碍物: {obstacle_name}({obstacle_distance:.2f}m)，正在停止...")
                turn_counter = 0

                # 减速停止
                for i in range(len(data.ctrl)):
                    data.ctrl[i] = 0.0
            else:
                # 无障碍物，正常巡航（两倍速度）
                data.ctrl[0] = 0.0  # 前左转向
                data.ctrl[1] = 0.0  # 前右转向
                data.ctrl[2] = CRUISE_SPEED  # 前左驱动
                data.ctrl[3] = CRUISE_SPEED  # 前右驱动
                data.ctrl[4] = CRUISE_SPEED  # 后左驱动
                data.ctrl[5] = CRUISE_SPEED  # 后右驱动

        elif CAR_STATE == "STOPPED":
            # 已停止状态：选择转向方向
            turn_counter += 1

            # 停止所有驱动
            for i in range(len(data.ctrl)):
                data.ctrl[i] = 0.0

            # 等待几帧确保完全停止
            if turn_counter > 10:
                print("选择转向方向...")

                # 随机选择向左或向右90度转向
                turn_angle = choose_90_degree_turn()

                # 记录转向方向
                if turn_angle > 0:
                    turn_direction = "右转90度"
                else:
                    turn_direction = "左转90度"

                print(f"将执行{turn_direction}")

                # 进入转向状态
                CAR_STATE = "TURNING"
                turn_counter = 0

        elif CAR_STATE == "TURNING":
            # 转向状态：执行90度转向
            turn_counter += 1

            # 设置90度转向角度
            data.ctrl[0] = turn_angle  # 前左转向
            data.ctrl[1] = turn_angle  # 前右转向

            # 低速前进，帮助转向
            data.ctrl[2] = TURN_SPEED  # 前左驱动
            data.ctrl[3] = TURN_SPEED  # 前右驱动
            data.ctrl[4] = TURN_SPEED  # 后左驱动
            data.ctrl[5] = TURN_SPEED  # 后右驱动

            # 显示转向信息
            if turn_counter % 30 == 0:
                print(f"正在执行{turn_direction}，进度: {turn_counter}/{TURN_DURATION}")

            # 检查转向是否完成
            if turn_counter > TURN_DURATION:
                print(f"{turn_direction}完成，准备恢复巡航")
                CAR_STATE = "RESUME"
                turn_counter = 0

        elif CAR_STATE == "RESUME":
            # 恢复状态：转向后恢复直行并恢复巡航速度
            turn_counter += 1

            # 逐渐回正转向
            if turn_counter < 20:
                # 逐渐减小转向角度
                decay_factor = 1.0 - (turn_counter / 20.0)
                current_angle = turn_angle * decay_factor
                data.ctrl[0] = current_angle
                data.ctrl[1] = current_angle

                # 逐渐加速到巡航速度
                speed_factor = turn_counter / 20.0
                current_speed = TURN_SPEED + (CRUISE_SPEED - TURN_SPEED) * speed_factor
                data.ctrl[2] = current_speed
                data.ctrl[3] = current_speed
                data.ctrl[4] = current_speed
                data.ctrl[5] = current_speed
            else:
                # 转向已回正，恢复两倍巡航速度
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
                data.ctrl[2] = CRUISE_SPEED
                data.ctrl[3] = CRUISE_SPEED
                data.ctrl[4] = CRUISE_SPEED
                data.ctrl[5] = CRUISE_SPEED

                # 检查前方是否安全
                obstacle_status, obstacle_distance, _ = check_front_obstacle()
                if obstacle_status == 0:
                    print("前方安全，恢复巡航")
                    CAR_STATE = "CRUISING"
                    turn_counter = 0
                else:
                    # 前方仍有障碍物，重新停止
                    print("转向后前方仍有障碍物，重新处理")
                    CAR_STATE = "STOPPED"
                    turn_counter = 0

        # 仿真步骤
        mujoco.mj_step(model, data)

        # 显示信息
        vel = np.linalg.norm(data.qvel[:3])

        # 获取当前转向角度
        current_steer = (data.ctrl[0] + data.ctrl[1]) / 2

        # 显示状态信息
        status_info = f"状态: {CAR_STATE}, 速度: {vel:7.5f} m/s"
        if abs(current_steer) > 0.01:
            status_info += f", 转向: {math.degrees(current_steer):.1f}度"

        # 如果正在巡航，显示障碍物检测信息
        if CAR_STATE == "CRUISING":
            obstacle_status, obstacle_distance, obstacle_name = check_front_obstacle()
            if obstacle_status > 0 and obstacle_name:
                status_info += f", 前方障碍: {obstacle_name}({obstacle_distance:.2f}m)"

        print(f"\r{status_info}", end='', flush=True)

        # 同步视图
        viewer.sync()

print("\n程序结束")