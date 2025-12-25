"""
自动巡航小车 - 带前方障碍物检测与避障
- 提高速度至0.00075 m/s（提高50%）
- 前方0.5米内有障碍物时停止
- 持续扫描前方障碍物
- R 键可复位
"""
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
import math

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
# 提高50%的速度：0.0005 * 1.5 = 0.00075
CRUISE_SPEED = 0.00075  # 提高50%的巡航速度
OBSTACLE_DISTANCE_THRESHOLD = 0.5  # 障碍物检测距离改为0.5米
SAFE_DISTANCE = 0.2  # 安全距离，小于此距离时紧急停止（也相应减小）

# 障碍物名称列表（根据模型文件中的障碍物定义）
OBSTACLE_NAMES = [
    'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
    'obs_ball1', 'obs_ball2', 'obs_ball3',
    'wall1', 'wall2', 'front_dark_box'
]

# 提前获取所有障碍物的body id，避免每次循环都查找
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

# ------------------- 障碍物检测函数（简化版） -------------------
def check_front_obstacle():
    """检测小车前方是否有障碍物（简化版本）"""

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
            # 假设小车前向是x轴正方向
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

# ------------------- 主循环 -------------------
mujoco.mj_resetData(model, data)
auto_stop = False  # 自动停止标志
stop_counter = 0  # 停止计数器（用于避免频繁启停）

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -25

    while viewer.is_running():
        # 检查R键复位
        if KEYS.get(keyboard.KeyCode.from_char('r'), False):
            reset_car()
            auto_stop = False
            stop_counter = 0
            KEYS[keyboard.KeyCode.from_char('r')] = False

        # 检测前方障碍物
        obstacle_status, obstacle_distance, obstacle_name = check_front_obstacle()

        # 控制逻辑
        if auto_stop:
            # 当前处于自动停止状态
            if obstacle_status == 0:
                # 前方无障碍物，恢复行驶
                auto_stop = False
                stop_counter = 0
            else:
                # 前方仍有障碍物，保持停止
                stop_counter += 1
                # 所有驱动器设为0
                for i in range(len(data.ctrl)):
                    data.ctrl[i] = 0.0
        else:
            # 正常行驶状态
            if obstacle_status == 2:
                # 紧急停止（距离过近）
                auto_stop = True
                for i in range(len(data.ctrl)):
                    data.ctrl[i] = 0.0
                stop_counter = 20  # 由于速度提高，减少停止时间
            elif obstacle_status == 1:
                # 检测到前方障碍物，准备停止
                # 由于速度提高，提前触发停止（减少延迟帧数）
                if stop_counter > 5:  # 减少延迟帧数
                    auto_stop = True
                    for i in range(len(data.ctrl)):
                        data.ctrl[i] = 0.0
                else:
                    stop_counter += 1
                    # 正常巡航
                    data.ctrl[0] = 0.0  # 前左转向
                    data.ctrl[1] = 0.0  # 前右转向
                    data.ctrl[2] = CRUISE_SPEED  # 前左驱动
                    data.ctrl[3] = CRUISE_SPEED  # 前右驱动
                    data.ctrl[4] = CRUISE_SPEED  # 后左驱动
                    data.ctrl[5] = CRUISE_SPEED  # 后右驱动
            else:
                # 无障碍物，正常巡航
                stop_counter = 0
                data.ctrl[0] = 0.0  # 前左转向
                data.ctrl[1] = 0.0  # 前右转向
                data.ctrl[2] = CRUISE_SPEED  # 前左驱动
                data.ctrl[3] = CRUISE_SPEED  # 前右驱动
                data.ctrl[4] = CRUISE_SPEED  # 后左驱动
                data.ctrl[5] = CRUISE_SPEED  # 后右驱动

        # 仿真步骤
        mujoco.mj_step(model, data)

        # 显示信息
        vel = np.linalg.norm(data.qvel[:3])
        status_text = "前进"
        if auto_stop:
            status_text = "停止"
        elif obstacle_status > 0:
            status_text = "减速"

        obstacle_info = ""
        if obstacle_status > 0 and obstacle_name:
            obstacle_info = f", 障碍物: {obstacle_name}({obstacle_distance:.2f}m)"

        print(f"\r状态: {status_text}, 速度: {vel:7.5f} m/s{obstacle_info}", end='', flush=True)

        # 同步视图
        viewer.sync()

print()