"""
自动巡航小车 - 智能绕障与路径记忆系统
- 巡航速度：0.003 m/s
- 前方障碍物检测与智能路径规划
- 路径记忆与学习功能
- 绕障决策优化
- R 键复位
- 空格键强制截停 / 解除锁定
"""
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
import math
import random
import time
import json
from collections import deque

# ------------------- 键盘监听 -------------------
KEYS = {
    keyboard.KeyCode.from_char('r'): False,
    keyboard.KeyCode.from_char('d'): False,  # 调试模式开关
    keyboard.KeyCode.from_char('s'): False,  # 保存路径记录
    keyboard.Key.space: False,               # 强制截停
}

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
OBSTACLE_DISTANCE_THRESHOLD = 0.7
SAFE_DISTANCE = 0.3
TURN_ANGLE = 0.3
TURN_DURATION = 50
SCAN_RANGE = 1.0

# 新增：智能绕障参数
PATH_MEMORY_SIZE = 50  # 路径记忆大小
EXPLORATION_RATE = 0.3  # 探索率（尝试新路径的概率）
LEARNING_RATE = 0.1  # 学习率
PATH_REWARD = 1.0  # 成功路径奖励
PATH_PENALTY = -0.5  # 失败路径惩罚

# 新增：路径评分参数
DIRECTION_SCORES = {
    "forward": 1.0,  # 直行优先
    "slight_left": 0.9,
    "slight_right": 0.9,
    "left": 0.8,
    "right": 0.8,
    "sharp_left": 0.6,
    "sharp_right": 0.6,
    "backward": 0.3,  # 后退最后考虑
}

# 新增：方向定义
DIRECTIONS = {
    "forward": 0,
    "slight_left": math.radians(15),
    "slight_right": math.radians(-15),
    "left": math.radians(30),
    "right": math.radians(-30),
    "sharp_left": math.radians(60),
    "sharp_right": math.radians(-60),
    "backward": math.radians(180),
}

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

# ------------------- 路径记忆类 -------------------
class PathMemory:
    """路径记忆与学习系统"""

    def __init__(self, memory_size=50):
        self.memory = deque(maxlen=memory_size)
        self.path_scores = {}  # 路径评分字典
        self.obstacle_history = {}  # 障碍物历史
        self.successful_paths = []  # 成功路径记录
        self.debug_mode = False

    def add_experience(self, position, direction, success, distance_traveled):
        """添加路径经验"""
        key = self._create_key(position, direction)

        # 更新路径评分
        if key in self.path_scores:
            if success:
                self.path_scores[key] += PATH_REWARD * LEARNING_RATE
            else:
                self.path_scores[key] += PATH_PENALTY * LEARNING_RATE
        else:
            if success:
                self.path_scores[key] = PATH_REWARD
            else:
                self.path_scores[key] = PATH_PENALTY

        # 记录经验
        experience = {
            'position': tuple(position[:2]),  # 只记录x,y坐标
            'direction': direction,
            'success': success,
            'distance': distance_traveled,
            'timestamp': time.time()
        }
        self.memory.append(experience)

        if self.debug_mode:
            print(f"路径经验: {direction}, 成功: {success}, 评分: {self.path_scores.get(key, 0):.2f}")

    def get_best_direction(self, position, available_directions):
        """获取最佳方向（基于历史经验）"""
        if random.random() < EXPLORATION_RATE:
            # 探索：随机选择一个方向
            return random.choice(available_directions)

        # 利用：选择评分最高的方向
        best_direction = None
        best_score = -float('inf')

        for direction in available_directions:
            key = self._create_key(position, direction)
            base_score = DIRECTION_SCORES.get(direction, 0.5)
            memory_score = self.path_scores.get(key, 0)
            total_score = base_score + memory_score

            if total_score > best_score:
                best_score = total_score
                best_direction = direction

        return best_direction or random.choice(available_directions)

    def record_obstacle(self, obstacle_name, position):
        """记录障碍物位置"""
        key = f"{obstacle_name}_{int(position[0]*10)}_{int(position[1]*10)}"
        self.obstacle_history[key] = {
            'name': obstacle_name,
            'position': tuple(position[:2]),
            'timestamp': time.time(),
            'count': self.obstacle_history.get(key, {}).get('count', 0) + 1
        }

    def is_recent_obstacle(self, position, threshold=0.5):
        """检查位置附近是否有近期遇到的障碍物"""
        for key, data in self.obstacle_history.items():
            obs_pos = data['position']
            distance = math.sqrt((obs_pos[0] - position[0])**2 + (obs_pos[1] - position[1])**2)
            if distance < threshold and (time.time() - data['timestamp']) < 10:
                return True
        return False

    def record_successful_path(self, start_pos, end_pos, directions):
        """记录成功路径"""
        path = {
            'start': tuple(start_pos[:2]),
            'end': tuple(end_pos[:2]),
            'directions': directions[:],
            'length': len(directions),
            'timestamp': time.time()
        }
        self.successful_paths.append(path)

    def save_to_file(self, filename="path_memory.json"):
        """保存路径记忆到文件"""
        data = {
            'path_scores': self.path_scores,
            'obstacle_history': self.obstacle_history,
            'successful_paths': self.successful_paths[-10:],  # 只保存最近10条
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"路径记忆已保存到 {filename}")

    def load_from_file(self, filename="path_memory.json"):
        """从文件加载路径记忆"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.path_scores = data.get('path_scores', {})
            self.obstacle_history = data.get('obstacle_history', {})
            self.successful_paths = data.get('successful_paths', [])
            print(f"已从 {filename} 加载路径记忆")
        except FileNotFoundError:
            print(f"未找到记忆文件 {filename}，从头开始学习")

    def _create_key(self, position, direction):
        """创建记忆键"""
        x, y = int(position[0]*10), int(position[1]*10)
        return f"{x}_{y}_{direction}"

    def toggle_debug(self):
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
        print(f"调试模式: {'开启' if self.debug_mode else '关闭'}")

# 初始化路径记忆系统
path_memory = PathMemory(PATH_MEMORY_SIZE)
path_memory.load_from_file()

# ------------------- 复位函数 -------------------
def reset_car():
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.03
    print("\n>>> 已复位 <<<")

    # 重置状态变量
    global CAR_STATE, turn_counter, scan_counter, deceleration_counter
    global current_speed, turn_angle, turn_direction, path_history
    global last_success_pos, distance_since_last_obstacle, LOCKED_BY_SPACE

    CAR_STATE = "CRUISING"
    turn_counter = 0
    scan_counter = 0
    deceleration_counter = 0
    current_speed = CRUISE_SPEED
    turn_angle = 0
    turn_direction = ""
    path_history = []  # 重置路径历史
    last_success_pos = get_car_position()  # 记录起始位置
    distance_since_last_obstacle = 0.0
    LOCKED_BY_SPACE = False

# ------------------- 小车位置获取 -------------------
def get_car_position():
    """获取小车当前位置"""
    return data.body(chassis_body_id).xpos.copy()

def get_car_velocity():
    """获取小车当前速度"""
    return np.linalg.norm(data.qvel[:3])

# ------------------- 障碍物检测函数（增强版） -------------------
def check_front_obstacle(direction_angle=0, scan_width=0.5):
    """检测指定方向是否有障碍物"""

    chassis_pos = get_car_position()

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
    obstacle_pos = None

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

            # 计算横向距离（考虑扫描宽度）
            cross_product = np.cross([forward_direction[0], forward_direction[1], 0],
                                     [obstacle_direction[0], obstacle_direction[1], 0])
            lateral_distance = abs(cross_product[2]) * distance

            if angle_diff < math.radians(45) and lateral_distance < scan_width:
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = obs_name
                    obstacle_pos = obs_pos.copy()

    if closest_obstacle is not None:
        if min_distance < SAFE_DISTANCE:
            return 2, min_distance, closest_obstacle, obstacle_pos
        else:
            return 1, min_distance, closest_obstacle, obstacle_pos

    return 0, 0, None, None

# ------------------- 多方向路径扫描 -------------------
def scan_all_directions():
    """扫描所有可能的方向"""

    directions_info = {}
    chassis_pos = get_car_position()

    # 检查每个定义的方向
    for dir_name, dir_angle in DIRECTIONS.items():
        obstacle_status, distance, obs_name, _ = check_front_obstacle(
            dir_angle,
            scan_width=0.4 if "sharp" in dir_name else 0.3
        )

        # 计算方向得分
        base_score = DIRECTION_SCORES.get(dir_name, 0.5)

        if obstacle_status == 0:
            # 无障碍物，高分
            safety_score = 1.0
        elif obstacle_status == 1 and distance > 0.5:
            # 有障碍物但距离较远
            safety_score = 0.6
        else:
            # 有近距离障碍物
            safety_score = 0.2

        # 结合历史经验
        memory_score = 0
        if dir_name in ["forward", "slight_left", "slight_right"]:
            memory_key = path_memory._create_key(chassis_pos, dir_name)
            memory_score = path_memory.path_scores.get(memory_key, 0)

        total_score = base_score * 0.4 + safety_score * 0.4 + memory_score * 0.2

        directions_info[dir_name] = {
            'angle': dir_angle,
            'status': obstacle_status,
            'distance': distance,
            'obstacle': obs_name,
            'score': total_score
        }

    return directions_info

# ------------------- 智能路径选择 -------------------
def choose_intelligent_path():
    """智能选择最佳路径"""

    # 扫描所有方向
    directions_info = scan_all_directions()
    chassis_pos = get_car_position()

    # 过滤安全方向
    safe_directions = []
    for dir_name, info in directions_info.items():
        if info['status'] == 0 or (info['status'] == 1 and info['distance'] > 0.5):
            safe_directions.append(dir_name)

    # 如果没有安全方向，返回需要后退
    if not safe_directions:
        # 检查是否已经尝试过很多次
        if len(path_history) > 8:
            return "backward", "后退并重新规划"
        else:
            # 选择障碍物最远的方向
            best_dir = max(directions_info.items(), key=lambda x: x[1]['distance'])[0]
            return best_dir, f"强制{best_dir}(距离:{directions_info[best_dir]['distance']:.2f}m)"

    # 使用路径记忆选择最佳方向
    best_direction = path_memory.get_best_direction(chassis_pos, safe_directions)

    # 获取方向信息
    info = directions_info[best_direction]

    # 生成描述文本
    if best_direction == "forward":
        direction_text = "直行"
    elif best_direction == "backward":
        direction_text = "后退"
    else:
        angle_deg = math.degrees(info['angle'])
        if "left" in best_direction:
            direction_text = f"左转{abs(angle_deg):.0f}度"
        else:
            direction_text = f"右转{abs(angle_deg):.0f}度"

    return best_direction, direction_text

# ------------------- 路径历史记录 -------------------
def update_path_history(direction, success):
    """更新路径历史"""
    global path_history, last_success_pos, distance_since_last_obstacle

    current_pos = get_car_position()
    path_history.append({
        'direction': direction,
        'position': tuple(current_pos[:2]),
        'success': success,
        'time': time.time()
    })

    # 只保留最近20条记录
    if len(path_history) > 20:
        path_history = path_history[-20:]

    # 更新距离
    if success:
        distance_since_last_obstacle += get_car_velocity() * 0.002  # 近似距离增量

    # 如果成功行驶了一段距离，记录成功路径
    if success and distance_since_last_obstacle > 1.0:
        path_memory.record_successful_path(last_success_pos, current_pos,
                                          [h['direction'] for h in path_history[-5:]])
        last_success_pos = current_pos.copy()
        distance_since_last_obstacle = 0.0

# ------------------- 后退操作 -------------------
def perform_backup(backup_counter):
    """执行后退操作"""

    if backup_counter < 40:
        # 后退（负速度）
        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0
        data.ctrl[2] = -TURN_SPEED * 0.4
        data.ctrl[3] = -TURN_SPEED * 0.4
        data.ctrl[4] = -TURN_SPEED * 0.4
        data.ctrl[5] = -TURN_SPEED * 0.4
        return False, backup_counter + 1
    else:
        # 后退完成
        for i in range(len(data.ctrl)):
            data.ctrl[i] = 0.0
        return True, 0

# ------------------- 主循环 -------------------
mujoco.mj_resetData(model, data)

# 状态变量
CAR_STATE = "CRUISING"
turn_counter = 0
turn_angle = 0
turn_direction = ""
scan_counter = 0
deceleration_counter = 0
backup_counter = 0
current_speed = CRUISE_SPEED
path_history = []  # 路径历史记录
last_success_pos = get_car_position()  # 最后成功位置
distance_since_last_obstacle = 0.0
last_obstacle_time = 0  # 上次遇到障碍物的时间
LOCKED_BY_SPACE = False  # 空格触发的强制锁标志

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -25

    print("=== 智能绕障小车启动 ===")
    print("控制说明:")
    print("  R - 复位小车")
    print("  D - 切换调试模式")
    print("  S - 保存路径记忆")
    print("  空格 - 强制截停 / 解除锁定")
    print("======================")

    while viewer.is_running():
        # 检查键盘输入
        if KEYS.get(keyboard.KeyCode.from_char('r'), False):
            reset_car()
            KEYS[keyboard.KeyCode.from_char('r')] = False

        if KEYS.get(keyboard.KeyCode.from_char('d'), False):
            path_memory.toggle_debug()
            KEYS[keyboard.KeyCode.from_char('d')] = False

        if KEYS.get(keyboard.KeyCode.from_char('s'), False):
            path_memory.save_to_file()
            KEYS[keyboard.KeyCode.from_char('s')] = False

        # ===== 强制截停（最高优先级） =====
        if KEYS.get(keyboard.Key.space, False):
            KEYS[keyboard.Key.space] = False   # 消抖
            if CAR_STATE == "LOCK":
                # 解除锁定
                CAR_STATE = "CRUISING"
                LOCKED_BY_SPACE = False
                print("\n>>> 强制截停解除，恢复巡航 <<<")
            else:
                # 进入强制锁
                CAR_STATE = "LOCK"
                LOCKED_BY_SPACE = True
                # 瞬间清零所有电机
                for i in range(len(data.ctrl)):
                    data.ctrl[i] = 0.0
                current_speed = 0.0
                print("\n>>> 强制截停激活！按空格再次解除 <<<")

        # ===== 如果处于 LOCK 状态，直接跳过整个状态机 =====
        if CAR_STATE == "LOCK":
            mujoco.mj_step(model, data)
            viewer.sync()
            continue

        # 获取小车当前位置和速度
        car_pos = get_car_position()
        car_vel = get_car_velocity()

        # 根据当前状态执行不同操作
        if CAR_STATE == "CRUISING":
            # 巡航状态
            obstacle_status, obstacle_distance, obstacle_name, obs_pos = check_front_obstacle()

            if obstacle_status == 2:
                # 紧急停止
                CAR_STATE = "STOPPED"
                print(f"\n⚠️ 紧急停止！障碍物距离: {obstacle_distance:.2f}m")

                # 记录障碍物
                if obs_pos is not None:
                    path_memory.record_obstacle(obstacle_name, obs_pos)

                # 记录失败的路径经验
                path_memory.add_experience(car_pos, "forward", False, distance_since_last_obstacle)

                turn_counter = 0
                current_speed = 0

                for i in range(len(data.ctrl)):
                    data.ctrl[i] = 0.0

            elif obstacle_status == 1:
                # 检测到障碍物，减速
                CAR_STATE = "DECELERATING"
                deceleration_counter = 0
                print(f"\n⚠️ 检测到障碍物: {obstacle_name}({obstacle_distance:.2f}m)，开始减速...")

                # 记录障碍物
                if obs_pos is not None:
                    path_memory.record_obstacle(obstacle_name, obs_pos)

                # 记录失败的路径经验
                path_memory.add_experience(car_pos, "forward", False, distance_since_last_obstacle)

                last_obstacle_time = time.time()
            else:
                # 无障碍物，高速巡航
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
                data.ctrl[2] = CRUISE_SPEED
                data.ctrl[3] = CRUISE_SPEED
                data.ctrl[4] = CRUISE_SPEED
                data.ctrl[5] = CRUISE_SPEED
                current_speed = CRUISE_SPEED

                # 更新路径历史
                update_path_history("forward", True)

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
                print("正在智能规划路径...")
                CAR_STATE = "PATH_PLANNING"
                turn_counter = 0

        elif CAR_STATE == "PATH_PLANNING":
            # 路径规划状态
            print("扫描环境并选择最佳路径...")

            # 智能选择路径
            chosen_direction, direction_text = choose_intelligent_path()

            if chosen_direction == "backward":
                # 需要后退
                print("路径受阻，执行后退操作")
                CAR_STATE = "BACKING_UP"
                backup_counter = 0
            else:
                # 找到可行路径
                turn_angle = DIRECTIONS[chosen_direction]
                turn_direction = direction_text

                print(f"选择路径: {turn_direction}")
                CAR_STATE = "TURNING"
                turn_counter = 0

        elif CAR_STATE == "BACKING_UP":
            # 后退状态
            backup_complete, backup_counter = perform_backup(backup_counter)

            if backup_complete:
                print("后退完成，重新规划路径")
                CAR_STATE = "PATH_PLANNING"
                update_path_history("backward", True)

        elif CAR_STATE == "TURNING":
            # 转向状态
            turn_counter += 1

            # 渐进式转向
            progress = min(1.0, turn_counter / 8.0)
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
                print(f"正在{turn_direction}，进度: {progress*100:.0f}%")

            if turn_counter > TURN_DURATION:
                print(f"{turn_direction}完成，开始验证路径...")
                CAR_STATE = "PATH_VERIFICATION"
                turn_counter = 0
                scan_counter = 0

        elif CAR_STATE == "PATH_VERIFICATION":
            # 路径验证状态
            scan_counter += 1

            # 保持转向角度，低速前进验证路径
            data.ctrl[0] = turn_angle * 0.5
            data.ctrl[1] = turn_angle * 0.5
            data.ctrl[2] = TURN_SPEED * 0.6
            data.ctrl[3] = TURN_SPEED * 0.6
            data.ctrl[4] = TURN_SPEED * 0.6
            data.ctrl[5] = TURN_SPEED * 0.6
            current_speed = TURN_SPEED * 0.6

            if scan_counter % 10 == 0:
                # 检查前方是否安全
                obstacle_status, obstacle_distance, obstacle_name, _ = check_front_obstacle()

                if obstacle_status == 0:
                    print("路径验证通过，准备恢复巡航")

                    # 记录成功的路径经验
                    chosen_direction_name = [k for k, v in DIRECTIONS.items() if abs(v - turn_angle) < 0.01][0]
                    path_memory.add_experience(car_pos, chosen_direction_name, True, distance_since_last_obstacle)

                    CAR_STATE = "RESUME"
                    turn_counter = 0
                else:
                    print(f"路径验证失败，检测到障碍物: {obstacle_name}({obstacle_distance:.2f}m)")
                    CAR_STATE = "STOPPED"
                    turn_counter = 0

            if scan_counter > 40:
                print("路径验证超时，尝试恢复巡航")
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

                obstacle_status, obstacle_distance, _, _ = check_front_obstacle()
                if obstacle_status == 0:
                    print("成功恢复巡航")
                    CAR_STATE = "CRUISING"
                    turn_counter = 0

                    # 更新路径历史
                    chosen_direction_name = [k for k, v in DIRECTIONS.items() if abs(v - turn_angle) < 0.01][0]
                    update_path_history(chosen_direction_name, True)
                else:
                    print("恢复巡航时检测到障碍物，重新处理")
                    CAR_STATE = "STOPPED"
                    turn_counter = 0

        # 仿真步骤
        mujoco.mj_step(model, data)

        # 显示信息
        vel = car_vel
        current_steer = (data.ctrl[0] + data.ctrl[1]) / 2

        # 显示状态信息
        status_info = f"状态: {CAR_STATE}, 速度: {vel:7.5f} m/s"
        if abs(current_steer) > 0.01:
            status_info += f", 转向: {math.degrees(current_steer):.1f}°"

        # 显示路径历史长度
        status_info += f", 历史: {len(path_history)}"

        # 显示记忆系统状态
        status_info += f", 记忆: {len(path_memory.memory)}"

        if path_memory.debug_mode and CAR_STATE == "CRUISING":
            obstacle_status, obstacle_distance, obstacle_name, _ = check_front_obstacle()
            if obstacle_status > 0 and obstacle_name:
                status_info += f", 障碍: {obstacle_name}({obstacle_distance:.2f}m)"

        print(f"\r{status_info}", end='', flush=True)

        # 同步视图
        viewer.sync()

print("\n程序结束，保存路径记忆...")
path_memory.save_to_file()