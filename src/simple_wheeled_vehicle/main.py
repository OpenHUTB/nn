"""
自动巡航小车 - 增强版智能绕障与路径记忆系统
- 巡航速度：0.003 m/s，可3倍加速至0.009 m/s
- 智能障碍检测与路径规划
- 强化学习路径记忆与自适应优化
- 空格键强制截停/恢复
- Shift键3倍加速
- R键复位，D键调试，S键保存
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
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Deque

# ------------------- 枚举定义 -------------------
class CarState(Enum):
    """小车状态枚举"""
    CRUISING = "巡航中"
    DECELERATING = "减速中"
    STOPPED = "已停止"
    PATH_PLANNING = "路径规划中"
    TURNING = "转向中"
    PATH_VERIFICATION = "路径验证中"
    RESUME = "恢复巡航"
    BACKING_UP = "后退中"
    EMERGENCY_STOP = "强制截停"

class Direction(Enum):
    """方向枚举"""
    FORWARD = "forward"
    SLIGHT_LEFT = "slight_left"
    SLIGHT_RIGHT = "slight_right"
    LEFT = "left"
    RIGHT = "right"
    SHARP_LEFT = "sharp_left"
    SHARP_RIGHT = "sharp_right"
    BACKWARD = "backward"

# ------------------- 数据类定义 -------------------
@dataclass
class DirectionInfo:
    """方向信息"""
    angle: float
    status: int
    distance: float
    obstacle: Optional[str]
    score: float

@dataclass
class PathExperience:
    """路径经验"""
    position: Tuple[float, float]
    direction: str
    success: bool
    distance: float
    timestamp: float

@dataclass
class ObstacleRecord:
    """障碍物记录"""
    name: str
    position: Tuple[float, float]
    timestamp: float
    count: int = 1

# ------------------- 参数配置类 -------------------
class Config:
    """系统配置参数"""
    # 速度参数
    BASE_CRUISE_SPEED = 0.003
    TURN_SPEED_RATIO = 0.4
    BOOST_MULTIPLIER = 3.0  # 3倍加速

    # 障碍物检测
    OBSTACLE_THRESHOLD = 0.7
    SAFE_DISTANCE = 0.3
    SCAN_RANGE = 1.0

    # 转向参数
    TURN_ANGLE = 0.3
    TURN_DURATION = 50

    # 路径记忆
    PATH_MEMORY_SIZE = 50
    EXPLORATION_RATE = 0.3
    LEARNING_RATE = 0.1
    PATH_REWARD = 1.0
    PATH_PENALTY = -0.5

    # 方向得分权重
    DIRECTION_SCORES = {
        "forward": 1.0,
        "slight_left": 0.9,
        "slight_right": 0.9,
        "left": 0.8,
        "right": 0.8,
        "sharp_left": 0.6,
        "sharp_right": 0.6,
        "backward": 0.3,
    }

    # 方向角度定义
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

    # 转向扫描宽度
    SCAN_WIDTHS = {
        "sharp": 0.4,
        "default": 0.3
    }

# ------------------- 键盘管理器 -------------------
class KeyboardManager:
    """键盘输入管理"""

    def __init__(self):
        self.keys = {
            keyboard.KeyCode.from_char('r'): False,
            keyboard.KeyCode.from_char('d'): False,
            keyboard.KeyCode.from_char('s'): False,
            keyboard.Key.space: False,
            keyboard.Key.shift: False,
            keyboard.Key.shift_l: False,
            keyboard.Key.shift_r: False,
        }
        self.listener = None
        self._start_listener()

    def _start_listener(self):
        """启动键盘监听"""
        def on_press(key):
            if key in self.keys:
                self.keys[key] = True
            elif isinstance(key, keyboard.Key) and key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
                self.keys[keyboard.Key.shift] = True

        def on_release(key):
            if key in self.keys:
                self.keys[key] = False
            elif isinstance(key, keyboard.Key) and key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
                self.keys[keyboard.Key.shift] = False

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.daemon = True
        self.listener.start()

    def is_pressed(self, key):
        """检查按键是否按下"""
        return self.keys.get(key, False)

    def reset_key(self, key):
        """重置按键状态"""
        if key in self.keys:
            self.keys[key] = False

# ------------------- 路径记忆系统 -------------------
class PathMemory:
    """增强版路径记忆与学习系统"""

    def __init__(self, memory_size: int = Config.PATH_MEMORY_SIZE):
        self.memory: Deque[PathExperience] = deque(maxlen=memory_size)
        self.path_scores: Dict[str, float] = {}
        self.obstacle_history: Dict[str, ObstacleRecord] = {}
        self.successful_paths: List[Dict] = []
        self.debug_mode = False
        self.learning_rate = Config.LEARNING_RATE

    def add_experience(self, position: np.ndarray, direction: str,
                      success: bool, distance_traveled: float) -> None:
        """添加并学习路径经验"""
        key = self._create_key(position, direction)

        # 强化学习更新
        reward = Config.PATH_REWARD if success else Config.PATH_PENALTY
        current_score = self.path_scores.get(key, 0)
        new_score = current_score + self.learning_rate * (reward - current_score)
        self.path_scores[key] = new_score

        # 记录经验
        experience = PathExperience(
            position=tuple(position[:2]),
            direction=direction,
            success=success,
            distance=distance_traveled,
            timestamp=time.time()
        )
        self.memory.append(experience)

        if self.debug_mode:
            status = "✓" if success else "✗"
            print(f"路径经验: {direction} {status}, 评分: {new_score:.2f}")

    def get_best_direction(self, position: np.ndarray,
                          available_directions: List[str]) -> str:
        """基于历史经验获取最佳方向"""
        # 探索策略
        if random.random() < Config.EXPLORATION_RATE:
            return random.choice(available_directions)

        # 利用策略：选择综合得分最高的方向
        best_direction = None
        best_score = -float('inf')

        for direction in available_directions:
            key = self._create_key(position, direction)
            base_score = Config.DIRECTION_SCORES.get(direction, 0.5)
            memory_score = self.path_scores.get(key, 0)

            # 综合得分：基础分 + 记忆分
            total_score = base_score * 0.6 + memory_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_direction = direction

        return best_direction or random.choice(available_directions)

    def record_obstacle(self, obstacle_name: str, position: np.ndarray) -> None:
        """记录障碍物位置"""
        key = f"{obstacle_name}_{int(position[0]*10)}_{int(position[1]*10)}"

        if key in self.obstacle_history:
            self.obstacle_history[key].count += 1
            self.obstacle_history[key].timestamp = time.time()
        else:
            self.obstacle_history[key] = ObstacleRecord(
                name=obstacle_name,
                position=tuple(position[:2]),
                timestamp=time.time()
            )

    def is_recent_obstacle(self, position: np.ndarray,
                          threshold: float = 0.5, time_window: float = 10.0) -> bool:
        """检查位置附近是否有近期遇到的障碍物"""
        current_time = time.time()

        for record in self.obstacle_history.values():
            obs_pos = record.position
            distance = math.dist(obs_pos, position[:2])

            if (distance < threshold and
                (current_time - record.timestamp) < time_window):
                return True

        return False

    def save_to_file(self, filename: str = "path_memory.json") -> None:
        """保存路径记忆到文件"""
        save_data = {
            'path_scores': self.path_scores,
            'obstacle_history': {k: vars(v) for k, v in self.obstacle_history.items()},
            'successful_paths': self.successful_paths[-10:],
            'timestamp': time.time()
        }

        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"✅ 路径记忆已保存到 {filename}")

    def load_from_file(self, filename: str = "path_memory.json") -> bool:
        """从文件加载路径记忆"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.path_scores = data.get('path_scores', {})

            # 恢复障碍物记录
            obs_history = data.get('obstacle_history', {})
            for key, obs_data in obs_history.items():
                self.obstacle_history[key] = ObstacleRecord(**obs_data)

            self.successful_paths = data.get('successful_paths', [])
            print(f"✅ 已从 {filename} 加载路径记忆")
            return True

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️  无法加载记忆文件: {e}")
            return False

    def _create_key(self, position: np.ndarray, direction: str) -> str:
        """创建记忆键"""
        x, y = int(position[0] * 10), int(position[1] * 10)
        return f"{x}_{y}_{direction}"

    def toggle_debug(self) -> None:
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
        status = "开启" if self.debug_mode else "关闭"
        print(f"🔧 调试模式: {status}")

# ------------------- 小车控制器 -------------------
class CarController:
    """小车运动控制器"""

    def __init__(self, model, data, config: Config):
        self.model = model
        self.data = data
        self.config = config

        # 获取车身ID
        self.chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # 预加载障碍物ID
        self.obstacle_ids = self._load_obstacle_ids()

    def _load_obstacle_ids(self) -> Dict[str, int]:
        """加载障碍物ID"""
        obstacle_names = [
            'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
            'obs_ball1', 'obs_ball2', 'obs_ball3',
            'wall1', 'wall2', 'front_dark_box'
        ]

        ids = {}
        for name in obstacle_names:
            obs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if obs_id != -1:
                ids[name] = obs_id

        return ids

    def get_position(self) -> np.ndarray:
        """获取小车当前位置"""
        return self.data.body(self.chassis_id).xpos.copy()

    def get_velocity(self) -> float:
        """获取小车当前速度"""
        return np.linalg.norm(self.data.qvel[:3])

    def set_control(self, steer_angle: float = 0.0,
                   speed: float = 0.0, all_wheels: bool = True) -> None:
        """设置小车控制参数"""
        # 转向控制
        self.data.ctrl[0] = steer_angle
        self.data.ctrl[1] = steer_angle

        # 速度控制
        if all_wheels:
            self.data.ctrl[2] = speed
            self.data.ctrl[3] = speed
            self.data.ctrl[4] = speed
            self.data.ctrl[5] = speed
        else:
            # 仅前轮驱动
            self.data.ctrl[2] = speed
            self.data.ctrl[3] = speed

    def emergency_stop(self) -> None:
        """紧急停止"""
        for i in range(len(self.data.ctrl)):
            self.data.ctrl[i] = 0.0

    def check_obstacle(self, direction_angle: float = 0,
                      scan_width: float = 0.3) -> Tuple[int, float, Optional[str], Optional[np.ndarray]]:
        """检测指定方向的障碍物"""
        chassis_pos = self.get_position()

        # 获取前进方向
        velocity = self.data.qvel[:2]
        if np.linalg.norm(velocity) < 0.0001:
            forward = np.array([1.0, 0.0])
        else:
            forward = velocity / np.linalg.norm(velocity)

        # 应用方向旋转
        if direction_angle != 0:
            cos_a, sin_a = math.cos(direction_angle), math.sin(direction_angle)
            forward = np.array([
                forward[0] * cos_a - forward[1] * sin_a,
                forward[0] * sin_a + forward[1] * cos_a
            ])

        min_distance = float('inf')
        closest_obstacle = None
        obstacle_pos = None

        for obs_name, obs_id in self.obstacle_ids.items():
            obs_pos = self.data.body(obs_id).xpos
            rel_pos = obs_pos[:2] - chassis_pos[:2]
            distance = np.linalg.norm(rel_pos)

            if 0 < distance < self.config.SCAN_RANGE:
                obs_dir = rel_pos / distance

                # 计算夹角
                dot_product = np.dot(obs_dir, forward)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle_diff = math.acos(dot_product)

                # 计算横向距离
                cross_z = np.cross([forward[0], forward[1], 0],
                                  [obs_dir[0], obs_dir[1], 0])[2]
                lateral_dist = abs(cross_z) * distance

                # 判断是否在检测范围内
                if angle_diff < math.radians(45) and lateral_dist < scan_width:
                    if distance < min_distance:
                        min_distance = distance
                        closest_obstacle = obs_name
                        obstacle_pos = obs_pos.copy()

        # 返回结果
        if closest_obstacle is not None:
            if min_distance < self.config.SAFE_DISTANCE:
                return 2, min_distance, closest_obstacle, obstacle_pos
            else:
                return 1, min_distance, closest_obstacle, obstacle_pos

        return 0, 0, None, None

# ------------------- 路径规划器 -------------------
class PathPlanner:
    """智能路径规划器"""

    def __init__(self, controller: CarController, memory: PathMemory):
        self.controller = controller
        self.memory = memory
        self.config = Config()

    def scan_directions(self) -> Dict[str, DirectionInfo]:
        """扫描所有可能方向"""
        directions_info = {}

        for dir_name, dir_angle in self.config.DIRECTIONS.items():
            # 确定扫描宽度
            scan_width = (self.config.SCAN_WIDTHS["sharp"]
                         if "sharp" in dir_name
                         else self.config.SCAN_WIDTHS["default"])

            # 检测障碍物
            status, distance, obs_name, _ = self.controller.check_obstacle(
                dir_angle, scan_width
            )

            # 计算安全得分
            if status == 0:
                safety_score = 1.0
            elif status == 1 and distance > 0.5:
                safety_score = 0.6
            else:
                safety_score = 0.2

            # 基础得分
            base_score = self.config.DIRECTION_SCORES.get(dir_name, 0.5)

            # 记忆得分
            memory_score = 0
            pos = self.controller.get_position()
            if dir_name in ["forward", "slight_left", "slight_right"]:
                key = self.memory._create_key(pos, dir_name)
                memory_score = self.memory.path_scores.get(key, 0)

            # 综合得分
            total_score = base_score * 0.4 + safety_score * 0.4 + memory_score * 0.2

            directions_info[dir_name] = DirectionInfo(
                angle=dir_angle,
                status=status,
                distance=distance,
                obstacle=obs_name,
                score=total_score
            )

        return directions_info

    def choose_best_path(self) -> Tuple[str, str]:
        """智能选择最佳路径"""
        # 扫描环境
        directions_info = self.scan_directions()
        position = self.controller.get_position()

        # 筛选安全方向
        safe_directions = [
            dir_name for dir_name, info in directions_info.items()
            if info.status == 0 or (info.status == 1 and info.distance > 0.5)
        ]

        # 无安全方向时的处理
        if not safe_directions:
            # 尝试选择障碍物最远的方向
            best_dir = max(directions_info.items(),
                          key=lambda x: x[1].distance)[0]
            dist = directions_info[best_dir].distance
            return best_dir, f"强制{best_dir}(距离:{dist:.2f}m)"

        # 使用记忆系统选择最佳方向
        best_direction = self.memory.get_best_direction(position, safe_directions)
        info = directions_info[best_direction]

        # 生成描述文本
        if best_direction == "forward":
            desc = "直行"
        elif best_direction == "backward":
            desc = "后退"
        else:
            angle_deg = math.degrees(info.angle)
            direction = "左" if "left" in best_direction else "右"
            desc = f"{direction}转{abs(angle_deg):.0f}度"

        return best_direction, desc

# ------------------- 主控制系统 -------------------
class PatrolSystem:
    """主控制系统"""

    def __init__(self, model_path: str = "wheeled_car.xml"):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 初始化组件
        self.config = Config()
        self.keyboard = KeyboardManager()
        self.controller = CarController(self.model, self.data, self.config)
        self.memory = PathMemory()
        self.planner = PathPlanner(self.controller, self.memory)

        # 状态变量
        self.state = CarState.CRUISING
        self.previous_state = None  # 用于强制截停恢复

        # 控制变量
        self.turn_counter = 0
        self.turn_angle = 0
        self.turn_direction = ""
        self.scan_counter = 0
        self.deceleration_counter = 0
        self.backup_counter = 0

        # 速度管理
        self.is_boosting = False
        self.current_cruise_speed = self.config.BASE_CRUISE_SPEED
        self.current_turn_speed = self.config.BASE_CRUISE_SPEED * self.config.TURN_SPEED_RATIO

        # 路径历史
        self.path_history = []
        self.last_success_pos = self.controller.get_position()
        self.distance_since_obstacle = 0.0

        # 加载记忆
        self.memory.load_from_file()

    def reset(self) -> None:
        """复位小车"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.03  # 确保离地高度

        # 重置状态
        self.state = CarState.CRUISING
        self.previous_state = None

        # 重置控制变量
        self.turn_counter = 0
        self.turn_angle = 0
        self.turn_direction = ""
        self.scan_counter = 0
        self.deceleration_counter = 0
        self.backup_counter = 0

        # 重置速度
        self.is_boosting = False
        self._update_speeds()

        # 重置路径记录
        self.path_history.clear()
        self.last_success_pos = self.controller.get_position()
        self.distance_since_obstacle = 0.0

        print("\n🔄 小车已复位")

    def _update_speeds(self) -> None:
        """更新当前速度参数"""
        multiplier = self.config.BOOST_MULTIPLIER if self.is_boosting else 1.0
        self.current_cruise_speed = self.config.BASE_CRUISE_SPEED * multiplier
        self.current_turn_speed = (self.config.BASE_CRUISE_SPEED *
                                  self.config.TURN_SPEED_RATIO * multiplier)

    def toggle_emergency_stop(self) -> None:
        """切换强制截停状态"""
        if self.state == CarState.EMERGENCY_STOP:
            # 恢复之前的状态
            self.state = self.previous_state or CarState.CRUISING
            self.previous_state = None
            print("\n✅ 强制截停解除，恢复运行")
        else:
            # 进入强制截停
            self.previous_state = self.state
            self.state = CarState.EMERGENCY_STOP
            self.controller.emergency_stop()
            print("\n🚨 强制截停已激活")

    def update_path_history(self, direction: str, success: bool) -> None:
        """更新路径历史记录"""
        current_pos = self.controller.get_position()

        self.path_history.append({
            'direction': direction,
            'position': tuple(current_pos[:2]),
            'success': success,
            'time': time.time()
        })

        # 限制历史记录长度
        if len(self.path_history) > 20:
            self.path_history = self.path_history[-20:]

        # 更新距离
        if success:
            self.distance_since_obstacle += self.controller.get_velocity() * 0.002

        # 记录成功路径
        if success and self.distance_since_obstacle > 1.0:
            directions = [h['direction'] for h in self.path_history[-5:]]
            self.memory.successful_paths.append({
                'start': self.last_success_pos[:2],
                'end': current_pos[:2],
                'directions': directions,
                'timestamp': time.time()
            })
            self.last_success_pos = current_pos.copy()
            self.distance_since_obstacle = 0.0

    def handle_cruising(self) -> None:
        """处理巡航状态"""
        status, distance, obs_name, obs_pos = self.controller.check_obstacle()

        if status == 2:  # 紧急障碍
            self.state = CarState.STOPPED
            print(f"\n⚠️ 紧急停止！障碍物距离: {distance:.2f}m")

            if obs_pos is not None:
                self.memory.record_obstacle(obs_name, obs_pos)

            self.memory.add_experience(
                self.controller.get_position(),
                "forward",
                False,
                self.distance_since_obstacle
            )

            self.controller.emergency_stop()

        elif status == 1:  # 检测到障碍物
            self.state = CarState.DECELERATING
            self.deceleration_counter = 0
            print(f"\n⚠️ 检测到障碍物: {obs_name}({distance:.2f}m)，开始减速...")

            if obs_pos is not None:
                self.memory.record_obstacle(obs_name, obs_pos)

            self.memory.add_experience(
                self.controller.get_position(),
                "forward",
                False,
                self.distance_since_obstacle
            )

        else:  # 安全巡航
            self.controller.set_control(
                speed=self.current_cruise_speed,
                all_wheels=True
            )
            self.update_path_history("forward", True)

    def handle_decelerating(self) -> None:
        """处理减速状态"""
        self.deceleration_counter += 1
        progress = min(1.0, self.deceleration_counter / 15.0)
        current_speed = self.current_cruise_speed * (1.0 - progress)

        self.controller.set_control(speed=current_speed)

        if self.deceleration_counter > 20:
            self.state = CarState.STOPPED
            print("减速完成，准备规划路径")
            self.turn_counter = 0

    def handle_stopped(self) -> None:
        """处理停止状态"""
        self.turn_counter += 1
        self.controller.emergency_stop()

        if self.turn_counter > 10:
            print("正在智能规划路径...")
            self.state = CarState.PATH_PLANNING
            self.turn_counter = 0

    def handle_path_planning(self) -> None:
        """处理路径规划"""
        chosen_direction, direction_text = self.planner.choose_best_path()

        if chosen_direction == "backward":
            print("路径受阻，执行后退操作")
            self.state = CarState.BACKING_UP
            self.backup_counter = 0
        else:
            self.turn_angle = self.config.DIRECTIONS[chosen_direction]
            self.turn_direction = direction_text
            print(f"选择路径: {self.turn_direction}")
            self.state = CarState.TURNING
            self.turn_counter = 0

    def handle_backing_up(self) -> None:
        """处理后撤"""
        if self.backup_counter < 40:
            speed = -self.current_turn_speed * 0.4
            self.controller.set_control(speed=speed)
            self.backup_counter += 1
        else:
            self.controller.emergency_stop()
            print("后退完成，重新规划路径")
            self.state = CarState.PATH_PLANNING
            self.update_path_history("backward", True)

    def handle_turning(self) -> None:
        """处理转向"""
        self.turn_counter += 1
        progress = min(1.0, self.turn_counter / 8.0)

        # 渐进转向
        current_angle = self.turn_angle * progress
        self.controller.set_control(steer_angle=current_angle)

        # 渐进加速
        if self.turn_counter > 5:
            speed_progress = min(1.0, (self.turn_counter - 5) / 15.0)
            current_speed = self.current_turn_speed * speed_progress
            self.controller.set_control(
                steer_angle=current_angle,
                speed=current_speed
            )

        # 状态更新
        if self.turn_counter % 15 == 0:
            print(f"正在{self.turn_direction}，进度: {progress*100:.0f}%")

        if self.turn_counter > self.config.TURN_DURATION:
            print(f"{self.turn_direction}完成，开始验证路径...")
            self.state = CarState.PATH_VERIFICATION
            self.turn_counter = 0
            self.scan_counter = 0

    def handle_path_verification(self) -> None:
        """处理路径验证"""
        self.scan_counter += 1

        # 低速验证路径
        self.controller.set_control(
            steer_angle=self.turn_angle * 0.5,
            speed=self.current_turn_speed * 0.6
        )

        if self.scan_counter % 10 == 0:
            status, distance, obs_name, _ = self.controller.check_obstacle()

            if status == 0:  # 路径安全
                print("路径验证通过，准备恢复巡航")

                # 记录成功经验
                for dir_name, angle in self.config.DIRECTIONS.items():
                    if abs(angle - self.turn_angle) < 0.01:
                        self.memory.add_experience(
                            self.controller.get_position(),
                            dir_name,
                            True,
                            self.distance_since_obstacle
                        )
                        break

                self.state = CarState.RESUME
                self.turn_counter = 0
            else:  # 路径不安全
                print(f"路径验证失败，检测到障碍物: {obs_name}({distance:.2f}m)")
                self.state = CarState.STOPPED
                self.turn_counter = 0

        if self.scan_counter > 40:
            print("路径验证超时，尝试恢复巡航")
            self.state = CarState.RESUME
            self.turn_counter = 0

    def handle_resume(self) -> None:
        """处理恢复巡航"""
        self.turn_counter += 1
        progress = min(1.0, self.turn_counter / 15.0)

        # 渐进恢复
        current_angle = self.turn_angle * (1.0 - progress)
        current_speed = (self.current_turn_speed +
                        (self.current_cruise_speed - self.current_turn_speed) * progress)

        self.controller.set_control(
            steer_angle=current_angle,
            speed=current_speed
        )

        if self.turn_counter > 20:
            # 完全恢复巡航
            self.controller.set_control(speed=self.current_cruise_speed)

            # 检查前方安全
            status, _, _, _ = self.controller.check_obstacle()
            if status == 0:
                print("成功恢复巡航")
                self.state = CarState.CRUISING
                self.turn_counter = 0

                # 记录路径历史
                for dir_name, angle in self.config.DIRECTIONS.items():
                    if abs(angle - self.turn_angle) < 0.01:
                        self.update_path_history(dir_name, True)
                        break
            else:
                print("恢复巡航时检测到障碍物，重新处理")
                self.state = CarState.STOPPED
                self.turn_counter = 0

    def run(self) -> None:
        """运行主循环"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -25

            # 显示控制说明
            print("=" * 50)
            print("🚗 增强版智能绕障小车启动")
            print("=" * 50)
            print("控制说明:")
            print("  R        - 复位小车")
            print("  D        - 切换调试模式")
            print("  S        - 保存路径记忆")
            print("  空格键    - 强制截停/恢复")
            print("  Shift键  - 3倍加速行驶")
            print("=" * 50)

            try:
                while viewer.is_running():
                    # 处理键盘输入
                    self._handle_keyboard()

                    # 更新速度参数
                    self._update_speeds()

                    # 强制截停状态处理
                    if self.state == CarState.EMERGENCY_STOP:
                        self.controller.emergency_stop()
                    else:
                        # 状态机处理
                        self._handle_state()

                    # 执行仿真步
                    mujoco.mj_step(self.model, self.data)

                    # 显示状态信息
                    self._display_status()

                    # 同步视图
                    viewer.sync()

            except KeyboardInterrupt:
                print("\n\n⚠️ 用户中断程序")
            finally:
                print("\n保存路径记忆...")
                self.memory.save_to_file()
                print("程序结束")

    def _handle_keyboard(self) -> None:
        """处理键盘输入"""
        if self.keyboard.is_pressed(keyboard.KeyCode.from_char('r')):
            self.reset()
            self.keyboard.reset_key(keyboard.KeyCode.from_char('r'))

        if self.keyboard.is_pressed(keyboard.KeyCode.from_char('d')):
            self.memory.toggle_debug()
            self.keyboard.reset_key(keyboard.KeyCode.from_char('d'))

        if self.keyboard.is_pressed(keyboard.KeyCode.from_char('s')):
            self.memory.save_to_file()
            self.keyboard.reset_key(keyboard.KeyCode.from_char('s'))

        if self.keyboard.is_pressed(keyboard.Key.space):
            self.toggle_emergency_stop()
            self.keyboard.reset_key(keyboard.Key.space)

        # 更新加速状态
        self.is_boosting = self.keyboard.is_pressed(keyboard.Key.shift)

    def _handle_state(self) -> None:
        """处理状态机"""
        state_handlers = {
            CarState.CRUISING: self.handle_cruising,
            CarState.DECELERATING: self.handle_decelerating,
            CarState.STOPPED: self.handle_stopped,
            CarState.PATH_PLANNING: self.handle_path_planning,
            CarState.TURNING: self.handle_turning,
            CarState.PATH_VERIFICATION: self.handle_path_verification,
            CarState.RESUME: self.handle_resume,
            CarState.BACKING_UP: self.handle_backing_up,
        }

        handler = state_handlers.get(self.state)
        if handler:
            handler()

    def _display_status(self) -> None:
        """显示状态信息"""
        vel = self.controller.get_velocity()
        steer = (self.data.ctrl[0] + self.data.ctrl[1]) / 2

        # 基础状态信息
        info_parts = [
            f"状态: {self.state.value}",
            f"速度: {vel:7.5f} m/s",
        ]

        # 转向信息
        if abs(steer) > 0.01:
            info_parts.append(f"转向: {math.degrees(steer):.1f}°")

        # 系统信息
        info_parts.extend([
            f"路径历史: {len(self.path_history)}",
            f"路径记忆: {len(self.memory.memory)}",
        ])

        # 加速状态
        if self.is_boosting:
            info_parts.append(f"加速: {self.config.BOOST_MULTIPLIER}倍")

        # 调试信息
        if (self.memory.debug_mode and
            self.state == CarState.CRUISING):
            status, distance, obs_name, _ = self.controller.check_obstacle()
            if status > 0 and obs_name:
                info_parts.append(f"障碍: {obs_name}({distance:.2f}m)")

        # 输出状态行
        status_line = ", ".join(info_parts)
        print(f"\r{status_line}", end='', flush=True)

# ------------------- 主程序入口 -------------------
def main():
    """主程序"""
    try:
        system = PatrolSystem("wheeled_car.xml")
        system.run()
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()