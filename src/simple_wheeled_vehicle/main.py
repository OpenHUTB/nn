"""
自动巡航小车 - 增强版智能绕障与路径记忆系统
版本：2.0
系统功能：
1. 基础巡航：0.003 m/s基础速度，可3倍加速至0.009 m/s
2. 智能避障：多方向障碍物检测与风险评估
3. 路径记忆：基于强化学习的路径记忆与自适应优化
4. 实时控制：空格键强制截停/恢复，Shift键3倍加速
5. 状态管理：R键复位，D键调试，S键保存记忆

系统架构：
PatrolSystem (主控制系统) ─┬─ KeyboardManager (键盘输入)
                           ├─ CarController (运动控制)
                           ├─ PathPlanner (路径规划)
                           └─ PathMemory (记忆学习)

依赖库：
- mujoco：物理仿真引擎
- pynput：键盘监听
- numpy：数值计算
- 其他标准库
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

# ============================================================================
# 枚举类型定义
# ============================================================================

class CarState(Enum):
    """
    小车状态枚举
    定义小车在自动驾驶过程中的各种状态
    """
    CRUISING = "巡航中"            # 正常前进状态
    DECELERATING = "减速中"        # 检测到障碍物，正在减速
    STOPPED = "已停止"             # 完全停止，等待路径规划
    PATH_PLANNING = "路径规划中"   # 正在计算最佳绕障路径
    TURNING = "转向中"             # 执行转向操作
    PATH_VERIFICATION = "路径验证中" # 验证转向后路径的安全性
    RESUME = "恢复巡航"            # 从转向状态恢复到正常巡航
    BACKING_UP = "后退中"          # 执行后退操作
    EMERGENCY_STOP = "强制截停"    # 手动强制停止状态

class Direction(Enum):
    """
    方向枚举
    定义小车可能的移动方向
    """
    FORWARD = "forward"        # 直行
    SLIGHT_LEFT = "slight_left"  # 轻微左转(15°)
    SLIGHT_RIGHT = "slight_right" # 轻微右转(15°)
    LEFT = "left"              # 左转(30°)
    RIGHT = "right"            # 右转(30°)
    SHARP_LEFT = "sharp_left"  # 急左转(60°)
    SHARP_RIGHT = "sharp_right" # 急右转(60°)
    BACKWARD = "backward"      # 后退

# ============================================================================
# 数据类定义（使用Python 3.7+的dataclass）
# ============================================================================

@dataclass
class DirectionInfo:
    """
    方向信息数据类
    存储每个方向的详细信息，用于路径规划决策

    属性：
    - angle: 方向角度（弧度）
    - status: 障碍物状态（0=安全，1=有障碍但安全，2=危险）
    - distance: 到最近障碍物的距离
    - obstacle: 障碍物名称（如果有）
    - score: 该方向的综合评分
    """
    angle: float
    status: int
    distance: float
    obstacle: Optional[str]
    score: float

@dataclass
class PathExperience:
    """
    路径经验数据类
    存储单次路径选择的经验，用于强化学习

    属性：
    - position: 位置坐标(x,y)
    - direction: 选择的方向
    - success: 是否成功
    - distance: 行驶距离
    - timestamp: 时间戳
    """
    position: Tuple[float, float]
    direction: str
    success: bool
    distance: float
    timestamp: float

@dataclass
class ObstacleRecord:
    """
    障碍物记录数据类
    记录遇到的障碍物信息

    属性：
    - name: 障碍物名称
    - position: 障碍物位置(x,y)
    - timestamp: 遇到时间
    - count: 遇到次数
    """
    name: str
    position: Tuple[float, float]
    timestamp: float
    count: int = 1

# ============================================================================
# 系统配置类
# ============================================================================

class Config:
    """
    系统配置参数类
    集中管理所有可调参数，便于维护和调整

    设计原则：
    1. 参数分组：按功能模块分组参数
    2. 命名清晰：使用大写和下划线
    3. 注释完整：每个参数都有用途说明
    """

    # ------------------ 速度参数 ------------------
    BASE_CRUISE_SPEED = 0.003     # 基础巡航速度 (m/s)
    TURN_SPEED_RATIO = 0.4        # 转向时速度与巡航速度的比例
    BOOST_MULTIPLIER = 3.0        # 加速倍率（按住Shift时生效）

    # ------------------ 障碍物检测参数 ------------------
    OBSTACLE_THRESHOLD = 0.7      # 障碍物检测阈值(m)，小于此值触发避障
    SAFE_DISTANCE = 0.3           # 安全距离(m)，小于此值紧急停止
    SCAN_RANGE = 1.0              # 障碍物扫描范围(m)

    # ------------------ 转向控制参数 ------------------
    TURN_ANGLE = 0.3              # 最大转向角度(弧度)
    TURN_DURATION = 50            # 转向持续时间(仿真步数)

    # ------------------ 路径记忆参数 ------------------
    PATH_MEMORY_SIZE = 50         # 路径记忆容量（经验条数）
    EXPLORATION_RATE = 0.3        # 探索率：随机尝试新路径的概率
    LEARNING_RATE = 0.1           # 学习率：经验更新的速度
    PATH_REWARD = 1.0             # 成功路径奖励值
    PATH_PENALTY = -0.5           # 失败路径惩罚值

    # ------------------ 方向评分权重 ------------------
    # 各方向的基准评分，用于引导小车优先选择某些方向
    DIRECTION_SCORES = {
        "forward": 1.0,           # 直行优先（最高分）
        "slight_left": 0.9,       # 轻微左转
        "slight_right": 0.9,      # 轻微右转
        "left": 0.8,              # 左转
        "right": 0.8,             # 右转
        "sharp_left": 0.6,        # 急左转
        "sharp_right": 0.6,       # 急右转
        "backward": 0.3,          # 后退（最低分，最后考虑）
    }

    # ------------------ 方向角度定义 ------------------
    # 各方向对应的转向角度（弧度制）
    DIRECTIONS = {
        "forward": 0,                    # 直行：0°
        "slight_left": math.radians(15),  # 轻微左转：15°
        "slight_right": math.radians(-15),# 轻微右转：-15°
        "left": math.radians(30),        # 左转：30°
        "right": math.radians(-30),      # 右转：-30°
        "sharp_left": math.radians(60),  # 急左转：60°
        "sharp_right": math.radians(-60),# 急右转：-60°
        "backward": math.radians(180),   # 后退：180°
    }

    # ------------------ 扫描宽度参数 ------------------
    # 不同转向角度下的障碍物扫描宽度
    SCAN_WIDTHS = {
        "sharp": 0.4,     # 急转向时使用较宽扫描范围
        "default": 0.3    # 普通转向使用标准扫描范围
    }

# ============================================================================
# 键盘管理器
# ============================================================================

class KeyboardManager:
    """
    键盘输入管理器
    负责监听和处理键盘事件，提供线程安全的按键状态查询

    功能特点：
    1. 异步监听：不阻塞主线程
    2. 状态缓存：记录按键按下/释放状态
    3. 多键支持：同时处理多个按键
    4. 防抖处理：避免重复触发
    """

    def __init__(self):
        """初始化键盘管理器"""
        # 按键状态字典：key -> bool（是否按下）
        self.keys = {
            keyboard.KeyCode.from_char('r'): False,  # 复位键
            keyboard.KeyCode.from_char('d'): False,  # 调试模式切换键
            keyboard.KeyCode.from_char('s'): False,  # 保存记忆键
            keyboard.Key.space: False,               # 空格键（强制截停）
            keyboard.Key.shift: False,               # Shift键（加速）
            keyboard.Key.shift_l: False,             # 左Shift键
            keyboard.Key.shift_r: False,             # 右Shift键
        }
        self.listener = None  # 键盘监听器对象
        self._start_listener()  # 启动监听

    def _start_listener(self):
        """
        启动键盘监听器（私有方法）
        创建并启动后台线程监听键盘事件
        """
        def on_press(key):
            """
            按键按下回调函数
            key: 按下的键对象
            """
            if key in self.keys:
                self.keys[key] = True
            # 处理Shift键变体：左Shift和右Shift都映射到Shift
            elif isinstance(key, keyboard.Key) and key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
                self.keys[keyboard.Key.shift] = True

        def on_release(key):
            """
            按键释放回调函数
            key: 释放的键对象
            """
            if key in self.keys:
                self.keys[key] = False
            # 处理Shift键变体
            elif isinstance(key, keyboard.Key) and key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
                self.keys[keyboard.Key.shift] = False

        # 创建并启动监听器（守护线程，主程序退出时自动结束）
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.daemon = True
        self.listener.start()

    def is_pressed(self, key):
        """
        检查按键是否处于按下状态

        参数：
        key: 要检查的按键

        返回：
        bool: True表示按键按下，False表示未按下
        """
        return self.keys.get(key, False)

    def reset_key(self, key):
        """
        重置按键状态（用于单次触发后清除状态）

        参数：
        key: 要重置的按键
        """
        if key in self.keys:
            self.keys[key] = False

# ============================================================================
# 路径记忆系统
# ============================================================================

class PathMemory:
    """
    增强版路径记忆与学习系统
    基于强化学习的路径记忆，通过经验积累优化决策

    核心算法：
    1. Q-learning变体：状态(位置+方向) -> 动作价值
    2. ε-贪心策略：平衡探索与利用
    3. 经验回放：存储并重用历史经验

    数据结构：
    - memory: 经验回放缓冲区（双端队列）
    - path_scores: 路径评分字典（Q值表）
    - obstacle_history: 障碍物位置记录
    - successful_paths: 成功路径历史
    """

    def __init__(self, memory_size: int = Config.PATH_MEMORY_SIZE):
        """
        初始化路径记忆系统

        参数：
        memory_size: 经验记忆容量
        """
        self.memory: Deque[PathExperience] = deque(maxlen=memory_size)
        self.path_scores: Dict[str, float] = {}  # 路径评分（Q值）
        self.obstacle_history: Dict[str, ObstacleRecord] = {}  # 障碍物记录
        self.successful_paths: List[Dict] = []  # 成功路径列表
        self.debug_mode = False  # 调试模式标志
        self.learning_rate = Config.LEARNING_RATE  # 学习率

    def add_experience(self, position: np.ndarray, direction: str,
                      success: bool, distance_traveled: float) -> None:
        """
        添加并学习路径经验（核心学习函数）

        参数：
        position: 当前位置坐标
        direction: 选择的方向
        success: 是否成功（是否撞到障碍物）
        distance_traveled: 行驶距离

        算法步骤：
        1. 创建状态键（位置+方向）
        2. 计算奖励（成功+1，失败-0.5）
        3. 更新Q值：Q_new = Q_old + α * (reward - Q_old)
        4. 存储经验到回放缓冲区
        """
        # 创建唯一状态键（网格化位置+方向）
        key = self._create_key(position, direction)

        # 强化学习更新：使用时间差分(TD)更新
        reward = Config.PATH_REWARD if success else Config.PATH_PENALTY
        current_score = self.path_scores.get(key, 0)
        # Q-learning更新公式：Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
        new_score = current_score + self.learning_rate * (reward - current_score)
        self.path_scores[key] = new_score

        # 记录经验到回放缓冲区
        experience = PathExperience(
            position=tuple(position[:2]),  # 只记录x,y坐标（忽略高度）
            direction=direction,
            success=success,
            distance=distance_traveled,
            timestamp=time.time()
        )
        self.memory.append(experience)

        # 调试输出
        if self.debug_mode:
            status = "✓" if success else "✗"
            print(f"路径经验: {direction} {status}, 评分: {new_score:.2f}")

    def get_best_direction(self, position: np.ndarray,
                          available_directions: List[str]) -> str:
        """
        基于历史经验获取最佳方向（决策函数）

        参数：
        position: 当前位置
        available_directions: 可行的方向列表

        返回：
        str: 最佳方向名称

        决策策略：
        1. ε-贪心策略：以EXPLORATION_RATE概率随机探索
        2. 利用策略：选择综合评分最高的方向
        3. 综合评分 = 0.6*基础分 + 0.4*记忆分
        """
        # 探索策略：随机选择一个方向（鼓励探索新路径）
        if random.random() < Config.EXPLORATION_RATE:
            return random.choice(available_directions)

        # 利用策略：选择综合得分最高的方向
        best_direction = None
        best_score = -float('inf')  # 初始化为负无穷

        for direction in available_directions:
            # 获取该方向的历史评分（Q值）
            key = self._create_key(position, direction)
            base_score = Config.DIRECTION_SCORES.get(direction, 0.5)
            memory_score = self.path_scores.get(key, 0)

            # 综合得分：加权平均（可调整权重）
            total_score = base_score * 0.6 + memory_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_direction = direction

        # 如果所有方向都没有评分，随机选择一个
        return best_direction or random.choice(available_directions)

    def record_obstacle(self, obstacle_name: str, position: np.ndarray) -> None:
        """
        记录障碍物位置（环境建模）

        参数：
        obstacle_name: 障碍物名称
        position: 障碍物位置

        功能：
        1. 记录障碍物位置和时间
        2. 统计遇到次数
        3. 用于后续路径规划时避开已知障碍
        """
        # 创建障碍物键（名称+网格化位置）
        key = f"{obstacle_name}_{int(position[0]*10)}_{int(position[1]*10)}"

        # 更新或创建障碍物记录
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
        """
        检查位置附近是否有近期遇到的障碍物

        参数：
        position: 要检查的位置
        threshold: 距离阈值（米）
        time_window: 时间窗口（秒）

        返回：
        bool: True表示附近有近期障碍物

        用途：避免重复尝试已知的危险路径
        """
        current_time = time.time()

        # 遍历所有障碍物记录
        for record in self.obstacle_history.values():
            obs_pos = record.position
            # 计算欧几里得距离
            distance = math.dist(obs_pos, position[:2])

            # 检查是否在时空范围内
            if (distance < threshold and
                (current_time - record.timestamp) < time_window):
                return True

        return False

    def save_to_file(self, filename: str = "path_memory.json") -> None:
        """
        保存路径记忆到JSON文件

        参数：
        filename: 文件名

        保存内容：
        1. 路径评分表
        2. 障碍物历史记录
        3. 最近的成功路径
        4. 时间戳
        """
        # 准备保存数据
        save_data = {
            'path_scores': self.path_scores,
            'obstacle_history': {k: vars(v) for k, v in self.obstacle_history.items()},
            'successful_paths': self.successful_paths[-10:],  # 只保存最近10条
            'timestamp': time.time()
        }

        # 写入文件
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"✅ 路径记忆已保存到 {filename}")

    def load_from_file(self, filename: str = "path_memory.json") -> bool:
        """
        从JSON文件加载路径记忆

        参数：
        filename: 文件名

        返回：
        bool: 是否成功加载

        异常处理：文件不存在或格式错误时不影响程序运行
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # 恢复数据
            self.path_scores = data.get('path_scores', {})

            # 恢复障碍物记录（需要特殊处理dataclass）
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
        """
        创建记忆键（私有方法）

        参数：
        position: 位置坐标
        direction: 方向

        返回：
        str: 格式为"x_y_direction"的键

        设计说明：
        1. 网格化：将连续位置离散化为10cm网格
        2. 简化：忽略高度信息
        3. 可读：键格式便于调试
        """
        x, y = int(position[0] * 10), int(position[1] * 10)  # 10倍放大，取整
        return f"{x}_{y}_{direction}"

    def toggle_debug(self) -> None:
        """
        切换调试模式

        功能：
        1. 开启/关闭调试信息输出
        2. 显示学习过程中的评分变化
        3. 帮助调试路径选择逻辑
        """
        self.debug_mode = not self.debug_mode
        status = "开启" if self.debug_mode else "关闭"
        print(f"🔧 调试模式: {status}")

# ============================================================================
# 小车控制器
# ============================================================================

class CarController:
    """
    小车运动控制器
    负责与MuJoCo仿真引擎交互，控制小车运动

    功能模块：
    1. 位置/速度获取
    2. 电机控制（转向、驱动）
    3. 障碍物检测
    4. 紧急停止
    """

    def __init__(self, model, data, config: Config):
        """
        初始化小车控制器

        参数：
        model: MuJoCo模型对象
        data: MuJoCo数据对象
        config: 配置参数对象
        """
        self.model = model
        self.data = data
        self.config = config

        # 获取车身ID（用于位置查询）
        self.chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # 预加载障碍物ID（提高检测效率）
        self.obstacle_ids = self._load_obstacle_ids()

    def _load_obstacle_ids(self) -> Dict[str, int]:
        """
        加载障碍物ID（私有方法）

        返回：
        Dict[str, int]: 障碍物名称到ID的映射

        优化：提前加载，避免每次检测时都查询
        """
        # 障碍物名称列表（与XML文件中的名称对应）
        obstacle_names = [
            'obs_box1', 'obs_box2', 'obs_box3', 'obs_box4',
            'obs_ball1', 'obs_ball2', 'obs_ball3',
            'wall1', 'wall2', 'front_dark_box'
        ]

        ids = {}
        for name in obstacle_names:
            obs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if obs_id != -1:  # -1表示未找到
                ids[name] = obs_id

        return ids

    def get_position(self) -> np.ndarray:
        """
        获取小车当前位置

        返回：
        np.ndarray: 三维位置向量[x, y, z]
        """
        return self.data.body(self.chassis_id).xpos.copy()

    def get_velocity(self) -> float:
        """
        获取小车当前速度

        返回：
        float: 速度大小（m/s）
        """
        return np.linalg.norm(self.data.qvel[:3])

    def set_control(self, steer_angle: float = 0.0,
                   speed: float = 0.0, all_wheels: bool = True) -> None:
        """
        设置小车控制参数（核心控制函数）

        参数：
        steer_angle: 转向角度（弧度）
        speed: 驱动速度（m/s）
        all_wheels: 是否所有轮子都驱动（True=四驱，False=前驱）

        控制映射：
        ctrl[0]: 左前轮转向
        ctrl[1]: 右前轮转向
        ctrl[2]: 左前轮驱动
        ctrl[3]: 右前轮驱动
        ctrl[4]: 左后轮驱动
        ctrl[5]: 右后轮驱动
        """
        # 转向控制（前轮转向）
        self.data.ctrl[0] = steer_angle
        self.data.ctrl[1] = steer_angle

        # 速度控制（驱动电机）
        if all_wheels:
            # 四轮驱动
            self.data.ctrl[2] = speed
            self.data.ctrl[3] = speed
            self.data.ctrl[4] = speed
            self.data.ctrl[5] = speed
        else:
            # 仅前轮驱动
            self.data.ctrl[2] = speed
            self.data.ctrl[3] = speed

    def emergency_stop(self) -> None:
        """
        紧急停止
        将所有控制信号设置为0，立即停止小车
        """
        for i in range(len(self.data.ctrl)):
            self.data.ctrl[i] = 0.0

    def check_obstacle(self, direction_angle: float = 0,
                      scan_width: float = 0.3) -> Tuple[int, float, Optional[str], Optional[np.ndarray]]:
        """
        检测指定方向的障碍物（核心检测函数）

        参数：
        direction_angle: 检测方向角度（相对于前进方向）
        scan_width: 扫描宽度（扇形区域的宽度）

        返回：
        tuple: (状态码, 最小距离, 最近障碍物名称, 障碍物位置)
        状态码：
        0 = 无障碍物
        1 = 有障碍物但在安全距离外
        2 = 障碍物在安全距离内（危险）

        算法原理：
        1. 计算检测方向向量
        2. 遍历所有障碍物，计算相对位置
        3. 判断是否在检测扇形区域内
        4. 返回最近的障碍物信息
        """
        chassis_pos = self.get_position()

        # 获取前进方向向量（归一化）
        velocity = self.data.qvel[:2]  # 只考虑xy平面
        if np.linalg.norm(velocity) < 0.0001:  # 近似静止
            forward = np.array([1.0, 0.0])  # 默认朝x轴正方向
        else:
            forward = velocity / np.linalg.norm(velocity)

        # 应用方向旋转（计算检测方向）
        if direction_angle != 0:
            cos_a, sin_a = math.cos(direction_angle), math.sin(direction_angle)
            # 二维旋转矩阵：[cosθ -sinθ; sinθ cosθ]
            forward = np.array([
                forward[0] * cos_a - forward[1] * sin_a,
                forward[0] * sin_a + forward[1] * cos_a
            ])

        # 初始化检测结果
        min_distance = float('inf')
        closest_obstacle = None
        obstacle_pos = None

        # 遍历所有障碍物
        for obs_name, obs_id in self.obstacle_ids.items():
            obs_pos = self.data.body(obs_id).xpos
            rel_pos = obs_pos[:2] - chassis_pos[:2]  # 相对位置向量
            distance = np.linalg.norm(rel_pos)  # 欧几里得距离

            # 距离过滤：只考虑扫描范围内的障碍物
            if 0 < distance < self.config.SCAN_RANGE:
                obs_dir = rel_pos / distance  # 障碍物方向向量（归一化）

                # 计算夹角（点积公式）
                dot_product = np.dot(obs_dir, forward)
                dot_product = np.clip(dot_product, -1.0, 1.0)  # 数值稳定性
                angle_diff = math.acos(dot_product)

                # 计算横向距离（叉积的z分量）
                cross_z = np.cross([forward[0], forward[1], 0],
                                  [obs_dir[0], obs_dir[1], 0])[2]
                lateral_dist = abs(cross_z) * distance

                # 判断是否在检测区域内（扇形区域）
                # 条件1：夹角小于45°
                # 条件2：横向距离小于扫描宽度
                if angle_diff < math.radians(45) and lateral_dist < scan_width:
                    if distance < min_distance:  # 找到更近的障碍物
                        min_distance = distance
                        closest_obstacle = obs_name
                        obstacle_pos = obs_pos.copy()

        # 返回检测结果
        if closest_obstacle is not None:
            if min_distance < self.config.SAFE_DISTANCE:
                return 2, min_distance, closest_obstacle, obstacle_pos
            else:
                return 1, min_distance, closest_obstacle, obstacle_pos

        # 无障碍物
        return 0, 0, None, None

# ============================================================================
# 路径规划器
# ============================================================================

class PathPlanner:
    """
    智能路径规划器
    负责评估各方向的安全性并选择最佳路径

    规划流程：
    1. 扫描所有预定义方向
    2. 评估每个方向的安全性得分
    3. 结合历史经验计算综合得分
    4. 选择得分最高的安全方向
    """

    def __init__(self, controller: CarController, memory: PathMemory):
        """
        初始化路径规划器

        参数：
        controller: 小车控制器
        memory: 路径记忆系统
        """
        self.controller = controller
        self.memory = memory
        self.config = Config()  # 本地配置副本

    def scan_directions(self) -> Dict[str, DirectionInfo]:
        """
        扫描所有可能方向

        返回：
        Dict[str, DirectionInfo]: 方向名称到方向信息的映射

        扫描逻辑：
        1. 对每个预定义方向进行障碍物检测
        2. 计算安全得分（基于障碍物距离）
        3. 结合基础得分和记忆得分
        4. 返回完整的评分信息
        """
        directions_info = {}

        # 遍历所有预定义方向
        for dir_name, dir_angle in self.config.DIRECTIONS.items():
            # 确定扫描宽度（急转向用较宽范围）
            scan_width = (self.config.SCAN_WIDTHS["sharp"]
                         if "sharp" in dir_name
                         else self.config.SCAN_WIDTHS["default"])

            # 检测该方向的障碍物
            status, distance, obs_name, _ = self.controller.check_obstacle(
                dir_angle, scan_width
            )

            # 计算安全得分（基于障碍物状态和距离）
            if status == 0:  # 无障碍物
                safety_score = 1.0
            elif status == 1 and distance > 0.5:  # 有障碍物但距离较远
                safety_score = 0.6
            else:  # 有近距离障碍物
                safety_score = 0.2

            # 基础得分（方向偏好）
            base_score = self.config.DIRECTION_SCORES.get(dir_name, 0.5)

            # 记忆得分（历史经验）
            memory_score = 0
            pos = self.controller.get_position()
            # 主要方向（直行、轻微转向）有记忆评分
            if dir_name in ["forward", "slight_left", "slight_right"]:
                key = self.memory._create_key(pos, dir_name)
                memory_score = self.memory.path_scores.get(key, 0)

            # 综合得分（加权平均）
            total_score = base_score * 0.4 + safety_score * 0.4 + memory_score * 0.2

            # 存储方向信息
            directions_info[dir_name] = DirectionInfo(
                angle=dir_angle,
                status=status,
                distance=distance,
                obstacle=obs_name,
                score=total_score
            )

        return directions_info

    def choose_best_path(self) -> Tuple[str, str]:
        """
        智能选择最佳路径（核心决策函数）

        返回：
        tuple: (方向名称, 方向描述文本)

        决策流程：
        1. 扫描环境获取所有方向信息
        2. 筛选安全方向（无障碍或距离足够远）
        3. 若无安全方向，选择障碍物最远的方向
        4. 使用记忆系统选择最佳方向
        5. 生成描述文本
        """
        # 步骤1：扫描环境
        directions_info = self.scan_directions()
        position = self.controller.get_position()

        # 步骤2：筛选安全方向
        safe_directions = [
            dir_name for dir_name, info in directions_info.items()
            if info.status == 0 or (info.status == 1 and info.distance > 0.5)
        ]

        # 步骤3：无安全方向时的应急处理
        if not safe_directions:
            # 选择障碍物最远的方向（"最小化危险"策略）
            best_dir = max(directions_info.items(),
                          key=lambda x: x[1].distance)[0]
            dist = directions_info[best_dir].distance
            return best_dir, f"强制{best_dir}(距离:{dist:.2f}m)"

        # 步骤4：使用记忆系统选择最佳方向
        best_direction = self.memory.get_best_direction(position, safe_directions)
        info = directions_info[best_direction]

        # 步骤5：生成用户友好的描述文本
        if best_direction == "forward":
            desc = "直行"
        elif best_direction == "backward":
            desc = "后退"
        else:
            angle_deg = math.degrees(info.angle)
            direction = "左" if "left" in best_direction else "右"
            desc = f"{direction}转{abs(angle_deg):.0f}度"

        return best_direction, desc

# ============================================================================
# 主控制系统
# ============================================================================

class PatrolSystem:
    """
    主控制系统
    集成所有模块，实现完整的自动驾驶逻辑

    系统架构：
    ┌─────────────────────────────────────┐
    │           PatrolSystem             │
    ├─────────────┬─────────┬────────────┤
    │ Keyboard    │ Car     │ Path       │
    │ Manager     │ Controller │ Planner   │
    ├─────────────┼─────────┼────────────┤
    │ Path Memory │ Config  │ State      │
    │ System      │         │ Machine    │
    └─────────────┴─────────┴────────────┘

    状态机设计：
    CRUISING → DECELERATING → STOPPED → PATH_PLANNING → TURNING
        ↑           ↓            ↑           ↓            ↓
        └───────────┴────────────┴───────────┴────────────┘
    """

    def __init__(self, model_path: str = "wheeled_car.xml"):
        """
        初始化主控制系统

        参数：
        model_path: MuJoCo模型文件路径
        """
        # ------------------ 初始化MuJoCo ------------------
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # ------------------ 初始化子系统 ------------------
        self.config = Config()  # 配置参数
        self.keyboard = KeyboardManager()  # 键盘管理器
        self.controller = CarController(self.model, self.data, self.config)  # 运动控制器
        self.memory = PathMemory()  # 路径记忆系统
        self.planner = PathPlanner(self.controller, self.memory)  # 路径规划器

        # ------------------ 状态变量 ------------------
        self.state = CarState.CRUISING  # 当前状态
        self.previous_state = None  # 强制截停前的状态（用于恢复）

        # ------------------ 控制变量 ------------------
        self.turn_counter = 0  # 转向计时器
        self.turn_angle = 0  # 目标转向角度
        self.turn_direction = ""  # 转向描述文本
        self.scan_counter = 0  # 路径验证扫描计数器
        self.deceleration_counter = 0  # 减速计时器
        self.backup_counter = 0  # 后退计时器

        # ------------------ 速度管理 ------------------
        self.is_boosting = False  # 是否正在加速
        self.current_cruise_speed = self.config.BASE_CRUISE_SPEED  # 当前巡航速度
        self.current_turn_speed = self.config.BASE_CRUISE_SPEED * self.config.TURN_SPEED_RATIO  # 当前转向速度

        # ------------------ 路径历史 ------------------
        self.path_history = []  # 路径选择历史记录
        self.last_success_pos = self.controller.get_position()  # 最后成功位置
        self.distance_since_obstacle = 0.0  # 上次遇到障碍物后的行驶距离

        # ------------------ 加载记忆 ------------------
        self.memory.load_from_file()

    def reset(self) -> None:
        """
        复位小车（重置所有状态）

        功能：
        1. 重置MuJoCo仿真状态
        2. 重置所有控制变量
        3. 重置速度和路径记录
        4. 恢复到初始巡航状态
        """
        # 重置MuJoCo仿真
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.03  # 设置离地高度（防止陷入地面）

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
        """
        更新当前速度参数（私有方法）

        根据加速状态计算实际速度：
        1. 正常状态：使用基础速度
        2. 加速状态：速度乘以加速倍率
        """
        multiplier = self.config.BOOST_MULTIPLIER if self.is_boosting else 1.0
        self.current_cruise_speed = self.config.BASE_CRUISE_SPEED * multiplier
        self.current_turn_speed = (self.config.BASE_CRUISE_SPEED *
                                  self.config.TURN_SPEED_RATIO * multiplier)

    def toggle_emergency_stop(self) -> None:
        """
        切换强制截停状态

        功能：
        1. 第一次按空格：进入强制截停，保存当前状态
        2. 第二次按空格：退出强制截停，恢复之前状态
        """
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
        """
        更新路径历史记录

        参数：
        direction: 选择的方向
        success: 是否成功

        功能：
        1. 记录每次路径选择
        2. 更新行驶距离
        3. 当成功行驶一定距离后，记录为成功路径
        """
        current_pos = self.controller.get_position()

        # 添加历史记录
        self.path_history.append({
            'direction': direction,
            'position': tuple(current_pos[:2]),
            'success': success,
            'time': time.time()
        })

        # 限制历史记录长度（避免内存无限增长）
        if len(self.path_history) > 20:
            self.path_history = self.path_history[-20:]

        # 更新行驶距离（近似计算：速度×时间）
        if success:
            # 0.002是仿真步长，这里近似计算每步行驶距离
            self.distance_since_obstacle += self.controller.get_velocity() * 0.002

        # 记录成功路径（当成功行驶超过1米时）
        if success and self.distance_since_obstacle > 1.0:
            directions = [h['direction'] for h in self.path_history[-5:]]  # 最近5次方向
            self.memory.successful_paths.append({
                'start': self.last_success_pos[:2],
                'end': current_pos[:2],
                'directions': directions,
                'timestamp': time.time()
            })
            self.last_success_pos = current_pos.copy()
            self.distance_since_obstacle = 0.0

    # ==================== 状态处理函数 ====================
    # 以下函数按状态机设计模式实现，每个函数处理一个特定状态

    def handle_cruising(self) -> None:
        """处理巡航状态（正常前进）"""
        # 检测前方障碍物
        status, distance, obs_name, obs_pos = self.controller.check_obstacle()

        if status == 2:  # 紧急障碍（距离过近）
            self.state = CarState.STOPPED
            print(f"\n⚠️ 紧急停止！障碍物距离: {distance:.2f}m")

            # 记录障碍物
            if obs_pos is not None:
                self.memory.record_obstacle(obs_name, obs_pos)

            # 记录失败经验
            self.memory.add_experience(
                self.controller.get_position(),
                "forward",
                False,
                self.distance_since_obstacle
            )

            self.controller.emergency_stop()

        elif status == 1:  # 检测到障碍物（但还有安全距离）
            self.state = CarState.DECELERATING
            self.deceleration_counter = 0
            print(f"\n⚠️ 检测到障碍物: {obs_name}({distance:.2f}m)，开始减速...")

            # 记录障碍物
            if obs_pos is not None:
                self.memory.record_obstacle(obs_name, obs_pos)

            # 记录失败经验
            self.memory.add_experience(
                self.controller.get_position(),
                "forward",
                False,
                self.distance_since_obstacle
            )

        else:  # 无障碍物，正常巡航
            self.controller.set_control(
                speed=self.current_cruise_speed,
                all_wheels=True
            )
            self.update_path_history("forward", True)

    def handle_decelerating(self) -> None:
        """处理减速状态（平滑减速到停止）"""
        self.deceleration_counter += 1
        # 计算减速进度（0.0到1.0）
        progress = min(1.0, self.deceleration_counter / 15.0)
        # 线性减速：速度从当前速度降到0
        current_speed = self.current_cruise_speed * (1.0 - progress)

        self.controller.set_control(speed=current_speed)

        # 减速完成，进入停止状态
        if self.deceleration_counter > 20:
            self.state = CarState.STOPPED
            print("减速完成，准备规划路径")
            self.turn_counter = 0

    def handle_stopped(self) -> None:
        """处理停止状态（等待后开始路径规划）"""
        self.turn_counter += 1
        self.controller.emergency_stop()  # 确保完全停止

        # 短暂等待后开始路径规划
        if self.turn_counter > 10:
            print("正在智能规划路径...")
            self.state = CarState.PATH_PLANNING
            self.turn_counter = 0

    def handle_path_planning(self) -> None:
        """处理路径规划状态（选择最佳绕障路径）"""
        # 使用路径规划器选择最佳方向
        chosen_direction, direction_text = self.planner.choose_best_path()

        if chosen_direction == "backward":
            # 需要后退（所有方向都不安全）
            print("路径受阻，执行后退操作")
            self.state = CarState.BACKING_UP
            self.backup_counter = 0
        else:
            # 找到可行路径
            self.turn_angle = self.config.DIRECTIONS[chosen_direction]
            self.turn_direction = direction_text
            print(f"选择路径: {self.turn_direction}")
            self.state = CarState.TURNING
            self.turn_counter = 0

    def handle_backing_up(self) -> None:
        """处理后撤状态（缓慢后退一定距离）"""
        if self.backup_counter < 40:
            # 后退速度为转向速度的40%（负值表示后退）
            speed = -self.current_turn_speed * 0.4
            self.controller.set_control(speed=speed)
            self.backup_counter += 1
        else:
            # 后退完成
            self.controller.emergency_stop()
            print("后退完成，重新规划路径")
            self.state = CarState.PATH_PLANNING
            self.update_path_history("backward", True)

    def handle_turning(self) -> None:
        """处理转向状态（执行转向操作）"""
        self.turn_counter += 1
        # 计算转向进度（0.0到1.0）
        progress = min(1.0, self.turn_counter / 8.0)

        # 渐进转向：角度从0线性增加到目标角度
        current_angle = self.turn_angle * progress
        self.controller.set_control(steer_angle=current_angle)

        # 转向5步后开始缓慢加速
        if self.turn_counter > 5:
            speed_progress = min(1.0, (self.turn_counter - 5) / 15.0)
            current_speed = self.current_turn_speed * speed_progress
            self.controller.set_control(
                steer_angle=current_angle,
                speed=current_speed
            )

        # 定期显示转向进度
        if self.turn_counter % 15 == 0:
            print(f"正在{self.turn_direction}，进度: {progress*100:.0f}%")

        # 转向完成，进入路径验证
        if self.turn_counter > self.config.TURN_DURATION:
            print(f"{self.turn_direction}完成，开始验证路径...")
            self.state = CarState.PATH_VERIFICATION
            self.turn_counter = 0
            self.scan_counter = 0

    def handle_path_verification(self) -> None:
        """处理路径验证状态（验证转向后路径是否安全）"""
        self.scan_counter += 1

        # 保持转向角度，低速前进以验证路径
        self.controller.set_control(
            steer_angle=self.turn_angle * 0.5,  # 稍微减小转向角度
            speed=self.current_turn_speed * 0.6  # 低速验证
        )

        # 定期检查前方是否安全
        if self.scan_counter % 10 == 0:
            status, distance, obs_name, _ = self.controller.check_obstacle()

            if status == 0:  # 路径安全
                print("路径验证通过，准备恢复巡航")

                # 记录成功经验
                for dir_name, angle in self.config.DIRECTIONS.items():
                    if abs(angle - self.turn_angle) < 0.01:  # 找到对应的方向名称
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

        # 验证超时处理
        if self.scan_counter > 40:
            print("路径验证超时，尝试恢复巡航")
            self.state = CarState.RESUME
            self.turn_counter = 0

    def handle_resume(self) -> None:
        """处理恢复巡航状态（从转向状态平滑恢复到正常巡航）"""
        self.turn_counter += 1
        progress = min(1.0, self.turn_counter / 15.0)

        # 渐进恢复：转向角度归零，速度增加到巡航速度
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

            # 检查前方是否安全
            status, _, _, _ = self.controller.check_obstacle()
            if status == 0:  # 安全，恢复巡航
                print("成功恢复巡航")
                self.state = CarState.CRUISING
                self.turn_counter = 0

                # 记录路径历史
                for dir_name, angle in self.config.DIRECTIONS.items():
                    if abs(angle - self.turn_angle) < 0.01:
                        self.update_path_history(dir_name, True)
                        break
            else:  # 不安全，重新处理
                print("恢复巡航时检测到障碍物，重新处理")
                self.state = CarState.STOPPED
                self.turn_counter = 0

    # ==================== 主循环函数 ====================

    def run(self) -> None:
        """
        运行主循环（程序入口点）

        主循环流程：
        1. 初始化MuJoCo视图器
        2. 显示控制说明
        3. 进入主循环：
           a. 处理键盘输入
           b. 更新速度参数
           c. 根据状态执行相应处理
           d. 执行仿真步
           e. 显示状态信息
           f. 同步视图
        4. 清理退出
        """
        # 启动MuJoCo被动视图器（非阻塞模式）
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 设置相机视角
            viewer.cam.distance = 2.5  # 相机距离
            viewer.cam.elevation = -25  # 相机俯角

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
                # 主循环
                while viewer.is_running():
                    # 1. 处理键盘输入
                    self._handle_keyboard()

                    # 2. 更新速度参数（根据加速状态）
                    self._update_speeds()

                    # 3. 状态处理
                    if self.state == CarState.EMERGENCY_STOP:
                        # 强制截停状态：只执行紧急停止
                        self.controller.emergency_stop()
                    else:
                        # 正常状态：执行状态机处理
                        self._handle_state()

                    # 4. 执行仿真步（推进物理仿真）
                    mujoco.mj_step(self.model, self.data)

                    # 5. 显示状态信息
                    self._display_status()

                    # 6. 同步视图（更新显示）
                    viewer.sync()

            except KeyboardInterrupt:
                # 用户中断（Ctrl+C）
                print("\n\n⚠️ 用户中断程序")
            finally:
                # 清理退出
                print("\n保存路径记忆...")
                self.memory.save_to_file()
                print("程序结束")

    def _handle_keyboard(self) -> None:
        """
        处理键盘输入（私有方法）

        检查所有功能键的状态并执行相应操作
        使用reset_key避免重复触发
        """
        # R键：复位
        if self.keyboard.is_pressed(keyboard.KeyCode.from_char('r')):
            self.reset()
            self.keyboard.reset_key(keyboard.KeyCode.from_char('r'))

        # D键：切换调试模式
        if self.keyboard.is_pressed(keyboard.KeyCode.from_char('d')):
            self.memory.toggle_debug()
            self.keyboard.reset_key(keyboard.KeyCode.from_char('d'))

        # S键：保存记忆
        if self.keyboard.is_pressed(keyboard.KeyCode.from_char('s')):
            self.memory.save_to_file()
            self.keyboard.reset_key(keyboard.KeyCode.from_char('s'))

        # 空格键：强制截停/恢复
        if self.keyboard.is_pressed(keyboard.Key.space):
            self.toggle_emergency_stop()
            self.keyboard.reset_key(keyboard.Key.space)

        # Shift键：更新加速状态（按住生效）
        self.is_boosting = self.keyboard.is_pressed(keyboard.Key.shift)

    def _handle_state(self) -> None:
        """
        状态机分发器（私有方法）

        根据当前状态调用相应的处理函数
        使用字典映射避免复杂的if-elif链
        """
        # 状态处理函数映射表
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

        # 查找并执行对应的处理函数
        handler = state_handlers.get(self.state)
        if handler:
            handler()

    def _display_status(self) -> None:
        """
        显示状态信息（私有方法）

        在控制台显示单行状态信息，使用回车符\r实现原地更新
        信息包括：状态、速度、转向、历史记录、加速状态等
        """
        # 获取当前速度和转向角度
        vel = self.controller.get_velocity()
        steer = (self.data.ctrl[0] + self.data.ctrl[1]) / 2

        # 构建信息部分列表
        info_parts = [
            f"状态: {self.state.value}",
            f"速度: {vel:7.5f} m/s",
        ]

        # 添加转向信息（如果有明显转向）
        if abs(steer) > 0.01:
            info_parts.append(f"转向: {math.degrees(steer):.1f}°")

        # 添加系统信息
        info_parts.extend([
            f"路径历史: {len(self.path_history)}",
            f"路径记忆: {len(self.memory.memory)}",
        ])

        # 添加加速状态
        if self.is_boosting:
            info_parts.append(f"加速: {self.config.BOOST_MULTIPLIER}倍")

        # 调试信息（仅在调试模式且巡航状态下显示障碍物信息）
        if (self.memory.debug_mode and
            self.state == CarState.CRUISING):
            status, distance, obs_name, _ = self.controller.check_obstacle()
            if status > 0 and obs_name:
                info_parts.append(f"障碍: {obs_name}({distance:.2f}m)")

        # 组合并输出状态行（\r回到行首，实现原地更新）
        status_line = ", ".join(info_parts)
        print(f"\r{status_line}", end='', flush=True)

# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """
    主程序入口函数

    职责：
    1. 创建主控制系统实例
    2. 运行主循环
    3. 捕获并处理异常
    4. 确保资源正确清理
    """
    try:
        # 创建主控制系统实例
        system = PatrolSystem("wheeled_car.xml")
        # 运行主循环
        system.run()
    except Exception as e:
        # 异常处理：显示错误信息并打印堆栈跟踪
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()

# Python标准入口点
if __name__ == "__main__":
    main()