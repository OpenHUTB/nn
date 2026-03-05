#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂控制器（完整版）
核心特性：智能避障+安全防护+远程交互+离线分析+自适应控制
"""

import sys
import time
import signal
import threading
import json
import numpy as np
import mujoco
import socket
import websockets
import asyncio
import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import splprep, splev
from queue import Queue

# ====================== 全局配置与常量 ======================
# 硬件参数
JOINT_COUNT = 5
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
MIN_LOAD, MAX_LOAD = 0.0, 2.0

# 关节极限
JOINT_LIMITS = np.array([[-np.pi, np.pi], [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2],
                         [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2]], dtype=np.float64)
MAX_VEL = np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64)
MAX_ACC = np.array([2.0, 1.6, 1.6, 1.2, 1.2], dtype=np.float64)
MAX_TORQUE = np.array([15.0, 12.0, 10.0, 8.0, 5.0], dtype=np.float64)

# 时间配置
SIM_DT = 0.0005
CTRL_FREQ = 2000
CTRL_DT = 1.0 / CTRL_FREQ
FPS = 60
SLEEP_DT = 1.0 / FPS

# 碰撞检测阈值
COLLISION_THRESHOLD = 0.01
COLLISION_FORCE_THRESHOLD = 5.0

# 安全阈值
SAFETY_THRESHOLDS = {
    "max_joint_error": DEG2RAD(5.0),  # 5度
    "max_torque_ratio": 0.9,  # 最大扭矩90%
    "max_load_ratio": 0.9,  # 最大负载90%
    "max_velocity_ratio": 0.95,  # 最大速度95%
}

# 网络配置
WS_HOST = "0.0.0.0"
WS_PORT = 8765
WS_BUFFER_SIZE = 1024

# 目录配置
DIR_CONFIG = {
    "trajectories": Path("trajectories"),
    "params": Path("params"),
    "logs": Path("logs"),
    "data": Path("data"),
    "replay": Path("replay")
}
for dir_path in DIR_CONFIG.values():
    dir_path.mkdir(exist_ok=True)


# ====================== 配置管理模块 ======================
@dataclass
class ControlConfig:
    """控制参数配置类"""
    # 基础控制
    kp_base: float = 120.0
    kd_base: float = 8.0
    kp_load_gain: float = 1.8
    kd_load_gain: float = 1.5
    ff_vel: float = 0.7
    ff_acc: float = 0.5

    # 误差补偿
    backlash: np.ndarray = np.array([0.001, 0.001, 0.002, 0.002, 0.003])
    friction: np.ndarray = np.array([0.1, 0.08, 0.08, 0.06, 0.06])
    gravity_comp: bool = True

    # 刚度阻尼
    stiffness_base: np.ndarray = np.array([200.0, 180.0, 150.0, 120.0, 80.0])
    stiffness_load_gain: float = 1.8
    stiffness_error_gain: float = 1.5
    stiffness_min: np.ndarray = np.array([100.0, 90.0, 75.0, 60.0, 40.0])
    stiffness_max: np.ndarray = np.array([300.0, 270.0, 225.0, 180.0, 120.0])
    damping_ratio: float = 0.04

    # 轨迹平滑
    smooth_factor: float = 0.1
    jerk_limit: np.ndarray = np.array([10.0, 8.0, 8.0, 6.0, 6.0])

    # 智能避障
    obstacle_margin: float = 0.02  # 避障余量2cm
    path_resolution: float = 0.01  # 路径分辨率1cm
    astar_heuristic_weight: float = 1.0

    def to_dict(self):
        """转换为字典"""
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    @classmethod
    def from_dict(cls, data):
        """从字典加载"""
        data = data.copy()
        array_keys = ['backlash', 'friction', 'stiffness_base', 'stiffness_min',
                      'stiffness_max', 'jerk_limit']
        for key in array_keys:
            if key in data and isinstance(data[key], list):
                data[key] = np.array(data[key], dtype=np.float64)
        return cls(**data)


# 全局配置实例
CFG = ControlConfig()


# ====================== 工具函数模块 ======================
class Utils:
    """工具函数类"""
    _lock = threading.Lock()
    _perf_metrics = {"ctrl_time": deque(maxlen=1000), "step_time": deque(maxlen=1000)}

    @classmethod
    @contextmanager
    def lock(cls):
        """线程锁上下文管理器"""
        with cls._lock:
            yield

    @classmethod
    def log(cls, msg, level="INFO"):
        """分级日志系统"""
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            log_msg = f"[{ts}] [{level}] {msg}"

            # 控制台输出
            print(log_msg)

            # 文件日志
            log_files = {
                "INFO": DIR_CONFIG["logs"] / "arm.log",
                "ERROR": DIR_CONFIG["logs"] / "error.log",
                "COLLISION": DIR_CONFIG["logs"] / "collision.log",
                "PERF": DIR_CONFIG["logs"] / "performance.log",
                "SAFETY": DIR_CONFIG["logs"] / "safety.log",
                "NETWORK": DIR_CONFIG["logs"] / "network.log"
            }

            # 写入通用日志
            with open(log_files["INFO"], "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")

            # 写入特殊级别日志
            if level in log_files and level != "INFO":
                with open(log_files[level], "a", encoding="utf-8") as f:
                    f.write(log_msg + "\n")

        except Exception as e:
            print(f"日志写入失败: {e}")

    @classmethod
    def deg2rad(cls, x):
        """角度转弧度"""
        try:
            return np.asarray(x, np.float64) * DEG2RAD
        except:
            return np.zeros(JOINT_COUNT) if isinstance(x, (list, np.ndarray)) else 0.0

    @classmethod
    def rad2deg(cls, x):
        """弧度转角度"""
        try:
            return np.asarray(x, np.float64) * RAD2DEG
        except:
            return np.zeros(JOINT_COUNT) if isinstance(x, (list, np.ndarray)) else 0.0

    @classmethod
    def record_perf(cls, metric_name, value):
        """记录性能指标"""
        if metric_name in cls._perf_metrics:
            cls._perf_metrics[metric_name].append(value)

    @classmethod
    def get_perf_stats(cls):
        """获取性能统计"""
        stats = {}
        for name, values in cls._perf_metrics.items():
            if values:
                stats[name] = {
                    "avg": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values),
                    "std": np.std(values)
                }
        return stats

    @classmethod
    def compress_data(cls, data):
        """数据压缩"""
        try:
            serialized = pickle.dumps(data)
            compressed = zlib.compress(serialized)
            return compressed
        except Exception as e:
            cls.log(f"数据压缩失败: {e}", "ERROR")
            return b""

    @classmethod
    def decompress_data(cls, compressed_data):
        """数据解压"""
        try:
            decompressed = zlib.decompress(compressed_data)
            data = pickle.loads(decompressed)
            return data
        except Exception as e:
            cls.log(f"数据解压失败: {e}", "ERROR")
            return None


# ====================== 参数持久化模块 ======================
class ParamPersistence:
    """参数保存/加载模块"""

    @staticmethod
    def save_params(config: ControlConfig, name="default"):
        """保存参数到JSON文件"""
        try:
            file_path = DIR_CONFIG["params"] / f"{name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, indent=2)
            Utils.log(f"参数已保存: {file_path}")
            return True
        except Exception as e:
            Utils.log(f"保存参数失败: {e}", "ERROR")
            return False

    @staticmethod
    def load_params(name="default"):
        """从JSON文件加载参数"""
        try:
            file_path = DIR_CONFIG["params"] / f"{name}.json"
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_config = ControlConfig.from_dict(data)

            # 全局配置更新
            global CFG
            for key, value in new_config.__dict__.items():
                setattr(CFG, key, value)

            Utils.log(f"参数已加载: {file_path}")
            return True
        except Exception as e:
            Utils.log(f"加载参数失败: {e}", "ERROR")
            return False


# ====================== 智能避障模块（新增核心功能） ======================
class ObstacleAvoidance:
    """A*算法智能避障模块"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.obstacles = self._detect_obstacles()
        self.path_queue = Queue()

    def _detect_obstacles(self):
        """检测环境中的障碍物"""
        obstacles = []

        # 预定义障碍物
        obstacle_names = ['obstacle1', 'obstacle2']
        for name in obstacle_names:
            obs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if obs_id >= 0:
                pos = self.data.geom_xpos[obs_id].copy()
                size = self.model.geom_size[obs_id].copy()
                obstacles.append({
                    "name": name,
                    "id": obs_id,
                    "position": pos,
                    "size": size,
                    "type": self.model.geom_type[obs_id]
                })

        Utils.log(f"检测到 {len(obstacles)} 个障碍物")
        return obstacles

    def _fk(self, joint_angles):
        """正运动学计算末端位置"""
        # 简化版正运动学（实际应用需替换为精确模型）
        link_lengths = [0.18, 0.18, 0.18, 0.18, 0.18]

        x, y, z = 0.0, 0.0, 0.1  # 基座高度

        # 关节1 (绕Z轴旋转)
        theta1 = joint_angles[0]
        x += link_lengths[0] * np.cos(theta1)
        y += link_lengths[0] * np.sin(theta1)

        # 关节2-5 (绕Y轴旋转)
        for i in range(1, 5):
            theta = joint_angles[i]
            z += link_lengths[i] * np.cos(theta)
            x += link_lengths[i] * np.sin(theta) * np.cos(theta1)
            y += link_lengths[i] * np.sin(theta) * np.sin(theta1)

        return np.array([x, y, z])

    def _is_collision(self, position):
        """检测位置是否与障碍物碰撞"""
        for obs in self.obstacles:
            obs_pos = obs["position"]
            obs_size = obs["size"][0] + CFG.obstacle_margin

            # 计算距离
            dist = np.linalg.norm(position - obs_pos)

            # 碰撞检测
            if dist < obs_size:
                return True, obs["name"]

        return False, None

    def _heuristic(self, current, goal):
        """A*启发函数"""
        return CFG.astar_heuristic_weight * np.linalg.norm(current - goal)

    def _get_neighbors(self, current, step_size=0.05):
        """获取邻居节点"""
        neighbors = []

        # 生成周围8个方向+上下的邻居
        directions = [
            (step_size, 0, 0), (-step_size, 0, 0),
            (0, step_size, 0), (0, -step_size, 0),
            (0, 0, step_size), (0, 0, -step_size),
            (step_size, step_size, 0), (step_size, -step_size, 0),
            (-step_size, step_size, 0), (-step_size, -step_size, 0)
        ]

        for dx, dy, dz in directions:
            neighbor = current + np.array([dx, dy, dz])
            # 检查是否在工作空间内
            if (0 <= neighbor[0] <= 1.0 and
                    -0.5 <= neighbor[1] <= 0.5 and
                    0.1 <= neighbor[2] <= 1.0):
                collision, _ = self._is_collision(neighbor)
                if not collision:
                    neighbors.append(neighbor)

        return neighbors

    def astar_path_planning(self, start_joints, goal_joints):
        """A*算法路径规划"""
        # 计算起始和目标末端位置
        start_pos = self._fk(start_joints)
        goal_pos = self._fk(goal_joints)

        Utils.log(f"避障规划: {start_pos} → {goal_pos}")

        # 检查起点/终点是否碰撞
        start_collision, _ = self._is_collision(start_pos)
        goal_collision, _ = self._is_collision(goal_pos)

        if start_collision:
            Utils.log("起点在障碍物内", "ERROR")
            return None
        if goal_collision:
            Utils.log("目标点在障碍物内", "ERROR")
            return None

        # A*算法初始化
        open_set = {tuple(start_pos): (0, self._heuristic(start_pos, goal_pos))}
        closed_set = set()
        came_from = {}
        g_score = {tuple(start_pos): 0}

        while open_set:
            # 选择f_score最小的节点
            current = min(open_set.keys(), key=lambda x: open_set[x][0] + open_set[x][1])
            current_arr = np.array(current)

            # 到达目标
            if np.linalg.norm(current_arr - goal_pos) < CFG.path_resolution:
                # 重构路径
                path = [current_arr]
                while tuple(current_arr) in came_from:
                    current_arr = came_from[tuple(current_arr)]
                    path.append(current_arr)
                path.reverse()

                # 路径平滑
                smooth_path = self._smooth_path(path)

                Utils.log(f"避障路径规划完成，路径点数量: {len(smooth_path)}")
                return smooth_path

            # 移到closed set
            del open_set[tuple(current)]
            closed_set.add(tuple(current))

            # 遍历邻居
            for neighbor in self._get_neighbors(current_arr):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple in closed_set:
                    continue

                # 计算g_score
                tentative_g = g_score[tuple(current)] + np.linalg.norm(neighbor - current_arr)

                if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current_arr
                    g_score[neighbor_tuple] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_pos)
                    open_set[neighbor_tuple] = (tentative_g, f_score)

        Utils.log("未找到可行路径", "ERROR")
        return None

    def _smooth_path(self, path):
        """路径平滑"""
        if len(path) < 3:
            return path

        # 转换为数组
        path_arr = np.array(path)

        # B样条平滑
        try:
            tck, u = splprep([path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]], s=1.0)
            u_new = np.linspace(0, 1, max(10, len(path) * 2))
            smooth = splev(u_new, tck)
            smooth_path = np.column_stack(smooth)

            # 碰撞检查
            final_path = [path[0]]
            for point in smooth_path:
                collision, _ = self._is_collision(point)
                if not collision:
                    final_path.append(point)
            final_path.append(path[-1])

            return final_path
        except Exception as e:
            Utils.log(f"路径平滑失败: {e}", "ERROR")
            return path

    def plan_safe_trajectory(self, start_joints, goal_joints):
        """规划安全轨迹（避障）"""
        # 首先检查直线路径是否可行
        straight_collision = False
        step_count = 50
        for i in range(step_count + 1):
            t = i / step_count
            mid_joints = start_joints * (1 - t) + goal_joints * t
            mid_pos = self._fk(mid_joints)
            collision, _ = self._is_collision(mid_pos)
            if collision:
                straight_collision = True
                break

        if not straight_collision:
            # 直线路径可行
            return None

        # 需要避障规划
        safe_path = self.astar_path_planning(start_joints, goal_joints)
        if safe_path is None:
            return None

        # 将笛卡尔路径转换为关节空间轨迹
        joint_trajectory = []
        for cart_pos in safe_path:
            # 简化版逆运动学（实际应用需替换为精确解）
            joint_angles = self._ik(cart_pos)
            if joint_angles is not None:
                joint_trajectory.append(joint_angles)

        return np.array(joint_trajectory)

    def _ik(self, target_pos):
        """简化版逆运动学"""
        try:
            # 计算关节1 (绕Z轴)
            theta1 = np.arctan2(target_pos[1], target_pos[0])

            # 计算剩余关节角度（简化）
            x_proj = np.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
            z_diff = target_pos[2] - 0.1

            theta2 = np.arctan2(z_diff, x_proj)
            theta3 = theta4 = theta5 = theta2 / 3

            joint_angles = np.array([theta1, theta2, theta3, theta4, theta5])

            # 限制在关节范围内
            joint_angles = np.clip(joint_angles, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

            return joint_angles
        except Exception as e:
            Utils.log(f"逆运动学求解失败: {e}", "ERROR")
            return None


# ====================== 安全防护模块（新增核心功能） ======================
class SafetyMonitor:
    """多级安全防护模块"""

    def __init__(self):
        self.safety_level = 0  # 0:正常, 1:警告, 2:暂停, 3:急停
        self.safety_events = deque(maxlen=100)
        self.last_safety_check = time.time()

    def check_safety(self, controller):
        """安全检查"""
        current_time = time.time()
        if current_time - self.last_safety_check < 0.01:  # 10ms检查一次
            return self.safety_level

        safety_breaches = []

        # 1. 关节误差检查
        max_error = np.max(np.abs(controller.err))
        if max_error > SAFETY_THRESHOLDS["max_joint_error"]:
            safety_breaches.append(
                f"关节误差超限: {Utils.rad2deg(max_error):.2f}° > {Utils.rad2deg(SAFETY_THRESHOLDS['max_joint_error']):.2f}°")

        # 2. 扭矩检查
        qpos, qvel = controller.get_states()
        load_factor = np.clip(controller.load_actual / MAX_LOAD, 0.0, 1.0)
        kp = CFG.kp_base * (1 + load_factor * (CFG.kp_load_gain - 1))
        pd = kp * controller.err
        torque_ratio = np.max(np.abs(pd) / MAX_TORQUE)
        if torque_ratio > SAFETY_THRESHOLDS["max_torque_ratio"]:
            safety_breaches.append(
                f"扭矩超限: {torque_ratio * 100:.1f}% > {SAFETY_THRESHOLDS['max_torque_ratio'] * 100:.1f}%")

        # 3. 负载检查
        load_ratio = controller.load_actual / MAX_LOAD
        if load_ratio > SAFETY_THRESHOLDS["max_load_ratio"]:
            safety_breaches.append(
                f"负载超限: {load_ratio * 100:.1f}% > {SAFETY_THRESHOLDS['max_load_ratio'] * 100:.1f}%")

        # 4. 速度检查
        vel_ratio = np.max(np.abs(qvel) / MAX_VEL)
        if vel_ratio > SAFETY_THRESHOLDS["max_velocity_ratio"]:
            safety_breaches.append(
                f"速度超限: {vel_ratio * 100:.1f}% > {SAFETY_THRESHOLDS['max_velocity_ratio'] * 100:.1f}%")

        # 5. 碰撞检查
        if controller.collision_detector and controller.collision_detector.collision_detected:
            safety_breaches.append("碰撞检测触发")

        # 更新安全级别
        if not safety_breaches:
            self.safety_level = 0
        elif len(safety_breaches) == 1:
            self.safety_level = 1  # 警告
        elif len(safety_breaches) <= 2:
            self.safety_level = 2  # 暂停
        else:
            self.safety_level = 3  # 急停

        # 记录安全事件
        if safety_breaches:
            event = {
                "time": current_time,
                "level": self.safety_level,
                "breaches": safety_breaches
            }
            self.safety_events.append(event)

            # 记录安全日志
            level_str = ["正常", "警告", "暂停", "急停"][self.safety_level]
            Utils.log(f"安全{level_str}: {'; '.join(safety_breaches)}", "SAFETY")

            # 执行安全动作
            if self.safety_level >= 2:
                controller.pause()
            if self.safety_level >= 3:
                controller.emergency_stop()

        self.last_safety_check = current_time
        return self.safety_level

    def get_safety_status(self):
        """获取安全状态"""
        return {
            "level": self.safety_level,
            "level_str": ["正常", "警告", "暂停", "急停"][self.safety_level],
            "events": list(self.safety_events),
            "event_count": len(self.safety_events)
        }

    def reset_safety_level(self, level=0):
        """重置安全级别"""
        self.safety_level = level
        Utils.log(f"安全级别已重置为: {['正常', '警告', '暂停', '急停'][level]}")


# ====================== 自适应控制模块（新增核心功能） ======================
class AdaptiveControl:
    """PID参数自整定模块"""

    def __init__(self):
        self.tuning_active = False
        self.tuning_history = deque(maxlen=1000)
        self.best_params = CFG.to_dict()
        self.error_integral = np.zeros(JOINT_COUNT)
        self.error_derivative = np.zeros(JOINT_COUNT)
        self.last_error = np.zeros(JOINT_COUNT)

    def start_tuning(self):
        """开始参数自整定"""
        self.tuning_active = True
        self.tuning_history.clear()
        self.error_integral = np.zeros(JOINT_COUNT)
        Utils.log("开始PID参数自整定")

    def stop_tuning(self, save_best=True):
        """停止参数自整定"""
        self.tuning_active = False
        if save_best and self.tuning_history:
            # 保存最优参数
            ParamPersistence.save_params(ControlConfig.from_dict(self.best_params), "tuned")
        Utils.log("停止PID参数自整定")

    def update_params(self, controller):
        """自适应更新参数"""
        if not self.tuning_active:
            return

        # 获取当前误差
        error = controller.err
        qvel, _ = controller.get_states()

        # 计算误差积分和微分
        self.error_integral += error * CTRL_DT
        self.error_derivative = (error - self.last_error) / CTRL_DT

        # 性能指标：ISE (Integral of Squared Error)
        ise = np.sum(error ** 2)
        velocity_penalty = np.sum(np.abs(qvel) / MAX_VEL) * 0.1
        performance = ise + velocity_penalty

        # 记录历史
        self.tuning_history.append({
            "time": time.time(),
            "error": error.copy(),
            "ise": ise,
            "performance": performance,
            "params": CFG.to_dict()
        })

        # 简单的参数调整策略（可替换为更复杂的算法）
        if len(self.tuning_history) > 100:
            # 计算平均性能
            recent_perf = np.mean([h["performance"] for h in list(self.tuning_history)[-50:]])
            best_perf = np.min([h["performance"] for h in self.tuning_history])

            # 更新最优参数
            if recent_perf < best_perf:
                self.best_params = CFG.to_dict()

            # 参数调整
            if np.max(np.abs(error)) > DEG2RAD(1.0):  # 误差大于1度
                # 增大KP
                CFG.kp_base = min(200.0, CFG.kp_base * 1.01)
            else:
                # 减小KP防止超调
                CFG.kp_base = max(50.0, CFG.kp_base * 0.99)

            # 根据误差微分调整KD
            avg_deriv = np.mean(np.abs(self.error_derivative))
            if avg_deriv > DEG2RAD(5.0):  # 误差变化快
                CFG.kd_base = min(20.0, CFG.kd_base * 1.02)
            else:
                CFG.kd_base = max(2.0, CFG.kd_base * 0.98)

            # 每500步打印一次调整结果
            if len(self.tuning_history) % 500 == 0:
                Utils.log(f"自整定更新 - KP: {CFG.kp_base:.1f}, KD: {CFG.kd_base:.1f}, 性能: {recent_perf:.4f}")

        self.last_error = error.copy()

    def get_tuning_status(self):
        """获取自整定状态"""
        if not self.tuning_history:
            return {"active": self.tuning_active, "history_count": 0}

        recent_perf = np.mean([h["performance"] for h in list(self.tuning_history)[-100:]])
        return {
            "active": self.tuning_active,
            "history_count": len(self.tuning_history),
            "current_performance": recent_perf,
            "best_performance": np.min([h["performance"] for h in self.tuning_history]),
            "current_params": CFG.to_dict(),
            "best_params": self.best_params
        }


# ====================== 轨迹规划模块 ======================
class TrajectoryPlanner:
    """轨迹规划模块"""
    _cache = {}
    _cache_max_size = 100

    @classmethod
    def smooth_trajectory(cls, traj_pos, traj_vel):
        """轨迹平滑"""
        if len(traj_pos) <= 2:
            return traj_pos, traj_vel

        smooth_pos = np.empty_like(traj_pos)
        smooth_vel = np.empty_like(traj_vel)

        smooth_pos[0] = traj_pos[0]
        smooth_vel[0] = traj_vel[0]

        alpha = 1 - CFG.smooth_factor
        smooth_pos[1:] = alpha * smooth_pos[:-1] + CFG.smooth_factor * traj_pos[1:]

        vel_diff = (smooth_pos[1:] - smooth_pos[:-1]) / CTRL_DT
        smooth_vel[1:] = np.clip(vel_diff, -MAX_VEL, MAX_VEL)

        if len(smooth_vel) > 2:
            jerk = (smooth_vel[2:] - smooth_vel[1:-1]) / CTRL_DT
            jerk_clipped = np.clip(jerk, -CFG.jerk_limit, CFG.jerk_limit)
            smooth_vel[2:] = smooth_vel[1:-1] + jerk_clipped * CTRL_DT

        return smooth_pos, smooth_vel

    @classmethod
    def plan_trajectory(cls, start, target, smooth=True, avoid_obstacles=True, obstacle_avoider=None):
        """梯形轨迹规划（支持避障）"""
        # 避障规划
        if avoid_obstacles and obstacle_avoider is not None:
            safe_traj = obstacle_avoider.plan_safe_trajectory(start, target)
            if safe_traj is not None and len(safe_traj) > 0:
                traj_pos = safe_traj
                traj_vel = np.zeros_like(traj_pos)

                # 计算速度
                traj_vel[1:] = (traj_pos[1:] - traj_pos[:-1]) / CTRL_DT
                traj_vel = np.clip(traj_vel, -MAX_VEL, MAX_VEL)

                if smooth:
                    traj_pos, traj_vel = cls.smooth_trajectory(traj_pos, traj_vel)

                return traj_pos, traj_vel

        # 常规轨迹规划
        start = np.clip(start, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)
        target = np.clip(target, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)

        cache_key = (start.tobytes(), target.tobytes(), smooth)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        traj_pos = np.zeros((0, JOINT_COUNT))
        traj_vel = np.zeros((0, JOINT_COUNT))
        max_len = 1

        for i in range(JOINT_COUNT):
            delta = target[i] - start[i]

            if abs(delta) < 1e-5:
                pos = np.array([target[i]])
                vel = np.array([0.0])
            else:
                dir = np.sign(delta)
                dist = abs(delta)
                accel_dist = (MAX_VEL[i] ** 2) / (2 * MAX_ACC[i])

                if dist <= 2 * accel_dist:
                    peak_vel = np.sqrt(dist * MAX_ACC[i])
                    accel_time = peak_vel / MAX_ACC[i]
                    total_time = 2 * accel_time
                else:
                    accel_time = MAX_VEL[i] / MAX_ACC[i]
                    uniform_time = (dist - 2 * accel_dist) / MAX_VEL[i]
                    total_time = 2 * accel_time + uniform_time

                t = np.arange(0, total_time + CTRL_DT, CTRL_DT)
                pos = np.empty_like(t)
                vel = np.empty_like(t)

                mask_acc = t <= accel_time
                mask_uni = (t > accel_time) & (
                            t <= accel_time + uniform_time) if dist > 2 * accel_dist else np.zeros_like(t, bool)
                mask_dec = ~(mask_acc | mask_uni)

                vel[mask_acc] = MAX_ACC[i] * t[mask_acc] * dir
                pos[mask_acc] = start[i] + 0.5 * MAX_ACC[i] * t[mask_acc] ** 2 * dir

                if dist > 2 * accel_dist:
                    t_uni = t[mask_uni] - accel_time
                    vel[mask_uni] = MAX_VEL[i] * dir
                    pos[mask_uni] = start[i] + (accel_dist + MAX_VEL[i] * t_uni) * dir

                    t_dec = t[mask_dec] - (accel_time + uniform_time)
                    vel[mask_dec] = (MAX_VEL[i] - MAX_ACC[i] * t_dec) * dir
                    pos[mask_dec] = start[i] + (dist - (accel_dist - 0.5 * MAX_ACC[i] * t_dec ** 2)) * dir
                else:
                    t_dec = t[mask_dec] - accel_time
                    vel[mask_dec] = (peak_vel - MAX_ACC[i] * t_dec) * dir
                    pos[mask_dec] = start[i] + (peak_vel * accel_time - 0.5 * MAX_ACC[i] * t_dec ** 2) * dir

                pos[-1], vel[-1] = target[i], 0.0

            if len(traj_pos) < len(pos):
                traj_pos = np.pad(traj_pos, ((0, len(pos) - len(traj_pos)), (0, 0)), 'constant')
                traj_vel = np.pad(traj_vel, ((0, len(pos) - len(traj_vel)), (0, 0)), 'constant')

            traj_pos[:len(pos), i] = pos
            traj_vel[:len(pos), i] = vel
            max_len = max(max_len, len(pos))

        if len(traj_pos) < max_len:
            pad = max_len - len(traj_pos)
            traj_pos = np.pad(traj_pos, ((0, pad), (0, 0)), 'constant', constant_values=target)
            traj_vel = np.pad(traj_vel, ((0, pad), (0, 0)), 'constant')

        if smooth:
            traj_pos, traj_vel = cls.smooth_trajectory(traj_pos, traj_vel)

        if len(cls._cache) >= cls._cache_max_size:
            cls._cache.pop(next(iter(cls._cache)))
        cls._cache[cache_key] = (traj_pos, traj_vel)

        return traj_pos, traj_vel

    @classmethod
    def save_traj(cls, traj_pos, traj_vel, name):
        """保存轨迹"""
        try:
            header = ['step'] + [f'j{i + 1}_pos' for i in range(JOINT_COUNT)] + [f'j{i + 1}_vel' for i in
                                                                                 range(JOINT_COUNT)]
            data = np.hstack([np.arange(len(traj_pos))[:, None], traj_pos, traj_vel])
            file_path = DIR_CONFIG["trajectories"] / f"{name}.csv"
            np.savetxt(file_path, data, delimiter=',', header=','.join(header), comments='')
            Utils.log(f"轨迹保存: {file_path}")
            return True
        except Exception as e:
            Utils.log(f"保存轨迹失败: {e}", "ERROR")
            return False

    @classmethod
    def load_traj(cls, name, smooth=True):
        """加载轨迹"""
        try:
            file_path = DIR_CONFIG["trajectories"] / f"{name}.csv"
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            if len(data) == 0:
                return np.array([]), np.array([])

            traj_pos = data[:, 1:JOINT_COUNT + 1]
            traj_vel = data[:, JOINT_COUNT + 1:]

            if smooth:
                traj_pos, traj_vel = cls.smooth_trajectory(traj_pos, traj_vel)

            return traj_pos, traj_vel
        except Exception as e:
            Utils.log(f"加载轨迹失败: {e}", "ERROR")
            return np.array([]), np.array([])

    @classmethod
    def clear_cache(cls):
        """清空缓存"""
        cls._cache.clear()
        Utils.log("轨迹缓存已清空")


# ====================== 碰撞检测模块 ======================
class CollisionDetector:
    """碰撞检测模块"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.collision_detected = False
        self.collision_history = deque(maxlen=100)

    def detect_collision(self, ee_id, link_geom_ids):
        """碰撞检测"""
        collision = False
        collision_info = []

        if ee_id >= 0:
            ee_pos = self.data.geom_xpos[ee_id]

            obstacle_names = ['obstacle1', 'obstacle2']
            for obs_name in obstacle_names:
                obs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obs_name)
                if obs_id >= 0:
                    obs_pos = self.data.geom_xpos[obs_id]
                    dist = np.linalg.norm(ee_pos - obs_pos)
                    if dist < COLLISION_THRESHOLD:
                        collision = True
                        collision_info.append(f"末端与{obs_name}距离过近: {dist:.4f}m")

        contact_forces = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, 0, contact_forces)
        max_force = np.max(np.abs(contact_forces[:3]))
        if max_force > COLLISION_FORCE_THRESHOLD:
            collision = True
            collision_info.append(f"接触力超限: {max_force:.2f}N")

        valid_links = [lid for lid in link_geom_ids if lid >= 0]
        if len(valid_links) > 1:
            link_positions = self.data.geom_xpos[valid_links]
            dist_matrix = np.linalg.norm(link_positions[:, None] - link_positions, axis=2)
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist = dist_matrix[min_dist_idx]

            if min_dist < 0.005:
                collision = True
                collision_info.append(f"连杆{min_dist_idx[0] + 1}与连杆{min_dist_idx[1] + 1}自碰撞: {min_dist:.4f}m")

        self.collision_detected = collision
        if collision:
            self.collision_history.append((time.time(), collision_info))
            for info in collision_info:
                Utils.log(f"碰撞检测: {info}", "COLLISION")

        return collision, collision_info


# ====================== 数据记录与可视化模块 ======================
class DataRecorder:
    """数据记录与可视化模块"""

    def __init__(self):
        self.enabled = False
        self.data = {
            'time': [], 'qpos': [], 'qvel': [], 'err': [], 'load': [],
            'stiffness': [], 'torque': [], 'collision': [], 'safety_level': []
        }
        self.record_count = 0
        self.sample_interval = 10
        self.replay_data = {}

    def start(self):
        """开始记录"""
        self.enabled = True
        self.reset()
        Utils.log("开始数据记录")

    def stop(self, save=True, plot=True, save_replay=True):
        """停止记录"""
        self.enabled = False
        Utils.log("停止数据记录")

        if save and self.record_count > 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self.save_data(f"run_{timestamp}")

            # 保存回放数据
            if save_replay:
                self.save_replay_data(f"replay_{timestamp}")

            if plot:
                self.plot_data(f"plot_{timestamp}")

    def reset(self):
        """重置记录"""
        self.data = {k: [] for k in self.data.keys()}
        self.record_count = 0

    def record(self, qpos, qvel, err, load, stiffness, torque, collision, safety_level):
        """记录数据"""
        if not self.enabled:
            return

        self.record_count += 1
        if self.record_count % self.sample_interval != 0:
            return

        self.data['time'].append(time.time())
        self.data['qpos'].append(qpos.copy())
        self.data['qvel'].append(qvel.copy())
        self.data['err'].append(err.copy())
        self.data['load'].append(load)
        self.data['stiffness'].append(stiffness.copy())
        self.data['torque'].append(torque.copy())
        self.data['collision'].append(collision)
        self.data['safety_level'].append(safety_level)

        # 保存回放数据
        self.replay_data[self.record_count] = {
            "qpos": qpos.copy(),
            "qvel": qvel.copy(),
            "timestamp": time.time()
        }

    def save_data(self, name):
        """保存数据"""
        try:
            save_data = {}
            for key, value in self.data.items():
                if key in ['time', 'load', 'collision', 'safety_level']:
                    save_data[key] = np.array(value)
                else:
                    save_data[key] = np.array(value, dtype=np.float64)

            file_path = DIR_CONFIG["data"] / f"{name}.npz"
            np.savez(file_path, **save_data)
            Utils.log(f"记录数据已保存: {file_path}")
            return True
        except Exception as e:
            Utils.log(f"保存数据失败: {e}", "ERROR")
            return False

    def save_replay_data(self, name):
        """保存回放数据"""
        try:
            file_path = DIR_CONFIG["replay"] / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(self.replay_data, f)
            Utils.log(f"回放数据已保存: {file_path}")
            return True
        except Exception as e:
            Utils.log(f"保存回放数据失败: {e}", "ERROR")
            return False

    def load_replay_data(self, name):
        """加载回放数据"""
        try:
            file_path = DIR_CONFIG["replay"] / f"{name}.pkl"
            with open(file_path, 'rb') as f:
                self.replay_data = pickle.load(f)
            Utils.log(f"回放数据已加载: {file_path}")
            return self.replay_data
        except Exception as e:
            Utils.log(f"加载回放数据失败: {e}", "ERROR")
            return {}

    def plot_data(self, name):
        """数据可视化"""
        try:
            if len(self.data['time']) < 10:
                Utils.log("数据量不足，无法绘图")
                return

            time = np.array(self.data['time'])
            time -= time[0]
            qpos = np.array(self.data['qpos'])
            err = np.array(self.data['err'])
            load = np.array(self.data['load'])
            collision = np.array(self.data['collision'])
            safety_level = np.array(self.data['safety_level'])

            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle('机械臂运行数据分析', fontsize=14)
            ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

            # 1. 关节角度
            for i in range(JOINT_COUNT):
                ax1.plot(time, Utils.rad2deg(qpos[:, i]), label=f'关节{i + 1}')
            ax1.set(xlabel='时间 (s)', ylabel='角度 (°)', title='关节角度')
            ax1.legend()
            ax1.grid(True)

            # 2. 跟踪误差
            for i in range(JOINT_COUNT):
                ax2.plot(time, Utils.rad2deg(np.abs(err[:, i])), label=f'关节{i + 1}')
            ax2.set(xlabel='时间 (s)', ylabel='误差 (°)', title='跟踪误差')
            ax2.legend()
            ax2.grid(True)

            # 3. 负载变化
            ax3.plot(time, load)
            ax3.set(xlabel='时间 (s)', ylabel='负载 (kg)', title='负载变化')
            ax3.grid(True)

            # 4. 碰撞事件
            collision_times = time[collision]
            ax4.scatter(collision_times, np.ones_like(collision_times),
                        color='red', marker='x', s=100, label='碰撞事件')
            ax4.set(xlabel='时间 (s)', ylabel='碰撞检测', title='碰撞事件')
            ax4.legend()
            ax4.grid(True)

            # 5. 安全级别
            ax5.plot(time, safety_level)
            ax5.set(xlabel='时间 (s)', ylabel='安全级别', title='安全级别变化')
            ax5.set_yticks([0, 1, 2, 3])
            ax5.set_yticklabels(['正常', '警告', '暂停', '急停'])
            ax5.grid(True)

            # 6. 误差统计
            err_magnitude = np.mean(Utils.rad2deg(np.abs(err)), axis=1)
            ax6.plot(time, err_magnitude)
            ax6.set(xlabel='时间 (s)', ylabel='平均误差 (°)', title='平均跟踪误差')
            ax6.grid(True)

            plt.tight_layout()
            file_path = DIR_CONFIG["data"] / f"{name}.png"
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()

            Utils.log(f"可视化图表已保存: {file_path}")
            return True
        except Exception as e:
            Utils.log(f"绘图失败: {e}", "ERROR")
            return False


# ====================== 远程交互模块（新增核心功能） ======================
class WebSocketServer:
    """WebSocket远程交互服务器"""

    def __init__(self, controller):
        self.controller = controller
        self.server = None
        self.connected_clients = set()
        self.running = False
        self.msg_queue = Queue(maxsize=100)

    async def handle_client(self, websocket, path):
        """处理客户端连接"""
        client_addr = websocket.remote_address
        self.connected_clients.add(websocket)
        Utils.log(f"客户端连接: {client_addr}", "NETWORK")

        try:
            async for message in websocket:
                try:
                    # 解析消息
                    msg = json.loads(message)
                    response = self.handle_command(msg)

                    # 发送响应
                    await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    error_msg = {"status": "error", "message": "无效的JSON格式"}
                    await websocket.send(json.dumps(error_msg))
                except Exception as e:
                    error_msg = {"status": "error", "message": str(e)}
                    await websocket.send(json.dumps(error_msg))

        except websockets.exceptions.ConnectionClosed:
            Utils.log(f"客户端断开连接: {client_addr}", "NETWORK")
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)

    def handle_command(self, msg):
        """处理远程命令"""
        cmd = msg.get("command", "")
        params = msg.get("params", {})
        response = {"status": "success", "command": cmd}

        try:
            if cmd == "get_status":
                qpos, qvel = self.controller.get_states()
                safety_status = self.controller.safety_monitor.get_safety_status()

                response["data"] = {
                    "joint_positions": Utils.rad2deg(qpos).tolist(),
                    "joint_velocities": Utils.rad2deg(qvel).tolist(),
                    "load_set": self.controller.load_set,
                    "load_actual": self.controller.load_actual,
                    "error": Utils.rad2deg(self.controller.err).tolist(),
                    "max_error": Utils.rad2deg(self.controller.max_err).tolist(),
                    "paused": self.controller.paused,
                    "emergency_stop": self.controller.emergency_stop,
                    "collision_detected": self.controller.collision_detector.collision_detected if self.controller.collision_detector else False,
                    "safety_status": safety_status,
                    "trajectory_queue_size": len(self.controller.traj_queue),
                    "step_count": self.controller.step
                }

            elif cmd == "move_to":
                target = params.get("target", [0, 0, 0, 0, 0])
                smooth = params.get("smooth", True)
                avoid_obstacles = params.get("avoid_obstacles", True)

                self.controller.move_to(target, smooth=smooth, avoid_obstacles=avoid_obstacles)
                response["message"] = f"已规划轨迹到目标位置: {target}"

            elif cmd == "control_joint":
                joint_idx = params.get("joint", 0)
                angle = params.get("angle", 0)
                self.controller.control_joint(joint_idx, angle)
                response["message"] = f"已控制关节{joint_idx + 1}到{angle}度"

            elif cmd == "set_load":
                load = params.get("load", 0.5)
                self.controller.set_load(load)
                response["message"] = f"负载已设置为{load}kg"

            elif cmd == "pause":
                self.controller.pause()
                response["message"] = "运动已暂停"

            elif cmd == "resume":
                self.controller.resume()
                response["message"] = "运动已恢复"

            elif cmd == "emergency_stop":
                self.controller.emergency_stop()
                response["message"] = "紧急停止已触发"

            elif cmd == "reset_collision":
                self.controller.reset_collision()
                response["message"] = "碰撞状态已重置"

            elif cmd == "reset_safety":
                self.controller.safety_monitor.reset_safety_level()
                response["message"] = "安全级别已重置"

            elif cmd == "start_tuning":
                self.controller.adaptive_control.start_tuning()
                response["message"] = "参数自整定已开始"

            elif cmd == "stop_tuning":
                self.controller.adaptive_control.stop_tuning()
                response["message"] = "参数自整定已停止"

            elif cmd == "start_recording":
                self.controller.data_recorder.start()
                response["message"] = "数据记录已开始"

            elif cmd == "stop_recording":
                self.controller.data_recorder.stop()
                response["message"] = "数据记录已停止并保存"

            elif cmd == "get_params":
                response["data"] = CFG.to_dict()

            elif cmd == "set_param":
                param = params.get("param")
                value = params.get("value")
                idx = params.get("index")

                self.controller.adjust_param(param, value, idx)
                response["message"] = f"参数{param}已更新为{value}"

            else:
                response = {"status": "error", "message": f"未知命令: {cmd}"}

        except Exception as e:
            response = {"status": "error", "message": str(e)}
            Utils.log(f"远程命令执行错误: {e}", "ERROR")

        return response

    async def broadcast_status(self):
        """广播状态信息"""
        while self.running:
            try:
                if self.connected_clients:
                    # 获取状态数据
                    qpos, qvel = self.controller.get_states()
                    status_data = {
                        "type": "status_update",
                        "data": {
                            "joint_positions": Utils.rad2deg(qpos).tolist(),
                            "load_actual": self.controller.load_actual,
                            "collision_detected": self.controller.collision_detector.collision_detected if self.controller.collision_detector else False,
                            "safety_level": self.controller.safety_monitor.safety_level,
                            "step": self.controller.step
                        }
                    }

                    # 广播到所有客户端
                    for client in list(self.connected_clients):
                        try:
                            await client.send(json.dumps(status_data))
                        except:
                            pass

                await asyncio.sleep(0.1)  # 100ms更新一次

            except Exception as e:
                Utils.log(f"广播状态失败: {e}", "NETWORK")

    def start(self):
        """启动WebSocket服务器"""
        self.running = True

        async def start_server():
            self.server = await websockets.serve(
                self.handle_client,
                WS_HOST,
                WS_PORT,
                max_size=WS_BUFFER_SIZE
            )
            Utils.log(f"WebSocket服务器已启动: ws://{WS_HOST}:{WS_PORT}", "NETWORK")

            # 启动广播任务
            asyncio.create_task(self.broadcast_status())

            await self.server.wait_closed()

        # 在新线程中运行事件循环
        self.server_thread = threading.Thread(
            target=asyncio.run,
            args=(start_server(),),
            daemon=True
        )
        self.server_thread.start()

    def stop(self):
        """停止WebSocket服务器"""
        self.running = False
        if self.server:
            self.server.close()
        Utils.log("WebSocket服务器已停止", "NETWORK")


# ====================== 离线分析模块（新增核心功能） ======================
class OfflineAnalyzer:
    """离线数据分析与轨迹回放模块"""

    def __init__(self):
        self.analysis_results = {}

    def analyze_run_data(self, file_name):
        """分析运行数据"""
        try:
            file_path = DIR_CONFIG["data"] / f"{file_name}.npz"
            data = np.load(file_path)

            # 基本统计
            time_data = data['time']
            qpos_data = data['qpos']
            err_data = data['err']
            load_data = data['load']
            safety_data = data['safety_level']

            # 计算统计指标
            total_time = time_data[-1] - time_data[0]
            avg_error = np.mean(Utils.rad2deg(np.abs(err_data)))
            max_error = np.max(Utils.rad2deg(np.abs(err_data)))
            avg_load = np.mean(load_data)
            safety_events = np.sum(safety_data > 0)

            # 关节性能分析
            joint_performance = {}
            for i in range(JOINT_COUNT):
                joint_err = Utils.rad2deg(np.abs(err_data[:, i]))
                joint_performance[f"joint_{i + 1}"] = {
                    "avg_error": np.mean(joint_err),
                    "max_error": np.max(joint_err),
                    "rms_error": np.sqrt(np.mean(joint_err ** 2))
                }

            # 保存分析结果
            self.analysis_results = {
                "file_name": file_name,
                "total_time": total_time,
                "sample_count": len(time_data),
                "avg_frequency": len(time_data) / total_time,
                "avg_error": avg_error,
                "max_error": max_error,
                "avg_load": avg_load,
                "safety_event_count": safety_events,
                "joint_performance": joint_performance,
                "collision_count": np.sum(data['collision'])
            }

            # 生成分析报告
            self.generate_report()

            Utils.log(f"离线分析完成: {file_name}")
            return self.analysis_results

        except Exception as e:
            Utils.log(f"分析数据失败: {e}", "ERROR")
            return {}

    def generate_report(self):
        """生成分析报告"""
        try:
            report = f"""
# 机械臂运行分析报告
## 基本信息
- 数据文件: {self.analysis_results['file_name']}
- 运行时长: {self.analysis_results['total_time']:.2f}秒
- 采样点数: {self.analysis_results['sample_count']}
- 平均采样频率: {self.analysis_results['avg_frequency']:.1f}Hz

## 性能指标
- 平均跟踪误差: {self.analysis_results['avg_error']:.3f}°
- 最大跟踪误差: {self.analysis_results['max_error']:.3f}°
- 平均负载: {self.analysis_results['avg_load']:.2f}kg
- 安全事件数量: {self.analysis_results['safety_event_count']}
- 碰撞事件数量: {self.analysis_results['collision_count']}

## 关节性能详情
"""
            for joint, perf in self.analysis_results['joint_performance'].items():
                report += f"- {joint}:\n"
                report += f"  - 平均误差: {perf['avg_error']:.3f}°\n"
                report += f"  - 最大误差: {perf['max_error']:.3f}°\n"
                report += f"  - RMS误差: {perf['rms_error']:.3f}°\n"

            # 保存报告
            file_path = DIR_CONFIG["data"] / f"{self.analysis_results['file_name']}_report.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)

            Utils.log(f"分析报告已保存: {file_path}")

        except Exception as e:
            Utils.log(f"生成报告失败: {e}", "ERROR")

    def replay_trajectory(self, file_name, speed=1.0):
        """轨迹回放"""
        try:
            # 加载回放数据
            file_path = DIR_CONFIG["replay"] / f"{file_name}.pkl"
            with open(file_path, 'rb') as f:
                replay_data = pickle.load(f)

            # 创建可视化窗口
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('关节角度 (°)')
            ax.set_title('机械臂轨迹回放')
            ax.grid(True)

            # 准备数据
            steps = sorted(replay_data.keys())
            timestamps = [replay_data[step]['timestamp'] for step in steps]
            base_time = timestamps[0]
            relative_times = [t - base_time for t in timestamps]

            joint_data = {}
            lines = {}
            for i in range(JOINT_COUNT):
                joint_data[i] = [Utils.rad2deg(replay_data[step]['qpos'][i]) for step in steps]
                line, = ax.plot([], [], label=f'关节{i + 1}')
                lines[i] = line

            ax.legend()
            ax.set_xlim(0, max(relative_times))
            ax.set_ylim(-180, 180)

            # 动画更新函数
            def update(frame):
                if frame >= len(steps):
                    return list(lines.values())

                for i in range(JOINT_COUNT):
                    lines[i].set_data(
                        relative_times[:frame + 1],
                        joint_data[i][:frame + 1]
                    )

                return list(lines.values())

            # 创建动画
            ani = FuncAnimation(
                fig,
                update,
                frames=len(steps),
                interval=10 / speed,  # 速度控制
                blit=True
            )

            # 保存动画
            save_path = DIR_CONFIG["replay"] / f"{file_name}_replay.gif"
            ani.save(save_path, writer='pillow', fps=30)
            plt.close()

            Utils.log(f"轨迹回放动画已保存: {save_path}")
            return True

        except Exception as e:
            Utils.log(f"轨迹回放失败: {e}", "ERROR")
            return False


# ====================== 核心控制器 ======================
class ArmController:
    """机械臂核心控制器"""

    def __init__(self):
        # 全局状态
        self.running = True
        self.paused = False
        self.emergency_stop = False

        # 初始化MuJoCo
        self.model, self.data = self._init_mujoco()

        # 初始化模块化组件
        self.collision_detector = CollisionDetector(self.model, self.data) if self.model else None
        self.obstacle_avoider = ObstacleAvoidance(self.model, self.data) if self.model else None
        self.safety_monitor = SafetyMonitor()
        self.adaptive_control = AdaptiveControl()
        self.data_recorder = DataRecorder()
        self.offline_analyzer = OfflineAnalyzer()

        # 初始化网络服务
        self.ws_server = WebSocketServer(self)

        # ID缓存
        self._init_ids()

        # 控制状态
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.target = np.zeros(JOINT_COUNT)

        # 轨迹队列
        self.traj_queue = deque()
        self.current_queue_idx = 0

        # 物理状态
        self.stiffness = CFG.stiffness_base.copy()
        self.damping = self.stiffness * CFG.damping_ratio
        self.load_set = 0.5
        self.load_actual = 0.5

        # 误差状态
        self.err = np.zeros(JOINT_COUNT)
        self.max_err = np.zeros(JOINT_COUNT)

        # 性能统计
        self.step = 0
        self.last_ctrl = time.time()
        self.last_status = time.time()
        self.fps_count = 0

        # Viewer
        self.viewer = None

    def _init_ids(self):
        """ID初始化"""
        if self.model is None:
            self.joint_ids = [-1] * JOINT_COUNT
            self.motor_ids = [-1] * JOINT_COUNT
            self.ee_id = -1
            self.link_geom_ids = [-1] * JOINT_COUNT
            return

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i + 1}')
                          for i in range(JOINT_COUNT)]
        self.motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i + 1}')
                          for i in range(JOINT_COUNT)]
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ee_geom')
        self.link_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'link{i + 1}')
                              for i in range(JOINT_COUNT)]

    def _init_mujoco(self):
        """MuJoCo初始化"""
        xml_template = """
<mujoco model="arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{sim_dt}" gravity="0 0 -9.81" collision="all"/>
    <default>
        <joint type="hinge" limited="true"/>
        <motor ctrllimited="true" ctrlrange="-1 1" gear="100"/>
        <geom contype="1" conaffinity="1" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1" contype="1" conaffinity="1"/>
        <body name="base" pos="0 0 0">
            <geom type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" axis="0 0 1" pos="0 0 0.1" range="{j1_min} {j1_max}"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1" type="cylinder" size="0.04 0.18" mass="0.8" rgba="0 0.8 0 0.8"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18" range="{j2_min} {j2_max}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2" type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                    <joint name="joint3" axis="0 1 0" pos="0 0 0.18" range="{j3_min} {j3_max}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3" type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                        <joint name="joint4" axis="0 1 0" pos="0 0 0.18" range="{j4_min} {j4_max}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4" type="cylinder" size="0.04 0.18" mass="0.4" rgba="0 0.8 0 0.8"/>
                            <joint name="joint5" axis="0 1 0" pos="0 0 0.18" range="{j5_min} {j5_max}"/>
                            <body name="ee" pos="0 0 0.18">
                                <geom name="ee_geom" type="sphere" size="0.04" mass="{load}" rgba="0.8 0.2 0.2 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        <geom name="obstacle1" type="sphere" size="0.05" pos="0.2 0.1 0.5" rgba="1 0 0 0.5"/>
        <geom name="obstacle2" type="cylinder" size="0.03 0.2" pos="-