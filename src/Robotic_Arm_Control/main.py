#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂控制器（终极优化版）
核心特性：模块化架构+极致性能+完整功能+工程化保障
"""

import sys
import time
import signal
import threading
import json
import numpy as np
import mujoco
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import matplotlib.pyplot as plt
from collections import deque

# ====================== 全局配置与常量（预计算+只读） ======================
# 硬件参数
JOINT_COUNT = 5
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
MIN_LOAD, MAX_LOAD = 0.0, 2.0

# 关节极限（预计算常量）
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

# 碰撞检测阈值（常量）
COLLISION_THRESHOLD = 0.01
COLLISION_FORCE_THRESHOLD = 5.0

# 目录配置（预创建）
DIR_CONFIG = {
    "trajectories": Path("trajectories"),
    "params": Path("params"),
    "logs": Path("logs"),
    "data": Path("data")
}
for dir_path in DIR_CONFIG.values():
    dir_path.mkdir(exist_ok=True)


# ====================== 配置管理模块（独立解耦） ======================
@dataclass
class ControlConfig:
    """控制参数配置类（独立管理）"""
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

    def to_dict(self):
        """转换为字典（支持序列化）"""
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    @classmethod
    def from_dict(cls, data):
        """从字典加载"""
        data = data.copy()
        for key in ['backlash', 'friction', 'stiffness_base', 'stiffness_min',
                    'stiffness_max', 'jerk_limit']:
            if key in data and isinstance(data[key], list):
                data[key] = np.array(data[key], dtype=np.float64)
        return cls(**data)

# 全局状态
RUNNING = True
PAUSED = False
EMERGENCY_STOP = False
COLLISION_DETECTED = False
LOCK = threading.Lock()

# 全局配置实例（单例）
CFG = ControlConfig()


# ====================== 工具函数模块（独立解耦） ======================
class Utils:
    """工具函数类（模块化管理）"""
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

            # 文件日志（按级别分类）
            log_files = {
                "INFO": DIR_CONFIG["logs"] / "arm.log",
                "ERROR": DIR_CONFIG["logs"] / "error.log",
                "COLLISION": DIR_CONFIG["logs"] / "collision.log",
                "PERF": DIR_CONFIG["logs"] / "performance.log"
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
        """角度转弧度（向量化+异常安全）"""
        try:
            return np.asarray(x, np.float64) * DEG2RAD
        except:
            return np.zeros(JOINT_COUNT) if isinstance(x, (list, np.ndarray)) else 0.0

    @classmethod
    def rad2deg(cls, x):
        """弧度转角度（向量化+异常安全）"""
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


# ====================== 参数持久化模块（独立解耦） ======================
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

# ====================== 轨迹规划增强 ======================
TRAJ_CACHE = {}

# ====================== 轨迹规划模块（独立解耦） ======================
class TrajectoryPlanner:
    """轨迹规划模块（独立封装）"""
    _cache = {}  # 轨迹缓存（LRU策略）
    _cache_max_size = 100  # 最大缓存数

    @classmethod
    def smooth_trajectory(cls, traj_pos, traj_vel):
        """轨迹平滑（向量化极致优化）"""
        if len(traj_pos) <= 2:
            return traj_pos, traj_vel

        # 预分配内存
        smooth_pos = np.empty_like(traj_pos)
        smooth_vel = np.empty_like(traj_vel)

        # 初始值
        smooth_pos[0] = traj_pos[0]
        smooth_vel[0] = traj_vel[0]

        # 向量化低通滤波
        alpha = 1 - CFG.smooth_factor
        smooth_pos[1:] = alpha * smooth_pos[:-1] + CFG.smooth_factor * traj_pos[1:]

        # 速度计算（向量化）
        vel_diff = (smooth_pos[1:] - smooth_pos[:-1]) / CTRL_DT
        smooth_vel[1:] = np.clip(vel_diff, -MAX_VEL, MAX_VEL)

        # 加加速度限制（向量化）
        if len(smooth_vel) > 2:
            jerk = (smooth_vel[2:] - smooth_vel[1:-1]) / CTRL_DT
            jerk_clipped = np.clip(jerk, -CFG.jerk_limit, CFG.jerk_limit)
            smooth_vel[2:] = smooth_vel[1:-1] + jerk_clipped * CTRL_DT

        return smooth_pos, smooth_vel

    @classmethod
    def plan_trajectory(cls, start, target, smooth=True):
        """梯形轨迹规划（极致优化）"""
        # 边界裁剪（向量化）
        start = np.clip(start, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)
        target = np.clip(target, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)

        # 缓存键（哈希优化）
        cache_key = (start.tobytes(), target.tobytes(), smooth)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # 预分配数组
        traj_pos = np.zeros((0, JOINT_COUNT))
        traj_vel = np.zeros((0, JOINT_COUNT))
        max_len = 1

        # 批量规划（向量化）
        for i in range(JOINT_COUNT):
            delta = target[i] - start[i]

            # 静态目标直接返回
            if abs(delta) < 1e-5:
                pos = np.array([target[i]])
                vel = np.array([0.0])
            else:
                dir = np.sign(delta)
                dist = abs(delta)
                accel_dist = (MAX_VEL[i] ** 2) / (2 * MAX_ACC[i])

                # 计算时间参数
                if dist <= 2 * accel_dist:
                    peak_vel = np.sqrt(dist * MAX_ACC[i])
                    accel_time = peak_vel / MAX_ACC[i]
                    total_time = 2 * accel_time
                else:
                    accel_time = MAX_VEL[i] / MAX_ACC[i]
                    uniform_time = (dist - 2 * accel_dist) / MAX_VEL[i]
                    total_time = 2 * accel_time + uniform_time

                # 时间序列（预分配）
                t = np.arange(0, total_time + CTRL_DT, CTRL_DT)
                pos = np.empty_like(t)
                vel = np.empty_like(t)

                # 向量化分段计算
                mask_acc = t <= accel_time
                mask_uni = (t > accel_time) & (
                            t <= accel_time + uniform_time) if dist > 2 * accel_dist else np.zeros_like(t, bool)
                mask_dec = ~(mask_acc | mask_uni)

                # 加速段
                vel[mask_acc] = MAX_ACC[i] * t[mask_acc] * dir
                pos[mask_acc] = start[i] + 0.5 * MAX_ACC[i] * t[mask_acc] ** 2 * dir

                # 匀速段
                if dist > 2 * accel_dist:
                    t_uni = t[mask_uni] - accel_time
                    vel[mask_uni] = MAX_VEL[i] * dir
                    pos[mask_uni] = start[i] + (accel_dist + MAX_VEL[i] * t_uni) * dir

                    # 减速段
                    t_dec = t[mask_dec] - (accel_time + uniform_time)
                    vel[mask_dec] = (MAX_VEL[i] - MAX_ACC[i] * t_dec) * dir
                    pos[mask_dec] = start[i] + (dist - (accel_dist - 0.5 * MAX_ACC[i] * t_dec ** 2)) * dir
                else:
                    # 减速段
                    t_dec = t[mask_dec] - accel_time
                    vel[mask_dec] = (peak_vel - MAX_ACC[i] * t_dec) * dir
                    pos[mask_dec] = start[i] + (peak_vel * accel_time - 0.5 * MAX_ACC[i] * t_dec ** 2) * dir

                # 确保终点准确
                pos[-1], vel[-1] = target[i], 0.0

            # 扩展维度
            if len(traj_pos) < len(pos):
                traj_pos = np.pad(traj_pos, ((0, len(pos) - len(traj_pos)), (0, 0)), 'constant')
                traj_vel = np.pad(traj_vel, ((0, len(pos) - len(traj_vel)), (0, 0)), 'constant')

            traj_pos[:len(pos), i] = pos
            traj_vel[:len(pos), i] = vel
            max_len = max(max_len, len(pos))

        # 统一长度
        if len(traj_pos) < max_len:
            pad = max_len - len(traj_pos)
            traj_pos = np.pad(traj_pos, ((0, pad), (0, 0)), 'constant', constant_values=target)
            traj_vel = np.pad(traj_vel, ((0, pad), (0, 0)), 'constant')

        # 轨迹平滑
        if smooth:
            traj_pos, traj_vel = cls.smooth_trajectory(traj_pos, traj_vel)

        # 缓存管理（LRU）
        if len(cls._cache) >= cls._cache_max_size:
            cls._cache.pop(next(iter(cls._cache)))
        cls._cache[cache_key] = (traj_pos, traj_vel)

        return traj_pos, traj_vel

    @classmethod
    def save_traj(cls, traj_pos, traj_vel, name):
        """保存轨迹（批量IO优化）"""
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
        """加载轨迹（批量IO优化）"""
        try:
            file_path = DIR_CONFIG["trajectories"] / f"{name}.csv"
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            if len(data) == 0:
                return np.array([]), np.array([])

            traj_pos = data[:, 1:JOINT_COUNT + 1]
            traj_vel = data[:, JOINT_COUNT + 1:]

            # 可选平滑
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


# ====================== 碰撞检测模块（独立解耦） ======================
class CollisionDetector:
    """碰撞检测模块（独立封装）"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.collision_detected = False
        self.collision_history = deque(maxlen=100)

    def detect_collision(self, ee_id, link_geom_ids):
        """碰撞检测（极致优化）"""
        collision = False
        collision_info = []

        # 1. 末端执行器距离检测
        if ee_id >= 0:
            ee_pos = self.data.geom_xpos[ee_id]

            # 障碍物检测（向量化）
            obstacle_names = ['obstacle1', 'obstacle2']
            for obs_name in obstacle_names:
                obs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obs_name)
                if obs_id >= 0:
                    obs_pos = self.data.geom_xpos[obs_id]
                    dist = np.linalg.norm(ee_pos - obs_pos)
                    if dist < COLLISION_THRESHOLD:
                        collision = True
                        collision_info.append(f"末端与{obs_name}距离过近: {dist:.4f}m")

        # 2. 接触力检测
        contact_forces = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, 0, contact_forces)
        max_force = np.max(np.abs(contact_forces[:3]))
        if max_force > COLLISION_FORCE_THRESHOLD:
            collision = True
            collision_info.append(f"接触力超限: {max_force:.2f}N")

        # 3. 自碰撞检测（向量化）
        valid_links = [lid for lid in link_geom_ids if lid >= 0]
        if len(valid_links) > 1:
            link_positions = self.data.geom_xpos[valid_links]
            # 计算距离矩阵
            dist_matrix = np.linalg.norm(link_positions[:, None] - link_positions, axis=2)
            # 排除自身距离
            np.fill_diagonal(dist_matrix, np.inf)
            # 检测近距离
            min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist = dist_matrix[min_dist_idx]

            if min_dist < 0.005:
                collision = True
                collision_info.append(f"连杆{min_dist_idx[0] + 1}与连杆{min_dist_idx[1] + 1}自碰撞: {min_dist:.4f}m")

        # 更新状态
        self.collision_detected = collision
        if collision:
            self.collision_history.append((time.time(), collision_info))
            for info in collision_info:
                Utils.log(f"碰撞检测: {info}", "COLLISION")

        return collision, collision_info


# ====================== 数据记录与可视化模块（独立解耦） ======================
class DataRecorder:
    """数据记录与可视化模块"""

    def __init__(self):
        self.enabled = False
        self.data = {
            'time': [], 'qpos': [], 'qvel': [], 'err': [], 'load': [],
            'stiffness': [], 'torque': [], 'collision': []
        }
        self.record_count = 0
        self.sample_interval = 10  # 采样间隔

    def start(self):
        """开始记录"""
        self.enabled = True
        self.reset()
        Utils.log("开始数据记录")

    def stop(self, save=True, plot=True):
        """停止记录"""
        self.enabled = False
        Utils.log("停止数据记录")

        if save and self.record_count > 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self.save_data(f"run_{timestamp}")
            if plot:
                self.plot_data(f"plot_{timestamp}")

    def reset(self):
        """重置记录"""
        self.data = {k: [] for k in self.data.keys()}
        self.record_count = 0

    def record(self, qpos, qvel, err, load, stiffness, torque, collision):
        """记录数据（采样优化）"""
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

    def save_data(self, name):
        """保存数据（NPZ格式）"""
        try:
            # 转换为numpy数组
            save_data = {}
            for key, value in self.data.items():
                if key in ['time', 'load', 'collision']:
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

    def plot_data(self, name):
        """数据可视化（批量绘图）"""
        try:
            if len(self.data['time']) < 10:
                Utils.log("数据量不足，无法绘图")
                return

            # 转换数据
            time = np.array(self.data['time'])
            time -= time[0]
            qpos = np.array(self.data['qpos'])
            err = np.array(self.data['err'])
            load = np.array(self.data['load'])
            collision = np.array(self.data['collision'])

            # 创建图表（批量配置）
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('机械臂运行数据', fontsize=14)
            ax1, ax2, ax3, ax4 = axes.flatten()

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

            # 保存图表
            plt.tight_layout()
            file_path = DIR_CONFIG["data"] / f"{name}.png"
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()

            Utils.log(f"可视化图表已保存: {file_path}")
            return True
        except Exception as e:
            Utils.log(f"绘图失败: {e}", "ERROR")
            return False

    # 缓存结果
    TRAJ_CACHE[cache_key] = (traj_pos, traj_vel)
    return traj_pos, traj_vel

# ====================== 核心控制器（模块化组合） ======================
class ArmController:
    """机械臂核心控制器（模块化组合）"""

def save_traj(traj_pos, traj_vel, name):
    try:
        header = ['step'] + [f'j{i + 1}_pos' for i in range(JOINT_COUNT)] + [f'j{i + 1}_vel' for i in
                                                                             range(JOINT_COUNT)]
        data = np.hstack([np.arange(len(traj_pos))[:, None], traj_pos, traj_vel])
        np.savetxt(f"trajectories/{name}.csv", data, delimiter=',', header=','.join(header), comments='')
        log(f"轨迹保存: trajectories/{name}.csv")
    except Exception as e:
        log(f"保存轨迹失败: {e}")


def load_traj(name):
    try:
        data = np.genfromtxt(f"trajectories/{name}.csv", delimiter=',', skip_header=1)
        if len(data) == 0:
            return np.array([]), np.array([])
        return data[:, 1:JOINT_COUNT + 1], data[:, JOINT_COUNT + 1:]
    except Exception as e:
        log(f"加载轨迹失败: {e}")
        return np.array([]), np.array([])


# ====================== 核心控制器（增强版） ======================
class ArmController:
    def __init__(self):
        # 全局状态（原子操作）
        self.running = True
        self.paused = False
        self.emergency_stop = False

        # 初始化MuJoCo
        self.model, self.data = self._init_mujoco()

        # 模块化组件初始化
        self.collision_detector = CollisionDetector(self.model, self.data) if self.model else None
        self.data_recorder = DataRecorder()

        # ID缓存（预计算）
        self._init_ids()

        # 控制状态（预分配内存）
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.target = np.zeros(JOINT_COUNT)

        # 轨迹队列（双端队列优化）
        self.traj_queue = deque()
        self.current_queue_idx = 0

        # 物理状态（预分配）
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
        """ID初始化（预计算）"""
        if self.model is None:
            self.joint_ids = [-1] * JOINT_COUNT
            self.motor_ids = [-1] * JOINT_COUNT
            self.ee_id = -1
            self.link_geom_ids = [-1] * JOINT_COUNT
            return

        # 批量获取ID
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i + 1}')
                          for i in range(JOINT_COUNT)]
        self.motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i + 1}')
                          for i in range(JOINT_COUNT)]
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ee_geom')
        self.link_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'link{i + 1}')
                              for i in range(JOINT_COUNT)]

    def _init_mujoco(self):
        """MuJoCo初始化（模板化）"""
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
        <geom name="obstacle2" type="cylinder" size="0.03 0.2" pos="-0.2 0.1 0.4" rgba="1 0 1 0.5"/>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint1"/>
        <motor name="motor2" joint="joint2"/>
        <motor name="motor3" joint="joint3"/>
        <motor name="motor4" joint="joint4"/>
        <motor name="motor5" joint="joint5"/>
    </actuator>
    <sensor>
        <force name="ee_force" site="ee"/>
        <torque name="ee_torque" site="ee"/>
    </sensor>
</mujoco>
        """

        # 模板参数
        xml_params = {
            "sim_dt": SIM_DT,
            "load": self.load_set,
            "j1_min": JOINT_LIMITS[0, 0], "j1_max": JOINT_LIMITS[0, 1],
            "j2_min": JOINT_LIMITS[1, 0], "j2_max": JOINT_LIMITS[1, 1],
            "j3_min": JOINT_LIMITS[2, 0], "j3_max": JOINT_LIMITS[2, 1],
            "j4_min": JOINT_LIMITS[3, 0], "j4_max": JOINT_LIMITS[3, 1],
            "j5_min": JOINT_LIMITS[4, 0], "j5_max": JOINT_LIMITS[4, 1],
        }

        try:
            xml = xml_template.format(**xml_params)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            Utils.log("MuJoCo模型初始化成功")
            return model, data
        except Exception as e:
            Utils.log(f"MuJoCo初始化失败: {e}", "ERROR")
            self.emergency_stop = True
            self.running = False
            return None, None

    def get_states(self):
        """获取关节状态（向量化）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT)

        # 批量获取状态
        qpos = np.array([self.data.qpos[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])

        return qpos, qvel

    def _calc_load(self):
        """计算负载（向量化）"""
        if self.data is None:
            return 0.0

        # 批量计算
        forces = np.abs([self.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qpos, _ = self.get_states()

        # 负载计算（向量化）
        load = np.sum(forces * np.sin(qpos)) / 9.81
        return np.clip(load, MIN_LOAD, MAX_LOAD)

    def control_step(self):
        """核心控制步（极致优化）"""
        # 性能计时
        ctrl_start = time.time()

        # 急停处理
        if self.emergency_stop:
            if self.data is not None:
                self.data.ctrl[:] = 0.0
            return

        # 暂停处理
        if self.paused:
            return

        # 频率控制
        now = time.time()
        if now - self.last_ctrl < CTRL_DT:
            return

        # 碰撞检测
        if self.collision_detector:
            collision, _ = self.collision_detector.detect_collision(self.ee_id, self.link_geom_ids)
            if collision and not self.paused:
                self.paused = True
                Utils.log("⚠️ 碰撞检测触发，已暂停运动", "COLLISION")

        # 获取状态
        qpos, qvel = self.get_states()
        self.load_actual = self._calc_load()

        # 轨迹队列处理
        if len(self.traj_queue) > 0 and self.traj_idx >= len(self.traj_pos):
            self.current_queue_idx += 1
            if self.current_queue_idx < len(self.traj_queue):
                self.traj_pos, self.traj_vel = self.traj_queue[self.current_queue_idx]
                self.target = self.traj_pos[-1] if len(self.traj_pos) > 0 else np.zeros(JOINT_COUNT)
                self.traj_idx = 0
                Utils.log(f"执行轨迹队列第{self.current_queue_idx + 1}/{len(self.traj_queue)}段")
            else:
                self.traj_queue.clear()
                self.current_queue_idx = 0
                Utils.log("轨迹队列执行完毕")

        # 目标位置
        if self.traj_idx < len(self.traj_pos):
            target_pos = self.traj_pos[self.traj_idx]
            target_vel = self.traj_vel[self.traj_idx]
            self.traj_idx += 1
        else:
            target_pos = self.target
            target_vel = np.zeros(JOINT_COUNT)

        # 误差计算
        self.err = target_pos - qpos
        self.max_err = np.maximum(self.max_err, np.abs(self.err))

        # PD+前馈控制（完全向量化）
        load_factor = np.clip(self.load_actual / MAX_LOAD, 0.0, 1.0)
        kp = CFG.kp_base * (1 + load_factor * (CFG.kp_load_gain - 1))
        kd = CFG.kd_base * (1 + load_factor * (CFG.kd_load_gain - 1))

        pd = kp * self.err + kd * (target_vel - qvel)
        ff = CFG.ff_vel * target_vel + CFG.ff_acc * (target_vel - qvel) / CTRL_DT

        # 误差补偿（向量化）
        vel_sign = np.sign(qvel)
        vel_zero = np.abs(qvel) < 1e-4
        vel_sign[vel_zero] = np.sign(self.err)[vel_zero]

        backlash = CFG.backlash * vel_sign
        friction = np.where(vel_zero, CFG.friction * np.sign(self.err), 0.0)
        gravity = 0.5 * np.sin(qpos) * self.load_actual if CFG.gravity_comp else 0.0
        comp = backlash + friction + gravity

        # 控制输出
        ctrl = pd + ff + comp
        ctrl = np.clip(ctrl, -MAX_TORQUE, MAX_TORQUE)

        # 应用控制（批量）
        valid_motors = [(i, mid) for i, mid in enumerate(self.motor_ids) if mid >= 0]
        for i, mid in valid_motors:
            self.data.ctrl[mid] = ctrl[i]

        # 自适应刚度阻尼（向量化）
        load_ratio = np.clip(self.load_actual / MAX_LOAD, 0.0, 1.0)
        err_norm = np.clip(np.abs(self.err) / Utils.deg2rad(1.0), 0.0, 1.0)

        target_stiff = CFG.stiffness_base * (1 + load_ratio * (CFG.stiffness_load_gain - 1)) * (
                    1 + err_norm * (CFG.stiffness_error_gain - 1))
        target_stiff = np.clip(target_stiff, CFG.stiffness_min, CFG.stiffness_max)

        self.stiffness = 0.95 * self.stiffness + 0.05 * target_stiff
        self.damping = self.stiffness * CFG.damping_ratio
        self.damping = np.clip(self.damping, CFG.stiffness_min * 0.02, CFG.stiffness_max * 0.08)

        # 应用刚度阻尼（批量）
        valid_joints = [(i, jid) for i, jid in enumerate(self.joint_ids) if jid >= 0]
        for i, jid in valid_joints:
            self.model.jnt_damping[jid] = self.damping[i]

        # 数据记录
        torque = np.array([self.data.ctrl[mid] if mid >= 0 else 0.0 for mid in self.motor_ids])
        self.data_recorder.record(qpos, qvel, self.err, self.load_actual,
                                  self.stiffness, torque,
                                  self.collision_detector.collision_detected if self.collision_detector else False)

        # 性能记录
        Utils.record_perf("ctrl_time", time.time() - ctrl_start)
        self.last_ctrl = now

    # ====================== 控制接口（简洁统一） ======================
    def move_to(self, target_deg, save=False, name="default", smooth=True):
        """移动到目标位置"""
        with Utils.lock():
            target_deg = np.array(target_deg, dtype=np.float64)
            if target_deg.shape != (JOINT_COUNT,):
                Utils.log(f"目标维度错误: {target_deg.shape}", "ERROR")
                return

            start, _ = self.get_states()
            target = Utils.deg2rad(target_deg)

            # 规划轨迹
            self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_trajectory(start, target, smooth)
            self.target = target
            self.traj_idx = 0

            # 保存轨迹
            if save:
                TrajectoryPlanner.save_traj(self.traj_pos, self.traj_vel, name)

            Utils.log(f"规划轨迹: {np.round(Utils.rad2deg(start), 1)}° → {np.round(Utils.rad2deg(target), 1)}°")

    def add_to_queue(self, target_deg_list, smooth=True):
        """添加轨迹队列"""
        with Utils.lock():
            if not isinstance(target_deg_list, list) or len(target_deg_list) == 0:
                Utils.log("轨迹队列参数错误", "ERROR")
                return

            start, _ = self.get_states()
            self.traj_queue.clear()

            # 批量添加轨迹
            for target_deg in target_deg_list:
                target = Utils.deg2rad(target_deg)
                traj_pos, traj_vel = TrajectoryPlanner.plan_trajectory(start, target, smooth)
                self.traj_queue.append((traj_pos, traj_vel))
                start = target

            self.current_queue_idx = 0
            if self.traj_queue:
                self.traj_pos, self.traj_vel = self.traj_queue[0]
                self.target = self.traj_pos[-1]
                self.traj_idx = 0

            Utils.log(f"轨迹队列已添加 {len(self.traj_queue)} 段轨迹")

    def control_joint(self, idx, target_deg, smooth=True):
        """单独控制关节"""
        with Utils.lock():
            if not (0 <= idx < JOINT_COUNT):
                Utils.log(f"无效关节索引: {idx}", "ERROR")
                return

            current, _ = self.get_states()
            target = current.copy()
            target[idx] = Utils.deg2rad(target_deg)

            self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_trajectory(current, target, smooth)
            self.target = target
            self.traj_idx = 0

            Utils.log(f"控制关节{idx + 1}: {np.round(Utils.rad2deg(current[idx]), 1)}° → {target_deg:.1f}°")

    def set_load(self, mass):
        """设置负载"""
        with Utils.lock():
            mass = float(mass)
            if not (MIN_LOAD <= mass <= MAX_LOAD):
                Utils.log(f"负载超出范围: {mass}kg (0-2kg)", "ERROR")
                return

            self.load_set = mass
            if self.ee_id >= 0 and self.model is not None:
                self.model.geom_mass[self.ee_id] = mass

            Utils.log(f"负载设置为: {mass}kg")

    def pause(self):
        """暂停"""
        with Utils.lock():
            self.paused = True
            Utils.log("运动已暂停")

    def resume(self):
        """恢复"""
        with Utils.lock():
            self.paused = False
            Utils.log("运动已恢复")

    def emergency_stop(self):
        """紧急停止"""
        with Utils.lock():
            self.emergency_stop = True
            self.paused = True
            self.running = False
            Utils.log("⚠️ 紧急停止已触发", "ERROR")

    def reset_collision(self):
        """重置碰撞状态"""
        with Utils.lock():
            if self.collision_detector:
                self.collision_detector.collision_detected = False
            self.paused = False
            Utils.log("碰撞状态已重置")

    def adjust_param(self, param, value, idx=None):
        """调整控制参数"""
        with Utils.lock():
            if not hasattr(CFG, param):
                Utils.log(f"无效参数: {param}", "ERROR")
                return

            current = getattr(CFG, param)
            if isinstance(current, np.ndarray):
                if idx is None:
                    setattr(CFG, param, np.full(JOINT_COUNT, value))
                    Utils.log(f"参数 {param} 全部更新为: {value}")
                elif 0 <= idx < JOINT_COUNT:
                    current[idx] = value
                    setattr(CFG, param, current)
                    Utils.log(f"参数 {param} 关节{idx + 1}更新为: {value}")
                else:
                    Utils.log(f"无效索引: {idx}", "ERROR")
            else:
                setattr(CFG, param, value)
                Utils.log(f"参数 {param} 更新为: {value}")

    def preset_pose(self, pose_name):
        """预设姿态"""
        poses = {
            'zero': [0, 0, 0, 0, 0],
            'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10],
            'test': [10, 20, 15, 5, 8],
            'avoid': [15, 25, 10, 5, 0]
        }

        if pose_name in poses:
            self.move_to(poses[pose_name])
        else:
            Utils.log(f"未知姿态: {pose_name} (支持: {list(poses.keys())})", "ERROR")

    def start_recording(self):
        """开始数据记录（新增）"""
        with lock():
            self.record_enabled = True
            self.data_recorder = {k: [] for k in self.data_recorder.keys()}
            self.record_count = 0
            log("开始数据记录")

    def stop_recording(self, save_data=True, plot_data=True):
        """停止数据记录（新增）"""
        with lock():
            self.record_enabled = False
            log("停止数据记录")

            if save_data:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                self._save_recorded_data(f"run_{timestamp}")
                if plot_data:
                    self._plot_data(f"plot_{timestamp}")

    def _print_status(self):
        """打印状态（低频更新）"""
        now = time.time()
        if now - self.last_status < 1.0:
            return

        # 性能统计
        fps = self.fps_count / (now - self.last_status)
        perf_stats = Utils.get_perf_stats()
        ctrl_time = perf_stats.get("ctrl_time", {}).get("avg", 0) * 1000

        # 状态信息
        qpos, qvel = self.get_states()
        err = Utils.rad2deg(self.err)
        max_err = Utils.rad2deg(self.max_err)

        # 状态文本
        status_flags = []
        if self.paused: status_flags.append("暂停")
        if self.emergency_stop: status_flags.append("急停")
        if self.collision_detector and self.collision_detector.collision_detected:
            status_flags.append("碰撞")

        status_str = " | ".join(status_flags) if status_flags else "运行中"

        # 打印状态
        Utils.log("=" * 80)
        Utils.log(f"状态: {status_str} | 步数: {self.step} | FPS: {fps:.1f} | 控制耗时: {ctrl_time:.2f}ms")
        Utils.log(f"负载: {self.load_set:.1f}kg(设定) | {self.load_actual:.1f}kg(实际)")
        Utils.log(
            f"角度: {np.round(Utils.rad2deg(qpos), 1)}° | 误差: {np.round(np.abs(err), 3)}°(最大:{np.round(max_err, 3)}°)")
        Utils.log(
            f"轨迹队列: {len(self.traj_queue)}段 | 当前段: {self.current_queue_idx + 1 if self.traj_queue else 0}")
        Utils.log("=" * 80)

        self.last_status = now
        self.fps_count = 0

    def _interactive_shell(self):
        """交互式命令行"""
        help_text = """
=== 机械臂控制器命令列表 ===
基础控制:
  help          - 查看帮助
  pause/resume  - 暂停/恢复运动
  stop          - 紧急停止
  reset_collision - 重置碰撞检测状态

运动控制:
  pose [名称]   - 预设姿态(zero/up/grasp/test/avoid)
  joint [索引] [角度] - 控制单个关节
  load [kg]     - 设置负载(0-2kg)
  queue [姿态1,姿态2,...] - 添加轨迹队列
  clear_queue   - 清空轨迹队列

参数管理:
  param [名] [值] [关节] - 调整控制参数
  save_params [名] - 保存参数到文件
  load_params [名] - 从文件加载参数

轨迹管理:
  save [名]     - 保存当前轨迹
  load_traj [名] - 加载轨迹文件

数据记录:
  record_start  - 开始数据记录
  record_stop   - 停止数据记录并保存
"""
        Utils.log(help_text)

        while self.running and not self.emergency_stop:
            try:
                cmd = input("> ").strip().lower()
                if not cmd:
                    continue

                parts = cmd.split()
                cmd_map = {
                    'help': lambda: Utils.log(help_text),
                    'pause': self.pause,
                    'resume': self.resume,
                    'stop': self.emergency_stop,
                    'reset_collision': self.reset_collision,
                    'pose': lambda: self.preset_pose(parts[1]) if len(parts) >= 2 else Utils.log("缺少姿态名称"),
                    'joint': lambda: self.control_joint(int(parts[1]) - 1, float(parts[2])) if len(
                        parts) >= 3 else Utils.log("参数错误: joint [索引] [角度]"),
                    'load': lambda: self.set_load(float(parts[1])) if len(parts) >= 2 else Utils.log(
                        "参数错误: load [kg]"),
                    'queue': lambda: self._handle_queue_cmd(parts[1:]) if len(parts) >= 2 else Utils.log(
                        "参数错误: queue [姿态1] [姿态2] ..."),
                    'clear_queue': lambda: self.traj_queue.clear() or Utils.log("轨迹队列已清空"),
                    'param': lambda: self.adjust_param(parts[1], float(parts[2]),
                                                       int(parts[3]) - 1 if len(parts) >= 4 else None) if len(
                        parts) >= 3 else Utils.log("参数错误: param [名] [值] [关节]"),
                    'save_params': lambda: ParamPersistence.save_params(CFG, parts[1]) if len(
                        parts) >= 2 else ParamPersistence.save_params(CFG),
                    'load_params': lambda: ParamPersistence.load_params(parts[1]) if len(
                        parts) >= 2 else ParamPersistence.load_params(),
                    'save': lambda: self.move_to(Utils.rad2deg(self.target), save=True, name=parts[1]) if len(
                        parts) >= 2 else Utils.log("参数错误: save [名称]"),
                    'load_traj': lambda: self._load_trajectory(parts[1]) if len(parts) >= 2 else Utils.log(
                        "参数错误: load_traj [名称]"),
                    'record_start': self.data_recorder.start,
                    'record_stop': lambda: self.data_recorder.stop()
                }

                if parts[0] in cmd_map:
                    cmd_map[parts[0]]()
                else:
                    Utils.log("未知命令，输入help查看帮助")

            except Exception as e:
                Utils.log(f"命令执行错误: {e}", "ERROR")

    def _handle_queue_cmd(self, pose_names):
        """处理队列命令"""
        pose_map = {
            'zero': [0, 0, 0, 0, 0], 'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10], 'test': [10, 20, 15, 5, 8],
            'avoid': [15, 25, 10, 5, 0]
        }

        targets = []
        for pose in pose_names:
            if pose in pose_map:
                targets.append(pose_map[pose])
            else:
                Utils.log(f"未知姿态: {pose}", "ERROR")

        if targets:
            self.add_to_queue(targets)

    def _load_trajectory(self, name):
        """加载轨迹"""
        traj_pos, traj_vel = TrajectoryPlanner.load_traj(name)
        if len(traj_pos) > 0:
            self.traj_pos = traj_pos
            self.traj_vel = traj_vel
            self.target = traj_pos[-1]
            self.traj_idx = 0
            Utils.log(f"加载轨迹: {name} (共{len(traj_pos)}步)")

    def run(self):
        """主运行循环"""
        # 初始化Viewer
        try:
            if self.model and self.data:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                Utils.log("Viewer启动成功")
            else:
                raise RuntimeError("模型未初始化")
        except Exception as e:
            Utils.log(f"Viewer启动失败: {e}", "ERROR")
            self.running = False
            return

        # 启动交互线程
        threading.Thread(target=self._interactive_shell, daemon=True).start()

        # 演示程序
        self._run_demo()

        # 主循环
        Utils.log("机械臂控制器启动成功！输入help查看命令列表")
        Utils.log(f"核心配置: 控制频率={CTRL_FREQ}Hz, 仿真步长={SIM_DT * 1000}ms")

        while self.running and self.viewer.is_running():
            step_start = time.time()

            try:
                # 控制步
                self.control_step()

                # 仿真步
                if self.data:
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.sync()

                # 状态打印
                self._print_status()

                # 性能统计
                self.step += 1
                self.fps_count += 1
                Utils.record_perf("step_time", time.time() - step_start)

                # 限流
                time.sleep(max(0, SLEEP_DT - (time.time() - step_start)))

            except Exception as e:
                Utils.log(f"主循环错误: {e}", "ERROR")
                continue

        # 资源清理
        self._cleanup()

        # 最终统计
        max_err = Utils.rad2deg(np.max(self.max_err))
        perf_stats = Utils.get_perf_stats()
        avg_ctrl_time = perf_stats.get("ctrl_time", {}).get("avg", 0) * 1000

        Utils.log("=" * 80)
        Utils.log("控制器停止运行 - 最终统计")
        Utils.log(f"总步数: {self.step} | 最大跟踪误差: {np.round(max_err, 3)}°")
        Utils.log(
            f"平均控制耗时: {avg_ctrl_time:.2f}ms | 平均帧率: {self.fps_count / max(1, self.step * SIM_DT):.1f}FPS")
        Utils.log("=" * 80)

    def _run_demo(self):
        """演示程序"""

        def demo_task():
            # 保存默认参数
            ParamPersistence.save_params(CFG)

            # 开始记录
            self.data_recorder.start()

            # 演示步骤
            demo_steps = [
                (2, lambda: self.preset_pose('zero')),
                (3, lambda: self.preset_pose('test')),
                (2, self.pause),
                (2, self.resume),
                (4, lambda: self.set_load(1.5)),
                (4, lambda: self.preset_pose('grasp')),
                (1, self.reset_collision),
                (1, lambda: self.control_joint(0, 10)),
                (3, lambda: self.set_load(0.2)),
                (3, lambda: self.add_to_queue(
                    [[0, 0, 0, 0, 0], [0, 30, 20, 10, 0], [15, 25, 10, 5, 0], [0, 45, 30, 20, 10]])),
                (8, lambda: self.data_recorder.stop()),
                (2, lambda: self.preset_pose('zero')),
            ]

            for delay, func in demo_steps:
                time.sleep(delay)
                if not self.running or self.emergency_stop:
                    break
                try:
                    func()
                except Exception as e:
                    Utils.log(f"演示步骤错误: {e}", "ERROR")

        threading.Thread(target=demo_task, daemon=True).start()

    def _cleanup(self):
        """资源清理"""
        # 停止记录
        if self.data_recorder.enabled:
            self.data_recorder.stop()

        # 关闭Viewer
        if self.viewer:
            self.viewer.close()

        # 清空缓存
        TrajectoryPlanner.clear_cache()

        # 保存最终参数
        ParamPersistence.save_params(CFG, "last_session")

        Utils.log("资源清理完成")


# ====================== 信号处理与主函数 ======================
def signal_handler(sig, frame):
    """信号处理"""
    Utils.log(f"收到信号 {sig}，正在优雅退出...")
    if 'controller' in globals():
        controller.emergency_stop()


def main():
    """主函数"""
    # 配置numpy
    np.set_printoptions(precision=3, suppress=True)

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 初始化控制器
    global controller
    controller = ArmController()

    # 运行控制器
    try:
        controller.run()
    except Exception as e:
        Utils.log(f"程序异常退出: {e}", "ERROR")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()