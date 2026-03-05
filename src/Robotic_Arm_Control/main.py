#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节运动性能优化控制器（最终优化版）
核心特性：
- 基础功能：梯形轨迹/自适应刚度/误差补偿/PD+前馈控制
- 增强功能：轨迹暂停恢复/单独关节控制/负载监测/轨迹管理/紧急停止/动态调参
- 工程特性：极致性能/模块化设计/自动化资源管理/完善异常处理
"""

import sys
import time
import signal
import threading
import csv
import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Final, Callable, Any
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from functools import lru_cache

# ====================== 常量定义（不可变） ======================
JOINT_COUNT: Final[int] = 5
LOG_FILE: Final[str] = "arm_controller.log"
TRAJECTORY_DIR: Final[str] = "trajectories"
MIN_LOAD: Final[float] = 0.0
MAX_LOAD: Final[float] = 2.0
DEG_TO_RAD: Final[float] = np.pi / 180.0
RAD_TO_DEG: Final[float] = 180.0 / np.pi

# 初始化目录
Path(TRAJECTORY_DIR).mkdir(exist_ok=True)


# ====================== 配置类（纯数据，优化内存） ======================
@dataclass(frozen=True)
class JointLimits:
    """关节物理极限配置（预计算常量）"""
    limits: np.ndarray = field(default_factory=lambda: np.array([
        [-np.pi, np.pi], [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2],
        [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2]
    ], dtype=np.float64))
    max_vel: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, 0.8, 0.6, 0.6]))
    max_acc: np.ndarray = field(default_factory=lambda: np.array([2.0, 1.6, 1.6, 1.2, 1.2]))
    max_torque: np.ndarray = field(default_factory=lambda: np.array([15.0, 12.0, 10.0, 8.0, 5.0]))


@dataclass
class ControlParams:
    """控制参数（支持动态调整，优化内存布局）"""
    # PD控制
    kp_base: float = 120.0
    kd_base: float = 8.0
    kp_load_gain: float = 1.8
    kd_load_gain: float = 1.5

    # 前馈控制
    ff_vel: float = 0.7
    ff_acc: float = 0.5

    # 误差补偿
    backlash: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.001, 0.002, 0.002, 0.003]))
    friction: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.08, 0.08, 0.06, 0.06]))
    gravity_comp: bool = True

    # 刚度阻尼
    stiffness_base: np.ndarray = field(default_factory=lambda: np.array([200.0, 180.0, 150.0, 120.0, 80.0]))
    stiffness_load_gain: float = 1.8
    stiffness_error_gain: float = 1.5
    stiffness_min: np.ndarray = field(default_factory=lambda: np.array([100.0, 90.0, 75.0, 60.0, 40.0]))
    stiffness_max: np.ndarray = field(default_factory=lambda: np.array([300.0, 270.0, 225.0, 180.0, 120.0]))
    damping_ratio: float = 0.04


@dataclass
class TimeConfig:
    """时间配置（预计算派生参数）"""
    sim_dt: float = 0.0005
    ctrl_freq: int = 2000
    fps: int = 60

    ctrl_dt: float = field(init=False)
    sleep_dt: float = field(init=False)

    def __post_init__(self):
        self.ctrl_dt = 1.0 / self.ctrl_freq
        self.sleep_dt = 1.0 / self.fps


# ====================== 全局状态（精简，线程安全） ======================
# 配置实例（单例）
JOINT_CFG = JointLimits()
CTRL_CFG = ControlParams()
TIME_CFG = TimeConfig()

# 全局状态（使用原子操作）
RUNNING: bool = True
PAUSED: bool = False
EMERGENCY_STOP: bool = False
GLOBAL_LOCK = threading.Lock()


# ====================== 工具函数（纯函数，极致优化） ======================
@contextmanager
def thread_safe():
    """线程安全上下文管理器（简化实现）"""
    with GLOBAL_LOCK:
        yield


def log(content: str):
    """标准化日志（优化IO操作）"""
    try:
        with thread_safe():
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            msg = f"[{ts}] {content}"
            with open(LOG_FILE, "a", encoding="utf-8", buffering=1) as f:
                f.write(f"{msg}\n")
            print(msg)
    except Exception as e:
        print(f"日志错误: {e} | 内容: {content}")


def deg2rad(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """角度转弧度（向量化，无分支）"""
    try:
        return np.asarray(deg, dtype=np.float64) * DEG_TO_RAD
    except:
        return 0.0 if np.isscalar(deg) else np.zeros(JOINT_COUNT)


def rad2deg(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """弧度转角度（向量化，无分支）"""
    try:
        return np.asarray(rad, dtype=np.float64) * RAD_TO_DEG
    except:
        return 0.0 if np.isscalar(rad) else np.zeros(JOINT_COUNT)


@lru_cache(maxsize=100)
def get_mujoco_id(model: mujoco.MjModel, obj_type: str, name: str) -> int:
    """获取MuJoCo ID（缓存优化）"""
    type_map = {'joint': mujoco.mjtObj.mjOBJ_JOINT, 'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
                'geom': mujoco.mjtObj.mjOBJ_GEOM}
    try:
        return mujoco.mj_name2id(model, type_map.get(obj_type, mujoco.mjtObj.mjOBJ_JOINT), name)
    except:
        return -1


def calculate_actual_load(controller: Any) -> float:
    """计算实际负载（优化算法）"""
    if controller.data is None:
        return 0.0

    # 向量化力矩计算
    forces = np.abs([controller.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in controller.joint_ids])
    qpos, _ = controller.get_joint_states()

    # 优化的负载估算公式
    load = np.sum(forces * np.sin(qpos)) / 9.81
    return np.clip(load, MIN_LOAD, MAX_LOAD)


# ====================== 轨迹规划器（纯算法，极致优化） ======================
class TrajectoryPlanner:
    """梯形轨迹规划器（无状态，缓存优化）"""
    _cache = {}

    @staticmethod
    def _plan_single(start: float, target: float, max_vel: float, max_acc: float, dt: float) -> Tuple[
        np.ndarray, np.ndarray]:
        """单关节轨迹规划（纯向量化，无循环）"""
        delta = target - start
        if abs(delta) < 1e-5:
            return np.array([target]), np.array([0.0])

        dir = np.sign(delta)
        dist = abs(delta)

        # 预计算轨迹参数
        accel_dist = (max_vel ** 2) / (2 * max_acc)
        if dist <= 2 * accel_dist:
            peak_vel = np.sqrt(dist * max_acc)
            accel_time = peak_vel / max_acc
            total_time = 2 * accel_time
        else:
            accel_time = max_vel / max_acc
            uniform_time = (dist - 2 * accel_dist) / max_vel
            total_time = 2 * accel_time + uniform_time

        # 时间序列（预分配内存）
        t = np.arange(0, total_time + dt, dt, dtype=np.float64)
        pos = np.empty_like(t)
        vel = np.empty_like(t)

        # 向量化分段计算（无分支）
        mask_acc = t <= accel_time
        mask_uni = (t > accel_time) & (t <= accel_time + uniform_time) if dist > 2 * accel_dist else np.zeros_like(t,
                                                                                                                   dtype=bool)
        mask_dec = ~(mask_acc | mask_uni)

        # 加速段
        vel[mask_acc] = max_acc * t[mask_acc] * dir
        pos[mask_acc] = start + 0.5 * max_acc * np.square(t[mask_acc]) * dir

        # 匀速段
        if dist > 2 * accel_dist:
            t_uni = t[mask_uni] - accel_time
            vel[mask_uni] = max_vel * dir
            pos[mask_uni] = start + (accel_dist + max_vel * t_uni) * dir

            # 减速段
            t_dec = t[mask_dec] - (accel_time + uniform_time)
            vel[mask_dec] = (max_vel - max_acc * t_dec) * dir
            pos[mask_dec] = start + (dist - (accel_dist - 0.5 * max_acc * np.square(t_dec))) * dir
        else:
            # 减速段（无匀速）
            t_dec = t[mask_dec] - accel_time
            vel[mask_dec] = (peak_vel - max_acc * t_dec) * dir
            pos[mask_dec] = start + (peak_vel * accel_time - 0.5 * max_acc * np.square(t_dec)) * dir

        # 强制终点
        pos[-1], vel[-1] = target, 0.0
        return pos, vel

    @classmethod
    def plan_joints(cls, start: np.ndarray, target: np.ndarray, dt: float = TIME_CFG.ctrl_dt) -> Tuple[
        np.ndarray, np.ndarray]:
        """多关节轨迹规划（批量处理，缓存优化）"""
        # 边界裁剪
        start = np.clip(start, JOINT_CFG.limits[:, 0] + 0.01, JOINT_CFG.limits[:, 1] - 0.01)
        target = np.clip(target, JOINT_CFG.limits[:, 0] + 0.01, JOINT_CFG.limits[:, 1] - 0.01)

        # 缓存键
        cache_key = (hash(start.tobytes()), hash(target.tobytes()))
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # 批量规划
        traj_pos, traj_vel, max_len = [], [], 1
        for i in range(JOINT_COUNT):
            pos, vel = cls._plan_single(start[i], target[i], JOINT_CFG.max_vel[i], JOINT_CFG.max_acc[i], dt)
            traj_pos.append(pos)
            traj_vel.append(vel)
            max_len = max(max_len, len(pos))

        # 统一长度（向量化填充）
        for i in range(JOINT_COUNT):
            if len(traj_pos[i]) < max_len:
                pad = max_len - len(traj_pos[i])
                traj_pos[i] = np.pad(traj_pos[i], (0, pad), 'constant', constant_values=target[i])
                traj_vel[i] = np.pad(traj_vel[i], (0, pad), 'constant')

        # 转换为数组并缓存
        result_pos, result_vel = np.array(traj_pos).T, np.array(traj_vel).T
        cls._cache[cache_key] = (result_pos, result_vel)
        return result_pos, result_vel

    @classmethod
    def plan_single_joint(cls, joint_idx: int, start_deg: float, target_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """单独关节规划（简化接口）"""
        if not (0 <= joint_idx < JOINT_COUNT):
            log(f"无效关节索引: {joint_idx}")
            return np.array([]), np.array([])

        start = np.zeros(JOINT_COUNT)
        target = np.zeros(JOINT_COUNT)
        start[joint_idx] = deg2rad(start_deg)
        target[joint_idx] = deg2rad(target_deg)

        return cls.plan_joints(start, target)

    @classmethod
    def save(cls, traj_pos: np.ndarray, traj_vel: np.ndarray, name: str):
        """保存轨迹（优化IO）"""
        path = Path(TRAJECTORY_DIR) / f"{name}.csv"
        try:
            header = ['step'] + [f'j{i + 1}_pos' for i in range(JOINT_COUNT)] + [f'j{i + 1}_vel' for i in
                                                                                 range(JOINT_COUNT)]
            data = np.hstack([np.arange(len(traj_pos))[:, None], traj_pos, traj_vel])
            np.savetxt(path, data, delimiter=',', header=','.join(header), comments='', fmt='%.6f')
            log(f"轨迹保存: {path}")
        except Exception as e:
            log(f"保存轨迹失败: {e}")

    @classmethod
    def load(cls, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载轨迹（优化IO）"""
        path = Path(TRAJECTORY_DIR) / f"{name}.csv"
        if not path.exists():
            log(f"轨迹文件不存在: {path}")
            return np.array([]), np.array([])

        try:
            data = np.genfromtxt(path, delimiter=',', skip_header=1)
            if len(data) == 0:
                return np.array([]), np.array([])

            traj_pos = data[:, 1:JOINT_COUNT + 1]
            traj_vel = data[:, JOINT_COUNT + 1:]
            log(f"轨迹加载: {path} (共{len(traj_pos)}步)")
            return traj_pos, traj_vel
        except Exception as e:
            log(f"加载轨迹失败: {e}")
            return np.array([]), np.array([])

    @classmethod
    def clear_cache(cls):
        """清理缓存（内存管理）"""
        cls._cache.clear()
        log("轨迹缓存已清理")


# ====================== 机械臂控制器（核心类，极致优化） ======================
class ArmController:
    """机械臂控制器（模块化，性能优化）"""

    def __init__(self):
        # MuJoCo核心对象
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer: Optional[mujoco.viewer.Viewer] = None

        # ID缓存
        self.joint_ids: List[int] = []
        self.motor_ids: List[int] = []
        self.ee_geom_id: int = -1

        # 控制状态（预分配内存）
        self.traj_pos: np.ndarray = np.zeros((1, JOINT_COUNT))
        self.traj_vel: np.ndarray = np.zeros((1, JOINT_COUNT))
        self.traj_idx: int = 0
        self.saved_traj_idx: int = 0
        self.target: np.ndarray = np.zeros(JOINT_COUNT)

        # 物理状态
        self.stiffness: np.ndarray = CTRL_CFG.stiffness_base.copy()
        self.damping: np.ndarray = self.stiffness * CTRL_CFG.damping_ratio
        self.load_set: float = 0.5
        self.load_actual: float = 0.5

        # 误差状态
        self.err: np.ndarray = np.zeros(JOINT_COUNT)
        self.max_err: np.ndarray = np.zeros(JOINT_COUNT)

        # 性能统计
        self.step: int = 0
        self.last_ctrl: float = time.time()
        self.last_print: float = time.time()
        self.fps_count: int = 0

        # 初始化流程
        self._init_mujoco()
        self._init_ids()
        self._reset()
        log("控制器初始化完成")

    def _init_mujoco(self):
        """初始化MuJoCo（优化XML生成）"""
        try:
            xml = self._generate_xml()
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            log("MuJoCo模型加载成功")
        except Exception as e:
            log(f"模型初始化失败: {e}")
            global RUNNING, EMERGENCY_STOP
            RUNNING = False
            EMERGENCY_STOP = True

    def _generate_xml(self) -> str:
        """生成MuJoCo XML（优化字符串拼接）"""
        link_masses = [0.8, 0.6, 0.6, 0.4, 0.2]
        friction = CTRL_CFG.friction

        xml_template = f"""
<mujoco model="arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{TIME_CFG.sim_dt}" gravity="0 0 -9.81" iterations="100" tolerance="1e-9"/>
    <default>
        <joint type="hinge" limited="true" margin="0.001"/>
        <motor ctrllimited="true" ctrlrange="-1 1" gear="100"/>
        <geom contype="1" conaffinity="1" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
    <asset>
        <material name="arm" rgba="0.0 0.8 0.0 0.8"/>
        <material name="ee" rgba="0.8 0.2 0.2 1"/>
    </asset>
    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" axis="0 0 1" pos="0 0 0.1" range="{JOINT_CFG.limits[0, 0]} {JOINT_CFG.limits[0, 1]}"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1" type="cylinder" size="0.04 0.18" mass="{link_masses[0]}" material="arm" friction="{friction[0]} {friction[0]} {friction[0]}"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18" range="{JOINT_CFG.limits[1, 0]} {JOINT_CFG.limits[1, 1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2" type="cylinder" size="0.04 0.18" mass="{link_masses[1]}" material="arm" friction="{friction[1]} {friction[1]} {friction[1]}"/>
                    <joint name="joint3" axis="0 1 0" pos="0 0 0.18" range="{JOINT_CFG.limits[2, 0]} {JOINT_CFG.limits[2, 1]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3" type="cylinder" size="0.04 0.18" mass="{link_masses[2]}" material="arm" friction="{friction[2]} {friction[2]} {friction[2]}"/>
                        <joint name="joint4" axis="0 1 0" pos="0 0 0.18" range="{JOINT_CFG.limits[3, 0]} {JOINT_CFG.limits[3, 1]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4" type="cylinder" size="0.04 0.18" mass="{link_masses[3]}" material="arm" friction="{friction[3]} {friction[3]} {friction[3]}"/>
                            <joint name="joint5" axis="0 1 0" pos="0 0 0.18" range="{JOINT_CFG.limits[4, 0]} {JOINT_CFG.limits[4, 1]}"/>
                            <body name="ee" pos="0 0 0.18">
                                <geom name="ee_geom" type="sphere" size="0.04" mass="{self.load_set}" material="ee"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint1"/>
        <motor name="motor2" joint="joint2"/>
        <motor name="motor3" joint="joint3"/>
        <motor name="motor4" joint="joint4"/>
        <motor name="motor5" joint="joint5"/>
    </actuator>
</mujoco>
        """
        return xml_template

    def _init_ids(self):
        """初始化ID（缓存优化）"""
        if self.model is None:
            return

        self.joint_ids = [get_mujoco_id(self.model, 'joint', f"joint{i + 1}") for i in range(JOINT_COUNT)]
        self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
        self.ee_geom_id = get_mujoco_id(self.model, 'geom', "ee_geom")

        # 初始化阻尼
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                self.model.jnt_damping[jid] = self.damping[i]

    def _reset(self):
        """重置状态（内存复用）"""
        self.target.fill(0.0)
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.saved_traj_idx = 0
        self.err.fill(0.0)
        self.max_err.fill(0.0)

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取关节状态（向量化，无分支）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT)

        qpos = np.array([self.data.qpos[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        return qpos, qvel

    def _calc_compensation(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """计算误差补偿（纯向量化）"""
        # 间隙补偿
        vel_sign = np.sign(qvel)
        vel_zero = np.abs(qvel) < 1e-4
        vel_sign[vel_zero] = np.sign(self.err)[vel_zero]
        backlash = CTRL_CFG.backlash * vel_sign

        # 摩擦补偿
        friction = np.where(vel_zero, CTRL_CFG.friction * np.sign(self.err), 0.0)

        # 重力补偿
        gravity = 0.5 * np.sin(qpos) * self.load_actual if CTRL_CFG.gravity_comp else 0.0

        return backlash + friction + gravity

    def _update_stiffness(self, qpos: np.ndarray, qvel: np.ndarray):
        """更新刚度阻尼（向量化，无分支）"""
        if self.data is None:
            return

        # 负载比例
        load_ratio = np.clip(self.load_actual / MAX_LOAD, 0.0, 1.0)

        # 误差比例
        err_norm = np.clip(np.abs(self.err) / deg2rad(1.0), 0.0, 1.0)

        # 计算目标刚度
        target_stiff = CTRL_CFG.stiffness_base * (1 + load_ratio * (CTRL_CFG.stiffness_load_gain - 1)) * \
                       (1 + err_norm * (CTRL_CFG.stiffness_error_gain - 1))
        target_stiff = np.clip(target_stiff, CTRL_CFG.stiffness_min, CTRL_CFG.stiffness_max)

        # 平滑更新
        self.stiffness = 0.95 * self.stiffness + 0.05 * target_stiff

        # 更新阻尼
        self.damping = self.stiffness * CTRL_CFG.damping_ratio
        self.damping = np.clip(self.damping, CTRL_CFG.stiffness_min * 0.02, CTRL_CFG.stiffness_max * 0.08)

        # 应用到模型
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                self.model.jnt_damping[jid] = self.damping[i]

    def control_step(self):
        """单步控制（核心逻辑，极致优化）"""
        global PAUSED, EMERGENCY_STOP

        # 急停处理
        if EMERGENCY_STOP:
            if self.data is not None:
                self.data.ctrl[:] = 0.0
            return

        # 暂停处理
        if PAUSED:
            self.saved_traj_idx = self.traj_idx
            return

        # 频率限流
        now = time.time()
        if now - self.last_ctrl < TIME_CFG.ctrl_dt:
            return

        # 获取状态
        qpos, qvel = self.get_joint_states()
        self.load_actual = calculate_actual_load(self)

        # 获取目标点
        if self.traj_idx < len(self.traj_pos):
            target_pos = self.traj_pos[self.traj_idx]
            target_vel = self.traj_vel[self.traj_idx]
            self.traj_idx += 1
        else:
            target_pos = self.target
            target_vel = np.zeros(JOINT_COUNT)

        # 计算误差
        self.err = target_pos - qpos
        self.max_err = np.maximum(self.max_err, np.abs(self.err))

        # PD控制
        load_factor = np.clip(self.load_actual / MAX_LOAD, 0.0, 1.0)
        kp = CTRL_CFG.kp_base * (1 + load_factor * (CTRL_CFG.kp_load_gain - 1))
        kd = CTRL_CFG.kd_base * (1 + load_factor * (CTRL_CFG.kd_load_gain - 1))
        pd = kp * self.err + kd * (target_vel - qvel)

        # 前馈控制
        ff = CTRL_CFG.ff_vel * target_vel + CTRL_CFG.ff_acc * (target_vel - qvel) / TIME_CFG.ctrl_dt

        # 误差补偿
        comp = self._calc_compensation(qpos, qvel)

        # 总控制输出
        ctrl = pd + ff + comp
        ctrl = np.clip(ctrl, -JOINT_CFG.max_torque, JOINT_CFG.max_torque)

        # 应用控制
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = ctrl[i]

        # 更新刚度
        self._update_stiffness(qpos, qvel)

        # 更新时间戳
        self.last_ctrl = now

    # ====================== 控制接口（简化，统一） ======================
    def set_load(self, mass: float):
        """设置负载（范围检查）"""
        with thread_safe():
            if not (MIN_LOAD <= mass <= MAX_LOAD):
                log(f"负载超出范围: {mass}kg (0-2kg)")
                return

            self.load_set = mass
            if self.ee_geom_id >= 0 and self.model is not None:
                self.model.geom_mass[self.ee_geom_id] = mass
            log(f"负载设置为: {mass}kg")

    def move_to(self, target_deg: Union[List[float], np.ndarray], save_traj: bool = False, traj_name: str = "default"):
        """移动到目标角度（简化接口）"""
        with thread_safe():
            target_deg = np.asarray(target_deg, dtype=np.float64)
            if target_deg.shape != (JOINT_COUNT,):
                log(f"目标角度维度错误: {target_deg.shape} (期望{JOINT_COUNT}维)")
                return

            start, _ = self.get_joint_states()
            target = deg2rad(target_deg)

            self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_joints(start, target)
            self.target = target
            self.traj_idx = 0
            self.saved_traj_idx = 0

            # 保存轨迹
            if save_traj:
                TrajectoryPlanner.save(self.traj_pos, self.traj_vel, traj_name)

            log(f"规划轨迹: {np.round(rad2deg(start), 1)}° → {np.round(rad2deg(target), 1)}°")

    def control_joint(self, joint_idx: int, target_deg: float):
        """单独控制关节（简化接口）"""
        if not (0 <= joint_idx < JOINT_COUNT):
            log(f"无效关节索引: {joint_idx}")
            return

        current, _ = self.get_joint_states()
        current_deg = rad2deg(current[joint_idx])

        self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_single_joint(joint_idx, current_deg, target_deg)
        self.target = current.copy()
        self.target[joint_idx] = deg2rad(target_deg)
        self.traj_idx = 0

        log(f"控制关节{joint_idx + 1}: {current_deg:.1f}° → {target_deg:.1f}°")

    def load_trajectory(self, name: str):
        """加载轨迹（简化接口）"""
        with thread_safe():
            traj_pos, traj_vel = TrajectoryPlanner.load(name)
            if len(traj_pos) == 0:
                return

            self.traj_pos = traj_pos
            self.traj_vel = traj_vel
            self.target = traj_pos[-1] if len(traj_pos) > 0 else np.zeros(JOINT_COUNT)
            self.traj_idx = 0
            log(f"加载轨迹: {name} (共{len(traj_pos)}步)")

    def pause(self):
        """暂停轨迹"""
        global PAUSED
        with thread_safe():
            PAUSED = True
            log("轨迹暂停")

    def resume(self):
        """恢复轨迹"""
        global PAUSED
        with thread_safe():
            PAUSED = False
            self.traj_idx = self.saved_traj_idx
            log(f"轨迹恢复（从第{self.saved_traj_idx}步开始）")

    def emergency_stop(self):
        """紧急停止"""
        global RUNNING, PAUSED, EMERGENCY_STOP
        with thread_safe():
            EMERGENCY_STOP = True
            PAUSED = True
            RUNNING = False
            log("⚠️ 紧急停止触发！")

    def adjust_param(self, param: str, value: float, joint_idx: Optional[int] = None):
        """调整控制参数（简化逻辑）"""
        with thread_safe():
            if not hasattr(CTRL_CFG, param):
                log(f"无效参数: {param}")
                return

            current = getattr(CTRL_CFG, param)
            if isinstance(current, np.ndarray):
                if joint_idx is None:
                    setattr(CTRL_CFG, param, np.full(JOINT_COUNT, value))
                    log(f"参数 {param} 所有关节更新为: {value}")
                elif 0 <= joint_idx < JOINT_COUNT:
                    current[joint_idx] = value
                    setattr(CTRL_CFG, param, current)
                    log(f"参数 {param} 关节{joint_idx + 1}更新为: {value}")
                else:
                    log(f"无效关节索引: {joint_idx}")
            else:
                setattr(CTRL_CFG, param, value)
                log(f"参数 {param} 更新为: {value}")

    def preset_pose(self, pose: str):
        """预设姿态（简化字典）"""
        poses = {
            'zero': [0, 0, 0, 0, 0],
            'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10],
            'test': [10, 20, 15, 5, 8]
        }

        if pose in poses:
            self.move_to(poses[pose])
        else:
            log(f"未知姿态: {pose} (支持: {list(poses.keys())})")

    def _print_status(self):
        """打印状态（优化格式）"""
        now = time.time()
        if now - self.last_print < 1.0:
            return

        # 计算FPS
        fps = self.fps_count / (now - self.last_print)

        # 获取状态
        qpos, qvel = self.get_joint_states()
        err = rad2deg(self.err)
        max_err = rad2deg(self.max_err)

        # 状态标记
        status = []
        if PAUSED:
            status.append("暂停")
        if EMERGENCY_STOP:
            status.append("紧急停止")
        status_str = " | ".join(status) if status else "运行中"

        # 打印
        log("=" * 80)
        log(f"状态 | {status_str} | 步数: {self.step} | FPS: {fps:.1f}")
        log(f"负载 | 设定: {self.load_set:.1f}kg | 实际: {self.load_actual:.1f}kg")
        log(f"角度: {np.round(rad2deg(qpos), 1)} ° | 速度: {np.round(rad2deg(qvel), 2)} °/s")
        log(f"误差: {np.round(np.abs(err), 3)} ° (最大: {np.round(max_err, 3)} °)")
        log(f"刚度: {np.round(self.stiffness, 0)} | 阻尼: {np.round(self.damping, 1)}")
        log("=" * 80)

        # 重置计数器
        self.last_print = now
        self.fps_count = 0

    def _start_interactive(self):
        """交互线程（简化命令解析）"""

        def interactive_task():
            help_text = """
命令列表：
  help          - 查看帮助
  pause/resume  - 暂停/恢复轨迹
  stop          - 紧急停止
  pose [名称]   - 预设姿态（zero/up/grasp/test）
  joint [索引] [角度] - 控制单个关节
  load [kg]     - 设置负载（0-2kg）
  param [名称] [值] [关节] - 调整参数
  save [名称]   - 保存轨迹
  load_traj [名称] - 加载轨迹
            """
            log(help_text)

            while RUNNING and not EMERGENCY_STOP:
                try:
                    cmd = input("> ").strip().lower()
                    if not cmd:
                        continue

                    parts = cmd.split()
                    if parts[0] == 'help':
                        log(help_text)
                    elif parts[0] == 'pause':
                        self.pause()
                    elif parts[0] == 'resume':
                        self.resume()
                    elif parts[0] == 'stop':
                        self.emergency_stop()
                    elif parts[0] == 'pose' and len(parts) == 2:
                        self.preset_pose(parts[1])
                    elif parts[0] == 'joint' and len(parts) == 3:
                        self.control_joint(int(parts[1]) - 1, float(parts[2]))
                    elif parts[0] == 'load' and len(parts) == 2:
                        self.set_load(float(parts[1]))
                    elif parts[0] == 'param' and len(parts) >= 3:
                        idx = int(parts[3]) - 1 if len(parts) == 4 else None
                        self.adjust_param(parts[1], float(parts[2]), idx)
                    elif parts[0] == 'save' and len(parts) == 2:
                        self.move_to(rad2deg(self.target), save_traj=True, traj_name=parts[1])
                    elif parts[0] == 'load_traj' and len(parts) == 2:
                        self.load_trajectory(parts[1])
                    else:
                        log("未知命令，输入help查看帮助")
                except:
                    continue

        # 启动交互线程
        thread = threading.Thread(target=interactive_task, daemon=True)
        thread.start()

    def run(self):
        """主运行循环（优化逻辑）"""
        global RUNNING

        # 初始化Viewer
        try:
            if self.model is None or self.data is None:
                raise RuntimeError("模型未初始化")

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            log("Viewer启动成功")
        except Exception as e:
            log(f"Viewer启动失败: {e}")
            RUNNING = False
            return

        # 启动交互线程
        self._start_interactive()

        # 主循环
        log("控制器启动 (按Ctrl+C退出)")
        while RUNNING and self.viewer.is_running():
            try:
                self.step += 1
                self.fps_count += 1

                # 控制计算
                self.control_step()

                # 仿真步进
                mujoco.mj_step(self.model, self.data)

                # 可视化同步
                self.viewer.sync()

                # 状态打印
                self._print_status()

                # 睡眠控制
                time.sleep(TIME_CFG.sleep_dt)

            except Exception as e:
                log(f"运行错误: {e}")
                continue

        # 资源清理
        self._cleanup()
        max_err = rad2deg(np.max(self.max_err))
        log(f"控制器停止 | 总步数: {self.step} | 最大误差: {np.round(max_err, 3)}°")

    def _start_interactive_thread(self):
        """新增：启动交互线程，支持运行时命令输入"""

        def interactive_task():
            log_info("\n======= 交互命令 =======")
            log_info("help - 查看帮助")
            log_info("pause - 暂停轨迹")
            log_info("resume - 恢复轨迹")
            log_info("stop - 紧急停止")
            log_info("load [kg] - 设置负载（如load 1.0）")
            log_info("joint [idx] [deg] - 控制单个关节（如joint 1 30）")
            log_info("pose [name] - 预设姿态（zero/up/grasp/test）")
            log_info("param [name] [value] [idx] - 调整参数（如param kp_base 150）")
            log_info("save [name] - 保存当前轨迹")
            log_info("load_traj [name] - 加载轨迹")
            log_info("=======================\n")

            while RUNNING and not EMERGENCY_STOP:
                try:
                    cmd = input("> ").strip().lower()
                    if not cmd:
                        continue

                    parts = cmd.split()
                    if parts[0] == 'help':
                        self._print_help()
                    elif parts[0] == 'pause':
                        self.pause_trajectory()
                    elif parts[0] == 'resume':
                        self.resume_trajectory()
                    elif parts[0] == 'stop':
                        self.emergency_stop()
                    elif parts[0] == 'load' and len(parts) == 2:
                        self.set_end_load(float(parts[1]))
                    elif parts[0] == 'joint' and len(parts) == 3:
                        self.control_single_joint(int(parts[1]) - 1, float(parts[2]))
                    elif parts[0] == 'pose' and len(parts) == 2:
                        self.preset_pose(parts[1])
                    elif parts[0] == 'param' and len(parts) >= 3:
                        idx = int(parts[3]) - 1 if len(parts) == 4 else None
                        self.adjust_control_param(parts[1], float(parts[2]), idx)
                    elif parts[0] == 'save' and len(parts) == 2:
                        self.move_to(rad2deg(self.target_rad), save_traj=True, traj_name=parts[1])
                    elif parts[0] == 'load_traj' and len(parts) == 2:
                        self.load_trajectory(parts[1])
                    else:
                        log_info("未知命令，输入help查看帮助")
                except KeyboardInterrupt:
                    continue
                except Exception as e:
                    log_info(f"命令执行错误: {e}")

        # 启动交互线程（守护线程）
        interactive_thread = threading.Thread(target=interactive_task, daemon=True)
        interactive_thread.start()

    def _print_help(self):
        """打印帮助信息"""
        help_text = """
    def _cleanup(self):
        """资源清理（自动化）"""
        if self.viewer:
            self.viewer.close()

        # 释放内存
        self.traj_pos = np.array([])
        self.traj_vel = np.array([])
        TrajectoryPlanner.clear_cache()

        # 重置对象
        self.model = None
        self.data = None


# ====================== 信号处理与演示 ======================
def signal_handler(sig, frame):
    """信号处理（简化）"""
    global RUNNING, EMERGENCY_STOP
    if RUNNING:
        log("收到退出信号，正在停止...")
        RUNNING = False
        EMERGENCY_STOP = True


def demo(controller: ArmController):
    """演示程序（简化流程）"""

    def demo_task():
        steps = [
            (2, 'pose', 'zero'),
            (3, 'pose', 'test'),
            (2, 'pause', None),
            (2, 'resume', None),
            (4, 'load', 1.5),
            (4, 'pose', 'grasp'),
            (1, 'joint', (0, 10)),
            (3, 'load', 0.2),
            (3, 'pose', 'zero'),
            (2, 'stop', None)
        ]

        for delay, action, param in steps:
            time.sleep(delay)
            if not RUNNING or EMERGENCY_STOP:
                break

            if action == 'pose':
                controller.preset_pose(param)
            elif action == 'load':
                controller.set_load(param)
            elif action == 'pause':
                controller.pause()
            elif action == 'resume':
                controller.resume()
            elif action == 'joint':
                controller.control_joint(*param)
            elif action == 'stop':
                controller.emergency_stop()

    thread = threading.Thread(target=demo_task, daemon=True)
    thread.start()


# ====================== 主函数（简化入口） ======================
def main():
    """主函数（标准化）"""
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        controller = ArmController()
        demo(controller)
        controller.run()
    except Exception as e:
        log(f"程序异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()