#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节运动性能优化控制器（增强版）
核心特性：
- 基础功能：梯形轨迹/自适应刚度/误差补偿/PD+前馈控制
- 新增功能：轨迹暂停恢复/单独关节控制/负载监测/轨迹保存加载/紧急停止/动态调参
- 工程特性：完整异常处理/资源管理/线程安全/性能优化
"""

import sys
import time
import signal
import threading
import csv
import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Final, Callable
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

# ====================== 常量定义（不可变） ======================
JOINT_COUNT: Final[int] = 5
LOG_FILE: Final[str] = "arm_controller.log"
TRAJECTORY_DIR: Final[str] = "trajectories"
MIN_LOAD: Final[float] = 0.0
MAX_LOAD: Final[float] = 2.0
DEG_TO_RAD_SCALE: Final[float] = np.pi / 180.0
RAD_TO_DEG_SCALE: Final[float] = 180.0 / np.pi

# 创建轨迹目录
Path(TRAJECTORY_DIR).mkdir(exist_ok=True)


# ====================== 配置类（纯数据，无逻辑） ======================
@dataclass(frozen=True)  # 不可变配置，避免运行时意外修改
class JointLimits:
    """关节物理极限配置"""
    limits_rad: np.ndarray = field(default_factory=lambda: np.array([
        [-np.pi, np.pi],  # 基座
        [-np.pi / 2, np.pi / 2],  # 大臂
        [-np.pi / 2, np.pi / 2],  # 中臂
        [-np.pi / 2, np.pi / 2],  # 小臂
        [-np.pi / 2, np.pi / 2],  # 末端
    ], dtype=np.float64))
    max_vel_rad: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, 0.8, 0.6, 0.6]))
    max_acc_rad: np.ndarray = field(default_factory=lambda: np.array([2.0, 1.6, 1.6, 1.2, 1.2]))
    max_torque: np.ndarray = field(default_factory=lambda: np.array([15.0, 12.0, 10.0, 8.0, 5.0]))


@dataclass
class ControlTuning:
    """控制算法参数调优（支持动态修改）"""
    # PD控制
    kp_base: float = 120.0
    kd_base: float = 8.0
    kp_load_gain: float = 1.8
    kd_load_gain: float = 1.5

    # 前馈控制
    ff_vel_gain: float = 0.7
    ff_acc_gain: float = 0.5

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
    """时间相关配置"""
    sim_dt: float = 0.0005
    ctrl_freq: int = 2000
    fps: int = 60

    # 派生参数（自动计算）
    ctrl_dt: float = field(init=False)
    sleep_dt: float = field(init=False)

    def __post_init__(self):
        self.ctrl_dt = 1.0 / self.ctrl_freq
        self.sleep_dt = 1.0 / self.fps


# ====================== 全局状态与工具 ======================
# 配置实例化
JOINT_LIMITS = JointLimits()
CONTROL_TUNING = ControlTuning()
TIME_CFG = TimeConfig()

# 全局状态
RUNNING: bool = True
PAUSED: bool = False  # 新增：轨迹暂停状态
EMERGENCY_STOP: bool = False  # 新增：紧急停止
GLOBAL_LOCK: threading.Lock = threading.Lock()


# ====================== 工具函数（纯函数，无副作用） ======================
def deg2rad(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """角度转弧度（带安全检查）"""
    try:
        deg_arr = np.asarray(deg, dtype=np.float64)
        return deg_arr * DEG_TO_RAD_SCALE
    except (ValueError, TypeError):
        return 0.0 if np.isscalar(deg) else np.zeros(JOINT_COUNT)


def rad2deg(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """弧度转角度（带安全检查）"""
    try:
        rad_arr = np.asarray(rad, dtype=np.float64)
        return rad_arr * RAD_TO_DEG_SCALE
    except (ValueError, TypeError):
        return 0.0 if np.isscalar(rad) else np.zeros(JOINT_COUNT)


def get_mujoco_id_safe(model: mujoco.MjModel, obj_type: str, name: str) -> int:
    """安全获取MuJoCo对象ID，避免崩溃"""
    type_map = {
        'joint': mujoco.mjtObj.mjOBJ_JOINT,
        'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
        'geom': mujoco.mjtObj.mjOBJ_GEOM
    }
    obj_type_enum = type_map.get(obj_type, mujoco.mjtObj.mjOBJ_JOINT)
    try:
        return mujoco.mj_name2id(model, obj_type_enum, name)
    except Exception:
        return -1


@contextmanager
def thread_safe_operation():
    """线程安全的操作上下文管理器"""
    with GLOBAL_LOCK:
        yield


def log_info(content: str):
    """标准化日志记录"""
    try:
        with thread_safe_operation():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_msg = f"[{timestamp}] {content}"
            # 写入日志文件
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{log_msg}\n")
            # 打印到控制台
            print(log_msg)
    except Exception as e:
        print(f"日志记录失败: {e} | 内容: {content}")


def calculate_actual_load(controller) -> float:
    """新增：计算末端实际负载（基于关节力矩反推）"""
    if controller.data is None or controller.model is None:
        return 0.0

    # 基于关节力矩和当前角度反推实际负载
    joint_forces = np.abs([controller.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in controller.joint_ids])
    qpos, _ = controller.get_joint_states()

    # 简化的负载计算模型（可根据实际机械臂参数调整）
    load_estimation = np.sum(joint_forces * np.sin(qpos)) / 9.81
    return np.clip(load_estimation, MIN_LOAD, MAX_LOAD)


# ====================== 轨迹规划器（增强版） ======================
class TrajectoryPlanner:
    """梯形速度轨迹规划器（增强版：支持单独关节、轨迹保存加载）"""
    _trajectory_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # 轨迹缓存

    @staticmethod
    def _plan_single(
            start: float,
            target: float,
            max_vel: float,
            max_acc: float,
            dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单关节轨迹规划（内部方法，性能优化）"""
        delta = target - start
        if abs(delta) < 1e-5:
            return np.array([target]), np.array([0.0])

        direction = np.sign(delta)
        dist = abs(delta)

        # 计算轨迹关键参数
        accel_dist = (max_vel ** 2) / (2 * max_acc)
        if dist <= 2 * accel_dist:
            # 无匀速段
            peak_vel = np.sqrt(dist * max_acc)
            accel_time = peak_vel / max_acc
            total_time = 2 * accel_time
        else:
            # 有匀速段
            accel_time = max_vel / max_acc
            uniform_time = (dist - 2 * accel_dist) / max_vel
            total_time = 2 * accel_time + uniform_time

        # 生成时间序列（预分配内存）
        t = np.arange(0, total_time + dt, dt, dtype=np.float64)
        pos = np.empty_like(t)
        vel = np.empty_like(t)

        # 向量化分段计算（避免循环）
        mask_acc = t <= accel_time
        mask_uni = (t > accel_time) & (t <= accel_time + uniform_time) if dist > 2 * accel_dist else np.zeros_like(t,
                                                                                                                   dtype=bool)
        mask_dec = ~(mask_acc | mask_uni)

        # 加速段
        vel[mask_acc] = max_acc * t[mask_acc] * direction
        pos[mask_acc] = start + 0.5 * max_acc * np.square(t[mask_acc]) * direction

        if dist > 2 * accel_dist:
            # 匀速段
            t_uni = t[mask_uni] - accel_time
            vel[mask_uni] = max_vel * direction
            pos[mask_uni] = start + (accel_dist + max_vel * t_uni) * direction

            # 减速段
            t_dec = t[mask_dec] - (accel_time + uniform_time)
        else:
            # 减速段（无匀速）
            t_dec = t[mask_dec] - accel_time

        vel[mask_dec] = (max_vel if dist > 2 * accel_dist else peak_vel) - max_acc * t_dec
        vel[mask_dec] *= direction

        pos_dec_base = start + dist * direction if dist > 2 * accel_dist else start + peak_vel * accel_time * direction
        pos[mask_dec] = pos_dec_base - 0.5 * max_acc * np.square(t_dec) * direction

        # 强制终点（避免浮点误差）
        pos[-1] = target
        vel[-1] = 0.0

        return pos, vel

    @classmethod
    def plan_joints(
            cls,
            start_rad: np.ndarray,
            target_rad: np.ndarray,
            dt: float = TIME_CFG.ctrl_dt,
            cache_key: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """多关节轨迹规划（新增缓存机制）"""
        # 检查缓存
        if cache_key and cache_key in cls._trajectory_cache:
            return cls._trajectory_cache[cache_key]

        # 安全检查与边界裁剪
        start_rad = np.clip(
            start_rad,
            JOINT_LIMITS.limits_rad[:, 0] + 0.01,
            JOINT_LIMITS.limits_rad[:, 1] - 0.01
        )
        target_rad = np.clip(
            target_rad,
            JOINT_LIMITS.limits_rad[:, 0] + 0.01,
            JOINT_LIMITS.limits_rad[:, 1] - 0.01
        )

        # 批量规划（预分配内存提升性能）
        traj_pos = []
        traj_vel = []
        max_len = 1

        for i in range(JOINT_COUNT):
            pos, vel = cls._plan_single(
                start_rad[i], target_rad[i],
                JOINT_LIMITS.max_vel_rad[i],
                JOINT_LIMITS.max_acc_rad[i],
                dt
            )
            traj_pos.append(pos)
            traj_vel.append(vel)
            max_len = max(max_len, len(pos))

        # 统一轨迹长度（向量化填充）
        for i in range(JOINT_COUNT):
            current_len = len(traj_pos[i])
            if current_len < max_len:
                pad_len = max_len - current_len
                traj_pos[i] = np.pad(traj_pos[i], (0, pad_len), 'constant', constant_values=target_rad[i])
                traj_vel[i] = np.pad(traj_vel[i], (0, pad_len), 'constant', constant_values=0.0)

        result_pos = np.array(traj_pos).T
        result_vel = np.array(traj_vel).T

        # 存入缓存
        if cache_key:
            cls._trajectory_cache[cache_key] = (result_pos, result_vel)

        return result_pos, result_vel

    @classmethod
    def plan_single_joint(
            cls,
            joint_idx: int,
            start_deg: float,
            target_deg: float,
            dt: float = TIME_CFG.ctrl_dt
    ) -> Tuple[np.ndarray, np.ndarray]:
        """新增：单独规划单个关节的轨迹"""
        if joint_idx < 0 or joint_idx >= JOINT_COUNT:
            log_info(f"无效关节索引: {joint_idx} (范围: 0-{JOINT_COUNT - 1})")
            return np.array([]), np.array([])

        # 获取当前所有关节状态
        start_rad_all = np.zeros(JOINT_COUNT)
        target_rad_all = np.zeros(JOINT_COUNT)

        # 只修改目标关节
        start_rad_all[joint_idx] = deg2rad(start_deg)
        target_rad_all[joint_idx] = deg2rad(target_deg)

        # 规划轨迹
        return cls.plan_joints(start_rad_all, target_rad_all, dt)

    @classmethod
    def save_trajectory(cls, traj_pos: np.ndarray, traj_vel: np.ndarray, filename: str):
        """新增：保存轨迹到CSV文件"""
        filepath = Path(TRAJECTORY_DIR) / f"{filename}.csv"
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                header = ['time_step'] + [f'joint{i + 1}_pos' for i in range(JOINT_COUNT)] + \
                         [f'joint{i + 1}_vel' for i in range(JOINT_COUNT)]
                writer.writerow(header)

                # 写入数据
                for i in range(len(traj_pos)):
                    row = [i] + traj_pos[i].tolist() + traj_vel[i].tolist()
                    writer.writerow(row)

            log_info(f"轨迹已保存到: {filepath}")
        except Exception as e:
            log_info(f"保存轨迹失败: {e}")

    @classmethod
    def load_trajectory(cls, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """新增：从CSV文件加载轨迹"""
        filepath = Path(TRAJECTORY_DIR) / f"{filename}.csv"
        if not filepath.exists():
            log_info(f"轨迹文件不存在: {filepath}")
            return np.array([]), np.array([])

        try:
            data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
            if len(data) == 0:
                return np.array([]), np.array([])

            # 提取位置和速度数据
            traj_pos = data[:, 1:JOINT_COUNT + 1]
            traj_vel = data[:, JOINT_COUNT + 1:]

            log_info(f"轨迹已从: {filepath} 加载 (共{len(traj_pos)}步)")
            return traj_pos, traj_vel
        except Exception as e:
            log_info(f"加载轨迹失败: {e}")
            return np.array([]), np.array([])


# ====================== 机械臂控制器（增强版核心类） ======================
class ArmController:
    """机械臂控制器（增强版：新增暂停/急停/单独控制/动态调参）"""

    def __init__(self):
        # MuJoCo核心对象
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer: Optional[mujoco.viewer.Viewer] = None

        # ID缓存（避免重复查询）
        self.joint_ids: List[int] = []
        self.motor_ids: List[int] = []
        self.ee_geom_id: int = -1

        # 控制状态
        self.traj_pos: np.ndarray = np.zeros((1, JOINT_COUNT))
        self.traj_vel: np.ndarray = np.zeros((1, JOINT_COUNT))
        self.traj_idx: int = 0
        self.target_rad: np.ndarray = np.zeros(JOINT_COUNT)
        self.saved_traj_idx: int = 0  # 新增：暂停时保存轨迹索引

        # 物理参数状态
        self.stiffness: np.ndarray = CONTROL_TUNING.stiffness_base.copy()
        self.damping: np.ndarray = CONTROL_TUNING.stiffness_base * CONTROL_TUNING.damping_ratio
        self.end_load_set: float = 0.5  # 设定负载
        self.end_load_actual: float = 0.5  # 新增：实际负载

        # 误差状态
        self.joint_err: np.ndarray = np.zeros(JOINT_COUNT)
        self.max_joint_err: np.ndarray = np.zeros(JOINT_COUNT)

        # 性能统计
        self.step_count: int = 0
        self.last_ctrl_time: float = time.time()
        self.last_print_time: float = time.time()
        self.fps_counter: int = 0

        # 初始化流程
        self._init_mujoco()
        self._init_ids()
        self._reset_state()
        log_info("增强版控制器初始化完成")

    def _init_mujoco(self):
        """初始化MuJoCo模型（异常安全）"""
        try:
            xml_str = self._generate_model_xml()
            self.model = mujoco.MjModel.from_xml_string(xml_str)
            self.data = mujoco.MjData(self.model)
            log_info("MuJoCo模型加载成功")
        except Exception as e:
            log_info(f"模型初始化失败: {e}")
            global RUNNING, EMERGENCY_STOP
            RUNNING = False
            EMERGENCY_STOP = True

    def _generate_model_xml(self) -> str:
        """生成优化的MuJoCo XML模型"""
        link_masses = [0.8, 0.6, 0.6, 0.4, 0.2]
        friction = CONTROL_TUNING.friction

        return f"""
<mujoco model="optimized_arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{TIME_CFG.sim_dt}" gravity="0 0 -9.81" iterations="100" tolerance="1e-9"/>

    <default>
        <joint type="hinge" limited="true" margin="0.001"/>
        <motor ctrllimited="true" ctrlrange="-1 1" gear="100"/>
        <geom contype="1" conaffinity="1" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>

    <asset>
        <material name="arm_mat" rgba="0.0 0.8 0.0 0.8"/>
        <material name="ee_mat" rgba="0.8 0.2 0.2 1"/>
    </asset>

    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>

        <!-- 基座 -->
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" axis="0 0 1" pos="0 0 0.1"
                   range="{JOINT_LIMITS.limits_rad[0, 0]} {JOINT_LIMITS.limits_rad[0, 1]}"/>

            <!-- 大臂 -->
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[0]}"
                      material="arm_mat" friction="{friction[0]} {friction[0]} {friction[0]}"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18"
                       range="{JOINT_LIMITS.limits_rad[1, 0]} {JOINT_LIMITS.limits_rad[1, 1]}"/>

                <!-- 中臂 -->
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[1]}"
                          material="arm_mat" friction="{friction[1]} {friction[1]} {friction[1]}"/>
                    <joint name="joint3" axis="0 1 0" pos="0 0 0.18"
                           range="{JOINT_LIMITS.limits_rad[2, 0]} {JOINT_LIMITS.limits_rad[2, 1]}"/>

                    <!-- 小臂 -->
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[2]}"
                              material="arm_mat" friction="{friction[2]} {friction[2]} {friction[2]}"/>
                        <joint name="joint4" axis="0 1 0" pos="0 0 0.18"
                               range="{JOINT_LIMITS.limits_rad[3, 0]} {JOINT_LIMITS.limits_rad[3, 1]}"/>

                        <!-- 末端 -->
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[3]}"
                                  material="arm_mat" friction="{friction[3]} {friction[3]} {friction[3]}"/>
                            <joint name="joint5" axis="0 1 0" pos="0 0 0.18"
                                   range="{JOINT_LIMITS.limits_rad[4, 0]} {JOINT_LIMITS.limits_rad[4, 1]}"/>

                            <!-- 末端执行器 -->
                            <body name="end_effector" pos="0 0 0.18">
                                <geom name="ee_geom" type="sphere" size="0.04" mass="{self.end_load_set}" 
                                      material="ee_mat" rgba="1.0 0.0 0.0 0.8"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- 执行器 -->
    <actuator>
        <motor name="motor1" joint="joint1"/>
        <motor name="motor2" joint="joint2"/>
        <motor name="motor3" joint="joint3"/>
        <motor name="motor4" joint="joint4"/>
        <motor name="motor5" joint="joint5"/>
    </actuator>
</mujoco>
        """

    def _init_ids(self):
        """初始化MuJoCo对象ID（带空值检查）"""
        if self.model is None:
            return

        # 缓存ID（避免重复查询）
        self.joint_ids = [get_mujoco_id_safe(self.model, 'joint', f"joint{i + 1}") for i in range(JOINT_COUNT)]
        self.motor_ids = [get_mujoco_id_safe(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
        self.ee_geom_id = get_mujoco_id_safe(self.model, 'geom', "ee_geom")

        # 初始化阻尼
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                self.model.jnt_damping[jid] = self.damping[i]

    def _reset_state(self):
        """重置控制器状态（无副作用）"""
        global PAUSED, EMERGENCY_STOP
        self.target_rad = np.zeros(JOINT_COUNT)
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.saved_traj_idx = 0
        self.joint_err = np.zeros(JOINT_COUNT)
        self.max_joint_err = np.zeros(JOINT_COUNT)
        PAUSED = False
        EMERGENCY_STOP = False

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取关节状态（位置/速度），带空值保护"""
        if self.data is None or self.model is None:
            return np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT)

        # 向量化获取状态（避免循环）
        qpos = np.array([self.data.qpos[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        return qpos, qvel

    def _calculate_compensation(self, qpos: np.ndarray, qvel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算误差补偿（纯计算，无状态修改）"""
        # 1. 间隙补偿（向量化）
        vel_sign = np.sign(qvel)
        vel_zero_mask = np.abs(qvel) < 1e-4
        vel_sign[vel_zero_mask] = np.sign(self.joint_err)[vel_zero_mask]
        backlash_comp = CONTROL_TUNING.backlash * vel_sign

        # 2. 摩擦补偿（向量化）
        friction_comp = np.where(
            vel_zero_mask,
            CONTROL_TUNING.friction * np.sign(self.joint_err),
            0.0
        )

        # 3. 重力补偿（使用实际负载）
        gravity_comp = np.zeros(JOINT_COUNT)
        if CONTROL_TUNING.gravity_comp:
            gravity_comp = 0.5 * np.sin(qpos) * self.end_load_actual

        total_comp = backlash_comp + friction_comp + gravity_comp
        return total_comp, gravity_comp

    def _update_stiffness_damping(self, qpos: np.ndarray, qvel: np.ndarray):
        """自适应更新刚度阻尼（性能优化）"""
        if self.data is None or self.model is None or EMERGENCY_STOP:
            return

        # 计算负载比例（使用实际负载）
        load_ratio = np.clip(self.end_load_actual / MAX_LOAD, 0.0, 1.0)

        # 计算误差比例
        err_norm = np.clip(np.abs(self.joint_err) / deg2rad(1.0), 0.0, 1.0)

        # 更新刚度（向量化计算）
        target_stiffness = CONTROL_TUNING.stiffness_base * \
                           (1 + load_ratio * (CONTROL_TUNING.stiffness_load_gain - 1)) * \
                           (1 + err_norm * (CONTROL_TUNING.stiffness_error_gain - 1))
        target_stiffness = np.clip(target_stiffness, CONTROL_TUNING.stiffness_min, CONTROL_TUNING.stiffness_max)
        self.stiffness = 0.95 * self.stiffness + 0.05 * target_stiffness

        # 更新阻尼
        self.damping = self.stiffness * CONTROL_TUNING.damping_ratio
        self.damping = np.clip(
            self.damping,
            CONTROL_TUNING.stiffness_min * 0.02,
            CONTROL_TUNING.stiffness_max * 0.08
        )

        # 应用到模型（批量更新）
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                self.model.jnt_damping[jid] = self.damping[i]

    def control_step(self):
        """单步控制计算（增强版：支持暂停/急停）"""
        global PAUSED, EMERGENCY_STOP

        # 急停处理
        if EMERGENCY_STOP:
            # 切断所有控制输出
            if self.data is not None:
                self.data.ctrl[:] = 0.0
            return

        # 暂停处理
        if PAUSED:
            self.saved_traj_idx = self.traj_idx  # 保存当前轨迹位置
            return

        current_time = time.time()
        # 控制频率限流（避免高频计算）
        if current_time - self.last_ctrl_time < TIME_CFG.ctrl_dt:
            return

        # 获取当前状态
        qpos, qvel = self.get_joint_states()

        # 更新实际负载
        self.end_load_actual = calculate_actual_load(self)

        # 获取目标轨迹点
        if self.traj_idx < len(self.traj_pos):
            target_pos = self.traj_pos[self.traj_idx]
            target_vel = self.traj_vel[self.traj_idx]
            self.traj_idx += 1
        else:
            target_pos = self.target_rad
            target_vel = np.zeros(JOINT_COUNT)

        # 计算误差（向量化）
        self.joint_err = target_pos - qpos
        self.max_joint_err = np.maximum(self.max_joint_err, np.abs(self.joint_err))

        # 自适应PD参数（使用实际负载）
        load_factor = np.clip(self.end_load_actual / MAX_LOAD, 0.0, 1.0)
        kp = CONTROL_TUNING.kp_base * (1 + load_factor * (CONTROL_TUNING.kp_load_gain - 1))
        kd = CONTROL_TUNING.kd_base * (1 + load_factor * (CONTROL_TUNING.kd_load_gain - 1))

        # PD控制 + 前馈控制（完全向量化）
        pd_ctrl = kp * self.joint_err + kd * (target_vel - qvel)
        ff_ctrl = CONTROL_TUNING.ff_vel_gain * target_vel + \
                  CONTROL_TUNING.ff_acc_gain * (target_vel - qvel) / TIME_CFG.ctrl_dt

        # 误差补偿
        comp_ctrl, _ = self._calculate_compensation(qpos, qvel)

        # 总控制输出（带扭矩限制）
        total_ctrl = pd_ctrl + ff_ctrl + comp_ctrl
        total_ctrl = np.clip(total_ctrl, -JOINT_LIMITS.max_torque, JOINT_LIMITS.max_torque)

        # 应用控制信号（批量更新）
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = total_ctrl[i]

        # 更新刚度阻尼
        self._update_stiffness_damping(qpos, qvel)

        # 更新时间戳
        self.last_ctrl_time = current_time

    # ====================== 新增功能接口 ======================
    def set_end_load(self, mass: float):
        """设置末端负载（带范围检查和线程安全）"""
        with thread_safe_operation():
            if not (MIN_LOAD <= mass <= MAX_LOAD):
                log_info(f"负载超出范围: {mass}kg (限制: {MIN_LOAD}-{MAX_LOAD}kg)")
                return

            self.end_load_set = mass
            if self.ee_geom_id >= 0 and self.model is not None:
                self.model.geom_mass[self.ee_geom_id] = mass
            log_info(f"末端负载设定为: {mass}kg")

    def move_to(self, target_deg: Union[List[float], np.ndarray], save_traj: bool = False, traj_name: str = "default"):
        """移动到目标角度（新增轨迹保存选项）"""
        with thread_safe_operation():
            # 参数检查
            target_deg_arr = np.asarray(target_deg, dtype=np.float64)
            if target_deg_arr.shape != (JOINT_COUNT,):
                log_info(f"目标角度维度错误: 期望{JOINT_COUNT}维，实际{target_deg_arr.shape}")
                return

            # 获取当前状态
            start_rad, _ = self.get_joint_states()
            target_rad = deg2rad(target_deg_arr)

            # 规划轨迹（使用缓存）
            cache_key = f"{traj_name}_{hash(tuple(start_rad))}_{hash(tuple(target_rad))}"
            self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_joints(
                start_rad, target_rad, cache_key=cache_key
            )

            # 保存轨迹
            if save_traj:
                TrajectoryPlanner.save_trajectory(self.traj_pos, self.traj_vel, traj_name)

            self.target_rad = target_rad
            self.traj_idx = 0
            self.saved_traj_idx = 0

            # 日志记录
            log_info(f"规划轨迹: {np.round(rad2deg(start_rad), 1)}° → {np.round(rad2deg(target_rad), 1)}°")

    def control_single_joint(self, joint_idx: int, target_deg: float):
        """新增：单独控制单个关节"""
        if joint_idx < 0 or joint_idx >= JOINT_COUNT:
            log_info(f"无效关节索引: {joint_idx} (范围: 0-{JOINT_COUNT - 1})")
            return

        # 获取当前关节状态
        current_rad, _ = self.get_joint_states()
        current_deg = rad2deg(current_rad[joint_idx])

        # 规划单个关节轨迹
        self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_single_joint(
            joint_idx, current_deg, target_deg
        )

        # 更新目标状态
        self.target_rad = current_rad.copy()
        self.target_rad[joint_idx] = deg2rad(target_deg)
        self.traj_idx = 0
        self.saved_traj_idx = 0

        log_info(f"单独控制关节{joint_idx + 1}: {current_deg:.1f}° → {target_deg:.1f}°")

    def load_trajectory(self, traj_name: str):
        """新增：加载预存轨迹"""
        with thread_safe_operation():
            traj_pos, traj_vel = TrajectoryPlanner.load_trajectory(traj_name)
            if len(traj_pos) == 0:
                return

            self.traj_pos = traj_pos
            self.traj_vel = traj_vel
            self.traj_idx = 0
            self.saved_traj_idx = 0

            # 设置最后一个点为目标位置
            self.target_rad = traj_pos[-1] if len(traj_pos) > 0 else np.zeros(JOINT_COUNT)
            log_info(f"加载轨迹: {traj_name} (共{len(traj_pos)}步)")

    def pause_trajectory(self):
        """新增：暂停轨迹执行"""
        global PAUSED
        with thread_safe_operation():
            PAUSED = True
            log_info("轨迹执行已暂停")

    def resume_trajectory(self):
        """新增：恢复轨迹执行"""
        global PAUSED
        with thread_safe_operation():
            PAUSED = False
            self.traj_idx = self.saved_traj_idx  # 恢复到暂停时的位置
            log_info(f"轨迹执行已恢复（从第{self.saved_traj_idx}步开始）")

    def emergency_stop(self):
        """新增：紧急停止"""
        global RUNNING, PAUSED, EMERGENCY_STOP
        with thread_safe_operation():
            EMERGENCY_STOP = True
            PAUSED = True
            RUNNING = False
            log_info("⚠️ 紧急停止已触发！所有控制输出已切断")

    def adjust_control_param(self, param_name: str, value: float, joint_idx: Optional[int] = None):
        """新增：动态调整控制参数"""
        with thread_safe_operation():
            # 检查参数是否存在
            if not hasattr(CONTROL_TUNING, param_name):
                log_info(f"无效参数名: {param_name}")
                return

            # 获取当前参数值
            current_value = getattr(CONTROL_TUNING, param_name)

            # 处理数组参数（如stiffness_base）
            if isinstance(current_value, np.ndarray):
                if joint_idx is None:
                    # 全部关节
                    setattr(CONTROL_TUNING, param_name, np.full(JOINT_COUNT, value))
                    log_info(f"参数 {param_name} 所有关节更新为: {value}")
                else:
                    # 单个关节
                    if 0 <= joint_idx < JOINT_COUNT:
                        current_value[joint_idx] = value
                        setattr(CONTROL_TUNING, param_name, current_value)
                        log_info(f"参数 {param_name} 关节{joint_idx + 1}更新为: {value}")
                    else:
                        log_info(f"无效关节索引: {joint_idx}")
            else:
                # 标量参数
                setattr(CONTROL_TUNING, param_name, value)
                log_info(f"参数 {param_name} 更新为: {value}")

    def preset_pose(self, pose_name: str):
        """预设姿态（扩展方便）"""
        poses: Dict[str, List[float]] = {
            'zero': [0, 0, 0, 0, 0],
            'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10],
            'test': [10, 20, 15, 5, 8]
        }

        if pose_name in poses:
            self.move_to(poses[pose_name])
        else:
            log_info(f"未知姿态: {pose_name} (支持: {list(poses.keys())})")

    def _print_status(self):
        """打印运行状态（增强版：显示实际负载、暂停/急停状态）"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        # 计算FPS
        fps = self.fps_counter / (current_time - self.last_print_time)

        # 获取状态
        qpos_deg, qvel_deg = rad2deg(self.get_joint_states())
        err_deg = rad2deg(self.joint_err)
        max_err_deg = rad2deg(self.max_joint_err)

        # 状态标记
        status_flags = []
        if PAUSED:
            status_flags.append("暂停")
        if EMERGENCY_STOP:
            status_flags.append("紧急停止")
        status_str = " | ".join(status_flags) if status_flags else "运行中"

        # 打印状态（结构化）
        log_info("=" * 80)
        log_info(f"运行状态 | {status_str} | 步数: {self.step_count} | FPS: {fps:.1f}")
        log_info(f"负载状态 | 设定: {self.end_load_set:.1f}kg | 实际: {self.end_load_actual:.1f}kg")
        log_info(f"关节角度: {np.round(qpos_deg, 1)} °")
        log_info(f"关节速度: {np.round(qvel_deg, 2)} °/s")
        log_info(f"定位误差: {np.round(np.abs(err_deg), 3)} ° (最大: {np.round(max_err_deg, 3)} °)")
        log_info(f"刚度: {np.round(self.stiffness, 0)} | 阻尼: {np.round(self.damping, 1)}")
        log_info("=" * 80)

        # 重置计数器
        self.last_print_time = current_time
        self.fps_counter = 0

    def run(self):
        """主运行循环（增强版：支持暂停/急停）"""
        global RUNNING

        # 初始化Viewer
        try:
            if self.model is None or self.data is None:
                raise RuntimeError("MuJoCo模型未初始化")

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            log_info("Viewer启动成功")
        except Exception as e:
            log_info(f"Viewer启动失败: {e}")
            RUNNING = False
            return

        # 启动交互线程
        self._start_interactive_thread()

        # 主循环（异常安全）
        log_info("增强版控制器开始运行 (按Ctrl+C退出，输入help查看交互命令)")
        while RUNNING and self.viewer.is_running():
            try:
                # 急停检查
                if EMERGENCY_STOP:
                    log_info("紧急停止状态，退出主循环")
                    break

                # 计数与计时
                self.step_count += 1
                self.fps_counter += 1

                # 控制计算
                self.control_step()

                # 仿真步进
                mujoco.mj_step(self.model, self.data)

                # 可视化同步
                self.viewer.sync()

                # 状态打印
                self._print_status()

                # 睡眠控制（精确延时）
                time.sleep(TIME_CFG.sleep_dt)

            except Exception as e:
                log_info(f"运行循环错误: {e}")
                continue

        # 资源清理（无论是否异常都执行）
        self._cleanup()
        max_err_deg = rad2deg(np.max(self.max_joint_err))
        log_info(f"控制器停止 | 总步数: {self.step_count} | 最大定位误差: {np.round(max_err_deg, 3)}°")

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
======= 增强版控制器交互命令 =======
基础控制：
  pause          - 暂停轨迹执行
  resume         - 恢复轨迹执行
  stop           - 紧急停止（切断所有控制）
  pose [name]    - 切换预设姿态
                   支持: zero(零位)、up(抬起)、grasp(抓取)、test(测试)

关节控制：
  joint [idx] [deg] - 单独控制单个关节
                   示例: joint 1 30 (控制关节1到30度)
  load [kg]         - 设置末端负载（0-2kg）
                   示例: load 1.5

参数调优：
  param [name] [value] [idx] - 动态调整控制参数
                   示例1: param kp_base 150 (调整基础KP值)
                   示例2: param stiffness_base 250 1 (调整关节1刚度)
                   支持参数: kp_base, kd_base, stiffness_base, damping_ratio等

轨迹管理：
  save [name]    - 保存当前规划的轨迹
                   示例: save grasp_pose
  load_traj [name] - 加载预存轨迹
                   示例: load_traj grasp_pose

其他：
  help           - 查看本帮助信息
===================================
        """
        log_info(help_text)

    def _cleanup(self):
        """资源清理（自动化，避免泄漏）"""
        if self.viewer:
            self.viewer.close()
        # 释放大数组内存
        self.traj_pos = np.array([])
        self.traj_vel = np.array([])
        # 清空缓存
        TrajectoryPlanner._trajectory_cache.clear()
        self.model = None
        self.data = None


# ====================== 信号处理与演示 ======================
def signal_handler(sig, frame):
    """优雅退出信号处理"""
    global RUNNING, EMERGENCY_STOP
    if RUNNING:
        log_info("收到退出信号，正在停止控制器...")
        RUNNING = False
        EMERGENCY_STOP = True


def demo_routine(controller: ArmController):
    """增强版演示程序（支持暂停/恢复）"""

    def demo_task():
        demo_steps = [
            (2, 'preset_pose', 'zero'),
            (3, 'preset_pose', 'test'),
            (2, 'pause', None),  # 暂停2秒
            (2, 'resume', None),  # 恢复
            (4, 'set_end_load', 1.5),
            (4, 'preset_pose', 'grasp'),
            (1, 'control_single_joint', (0, 10)),  # 单独控制关节1到10度
            (3, 'set_end_load', 0.2),
            (3, 'preset_pose', 'zero'),
            (2, 'stop', None)
        ]

        for delay, action, param in demo_steps:
            time.sleep(delay)
            if not RUNNING or EMERGENCY_STOP:
                break

            if action == 'preset_pose':
                controller.preset_pose(param)
            elif action == 'set_end_load':
                controller.set_end_load(param)
            elif action == 'pause':
                controller.pause_trajectory()
            elif action == 'resume':
                controller.resume_trajectory()
            elif action == 'control_single_joint':
                controller.control_single_joint(*param)
            elif action == 'stop':
                global RUNNING
                RUNNING = False

    # 启动演示线程（守护线程，主程序退出自动结束）
    demo_thread = threading.Thread(target=demo_task, daemon=True)
    demo_thread.start()


# ====================== 主函数（入口点） ======================
def main():
    """程序主入口（标准化）"""
    # 配置numpy打印格式
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # 注册信号处理（优雅退出）
    signal.signal(signal.SIGINT, signal_handler)

    # 初始化并运行控制器
    try:
        arm_ctrl = ArmController()
        demo_routine(arm_ctrl)
        arm_ctrl.run()
    except Exception as e:
        log_info(f"程序异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()