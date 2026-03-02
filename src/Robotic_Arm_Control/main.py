#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节运动性能优化控制器
核心优化：定位精度、运动平滑性、负载抗干扰、刚度阻尼自适应
兼容Mujoco仿真，修复geom/ joint属性违规问题
"""

import sys
import os
import time
import signal
import ctypes
import threading
import numpy as np
import mujoco
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union


# ====================== 配置类（结构化管理，提升可读性） ======================
@dataclass
class JointLimits:
    """关节物理限制配置"""
    count: int = 5
    names: List[str] = field(default_factory=lambda: ["joint1", "joint2", "joint3", "joint4", "joint5"])
    limits_rad: np.ndarray = field(default_factory=lambda: np.array([
        [-np.pi, np.pi],  # joint1（基座）
        [-np.pi / 2, np.pi / 2],  # joint2（大臂）
        [-np.pi / 2, np.pi / 2],  # joint3（中臂）
        [-np.pi / 2, np.pi / 2],  # joint4（小臂）
        [-np.pi / 2, np.pi / 2],  # joint5（末端）
    ], dtype=np.float64))
    max_velocity_rad: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64))
    max_accel_rad: np.ndarray = field(default_factory=lambda: np.array([2.0, 1.6, 1.6, 1.2, 1.2], dtype=np.float64))
    max_torque: np.ndarray = field(default_factory=lambda: np.array([15.0, 12.0, 10.0, 8.0, 5.0], dtype=np.float64))


@dataclass
class StiffnessDampingConfig:
    """刚度阻尼自适应配置"""
    base_stiffness: np.ndarray = field(default_factory=lambda: np.array([200.0, 180.0, 150.0, 120.0, 80.0]))
    load_stiffness_gain: float = 1.8
    error_stiffness_gain: float = 1.5
    min_stiffness: np.ndarray = field(default_factory=lambda: np.array([100.0, 90.0, 75.0, 60.0, 40.0]))
    max_stiffness: np.ndarray = field(default_factory=lambda: np.array([300.0, 270.0, 225.0, 180.0, 120.0]))
    stiffness_smoothing: float = 0.05

    base_damping: np.ndarray = field(default_factory=lambda: np.array([8.0, 7.0, 6.0, 5.0, 3.0]))
    viscous_damping_gain: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.1, 1.1, 1.0, 1.0]))
    damping_stiffness_ratio: float = 0.04
    min_damping: np.ndarray = field(default_factory=lambda: np.array([4.0, 3.5, 3.0, 2.5, 1.5]))
    max_damping: np.ndarray = field(default_factory=lambda: np.array([16.0, 14.0, 12.0, 10.0, 6.0]))


@dataclass
class ControlConfig:
    """控制参数配置"""
    simulation_timestep: float = 0.0005
    control_frequency: int = 2000
    control_timestep: float = field(init=False)
    fps: int = 60
    sleep_time: float = field(init=False)

    kp_base: float = 120.0
    kd_base: float = 8.0
    kp_load_gain: float = 1.8
    kd_load_gain: float = 1.5
    ff_vel_gain: float = 0.7
    ff_accel_gain: float = 0.5

    backlash_error: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.001, 0.002, 0.002, 0.003]))
    friction_coeff: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.08, 0.08, 0.06, 0.06]))
    gravity_compensation: bool = True
    comp_smoothing: float = 0.02

    traj_type: str = 'trapezoidal'
    position_tol: float = 1e-5
    velocity_tol: float = 1e-4
    accel_time_ratio: float = 0.2
    decel_time_ratio: float = 0.2

    def __post_init__(self):
        self.control_timestep = 1.0 / self.control_frequency
        self.sleep_time = 1.0 / self.fps


# ====================== 全局配置初始化 ======================
# 系统性能优化（降低干扰，提升实时性）
def optimize_system_performance():
    """系统级性能优化配置"""
    # Windows系统优化
    if os.name == 'nt':
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            os.system('chcp 65001 >nul 2>&1')
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)
        except Exception as e:
            log_perf(f"⚠️ Windows系统优化失败（不影响核心功能）: {e}")

    # 强制单线程，避免多线程竞争
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1'
    })


# Mujoco Viewer配置
def init_mujoco_viewer() -> Tuple[bool, Optional[any]]:
    """初始化Mujoco Viewer（兼容多版本）"""
    try:
        from mujoco import viewer as new_viewer
        return True, new_viewer
    except ImportError:
        try:
            import mujoco.viewer as legacy_viewer
            return False, legacy_viewer
        except ImportError as e:
            log_perf(f"⚠️ Mujoco Viewer导入失败（无法可视化）: {e}")
            return False, None


# 全局状态
optimize_system_performance()
MUJOCO_NEW_VIEWER, MUJOCO_VIEWER_MODULE = init_mujoco_viewer()
RUNNING = True
JOINT_CONFIG = JointLimits()
STIFF_DAMP_CONFIG = StiffnessDampingConfig()
CONTROL_CONFIG = ControlConfig()


# ====================== 信号处理（优雅退出） ======================
def signal_handler(sig, frame):
    """信号处理：优雅退出并清理资源"""
    global RUNNING
    if not RUNNING:
        sys.exit(0)
    print("\n⚠️  收到退出信号，正在优雅退出（保存日志+清理资源）...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ====================== 工具函数（性能优化+类型安全） ======================
def get_mujoco_id(model: mujoco.MjModel, obj_type: str, name: str) -> int:
    """兼容Mujoco版本的ID查询，提升鲁棒性"""
    if model is None:
        return -1

    type_map = {
        'joint': mujoco.mjtObj.mjOBJ_JOINT,
        'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
        'site': mujoco.mjtObj.mjOBJ_SITE,
        'geom': mujoco.mjtObj.mjOBJ_GEOM
    }

    try:
        obj_type_int = type_map.get(obj_type, mujoco.mjtObj.mjOBJ_JOINT)
        return mujoco.mj_name2id(model, int(obj_type_int), str(name))
    except Exception as e:
        log_perf(f"⚠️  查询{obj_type} {name} ID失败: {e}")
        return -1


def deg2rad(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """角度转弧度（高精度+容错）"""
    try:
        degrees_arr = np.array(degrees, dtype=np.float64)
        return np.deg2rad(degrees_arr)
    except Exception as e:
        log_perf(f"⚠️  角度转换失败: {e}")
        return 0.0 if np.isscalar(degrees) else np.zeros(JOINT_CONFIG.count, dtype=np.float64)


def rad2deg(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """弧度转角度（高精度+容错）"""
    try:
        radians_arr = np.array(radians, dtype=np.float64)
        return np.rad2deg(radians_arr)
    except Exception as e:
        log_perf(f"⚠️  弧度转换失败: {e}")
        return 0.0 if np.isscalar(radians) else np.zeros(JOINT_CONFIG.count, dtype=np.float64)


def log_perf(content: str, log_path: str = "arm_joint_perf.log") -> None:
    """高性能日志写入（线程安全）"""
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            f.write(f"[{timestamp}] {content}\n")
    except Exception as e:
        print(f"⚠️  写入性能日志失败: {e}")


def trapezoidal_velocity_planner(
        start_pos: float,
        target_pos: float,
        max_vel: float,
        max_accel: float,
        dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    梯形速度轨迹规划（核心平滑运动优化，无超调）
    优化点：向量化计算，减少循环开销
    """
    pos_error = target_pos - start_pos
    total_distance = abs(pos_error)

    if total_distance < CONTROL_CONFIG.position_tol:
        return np.array([target_pos]), np.array([0.0])

    # 计算关键参数
    accel_phase_vel = max_vel
    accel_phase_dist = (accel_phase_vel ** 2) / (2 * max_accel)
    total_accel_decel_dist = 2 * accel_phase_dist
    direction = np.sign(pos_error)

    # 无匀速阶段
    if total_distance <= total_accel_decel_dist:
        max_reached_vel = np.sqrt(total_distance * max_accel)
        accel_time = max_reached_vel / max_accel
        total_time = 2 * accel_time

        # 向量化时间序列
        t = np.arange(0, total_time + dt, dt)
        accel_mask = t <= accel_time

        vel = np.zeros_like(t)
        pos = np.zeros_like(t)

        # 加速阶段
        vel[accel_mask] = max_accel * t[accel_mask] * direction
        pos[accel_mask] = start_pos + 0.5 * max_accel * (t[accel_mask] ** 2) * direction

        # 减速阶段
        delta_t = t[~accel_mask] - accel_time
        vel[~accel_mask] = (max_reached_vel - max_accel * delta_t) * direction
        pos[~accel_mask] = start_pos + (max_reached_vel * accel_time - 0.5 * max_accel * (delta_t ** 2)) * direction

    # 有匀速阶段
    else:
        accel_time = max_vel / max_accel
        uniform_dist = total_distance - total_accel_decel_dist
        uniform_time = uniform_dist / max_vel
        total_time = 2 * accel_time + uniform_time

        # 向量化时间序列
        t = np.arange(0, total_time + dt, dt)
        accel_mask = t <= accel_time
        uniform_mask = (t > accel_time) & (t <= accel_time + uniform_time)
        decel_mask = t > accel_time + uniform_time

        vel = np.zeros_like(t)
        pos = np.zeros_like(t)

        # 加速阶段
        vel[accel_mask] = max_accel * t[accel_mask] * direction
        pos[accel_mask] = start_pos + 0.5 * max_accel * (t[accel_mask] ** 2) * direction

        # 匀速阶段
        delta_t_uniform = t[uniform_mask] - accel_time
        vel[uniform_mask] = max_vel * direction
        pos[uniform_mask] = start_pos + (accel_phase_dist + max_vel * delta_t_uniform) * direction

        # 减速阶段
        delta_t_decel = t[decel_mask] - (accel_time + uniform_time)
        vel[decel_mask] = (max_vel - max_accel * delta_t_decel) * direction
        pos[decel_mask] = start_pos + (
                    total_distance - (accel_phase_dist - 0.5 * max_accel * (delta_t_decel ** 2))) * direction

    # 强制收尾
    pos[-1] = target_pos
    vel[-1] = 0.0

    return pos, vel


# ====================== 机械臂模型生成（优化+合规） ======================
def create_arm_model() -> str:
    """
    生成高性能机械臂Mujoco XML模型
    优化点：
    1. 移除geom无效viscous属性，消除Schema违规
    2. 结构化参数注入，提升可维护性
    3. 高精度接触参数，降低运动干扰
    """
    end_effector_mass = 0.5
    link_masses = [0.8, 0.6, 0.6, 0.4, 0.2]
    friction_coeffs = CONTROL_CONFIG.friction_coeff
    joint_damping = STIFF_DAMP_CONFIG.base_damping * STIFF_DAMP_CONFIG.viscous_damping_gain

    xml_template = f"""
<mujoco model="high_perf_arm">
    <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
    <option timestep="{CONTROL_CONFIG.simulation_timestep}" gravity="0 0 -9.81" iterations="100" tolerance="1e-9"/>

    <default>
        <joint type="hinge" damping="{joint_damping[0]}" limited="true" margin="0.001"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100"/>
        <geom contype="1" conaffinity="1" rgba="0.2 0.8 0.2 1" solref="0.01 1" solimp="0.9 0.95 0.001"
              friction="{friction_coeffs[0]} {friction_coeffs[0]} {friction_coeffs[0]}"/>
    </default>

    <asset>
        <material name="arm_material" rgba="0.0 0.8 0.0 0.8"/>
        <material name="end_effector_material" rgba="0.8 0.2 0.2 1"/>
    </asset>

    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>

        <!-- 基座（joint1） -->
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0.1"
                   range="{JOINT_CONFIG.limits_rad[0, 0]} {JOINT_CONFIG.limits_rad[0, 1]}" damping="{joint_damping[0]}"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[0]}"
                      material="arm_material" friction="{friction_coeffs[1]} {friction_coeffs[1]} {friction_coeffs[1]}"/>

                <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0.18"
                       range="{JOINT_CONFIG.limits_rad[1, 0]} {JOINT_CONFIG.limits_rad[1, 1]}" damping="{joint_damping[1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[1]}"
                          material="arm_material" friction="{friction_coeffs[2]} {friction_coeffs[2]} {friction_coeffs[2]}"/>

                    <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0.18"
                           range="{JOINT_CONFIG.limits_rad[2, 0]} {JOINT_CONFIG.limits_rad[2, 1]}" damping="{joint_damping[2]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[2]}"
                              material="arm_material" friction="{friction_coeffs[3]} {friction_coeffs[3]} {friction_coeffs[3]}"/>

                        <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.18"
                               range="{JOINT_CONFIG.limits_rad[3, 0]} {JOINT_CONFIG.limits_rad[3, 1]}" damping="{joint_damping[3]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[3]}"
                                  material="arm_material" friction="{friction_coeffs[3]} {friction_coeffs[3]} {friction_coeffs[3]}"/>

                            <joint name="joint5" type="hinge" axis="0 1 0" pos="0 0 0.18"
                                   range="{JOINT_CONFIG.limits_rad[4, 0]} {JOINT_CONFIG.limits_rad[4, 1]}" damping="{joint_damping[4]}"/>
                            <body name="link5" pos="0 0 0.18">
                                <geom name="link5_geom" type="cylinder" size="0.03 0.09" mass="{link_masses[4]}"
                                      material="end_effector_material" friction="{friction_coeffs[4]} {friction_coeffs[4]} {friction_coeffs[4]}"/>
                                <body name="end_effector" pos="0 0 0.09">
                                    <site name="ee_site" pos="0 0 0" size="0.005"/>
                                    <geom name="ee_geom" type="sphere" size="0.04" mass="{end_effector_mass}" rgba="1.0 0.0 0.0 0.8"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="motor1" joint="joint1" ctrlrange="-1 1" gear="100"/>
        <motor name="motor2" joint="joint2" ctrlrange="-1 1" gear="100"/>
        <motor name="motor3" joint="joint3" ctrlrange="-1 1" gear="100"/>
        <motor name="motor4" joint="joint4" ctrlrange="-1 1" gear="100"/>
        <motor name="motor5" joint="joint5" ctrlrange="-1 1" gear="100"/>
    </actuator>
</mujoco>
    """
    return xml_template


# ====================== 机械臂控制器核心类（全面优化） ======================
class ArmJointPerfOptimizationController:
    def __init__(self):
        # 模型与数据初始化
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self._init_model()

        # ID缓存（避免重复查询）
        self.joint_ids: List[int] = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_CONFIG.names]
        self.motor_ids: List[int] = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in
                                     range(JOINT_CONFIG.count)]
        self.ee_site_id: int = get_mujoco_id(self.model, 'site', "ee_site")
        self.ee_geom_id: int = get_mujoco_id(self.model, 'geom', "ee_geom")

        # 状态变量（类型注解+初始化优化）
        self.viewer_inst: Optional[any] = None
        self.viewer_ready: bool = False
        self.last_control_time: float = time.time()
        self.last_print_time: float = time.time()
        self.step_count: int = 0
        self.fps_counter: int = 0
        self.total_sim_time: float = 0.0

        # 性能优化核心状态
        self.current_stiffness: np.ndarray = STIFF_DAMP_CONFIG.base_stiffness.copy()
        self.current_damping: np.ndarray = STIFF_DAMP_CONFIG.base_damping.copy() * STIFF_DAMP_CONFIG.viscous_damping_gain
        self.target_angles_rad: np.ndarray = np.zeros(JOINT_CONFIG.count, dtype=np.float64)
        self.planned_positions: np.ndarray = np.zeros((1, JOINT_CONFIG.count), dtype=np.float64)
        self.planned_velocities: np.ndarray = np.zeros((1, JOINT_CONFIG.count), dtype=np.float64)
        self.traj_step_idx: int = 0
        self.position_error: np.ndarray = np.zeros(JOINT_CONFIG.count, dtype=np.float64)
        self.trajectory_error: np.ndarray = np.zeros(JOINT_CONFIG.count, dtype=np.float64)
        self.max_position_error: np.ndarray = np.zeros(JOINT_CONFIG.count, dtype=np.float64)

        # 负载与补偿状态
        self.current_end_load: float = 0.5
        self.smoothed_joint_forces: np.ndarray = np.zeros(JOINT_CONFIG.count, dtype=np.float64)
        self.compensated_error: np.ndarray = np.zeros(JOINT_CONFIG.count, dtype=np.float64)
        self.gravity_comp_torque: np.ndarray = np.zeros(JOINT_CONFIG.count, dtype=np.float64)

        # 初始化轨迹
        self.set_joint_angles(np.zeros(JOINT_CONFIG.count), smooth=False, use_deg=False)
        self.plan_trajectory(np.zeros(JOINT_CONFIG.count), np.zeros(JOINT_CONFIG.count))
        log_perf("机械臂关节运动性能控制器初始化完成")

    def _init_model(self) -> None:
        """初始化Mujoco模型（异常处理强化）"""
        try:
            self.model = mujoco.MjModel.from_xml_string(create_arm_model())
            self.data = mujoco.MjData(self.model)
            log_perf("高性能机械臂模型初始化成功")
        except Exception as e:
            error_msg = f"模型初始化失败: {e}"
            print(f"❌ {error_msg}")
            log_perf(error_msg)
            global RUNNING
            RUNNING = False

    def get_current_joint_angles(self, use_deg: bool = True) -> np.ndarray:
        """获取当前关节角度（优化空值处理）"""
        if self.data is None:
            return np.zeros(JOINT_CONFIG.count, dtype=np.float64)

        current_rad = np.array([
            self.data.qpos[jid] if jid >= 0 else 0.0
            for jid in self.joint_ids
        ], dtype=np.float64)

        return rad2deg(current_rad) if use_deg else current_rad

    def get_current_joint_velocities(self, use_deg: bool = True) -> np.ndarray:
        """获取当前关节速度（优化空值处理）"""
        if self.data is None:
            return np.zeros(JOINT_CONFIG.count, dtype=np.float64)

        current_vel_rad = np.array([
            self.data.qvel[jid] if jid >= 0 else 0.0
            for jid in self.joint_ids
        ], dtype=np.float64)

        return rad2deg(current_vel_rad) if use_deg else current_vel_rad

    def get_joint_forces(self) -> np.ndarray:
        """获取平滑后的关节受力（优化滤波）"""
        if self.data is None:
            return np.zeros(JOINT_CONFIG.count, dtype=np.float64)

        raw_forces = np.array([
            abs(self.data.qfrc_actuator[jid]) if jid >= 0 else 0.0
            for jid in self.joint_ids
        ], dtype=np.float64)

        # 一阶低通滤波（优化平滑效果）
        self.smoothed_joint_forces = 0.95 * self.smoothed_joint_forces + 0.05 * raw_forces
        return self.smoothed_joint_forces

    def calculate_error_compensation(self) -> Tuple[np.ndarray, np.ndarray]:
        """多维度误差补偿（向量化优化）"""
        current_angles = self.get_current_joint_angles(use_deg=False)
        current_vels = self.get_current_joint_velocities(use_deg=False)

        # 1. 关节间隙补偿（向量化）
        vel_sign = np.sign(current_vels)
        vel_sign[np.abs(current_vels) < CONTROL_CONFIG.velocity_tol] = np.sign(self.position_error)[
            np.abs(current_vels) < CONTROL_CONFIG.velocity_tol
            ]
        backlash_comp = CONTROL_CONFIG.backlash_error * vel_sign

        # 2. 静摩擦补偿（向量化）
        friction_comp = np.where(
            np.abs(current_vels) < CONTROL_CONFIG.velocity_tol,
            CONTROL_CONFIG.friction_coeff * np.sign(self.position_error),
            0.0
        )

        # 3. 重力补偿（向量化）
        gravity_comp = np.zeros(JOINT_CONFIG.count, dtype=np.float64)
        if CONTROL_CONFIG.gravity_compensation:
            gravity_comp = 0.5 * np.sin(current_angles) * self.current_end_load

        # 平滑总补偿
        total_comp = backlash_comp + friction_comp + gravity_comp
        self.compensated_error = (1 - CONTROL_CONFIG.comp_smoothing) * self.compensated_error + \
                                 CONTROL_CONFIG.comp_smoothing * total_comp
        self.gravity_comp_torque = gravity_comp * 0.8

        return self.compensated_error, self.gravity_comp_torque

    def calculate_adaptive_stiffness_damping(self) -> Tuple[np.ndarray, np.ndarray]:
        """刚度阻尼自适应匹配（向量化优化）"""
        # 负载与误差归一化（向量化）
        current_forces = self.get_joint_forces()
        force_ratios = np.clip(current_forces / JOINT_CONFIG.max_torque, 0.0, 1.0)
        normalized_load = np.mean(force_ratios)

        angle_error = np.abs(self.position_error)
        normalized_error = np.clip(angle_error / deg2rad(1.0), 0.0, 1.0)

        # 自适应刚度（向量化计算）
        target_stiffness = STIFF_DAMP_CONFIG.base_stiffness * \
                           (1 + normalized_load * (STIFF_DAMP_CONFIG.load_stiffness_gain - 1)) * \
                           (1 + normalized_error[:, np.newaxis] * (STIFF_DAMP_CONFIG.error_stiffness_gain - 1))

        target_stiffness = np.clip(
            target_stiffness,
            STIFF_DAMP_CONFIG.min_stiffness,
            STIFF_DAMP_CONFIG.max_stiffness
        )

        # 平滑更新
        self.current_stiffness = (1 - STIFF_DAMP_CONFIG.stiffness_smoothing) * self.current_stiffness + \
                                 STIFF_DAMP_CONFIG.stiffness_smoothing * target_stiffness

        # 自适应阻尼（与刚度匹配）
        target_damping = self.current_stiffness * STIFF_DAMP_CONFIG.damping_stiffness_ratio
        target_damping = target_damping * STIFF_DAMP_CONFIG.viscous_damping_gain
        self.current_damping = np.clip(
            target_damping,
            STIFF_DAMP_CONFIG.min_damping,
            STIFF_DAMP_CONFIG.max_damping
        )

        # 批量更新模型阻尼（减少循环开销）
        valid_jids = [jid for jid in self.joint_ids if jid >= 0]
        valid_damping = self.current_damping[:len(valid_jids)]
        for jid, damping in zip(valid_jids, valid_damping):
            self.model.jnt_damping[jid] = damping

        return self.current_stiffness, self.current_damping

    def precision_pd_feedforward_control(self) -> None:
        """PD+前馈控制（核心优化：向量化+性能提升）"""
        if self.data is None or self.planned_positions.size == 0:
            return

        # 批量获取状态
        current_angles = self.get_current_joint_angles(use_deg=False)
        current_vels = self.get_current_joint_velocities(use_deg=False)
        compensated_error, gravity_comp_torque = self.calculate_error_compensation()
        self.calculate_adaptive_stiffness_damping()

        # 获取规划轨迹点（边界检查优化）
        if self.traj_step_idx < self.planned_positions.shape[0]:
            target_pos = self.planned_positions[self.traj_step_idx]
            target_vel = self.planned_velocities[self.traj_step_idx]
            self.traj_step_idx += 1
        else:
            target_pos = self.target_angles_rad
            target_vel = np.zeros(JOINT_CONFIG.count, dtype=np.float64)

        # 误差计算（向量化）
        self.position_error = target_pos - current_angles
        self.trajectory_error = self.position_error + (target_vel - current_vels) * CONTROL_CONFIG.control_timestep
        self.max_position_error = np.maximum(self.max_position_error, np.abs(self.position_error))

        # 自适应PD参数（向量化）
        normalized_load = np.clip(self.current_end_load / 2.0, 0.0, 1.0)
        kp = CONTROL_CONFIG.kp_base * (1 + normalized_load * (CONTROL_CONFIG.kp_load_gain - 1))
        kd = CONTROL_CONFIG.kd_base * (1 + normalized_load * (CONTROL_CONFIG.kd_load_gain - 1))

        # 控制计算（全向量化，消除循环）
        pd_control = kp * self.position_error + kd * (target_vel - current_vels)
        ff_control = CONTROL_CONFIG.ff_vel_gain * target_vel + \
                     CONTROL_CONFIG.ff_accel_gain * (target_vel - current_vels) / CONTROL_CONFIG.control_timestep
        total_control = pd_control + ff_control + gravity_comp_torque + compensated_error

        # 输出限幅（向量化）
        total_control = np.clip(total_control, -JOINT_CONFIG.max_torque, JOINT_CONFIG.max_torque)

        # 批量设置控制信号（减少循环）
        valid_mids = [(i, mid) for i, mid in enumerate(self.motor_ids) if mid >= 0]
        for i, mid in valid_mids:
            self.data.ctrl[mid] = total_control[i]

    def plan_trajectory(
            self,
            start_angles: Union[List[float], np.ndarray],
            target_angles: Union[List[float], np.ndarray],
            use_deg: bool = True
    ) -> None:
        """规划梯形速度轨迹（向量化优化，消除逐关节循环）"""
        # 角度转换与限位
        start_angles_rad = self.clamp_joint_angles(start_angles, use_deg=use_deg)
        target_angles_rad = self.clamp_joint_angles(target_angles, use_deg=use_deg)

        # 批量规划轨迹
        joint_trajectories = []
        joint_velocities = []
        max_length = 0

        for i in range(JOINT_CONFIG.count):
            pos_traj, vel_traj = trapezoidal_velocity_planner(
                start_angles_rad[i],
                target_angles_rad[i],
                JOINT_CONFIG.max_velocity_rad[i],
                JOINT_CONFIG.max_accel_rad[i],
                CONTROL_CONFIG.control_timestep
            )
            joint_trajectories.append(pos_traj)
            joint_velocities.append(vel_traj)
            max_length = max(max_length, len(pos_traj))

        # 统一轨迹长度（向量化填充）
        for i in range(JOINT_CONFIG.count):
            if len(joint_trajectories[i]) < max_length:
                pad_len = max_length - len(joint_trajectories[i])
                joint_trajectories[i] = np.pad(
                    joint_trajectories[i],
                    (0, pad_len),
                    'constant',
                    constant_values=target_angles_rad[i]
                )
                joint_velocities[i] = np.pad(
                    joint_velocities[i],
                    (0, pad_len),
                    'constant',
                    constant_values=0.0
                )

        self.planned_positions = np.array(joint_trajectories).T
        self.planned_velocities = np.array(joint_velocities).T
        self.traj_step_idx = 0
        self.target_angles_rad = target_angles_rad.copy()

        # 日志输出（格式化优化）
        info_msg = (
            f"轨迹规划完成：从{np.round(rad2deg(start_angles_rad), 2)}° "
            f"到{np.round(rad2deg(target_angles_rad), 2)}°，长度{max_length}步"
        )
        print(f"✅ {info_msg}")
        log_perf(info_msg)

    def clamp_joint_angles(self, angles: Union[List[float], np.ndarray], use_deg: bool = True) -> np.ndarray:
        """关节角度限位（向量化优化）"""
        angles_arr = np.array(angles, dtype=np.float64)
        angles_rad = deg2rad(angles_arr) if use_deg else angles_arr.copy()

        # 安全余量
        limit_margin = 0.01
        limits = JOINT_CONFIG.limits_rad.copy()
        limits[:, 0] += limit_margin
        limits[:, 1] -= limit_margin

        # 向量化限位
        clamped_rad = np.clip(angles_rad, limits[:, 0], limits[:, 1])
        return rad2deg(clamped_rad) if use_deg else clamped_rad

    def set_end_load(self, mass: float) -> None:
        """动态设置末端负载（异常处理强化）"""
        if not 0.0 <= mass <= 2.0:
            print(f"⚠️  负载超出限制（0-2.0kg），当前设置{mass}kg")
            return

        self.current_end_load = mass

        # 更新末端质量（空值检查）
        if self.ee_geom_id >= 0 and self.model is not None:
            self.model.geom_mass[self.ee_geom_id] = mass

        log_perf(f"末端负载更新为{mass}kg")
        print(f"✅ 末端负载更新为{mass}kg")

    def print_perf_status(self) -> None:
        """打印运动性能状态（优化频率控制）"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        # 性能计算（优化浮点精度）
        fps = self.fps_counter / max(1e-6, current_time - self.last_print_time)
        joint_angles = self.get_current_joint_angles(use_deg=True)
        joint_vels = self.get_current_joint_velocities(use_deg=True)
        pos_error_deg = rad2deg(self.position_error)
        max_pos_error_deg = rad2deg(self.max_position_error)
        stiffness, damping = self.calculate_adaptive_stiffness_damping()

        # 格式化输出（优化可读性）
        print("-" * 120)
        print(f"📊 运动性能统计 | 步数：{self.step_count:,} | 帧率：{fps:.1f} | 总仿真时间：{self.total_sim_time:.2f}s")
        print(f"🔧 关节状态 | 角度：{np.round(joint_angles, 2)}° | 速度：{np.round(joint_vels, 3)}°/s")
        print(
            f"🎯 精度指标 | 当前定位误差：{np.round(np.abs(pos_error_deg), 4)}° | 最大定位误差：{np.round(max_pos_error_deg, 4)}°")
        print(f"🔩 刚度阻尼 | 刚度：{np.round(stiffness, 1)} | 阻尼：{np.round(damping, 1)}")
        print(f"🏋️  负载状态 | 末端负载：{self.current_end_load:.2f}kg")
        print("-" * 120)

        # 状态重置
        self.last_print_time = current_time
        self.fps_counter = 0
        self.total_sim_time += current_time - self.last_print_time

    def init_viewer(self) -> bool:
        """初始化可视化窗口（兼容性优化）"""
        if self.model is None or self.data is None:
            return False

        try:
            if MUJOCO_NEW_VIEWER:
                self.viewer_inst = MUJOCO_VIEWER_MODULE.launch_passive(self.model, self.data)
            else:
                self.viewer_inst = MUJOCO_VIEWER_MODULE.Viewer(self.model, self.data)

            self.viewer_ready = True
            print("✅ 可视化窗口初始化成功")
            return True
        except Exception as e:
            print(f"❌ 可视化窗口初始化失败: {e}")
            return False

    def run(self) -> None:
        """运行运动性能优化主循环（性能优化核心）"""
        global RUNNING
        if not self.init_viewer():
            RUNNING = False
            return

        # 启动信息（优化可读性）
        print("=" * 120)
        print("🚀 机械臂关节运动性能优化控制器启动成功")
        print(
            f"✅ 控制频率：{CONTROL_CONFIG.control_frequency}Hz | 仿真步长：{CONTROL_CONFIG.simulation_timestep}s | 关节数量：{JOINT_CONFIG.count}")
        print(f"✅ 核心优化：梯形轨迹规划 | 自适应PD+前馈 | 刚度阻尼匹配 | 多维度误差补偿")
        print("=" * 120)

        # 主循环（优化时间控制）
        while RUNNING:
            try:
                current_time = time.time()
                self.fps_counter += 1
                self.step_count += 1

                # 高频控制更新（时间精度优化）
                if current_time - self.last_control_time >= CONTROL_CONFIG.control_timestep:
                    self.precision_pd_feedforward_control()
                    self.last_control_time = current_time

                # 仿真步进（空值检查强化）
                if self.model is not None and self.data is not None:
                    mujoco.mj_step(self.model, self.data)

                # 可视化同步（状态检查强化）
                if self.viewer_ready and self.viewer_inst:
                    self.viewer_inst.sync()

                # 状态打印
                self.print_perf_status()

                # 动态睡眠（精度优化）
                time_diff = current_time - self.last_control_time
                sleep_duration = max(1e-6, CONTROL_CONFIG.sleep_time - time_diff)
                time.sleep(sleep_duration)

            except Exception as e:
                error_msg = f"仿真步异常（步数{self.step_count}）: {e}"
                print(f"⚠️ {error_msg}")
                log_perf(error_msg)
                continue

        # 资源清理
        self.cleanup()
        final_msg = (
            f"仿真结束 | 总步数{self.step_count:,} | 总时间{self.total_sim_time:.2f}s | "
            f"最大定位误差{np.round(rad2deg(np.max(self.max_position_error)), 4)}°"
        )
        print(f"\n✅ {final_msg}")
        log_perf(final_msg)

    def cleanup(self) -> None:
        """清理资源（异常处理强化）"""
        if self.viewer_ready and self.viewer_inst:
            try:
                self.viewer_inst.close()
            except Exception as e:
                print(f"⚠️  可视化窗口关闭失败: {e}")

        # 资源释放
        self.model = None
        self.data = None
        global RUNNING
        RUNNING = False

    def preset_pose(self, pose_name: str) -> None:
        """预设运动姿态（配置化管理）"""
        pose_map = {
            'zero': [0, 0, 0, 0, 0],
            'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10],
            'test': [10, 20, 15, 5, 8]
        }

        if pose_name not in pose_map:
            print(f"⚠️  无效姿态，支持：{list(pose_map.keys())}")
            return

        self.set_joint_angles(pose_map[pose_name], smooth=True, use_deg=True)
        print(f"✅ 切换到{pose_name}姿态")


# ====================== 演示函数（优化线程管理） ======================
def perf_optimization_demo(controller: ArmJointPerfOptimizationController) -> None:
    """运动性能优化演示（线程安全）"""

    def demo_task():
        try:
            time.sleep(2)
            controller.preset_pose('zero')
            time.sleep(3)
            controller.preset_pose('test')
            time.sleep(4)
            controller.set_end_load(1.5)
            time.sleep(4)
            controller.preset_pose('grasp')
            time.sleep(3)
            controller.set_end_load(0.2)
            time.sleep(3)
            controller.preset_pose('zero')
            time.sleep(2)
            global RUNNING
            RUNNING = False
        except Exception as e:
            print(f"⚠️  演示任务异常: {e}")
            log_perf(f"演示任务异常: {e}")

    demo_thread = threading.Thread(target=demo_task)
    demo_thread.daemon = True
    demo_thread.start()


# ====================== 主入口（优化初始化） ======================
if __name__ == "__main__":
    # 优化numpy打印格式
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    # 初始化控制器并运行
    try:
        arm_controller = ArmJointPerfOptimizationController()
        perf_optimization_demo(arm_controller)
        arm_controller.run()
    except Exception as e:
        print(f"❌ 控制器运行异常: {e}")
        log_perf(f"控制器运行异常: {e}")
        sys.exit(1)