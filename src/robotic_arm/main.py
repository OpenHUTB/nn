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

# ====================== 全局配置（性能优化核心参数） ======================
# 系统适配与性能优化（降低干扰，提升控制实时性）
if os.name == 'nt':
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        os.system('chcp 65001 >nul 2>&1')
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)  # 提升线程优先级
    except Exception as e:
        print(f"⚠️ Windows系统优化失败（不影响核心功能）: {e}")
# 强制单线程，避免多线程竞争导致控制延迟
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Mujoco Viewer兼容配置
MUJOCO_NEW_VIEWER = False
try:
    from mujoco import viewer

    MUJOCO_NEW_VIEWER = True
except ImportError:
    try:
        import mujoco.viewer as viewer
    except ImportError as e:
        print(f"⚠️ Mujoco Viewer导入失败（无法可视化）: {e}")

# 关节基础参数（5自由度机械臂，可按需扩展）
JOINT_COUNT = 5
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5"]
# 关节角度限制（rad）
JOINT_LIMITS_RAD = np.array([
    [-np.pi, np.pi],  # joint1（基座）
    [-np.pi / 2, np.pi / 2],  # joint2（大臂）
    [-np.pi / 2, np.pi / 2],  # joint3（中臂）
    [-np.pi / 2, np.pi / 2],  # joint4（小臂）
    [-np.pi / 2, np.pi / 2],  # joint5（末端）
], dtype=np.float64)
# 关节运动性能限制（避免超调与过载）
JOINT_MAX_VELOCITY_RAD = np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64)
JOINT_MAX_ACCEL_RAD = np.array([2.0, 1.6, 1.6, 1.2, 1.2], dtype=np.float64)
JOINT_MAX_TORQUE = np.array([15.0, 12.0, 10.0, 8.0, 5.0], dtype=np.float64)

# 刚度自适应配置（提升定位精度与抗干扰能力）
STIFFNESS_PARAMS = {
    'base_stiffness': np.array([200.0, 180.0, 150.0, 120.0, 80.0]),
    'load_stiffness_gain': 1.8,
    'error_stiffness_gain': 1.5,
    'min_stiffness': np.array([100.0, 90.0, 75.0, 60.0, 40.0]),
    'max_stiffness': np.array([300.0, 270.0, 225.0, 180.0, 120.0]),
    'stiffness_smoothing': 0.05,  # 刚度平滑更新，避免运动抖动
}

# 阻尼自适应配置（粘性阻尼整合，提升运动平滑性）
DAMPING_PARAMS = {
    'base_damping': np.array([8.0, 7.0, 6.0, 5.0, 3.0]),  # 基础阻尼
    'viscous_damping_gain': np.array([1.2, 1.1, 1.1, 1.0, 1.0]),  # 粘性阻尼增益
    'damping_stiffness_ratio': 0.04,  # 阻尼与刚度匹配系数
    'min_damping': np.array([4.0, 3.5, 3.0, 2.5, 1.5]),
    'max_damping': np.array([16.0, 14.0, 12.0, 10.0, 6.0]),
}

# 仿真与控制性能配置（高频控制提升精度，微步长降低离散误差）
SIMULATION_TIMESTEP = 0.0005  # 仿真微步长
CONTROL_FREQUENCY = 2000  # 控制频率（2000Hz，高频实时控制）
CONTROL_TIMESTEP = 1.0 / CONTROL_FREQUENCY
FPS = 60  # 可视化帧率（不影响控制性能）
SLEEP_TIME = 1.0 / FPS
RUNNING = True  # 仿真运行标志

# PD+前馈控制参数（核心运动精度优化）
PD_FEEDFORWARD_PARAMS = {
    'kp_base': 120.0,  # 比例增益（提升静态定位精度）
    'kd_base': 8.0,  # 微分增益（抑制运动振动）
    'kp_load_gain': 1.8,  # 负载下比例增益放大
    'kd_load_gain': 1.5,  # 负载下微分增益放大
    'ff_vel_gain': 0.7,  # 速度前馈增益（补偿动态误差）
    'ff_accel_gain': 0.5,  # 加速度前馈增益（提升动态响应）
}

# 误差补偿配置（多维度补偿，消除系统误差）
ERROR_COMPENSATION_PARAMS = {
    'backlash_error': np.array([0.001, 0.001, 0.002, 0.002, 0.003]),  # 关节间隙误差
    'friction_coeff': np.array([0.1, 0.08, 0.08, 0.06, 0.06]),  # 静摩擦系数
    'gravity_compensation': True,  # 重力误差补偿
    'comp_smoothing': 0.02,  # 补偿量平滑
}

# 轨迹规划配置（梯形速度规划，无超调平滑运动）
TRAJECTORY_PARAMS = {
    'traj_type': 'trapezoidal',
    'position_tol': 1e-5,  # 位置公差（高精度定位判定）
    'velocity_tol': 1e-4,  # 速度公差（平稳停止判定）
    'accel_time_ratio': 0.2,  # 加速时间占比
    'decel_time_ratio': 0.2,  # 减速时间占比
}


# ====================== 信号处理（优雅退出，保护数据） ======================
def signal_handler(sig, frame):
    global RUNNING
    if not RUNNING:
        sys.exit(0)
    print("\n⚠️  收到退出信号，正在优雅退出（保存日志+清理资源）...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ====================== 工具函数（性能优化辅助） ======================
def get_mujoco_id(model, obj_type, name):
    """兼容Mujoco版本的ID查询，提升代码鲁棒性"""
    if model is None:
        return -1
    type_map = {
        'joint': mujoco.mjtObj.mjOBJ_JOINT,
        'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
        'site': mujoco.mjtObj.mjOBJ_SITE,
        'geom': mujoco.mjtObj.mjOBJ_GEOM
    }
    obj_type_int = type_map.get(obj_type, mujoco.mjtObj.mjOBJ_JOINT)
    try:
        return mujoco.mj_name2id(model, int(obj_type_int), str(name))
    except Exception as e:
        print(f"⚠️  查询{obj_type} {name} ID失败: {e}")
        return -1


def deg2rad(degrees):
    """角度转弧度（高精度转换，容错增强）"""
    try:
        degrees = np.array(degrees, dtype=np.float64)
        return np.deg2rad(degrees)
    except Exception as e:
        print(f"⚠️  角度转换失败: {e}")
        return 0.0 if np.isscalar(degrees) else np.zeros(JOINT_COUNT, dtype=np.float64)


def rad2deg(radians):
    """弧度转角度（高精度转换，容错增强）"""
    try:
        radians = np.array(radians, dtype=np.float64)
        return np.rad2deg(radians)
    except Exception as e:
        print(f"⚠️  弧度转换失败: {e}")
        return 0.0 if np.isscalar(radians) else np.zeros(JOINT_COUNT, dtype=np.float64)


def write_perf_log(content, log_path="arm_joint_perf.log"):
    """写入运动性能日志，便于后续分析优化"""
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            f.write(f"[{timestamp}] {content}\n")
    except Exception as e:
        print(f"⚠️  写入性能日志失败: {e}")


def trapezoidal_velocity_planner(start_pos, target_pos, max_vel, max_accel, dt):
    """
    梯形速度轨迹规划（核心平滑运动优化，无超调）
    :param start_pos: 起始位置（rad）
    :param target_pos: 目标位置（rad）
    :param max_vel: 最大速度（rad/s）
    :param max_accel: 最大加速度（rad/s²）
    :param dt: 时间步长（s）
    :return: 规划位置序列、速度序列
    """
    pos_error = target_pos - start_pos
    total_distance = abs(pos_error)
    if total_distance < TRAJECTORY_PARAMS['position_tol']:
        return np.array([target_pos]), np.array([0.0])

    # 计算梯形轨迹关键参数
    accel_phase_vel = max_vel
    accel_phase_dist = (accel_phase_vel ** 2) / (2 * max_accel)
    total_accel_decel_dist = 2 * accel_phase_dist
    direction = np.sign(pos_error)

    pos_list = []
    vel_list = []
    current_pos = start_pos
    current_vel = 0.0

    if total_distance <= total_accel_decel_dist:
        # 无匀速阶段：加速后立即减速
        max_reached_vel = np.sqrt(total_distance * max_accel)
        accel_time = max_reached_vel / max_accel
        total_time = 2 * accel_time

        t = 0.0
        while t < total_time + dt:
            if t <= accel_time:
                # 加速阶段
                current_vel = max_accel * t * direction
                current_pos = start_pos + 0.5 * max_accel * (t ** 2) * direction
            else:
                # 减速阶段
                delta_t = t - accel_time
                current_vel = (max_reached_vel - max_accel * delta_t) * direction
                current_pos = start_pos + (max_reached_vel * accel_time - 0.5 * max_accel * (delta_t ** 2)) * direction
            pos_list.append(current_pos)
            vel_list.append(current_vel)
            t += dt
    else:
        # 有匀速阶段：加速→匀速→减速
        accel_time = max_vel / max_accel
        uniform_dist = total_distance - total_accel_decel_dist
        uniform_time = uniform_dist / max_vel
        total_time = 2 * accel_time + uniform_time

        t = 0.0
        while t < total_time + dt:
            if t <= accel_time:
                # 加速阶段
                current_vel = max_accel * t * direction
                current_pos = start_pos + 0.5 * max_accel * (t ** 2) * direction
            elif t <= accel_time + uniform_time:
                # 匀速阶段
                current_vel = max_vel * direction
                delta_t = t - accel_time
                current_pos = start_pos + (accel_phase_dist + max_vel * delta_t) * direction
            else:
                # 减速阶段
                delta_t = t - (accel_time + uniform_time)
                current_vel = (max_vel - max_accel * delta_t) * direction
                delta_pos = accel_phase_dist - 0.5 * max_accel * (delta_t ** 2)
                current_pos = start_pos + (total_distance - delta_pos) * direction
            pos_list.append(current_pos)
            vel_list.append(current_vel)
            t += dt

    # 强制收尾，消除累积误差
    pos_list[-1] = target_pos
    vel_list[-1] = 0.0
    return np.array(pos_list), np.array(vel_list)


# ====================== 机械臂模型生成（性能优化+合规配置） ======================
def create_arm_model():
    """
    生成高性能机械臂Mujoco XML模型
    核心优化：
    1.  移除geom无效viscous属性，消除Schema违规
    2.  joint标签配置damping，整合粘性阻尼效果
    3.  高精度接触参数，降低运动干扰
    4.  合理惯量配置，提升控制响应速度
    """
    end_effector_mass = 0.5
    link_masses = [0.8, 0.6, 0.6, 0.4, 0.2]
    friction_coeffs = ERROR_COMPENSATION_PARAMS['friction_coeff']
    joint_damping = DAMPING_PARAMS['base_damping'] * DAMPING_PARAMS['viscous_damping_gain']

    xml = f"""
<mujoco model="high_perf_arm">
    <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
    <option timestep="{SIMULATION_TIMESTEP}" gravity="0 0 -9.81" iterations="100" tolerance="1e-9"/>

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
                   range="{JOINT_LIMITS_RAD[0, 0]} {JOINT_LIMITS_RAD[0, 1]}" damping="{joint_damping[0]}"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[0]}"
                      material="arm_material" friction="{friction_coeffs[1]} {friction_coeffs[1]} {friction_coeffs[1]}"/>

                <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0.18"
                       range="{JOINT_LIMITS_RAD[1, 0]} {JOINT_LIMITS_RAD[1, 1]}" damping="{joint_damping[1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[1]}"
                          material="arm_material" friction="{friction_coeffs[2]} {friction_coeffs[2]} {friction_coeffs[2]}"/>

                    <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0.18"
                           range="{JOINT_LIMITS_RAD[2, 0]} {JOINT_LIMITS_RAD[2, 1]}" damping="{joint_damping[2]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[2]}"
                              material="arm_material" friction="{friction_coeffs[3]} {friction_coeffs[3]} {friction_coeffs[3]}"/>

                        <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.18"
                               range="{JOINT_LIMITS_RAD[3, 0]} {JOINT_LIMITS_RAD[3, 1]}" damping="{joint_damping[3]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[3]}"
                                  material="arm_material" friction="{friction_coeffs[3]} {friction_coeffs[3]} {friction_coeffs[3]}"/>

                            <joint name="joint5" type="hinge" axis="0 1 0" pos="0 0 0.18"
                                   range="{JOINT_LIMITS_RAD[4, 0]} {JOINT_LIMITS_RAD[4, 1]}" damping="{joint_damping[4]}"/>
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
    return xml


# ====================== 机械臂关节运动性能优化控制器 ======================
class ArmJointPerfOptimizationController:
    def __init__(self):
        # 模型与数据初始化
        self.model = None
        self.data = None
        self.init_model()

        # 关节与执行器ID
        self.joint_ids = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_NAMES]
        self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
        self.ee_site_id = get_mujoco_id(self.model, 'site', "ee_site")

        # 运动状态变量
        self.viewer_inst = None
        self.viewer_ready = False
        self.last_control_time = time.time()
        self.last_print_time = time.time()
        self.step_count = 0
        self.fps_counter = 0
        self.total_sim_time = 0.0

        # 性能优化核心状态
        self.current_stiffness = STIFFNESS_PARAMS['base_stiffness'].copy()
        self.current_damping = DAMPING_PARAMS['base_damping'].copy() * DAMPING_PARAMS['viscous_damping_gain']
        self.target_angles_rad = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.planned_positions = np.zeros((1, JOINT_COUNT), dtype=np.float64)
        self.planned_velocities = np.zeros((1, JOINT_COUNT), dtype=np.float64)
        self.traj_step_idx = 0
        self.position_error = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.trajectory_error = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.max_position_error = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 负载与补偿状态
        self.current_end_load = 0.5
        self.smoothed_joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.compensated_error = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.gravity_comp_torque = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 初始化轨迹与零位
        self.set_joint_angles(np.zeros(JOINT_COUNT), smooth=False, use_deg=False)
        self.plan_trajectory(np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT))
        write_perf_log("机械臂关节运动性能控制器初始化完成")

    def init_model(self):
        """初始化Mujoco模型，确保无Schema违规"""
        try:
            self.model = mujoco.MjModel.from_xml_string(create_arm_model())
            self.data = mujoco.MjData(self.model)
            write_perf_log("高性能机械臂模型初始化成功")
        except Exception as e:
            error_msg = f"模型初始化失败: {e}"
            print(f"❌ {error_msg}")
            write_perf_log(error_msg)
            global RUNNING
            RUNNING = False

    def get_current_joint_angles(self, use_deg=True):
        """获取当前关节角度（高精度采集）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        current_rad = np.array([self.data.qpos[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        return rad2deg(current_rad) if use_deg else current_rad

    def get_current_joint_velocities(self, use_deg=True):
        """获取当前关节速度（用于速度闭环控制）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        current_vel_rad = np.array([self.data.qvel[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        return rad2deg(current_vel_rad) if use_deg else current_vel_rad

    def get_joint_forces(self):
        """获取平滑后的关节受力（用于负载检测）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                raw_force = abs(self.data.qfrc_actuator[jid])
                self.smoothed_joint_forces[i] = (1 - 0.05) * self.smoothed_joint_forces[i] + 0.05 * raw_force
                joint_forces[i] = self.smoothed_joint_forces[i]
        return joint_forces

    def calculate_error_compensation(self):
        """多维度误差补偿（提升定位精度）"""
        current_angles = self.get_current_joint_angles(use_deg=False)
        current_vels = self.get_current_joint_velocities(use_deg=False)

        # 1. 关节间隙补偿
        backlash_comp = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i in range(JOINT_COUNT):
            if abs(current_vels[i]) > TRAJECTORY_PARAMS['velocity_tol']:
                backlash_comp[i] = ERROR_COMPENSATION_PARAMS['backlash_error'][i] * np.sign(current_vels[i])
            else:
                backlash_comp[i] = ERROR_COMPENSATION_PARAMS['backlash_error'][i] * np.sign(self.position_error[i])

        # 2. 静摩擦补偿
        friction_comp = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i in range(JOINT_COUNT):
            if abs(current_vels[i]) < TRAJECTORY_PARAMS['velocity_tol']:
                friction_comp[i] = ERROR_COMPENSATION_PARAMS['friction_coeff'][i] * np.sign(self.position_error[i])

        # 3. 重力补偿
        gravity_comp = np.zeros(JOINT_COUNT, dtype=np.float64)
        if ERROR_COMPENSATION_PARAMS['gravity_compensation']:
            for i in range(JOINT_COUNT):
                gravity_comp[i] = 0.5 * np.sin(current_angles[i]) * self.current_end_load

        # 平滑总补偿
        total_comp = backlash_comp + friction_comp + gravity_comp
        self.compensated_error = (1 - ERROR_COMPENSATION_PARAMS['comp_smoothing']) * self.compensated_error + \
                                 ERROR_COMPENSATION_PARAMS['comp_smoothing'] * total_comp
        self.gravity_comp_torque = gravity_comp * 0.8

        return self.compensated_error, self.gravity_comp_torque

    def calculate_adaptive_stiffness_damping(self):
        """刚度阻尼自适应匹配（提升运动平滑性与抗干扰能力）"""
        # 负载归一化
        current_forces = self.get_joint_forces()
        force_ratios = current_forces / JOINT_MAX_TORQUE
        normalized_load = np.clip(np.mean(force_ratios), 0, 1)

        # 误差归一化
        angle_error = np.abs(self.position_error)
        normalized_error = np.clip(angle_error / deg2rad(1.0), 0, 1)

        # 自适应刚度
        target_stiffness = STIFFNESS_PARAMS['base_stiffness'] * \
                           (1 + normalized_load * (STIFFNESS_PARAMS['load_stiffness_gain'] - 1)) * \
                           (1 + normalized_error * (STIFFNESS_PARAMS['error_stiffness_gain'] - 1))
        target_stiffness = np.clip(target_stiffness, STIFFNESS_PARAMS['min_stiffness'],
                                   STIFFNESS_PARAMS['max_stiffness'])
        self.current_stiffness = (1 - STIFFNESS_PARAMS['stiffness_smoothing']) * self.current_stiffness + \
                                 STIFFNESS_PARAMS['stiffness_smoothing'] * target_stiffness

        # 自适应阻尼（与刚度匹配）
        target_damping = self.current_stiffness * DAMPING_PARAMS['damping_stiffness_ratio']
        target_damping = target_damping * DAMPING_PARAMS['viscous_damping_gain']
        self.current_damping = np.clip(target_damping, DAMPING_PARAMS['min_damping'], DAMPING_PARAMS['max_damping'])

        # 更新模型阻尼
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0 and self.model is not None:
                self.model.jnt_damping[jid] = self.current_damping[i]

        return self.current_stiffness, self.current_damping

    def precision_pd_feedforward_control(self):
        """PD+前馈控制（核心运动精度与平滑性优化）"""
        if self.data is None or self.planned_positions.shape[0] == 0:
            return

        # 获取当前状态与补偿
        current_angles = self.get_current_joint_angles(use_deg=False)
        current_vels = self.get_current_joint_velocities(use_deg=False)
        compensated_error, gravity_comp_torque = self.calculate_error_compensation()
        self.calculate_adaptive_stiffness_damping()

        # 获取规划轨迹点
        if self.traj_step_idx < self.planned_positions.shape[0]:
            target_pos = self.planned_positions[self.traj_step_idx]
            target_vel = self.planned_velocities[self.traj_step_idx]
            self.traj_step_idx += 1
        else:
            target_pos = self.target_angles_rad
            target_vel = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 计算误差
        self.position_error = target_pos - current_angles
        self.trajectory_error = target_pos - current_angles + (target_vel - current_vels) * CONTROL_TIMESTEP
        self.max_position_error = np.maximum(self.max_position_error, np.abs(self.position_error))

        # 自适应PD参数
        normalized_load = np.clip(self.current_end_load / 2.0, 0, 1)
        kp = PD_FEEDFORWARD_PARAMS['kp_base'] * (1 + normalized_load * (PD_FEEDFORWARD_PARAMS['kp_load_gain'] - 1))
        kd = PD_FEEDFORWARD_PARAMS['kd_base'] * (1 + normalized_load * (PD_FEEDFORWARD_PARAMS['kd_load_gain'] - 1))

        # PD控制 + 前馈补偿 + 重力补偿
        pd_control = kp * self.position_error + kd * (target_vel - current_vels)
        ff_control = PD_FEEDFORWARD_PARAMS['ff_vel_gain'] * target_vel + \
                     PD_FEEDFORWARD_PARAMS['ff_accel_gain'] * (target_vel - current_vels) / CONTROL_TIMESTEP
        total_control = pd_control + ff_control + gravity_comp_torque + compensated_error

        # 输出限幅
        for i in range(JOINT_COUNT):
            total_control[i] = np.clip(total_control[i], -JOINT_MAX_TORQUE[i], JOINT_MAX_TORQUE[i])

        # 设置控制信号
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = total_control[i]

    def plan_trajectory(self, start_angles, target_angles, use_deg=True):
        """规划梯形速度轨迹（平滑无超调）"""
        start_angles_rad = self.clamp_joint_angles(start_angles, use_deg=use_deg)
        target_angles_rad = self.clamp_joint_angles(target_angles, use_deg=use_deg)

        # 逐关节规划轨迹
        joint_planned_pos = []
        joint_planned_vel = []
        max_traj_length = 0
        for i in range(JOINT_COUNT):
            pos_traj, vel_traj = trapezoidal_velocity_planner(
                start_angles_rad[i],
                target_angles_rad[i],
                JOINT_MAX_VELOCITY_RAD[i],
                JOINT_MAX_ACCEL_RAD[i],
                CONTROL_TIMESTEP
            )
            joint_planned_pos.append(pos_traj)
            joint_planned_vel.append(vel_traj)
            max_traj_length = max(max_traj_length, len(pos_traj))

        # 统一轨迹长度
        for i in range(JOINT_COUNT):
            if len(joint_planned_pos[i]) < max_traj_length:
                pad_len = max_traj_length - len(joint_planned_pos[i])
                joint_planned_pos[i] = np.pad(joint_planned_pos[i], (0, pad_len), 'constant',
                                              constant_values=target_angles_rad[i])
                joint_planned_vel[i] = np.pad(joint_planned_vel[i], (0, pad_len), 'constant', constant_values=0.0)

        self.planned_positions = np.array(joint_planned_pos).T
        self.planned_velocities = np.array(joint_planned_vel).T
        self.traj_step_idx = 0
        self.target_angles_rad = target_angles_rad.copy()

        info_msg = f"轨迹规划完成：从{np.round(rad2deg(start_angles_rad), 2)}°到{np.round(rad2deg(target_angles_rad), 2)}°，长度{max_traj_length}步"
        print(f"✅ {info_msg}")
        write_perf_log(info_msg)

    def set_joint_angles(self, target_angles, smooth=True, use_deg=True):
        """设置关节目标角度（带限位保护）"""
        if len(target_angles) != JOINT_COUNT:
            raise ValueError(f"目标角度数量必须为{JOINT_COUNT}")

        target_angles_rad = self.clamp_joint_angles(target_angles, use_deg=use_deg)

        # 直接设置或平滑规划
        if not smooth:
            for i, jid in enumerate(self.joint_ids):
                if jid >= 0:
                    self.data.qpos[jid] = target_angles_rad[i]
                    self.data.qvel[jid] = 0.0
            mujoco.mj_forward(self.model, self.data)
        else:
            start_angles = self.get_current_joint_angles(use_deg=use_deg)
            self.plan_trajectory(start_angles, target_angles, use_deg=use_deg)

        self.target_angles_rad = target_angles_rad.copy()

    def clamp_joint_angles(self, angles, use_deg=True):
        """关节角度限位（避免超程）"""
        angles = np.array(angles, dtype=np.float64)
        angles_rad = deg2rad(angles) if use_deg else angles.copy()
        # 安全余量
        limit_margin = 0.01
        limits = JOINT_LIMITS_RAD.copy()
        limits[:, 0] += limit_margin
        limits[:, 1] -= limit_margin
        clamped_rad = np.clip(angles_rad, limits[:, 0], limits[:, 1])
        return rad2deg(clamped_rad) if use_deg else clamped_rad

    def set_end_load(self, mass):
        """动态设置末端负载（用于抗干扰测试）"""
        if mass < 0 or mass > 2.0:
            print(f"⚠️  负载超出限制（0-2.0kg），当前设置{mass}kg")
            return

        self.current_end_load = mass
        # 更新末端几何质量
        ee_geom_id = get_mujoco_id(self.model, 'geom', "ee_geom")
        if ee_geom_id >= 0:
            self.model.geom_mass[ee_geom_id] = mass
        write_perf_log(f"末端负载更新为{mass}kg")
        print(f"✅ 末端负载更新为{mass}kg")

    def get_total_sim_time(self):
        """获取累计仿真时间，优先使用MuJoCo内部时间"""
        if self.data is not None:
            return float(self.data.time)
        return self.total_sim_time

    def print_perf_status(self):
        """打印运动性能状态"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        self.fps_counter = max(1, self.fps_counter)
        fps = self.fps_counter / (current_time - self.last_print_time)
        self.total_sim_time = self.get_total_sim_time()
        joint_angles = self.get_current_joint_angles(use_deg=True)
        joint_vels = self.get_current_joint_velocities(use_deg=True)
        pos_error_deg = rad2deg(self.position_error)
        max_pos_error_deg = rad2deg(self.max_position_error)
        stiffness, damping = self.calculate_adaptive_stiffness_damping()

        # 格式化输出
        print("-" * 120)
        print(f"📊 运动性能统计 | 步数：{self.step_count:,} | 帧率：{fps:.1f} | 总仿真时间：{self.total_sim_time:.2f}s")
        print(f"🔧 关节状态 | 角度：{np.round(joint_angles, 2)}° | 速度：{np.round(joint_vels, 3)}°/s")
        print(
            f"🎯 精度指标 | 当前定位误差：{np.round(np.abs(pos_error_deg), 4)}° | 最大定位误差：{np.round(max_pos_error_deg, 4)}°")
        print(f"🔩 刚度阻尼 | 刚度：{np.round(stiffness, 1)} | 阻尼：{np.round(damping, 1)}")
        print(f"🏋️  负载状态 | 末端负载：{self.current_end_load:.2f}kg")
        print("-" * 120)

        self.last_print_time = current_time
        self.fps_counter = 0

    def init_viewer(self):
        """初始化可视化窗口"""
        if self.model is None or self.data is None:
            return False
        try:
            if MUJOCO_NEW_VIEWER:
                self.viewer_inst = viewer.launch_passive(self.model, self.data)
            else:
                self.viewer_inst = viewer.Viewer(self.model, self.data)
            self.viewer_ready = True
            print("✅ 可视化窗口初始化成功")
            return True
        except Exception as e:
            print(f"❌ 可视化窗口初始化失败: {e}")
            return False

    def run(self):
        """运行运动性能优化主循环"""
        global RUNNING
        if not self.init_viewer():
            RUNNING = False
            return

        print("=" * 120)
        print("🚀 机械臂关节运动性能优化控制器启动成功")
        print(f"✅ 控制频率：{CONTROL_FREQUENCY}Hz | 仿真步长：{SIMULATION_TIMESTEP}s | 关节数量：{JOINT_COUNT}")
        print(f"✅ 核心优化：梯形轨迹规划 | 自适应PD+前馈 | 刚度阻尼匹配 | 多维度误差补偿")
        print("=" * 120)

        # 主循环
        while RUNNING:
            try:
                current_time = time.time()
                self.fps_counter += 1
                self.step_count += 1

                # 高频控制更新
                if current_time - self.last_control_time >= CONTROL_TIMESTEP:
                    self.precision_pd_feedforward_control()
                    self.last_control_time = current_time

                # 仿真步进
                if self.model is not None and self.data is not None:
                    mujoco.mj_step(self.model, self.data)

                # 可视化同步
                if self.viewer_ready:
                    self.viewer_inst.sync()

                # 状态打印
                self.print_perf_status()

                # 动态睡眠
                time_diff = current_time - self.last_control_time
                if time_diff < SLEEP_TIME:
                    time.sleep(max(0.00001, SLEEP_TIME - time_diff))

            except Exception as e:
                error_msg = f"仿真步异常（步数{self.step_count}）: {e}"
                print(f"⚠️ {error_msg}")
                write_perf_log(error_msg)
                continue

        self.total_sim_time = self.get_total_sim_time()
        # 资源清理
        self.cleanup()
        final_msg = f"仿真结束 | 总步数{self.step_count:,} | 总时间{self.total_sim_time:.2f}s | 最大定位误差{np.round(rad2deg(np.max(self.max_position_error)), 4)}°"
        print(f"\n✅ {final_msg}")
        write_perf_log(final_msg)

    def cleanup(self):
        """清理资源"""
        if self.viewer_ready and self.viewer_inst:
            try:
                self.viewer_inst.close()
            except Exception as e:
                print(f"⚠️  可视化窗口关闭失败: {e}")
        self.model = None
        self.data = None
        global RUNNING
        RUNNING = False

    def preset_pose(self, pose_name):
        """预设运动姿态"""
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


# ====================== 演示函数 ======================
def perf_optimization_demo(controller):
    """运动性能优化演示"""

    def demo_task():
        time.sleep(2)
        # 1. 零位校准
        controller.preset_pose('zero')
        time.sleep(3)
        # 2. 精度测试姿态
        controller.preset_pose('test')
        time.sleep(4)
        # 3. 增加负载测试抗干扰
        controller.set_end_load(1.5)
        time.sleep(4)
        # 4. 抓取姿态
        controller.preset_pose('grasp')
        time.sleep(3)
        # 5. 降低负载
        controller.set_end_load(0.2)
        time.sleep(3)
        # 6. 复位零位
        controller.preset_pose('zero')
        time.sleep(2)
        # 结束演示
        global RUNNING
        RUNNING = False

    demo_thread = threading.Thread(target=demo_task)
    demo_thread.daemon = True
    demo_thread.start()


# ====================== 主入口 ======================
if __name__ == "__main__":
    # 优化numpy打印格式
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    # 初始化控制器
    arm_controller = ArmJointPerfOptimizationController()
    # 启动性能演示
    perf_optimization_demo(arm_controller)
    # 运行主循环
    arm_controller.run()
