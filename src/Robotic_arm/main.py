#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节精度性能优化控制器（修复geom标签viscous属性错误版）
核心修复：移除geom标签无效viscous属性，迁移至joint标签damping属性，保证XML Schema合规
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

# ====================== 全局配置（精度优化专用） ======================
# 系统适配（Windows优先，降低系统干扰影响精度）
if os.name == 'nt':
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        os.system('chcp 65001 >nul 2>&1')
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)
    except Exception as e:
        print(f"⚠️ Windows系统优化失败（不影响核心功能）: {e}")
    # 强制单线程，避免多线程竞争导致控制延迟，影响精度
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Mujoco Viewer兼容
MUJOCO_NEW_VIEWER = False
try:
    from mujoco import viewer

    MUJOCO_NEW_VIEWER = True
except ImportError:
    try:
        import mujoco.viewer as viewer
    except ImportError as e:
        print(f"⚠️ Mujoco Viewer导入失败（无法可视化）: {e}")

# 核心参数配置
JOINT_COUNT = 5
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5"]
JOINT_LIMITS_RAD = np.array([
    [-np.pi, np.pi],  # joint1（基座）
    [-np.pi / 2, np.pi / 2],  # joint2（大臂）
    [-np.pi / 2, np.pi / 2],  # joint3（中臂）
    [-np.pi / 2, np.pi / 2],  # joint4（小臂）
    [-np.pi / 2, np.pi / 2],  # joint5（末端）
], dtype=np.float64)
JOINT_MAX_VELOCITY_RAD = np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64)
JOINT_MAX_ACCEL_RAD = np.array([2.0, 1.6, 1.6, 1.2, 1.2], dtype=np.float64)  # 最大加速度（精度优化：限制加减速避免超调）
JOINT_MAX_TORQUE = np.array([15.0, 12.0, 10.0, 8.0, 5.0], dtype=np.float64)

# 刚度配置（兼容之前的优化，不影响精度）
STIFFNESS_PARAMS = {
    'base_stiffness': np.array([200.0, 180.0, 150.0, 120.0, 80.0]),
    'load_stiffness_gain': 1.8,
    'error_stiffness_gain': 1.5,
    'min_stiffness': np.array([100.0, 90.0, 75.0, 60.0, 40.0]),
    'max_stiffness': np.array([300.0, 270.0, 225.0, 180.0, 120.0]),
    'stiffness_smoothing': 0.05,
}

# 阻尼与惯量配置（优化：将粘性摩擦参数整合至joint damping）
DAMPING_INERTIA_PARAMS = {
    'base_damping': np.array([8.0, 7.0, 6.0, 5.0, 3.0]),  # 基础阻尼（对应原粘性摩擦需求）
    'viscous_damping_gain': np.array([1.2, 1.1, 1.1, 1.0, 1.0]),  # 粘性阻尼增益，补充原有viscous效果
    'damping_stiffness_ratio': 0.04,
    'armature_inertia': np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
}

# 仿真配置（精度优化：更小步长+更高控制频率，提升控制分辨率）
SIMULATION_TIMESTEP = 0.0005  # 微步长，降低离散化误差
CONTROL_FREQUENCY = 2000  # 高频控制，提升响应精度
CONTROL_TIMESTEP = 1.0 / CONTROL_FREQUENCY
FPS = 60
SLEEP_TIME = 1.0 / FPS
EPS = 1e-9  # 更小误差阈值，提升精度判断准确性
RUNNING = True
SIMULATION_START_TIME = None

# 高精度PD+前馈控制参数（核心精度优化）
PRECISION_PD_PARAMS = {
    'kp_base': 120.0,  # 更高比例增益，提升静态定位精度
    'kd_base': 8.0,  # 优化阻尼增益，抑制振动超调
    'kp_load_gain': 1.8,  # 负载下增益放大，维持精度
    'kd_load_gain': 1.5,  # 负载下阻尼优化，防止震荡
    'ff_gain': 0.7,  # 前馈增益，补偿动态误差
    'max_vel': JOINT_MAX_VELOCITY_RAD.copy(),
    'max_accel': JOINT_MAX_ACCEL_RAD.copy()
}

# 负载配置
LOAD_PARAMS = {
    'end_effector_mass': 0.5,
    'joint_loads': np.zeros(JOINT_COUNT),
    'max_allowed_load': 2.0,
    'load_smoothing_factor': 0.05  # 更小平滑系数，提升负载检测精度
}

# 误差补偿配置（核心精度优化：移除geom的viscous配置，保留摩擦系数用于误差计算）
ERROR_COMPENSATION_PARAMS = {
    'backlash_error': np.array([0.001, 0.001, 0.002, 0.002, 0.003]),  # 关节间隙误差（rad）
    'friction_coeff': np.array([0.1, 0.08, 0.08, 0.06, 0.06]),  # 静摩擦力系数（仅用于误差补偿计算）
    'gravity_compensation': True,  # 是否启用重力误差补偿
    'comp_smoothing': 0.02,  # 误差补偿平滑系数，避免突变
}

# 轨迹规划配置（精度优化：梯形速度规划参数）
TRAJECTORY_PLANNING_PARAMS = {
    'traj_type': 'trapezoidal',  # 梯形速度规划，无超调
    'acceleration_time': 0.2,  # 加速时间
    'deceleration_time': 0.2,  # 减速时间
    'position_tol': 1e-5,  # 位置公差（rad），高精度定位判定
    'velocity_tol': 1e-4  # 速度公差（rad/s），平稳停止判定
}

# 精度监测配置
PRECISION_MONITOR_PARAMS = {
    'log_precision_data': True,
    'log_path': 'arm_joint_precision_log.txt',
    'max_allowed_position_error': np.deg2rad(0.1),  # 最大允许定位误差（0.1度）
    'max_allowed_trajectory_error': np.deg2rad(0.2)  # 最大允许轨迹跟踪误差（0.2度）
}

# 可靠性配置（兼容之前的优化）
RELIABILITY_PARAMS = {
    'stall_detection_threshold': 0.005,  # 更高灵敏度，提升异常检测精度
    'stall_duration_threshold': 1.0,
    'overload_duration_threshold': 2.0,
    'max_angle_error': np.deg2rad(10.0),
    'auto_reset_on_error': True,
    'log_reliability_data': True,
    'reliability_log_path': 'arm_reliability_log.txt'
}


# ====================== 信号处理（优雅退出，避免精度数据丢失） ======================
def signal_handler(sig, frame):
    global RUNNING
    if not RUNNING:
        sys.exit(0)
    print("\n⚠️ 收到退出信号，正在优雅退出（保存精度日志+清理资源）...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ====================== 工具函数（精度优化专用） ======================
def get_mujoco_id(model, obj_type, name):
    """兼容所有Mujoco版本的ID查询（容错增强，提升精度稳定性）"""
    if model is None:
        return -1
    type_map = {
        'joint': mujoco.mjtObj.mjOBJ_JOINT,
        'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
        'site': mujoco.mjtObj.mjOBJ_SITE,
        'body': mujoco.mjtObj.mjOBJ_BODY,
        'geom': mujoco.mjtObj.mjOBJ_GEOM
    }
    obj_type_int = type_map.get(obj_type, mujoco.mjtObj.mjOBJ_JOINT)
    try:
        obj_id = mujoco.mj_name2id(model, int(obj_type_int), str(name))
        return obj_id if obj_id >= 0 else -1
    except Exception as e:
        print(f"⚠️ 查询{obj_type} {name} ID失败: {e}")
        return -1


def deg2rad(degrees):
    """角度转弧度（高精度转换，容错增强）"""
    try:
        degrees = np.array(degrees, dtype=np.float64)
        return np.deg2rad(degrees)
    except Exception as e:
        print(f"⚠️ 角度转换失败: {e}")
        return 0.0 if np.isscalar(degrees) else np.zeros(JOINT_COUNT, dtype=np.float64)


def rad2deg(radians):
    """弧度转角度（高精度转换，容错增强）"""
    try:
        radians = np.array(radians, dtype=np.float64)
        return np.rad2deg(radians)
    except Exception as e:
        print(f"⚠️ 弧度转换失败: {e}")
        return 0.0 if np.isscalar(radians) else np.zeros(JOINT_COUNT, dtype=np.float64)


def write_precision_log(content, log_path=PRECISION_MONITOR_PARAMS['log_path']):
    """写入精度日志（记录误差数据，便于精度分析与优化）"""
    if not PRECISION_MONITOR_PARAMS['log_precision_data']:
        return
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 毫秒级时间戳，提升日志精度
            f.write(f"[{timestamp}] {content}\n")
    except Exception as e:
        print(f"⚠️ 写入精度日志失败: {e}")


def write_reliability_log(content, log_path=RELIABILITY_PARAMS['reliability_log_path']):
    """写入可靠性日志（兼容之前的优化）"""
    if not RELIABILITY_PARAMS['log_reliability_data']:
        return
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {content}\n")
    except Exception as e:
        print(f"⚠️ 写入可靠性日志失败: {e}")


def trapezoidal_velocity_planner(start_pos, target_pos, max_vel, max_accel, dt):
    """
    梯形速度规划（精度优化核心：无超调平滑轨迹生成）
    :param start_pos: 起始位置（rad）
    :param target_pos: 目标位置（rad）
    :param max_vel: 最大速度（rad/s）
    :param max_accel: 最大加速度（rad/s²）
    :param dt: 时间步长（s）
    :return: 规划的位置序列、速度序列
    """
    pos_error = target_pos - start_pos
    total_distance = abs(pos_error)
    if total_distance < TRAJECTORY_PLANNING_PARAMS['position_tol']:
        return np.array([target_pos]), np.array([0.0])

    # 计算梯形速度规划关键参数
    accel_phase_vel = max_vel
    accel_phase_dist = (accel_phase_vel ** 2) / (2 * max_accel)
    total_accel_decel_dist = 2 * accel_phase_dist

    # 判定运动阶段（是否存在匀速阶段）
    pos_list = []
    vel_list = []
    current_pos = start_pos
    current_vel = 0.0
    direction = np.sign(pos_error)

    if total_distance <= total_accel_decel_dist:
        # 无匀速阶段：加速到最大速度前即开始减速
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

    # 最后强制设置为目标位置，消除累积误差
    pos_list[-1] = target_pos
    vel_list[-1] = 0.0
    return np.array(pos_list), np.array(vel_list)


# ====================== 机械臂模型生成（修复geom标签viscous属性，高精度配置） ======================
def create_arm_model_with_precision():
    """
    生成高精度机械臂XML模型（彻底修复Schema违规错误，兼容所有Mujoco版本）
    核心修复：
    1.  移除所有geom标签的viscous属性（该属性不被geom支持，消除Schema违规）
    2.  保留geom标签的friction属性（3个值，合法支持静摩擦功能）
    3.  将粘性摩擦需求迁移至joint标签的damping属性（合法归属），通过粘性阻尼增益补充效果
    4.  优化joint标签的damping参数，确保与原有粘性摩擦需求一致
    """
    end_effector_mass = LOAD_PARAMS['end_effector_mass']
    link1_geom_mass = 0.8
    link2_geom_mass = 0.6
    link3_geom_mass = 0.6
    link4_geom_mass = 0.4
    link5_geom_mass = 0.2

    base_stiffness = STIFFNESS_PARAMS['base_stiffness']
    base_damping = DAMPING_INERTIA_PARAMS['base_damping']
    viscous_damping_gain = DAMPING_INERTIA_PARAMS['viscous_damping_gain']
    armature_inertia = DAMPING_INERTIA_PARAMS['armature_inertia']
    friction_coeffs = ERROR_COMPENSATION_PARAMS['friction_coeff']

    # 计算最终关节阻尼（基础阻尼 + 粘性阻尼增益，等效原有viscous效果）
    joint_damping = base_damping * viscous_damping_gain

    xml = f"""
<mujoco model="arm_with_precision_optimization">
    <!-- 修复1：compiler标签仅保留合法属性 -->
    <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
    <!-- tolerance属性合法存放于option标签 -->
    <option timestep="{SIMULATION_TIMESTEP}" gravity="0 0 -9.81" iterations="100" tolerance="1e-9"/>

    <!-- 高精度默认配置：修复2：移除geom的viscous，保留friction；优化joint的damping -->
    <default>
        <!-- joint标签：配置合法属性，damping整合基础阻尼+粘性阻尼效果 -->
        <joint type="hinge" armature="{armature_inertia[0]}" damping="{joint_damping[0]}" 
               limited="true" margin="0.001"/> <!-- 更小间隙，提升精度 -->
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100"/>
        <!-- geom标签：仅保留合法属性，移除viscous，保留friction（3个值） -->
        <geom contype="1" conaffinity="1" rgba="0.2 0.8 0.2 1" solref="0.01 1" solimp="0.9 0.95 0.001"
              friction="{friction_coeffs[0]} {friction_coeffs[0]} {friction_coeffs[0]}"/> <!-- 高精度接触与静摩擦参数 -->
    </default>

    <!-- 材质配置 -->
    <asset>
        <material name="load_material" rgba="1.0 0.0 0.0 0.8"/>
        <material name="high_precision_material" rgba="0.0 0.8 0.0 0.8"/>
        <material name="end_effector_material" rgba="0.8 0.2 0.2 1"/>
    </asset>

    <worldbody>
        <!-- 地面（高精度几何，降低接触误差） -->
        <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1" solref="0.01 1"/>

        <!-- 机械臂基座（joint1） -->
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>

            <!-- 修复3：joint标签配置优化后的damping（整合粘性阻尼效果），无违规属性 -->
            <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0.1" 
                   range="{JOINT_LIMITS_RAD[0, 0]} {JOINT_LIMITS_RAD[0, 1]}" 
                   armature="{armature_inertia[0]}" damping="{joint_damping[0]}"/>
            <body name="link1" pos="0 0 0.1">
                <!-- 修复4：geom标签移除viscous，仅保留friction（3个值），合法合规 -->
                <geom name="link1_geom" type="cylinder" size="0.04 0.18" mass="{link1_geom_mass}"
                      material="high_precision_material"
                      friction="{friction_coeffs[1]} {friction_coeffs[1]} {friction_coeffs[1]}"/>

                <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                       range="{JOINT_LIMITS_RAD[1, 0]} {JOINT_LIMITS_RAD[1, 1]}" 
                       armature="{armature_inertia[1]}" damping="{joint_damping[1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2_geom" type="cylinder" size="0.04 0.18" mass="{link2_geom_mass}"
                          material="high_precision_material"
                          friction="{friction_coeffs[2]} {friction_coeffs[2]} {friction_coeffs[2]}"/>

                    <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                           range="{JOINT_LIMITS_RAD[2, 0]} {JOINT_LIMITS_RAD[2, 1]}" 
                           armature="{armature_inertia[2]}" damping="{joint_damping[2]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3_geom" type="cylinder" size="0.04 0.18" mass="{link3_geom_mass}"
                              friction="{friction_coeffs[3]} {friction_coeffs[3]} {friction_coeffs[3]}"/>

                        <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                               range="{JOINT_LIMITS_RAD[3, 0]} {JOINT_LIMITS_RAD[3, 1]}" 
                               armature="{armature_inertia[3]}" damping="{joint_damping[3]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4_geom" type="cylinder" size="0.04 0.18" mass="{link4_geom_mass}"
                                  friction="{friction_coeffs[3]} {friction_coeffs[3]} {friction_coeffs[3]}"/>

                            <joint name="joint5" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                                   range="{JOINT_LIMITS_RAD[4, 0]} {JOINT_LIMITS_RAD[4, 1]}" 
                                   armature="{armature_inertia[4]}" damping="{joint_damping[4]}"/>
                            <body name="link5" pos="0 0 0.18">
                                <geom name="link5_geom" type="cylinder" size="0.03 0.09" mass="{link5_geom_mass}"
                                      material="end_effector_material"
                                      friction="{friction_coeffs[4]} {friction_coeffs[4]} {friction_coeffs[4]}"/>

                                <!-- 末端执行器（高精度负载配置） -->
                                <body name="end_effector" pos="0 0 0.09">
                                    <site name="ee_site" pos="0 0 0" size="0.005"/> <!-- 更小站点，提升定位精度 -->
                                    <geom name="load_geom" type="sphere" size="0.04" mass="{end_effector_mass}" 
                                          rgba="1.0 0.0 0.0 0.8" material="load_material"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- 关节电机（无违规属性，兼容所有Mujoco版本） -->
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


# ====================== 核心控制器类（关节精度性能优化） ======================
class ArmJointPrecisionOptimizationController:
    def __init__(self):
        # 模型与数据初始化（高精度配置）
        self.model = None
        self.data = None
        try:
            self.model = mujoco.MjModel.from_xml_string(create_arm_model_with_precision())
            self.data = mujoco.MjData(self.model)
            write_precision_log("高精度模型初始化成功，geom viscous属性修复完成，精度优化配置加载完毕")
            write_reliability_log("高精度模型初始化成功，geom viscous属性修复完成，精度优化配置加载完毕")
        except Exception as e:
            error_msg = f"高精度模型初始化失败: {e}"
            print(f"❌ {error_msg}")
            write_precision_log(error_msg)
            write_reliability_log(error_msg)
            global RUNNING
            RUNNING = False
            return

        # 获取各类ID（容错增强）
        self.joint_ids = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_NAMES]
        self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
        self.ee_site_id = get_mujoco_id(self.model, 'site', "ee_site")
        self.load_geom_id = get_mujoco_id(self.model, 'geom', "load_geom")

        # 状态变量初始化
        self.viewer_inst = None
        self.viewer_ready = False
        self.last_control_time = time.time()
        self.last_print_time = time.time()
        self.fps_counter = 0
        self.step_count = 0
        self.total_simulation_time = 0.0

        # 精度相关核心状态
        self.current_stiffness = STIFFNESS_PARAMS['base_stiffness'].copy()
        self.current_damping = DAMPING_INERTIA_PARAMS['base_damping'].copy() * DAMPING_INERTIA_PARAMS[
            'viscous_damping_gain']
        self.target_angles_rad = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.planned_positions = np.zeros((1, JOINT_COUNT), dtype=np.float64)  # 规划位置序列
        self.planned_velocities = np.zeros((1, JOINT_COUNT), dtype=np.float64)  # 规划速度序列
        self.traj_step_idx = 0  # 轨迹步骤索引
        self.position_error = np.zeros(JOINT_COUNT, dtype=np.float64)  # 当前定位误差
        self.trajectory_error = np.zeros(JOINT_COUNT, dtype=np.float64)  # 当前轨迹跟踪误差
        self.max_position_error = np.zeros(JOINT_COUNT, dtype=np.float64)  # 最大定位误差
        self.max_trajectory_error = np.zeros(JOINT_COUNT, dtype=np.float64)  # 最大轨迹跟踪误差

        # 负载与受力状态
        self.current_end_load = LOAD_PARAMS['end_effector_mass']
        self.smoothed_joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.angle_error_history = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 可靠性状态（兼容之前的优化）
        self.overload_warning_flag = False
        self.stall_detection_flag = np.zeros(JOINT_COUNT, dtype=bool)
        self.stall_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.overload_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.error_reset_count = 0

        # 误差补偿状态
        self.compensated_error = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.gravity_compensation_torque = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 初始化关节角度与轨迹
        try:
            self.set_joint_angles(np.zeros(JOINT_COUNT), smooth=False, use_deg=False)
            self.plan_trajectory(np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT))
            write_precision_log("关节零位校准完成，初始轨迹规划成功")
            write_reliability_log("关节零位校准完成，初始轨迹规划成功")
        except Exception as e:
            error_msg = f"初始化关节角度或轨迹失败: {e}"
            print(f"⚠️ {error_msg}")
            write_precision_log(error_msg)
            write_reliability_log(error_msg)

        # 全局仿真开始时间
        global SIMULATION_START_TIME
        SIMULATION_START_TIME = time.time()
        write_precision_log(f"高精度仿真启动，控制频率：{CONTROL_FREQUENCY}Hz，步长：{SIMULATION_TIMESTEP}s")
        write_reliability_log(f"高精度仿真启动，控制频率：{CONTROL_FREQUENCY}Hz，步长：{SIMULATION_TIMESTEP}s")

    def get_current_joint_angles(self, use_deg=True):
        """获取当前关节角度（高精度采集，容错增强）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        current_rad = np.array([self.data.qpos[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        if use_deg:
            return rad2deg(current_rad)
        return current_rad

    def get_current_joint_velocities(self, use_deg=True):
        """获取当前关节速度（高精度采集，用于速度闭环控制）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        current_vel_rad = np.array([self.data.qvel[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        if use_deg:
            return rad2deg(current_vel_rad)
        return current_vel_rad

    def get_joint_forces(self):
        """获取关节实时受力（高精度平滑，避免抖动影响精度）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                raw_force = abs(self.data.qfrc_actuator[jid])
                self.smoothed_joint_forces[i] = (1 - LOAD_PARAMS['load_smoothing_factor']) * self.smoothed_joint_forces[
                    i] + \
                                                LOAD_PARAMS['load_smoothing_factor'] * raw_force
                joint_forces[i] = self.smoothed_joint_forces[i]
        return joint_forces

    def calculate_error_compensation(self):
        """
        核心精度优化：多维度误差补偿计算
        1.  关节间隙误差补偿
        2.  摩擦力误差补偿（静摩擦，基于friction_coeff）
        3.  重力误差补偿
        """
        current_angles = self.get_current_joint_angles(use_deg=False)
        current_vels = self.get_current_joint_velocities(use_deg=False)
        current_forces = self.get_joint_forces()

        # 1. 关节间隙误差补偿（根据运动方向补偿间隙）
        backlash_comp = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i in range(JOINT_COUNT):
            if abs(current_vels[i]) > TRAJECTORY_PLANNING_PARAMS['velocity_tol']:
                # 运动时，根据速度方向补偿间隙
                backlash_comp[i] = ERROR_COMPENSATION_PARAMS['backlash_error'][i] * np.sign(current_vels[i])
            else:
                # 静止时，补偿当前误差方向的间隙
                backlash_comp[i] = ERROR_COMPENSATION_PARAMS['backlash_error'][i] * np.sign(self.position_error[i])

        # 2. 摩擦力误差补偿（仅静摩擦，基于合法的friction_coeff）
        friction_comp = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i in range(JOINT_COUNT):
            # 静摩擦力补偿（速度为零时）
            if abs(current_vels[i]) < TRAJECTORY_PLANNING_PARAMS['velocity_tol']:
                friction_comp[i] = ERROR_COMPENSATION_PARAMS['friction_coeff'][i] * np.sign(self.position_error[i])

        # 3. 重力误差补偿（简化版，根据关节角度补偿重力扭矩）
        gravity_comp = np.zeros(JOINT_COUNT, dtype=np.float64)
        if ERROR_COMPENSATION_PARAMS['gravity_compensation']:
            for i in range(JOINT_COUNT):
                gravity_comp[i] = 0.5 * np.sin(current_angles[i]) * self.current_end_load  # 简化重力补偿模型

        # 总误差补偿（平滑处理，避免突变）
        total_comp = backlash_comp + friction_comp + gravity_comp
        self.compensated_error = (1 - ERROR_COMPENSATION_PARAMS['comp_smoothing']) * self.compensated_error + \
                                 ERROR_COMPENSATION_PARAMS['comp_smoothing'] * total_comp

        # 重力补偿扭矩（直接用于控制信号补偿）
        self.gravity_compensation_torque = gravity_comp * 0.8  # 重力扭矩补偿系数

        return self.compensated_error, self.gravity_compensation_torque

    def plan_trajectory(self, start_angles, target_angles, use_deg=True):
        """
        精度优化：规划高精度平滑轨迹（梯形速度规划）
        :param start_angles: 起始角度
        :param target_angles: 目标角度
        :param use_deg: 是否为角度单位
        """
        start_angles_rad = self.clamp_joint_angles(start_angles, use_deg=use_deg)
        target_angles_rad = self.clamp_joint_angles(target_angles, use_deg=use_deg)

        # 为每个关节规划梯形速度轨迹
        joint_planned_pos = []
        joint_planned_vel = []
        max_traj_length = 0
        for i in range(JOINT_COUNT):
            pos_traj, vel_traj = trapezoidal_velocity_planner(
                start_angles_rad[i],
                target_angles_rad[i],
                PRECISION_PD_PARAMS['max_vel'][i],
                PRECISION_PD_PARAMS['max_accel'][i],
                CONTROL_TIMESTEP
            )
            joint_planned_pos.append(pos_traj)
            joint_planned_vel.append(vel_traj)
            if len(pos_traj) > max_traj_length:
                max_traj_length = len(pos_traj)

        # 统一轨迹长度（补零）
        for i in range(JOINT_COUNT):
            if len(joint_planned_pos[i]) < max_traj_length:
                pad_length = max_traj_length - len(joint_planned_pos[i])
                joint_planned_pos[i] = np.pad(joint_planned_pos[i], (0, pad_length), 'constant',
                                              constant_values=target_angles_rad[i])
                joint_planned_vel[i] = np.pad(joint_planned_vel[i], (0, pad_length), 'constant', constant_values=0.0)

        # 转换为二维数组
        self.planned_positions = np.array(joint_planned_pos).T
        self.planned_velocities = np.array(joint_planned_vel).T
        self.traj_step_idx = 0
        self.target_angles_rad = target_angles_rad.copy()

        info_msg = f"轨迹规划完成：从{np.round(rad2deg(start_angles_rad), 2)}度到{np.round(rad2deg(target_angles_rad), 2)}度，轨迹长度：{max_traj_length}步"
        print(f"✅ {info_msg}")
        write_precision_log(info_msg)

    def precision_adaptive_pd_control(self):
        """
        核心精度优化：高精度PD+前馈控制（位置-速度双闭环）
        1.  自适应PD参数，根据负载与误差调整
        2.  误差前馈补偿，提升动态响应精度
        3.  重力扭矩补偿，抵消静态误差
        4.  输出限幅，防止超调与过载
        """
        if self.data is None or self.planned_positions.shape[0] == 0:
            return

        # 1. 获取当前状态与误差补偿
        current_angles = self.get_current_joint_angles(use_deg=False)
        current_vels = self.get_current_joint_velocities(use_deg=False)
        compensated_error, gravity_comp_torque = self.calculate_error_compensation()

        # 2. 获取规划轨迹点（防止索引越界）
        if self.traj_step_idx < self.planned_positions.shape[0]:
            target_pos = self.planned_positions[self.traj_step_idx]
            target_vel = self.planned_velocities[self.traj_step_idx]
            self.traj_step_idx += 1
        else:
            target_pos = self.target_angles_rad
            target_vel = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 3. 计算定位误差与轨迹跟踪误差
        self.position_error = target_pos - current_angles
        self.trajectory_error = target_pos - current_angles + (target_vel - current_vels) * CONTROL_TIMESTEP

        # 更新最大误差
        self.max_position_error = np.maximum(self.max_position_error, np.abs(self.position_error))
        self.max_trajectory_error = np.maximum(self.max_trajectory_error, np.abs(self.trajectory_error))

        # 4. 自适应PD参数计算（根据负载调整）
        normalized_load = min(self.current_end_load / LOAD_PARAMS['max_allowed_load'], 1.0)
        kp = PRECISION_PD_PARAMS['kp_base'] * (1 + normalized_load * (PRECISION_PD_PARAMS['kp_load_gain'] - 1))
        kd = PRECISION_PD_PARAMS['kd_base'] * (1 + normalized_load * (PRECISION_PD_PARAMS['kd_load_gain'] - 1))

        # 5. PD控制信号计算（位置-速度双闭环）
        pd_control = kp * self.position_error + kd * (target_vel - current_vels)

        # 6. 前馈补偿与重力补偿
        ff_control = PRECISION_PD_PARAMS['ff_gain'] * target_vel  # 速度前馈
        total_control = pd_control + ff_control + gravity_comp_torque + compensated_error

        # 7. 输出限幅（防止超调与过载）
        for i in range(JOINT_COUNT):
            total_control[i] = np.clip(total_control[i], -JOINT_MAX_TORQUE[i], JOINT_MAX_TORQUE[i])

        # 8. 更新关节阻尼（与刚度匹配，提升控制精度）
        self.calculate_adaptive_stiffness()
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0 and self.model is not None:
                self.model.jnt_damping[jid] = self.current_damping[i]

        # 9. 设置控制信号
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = total_control[i]

    def calculate_adaptive_stiffness(self):
        """自适应刚度计算（兼容之前的优化，辅助提升精度）"""
        normalized_load = min(self.current_end_load / LOAD_PARAMS['max_allowed_load'], 1.0)
        current_angles = self.get_current_joint_angles(use_deg=False)
        angle_error_rad = np.abs(self.target_angles_rad - current_angles)
        normalized_error = np.clip(angle_error_rad / RELIABILITY_PARAMS['max_angle_error'], 0.0, 1.0)

        # 目标刚度计算
        target_stiffness = STIFFNESS_PARAMS['base_stiffness'] * \
                           (1 + normalized_load * (STIFFNESS_PARAMS['load_stiffness_gain'] - 1)) * \
                           (1 + normalized_error * (STIFFNESS_PARAMS['error_stiffness_gain'] - 1))
        target_stiffness = np.clip(target_stiffness, STIFFNESS_PARAMS['min_stiffness'],
                                   STIFFNESS_PARAMS['max_stiffness'])

        # 刚度平滑更新
        self.current_stiffness = (1 - STIFFNESS_PARAMS['stiffness_smoothing']) * self.current_stiffness + \
                                 STIFFNESS_PARAMS['stiffness_smoothing'] * target_stiffness

        # 阻尼与刚度匹配（整合粘性阻尼增益）
        target_damping = self.current_stiffness * DAMPING_INERTIA_PARAMS['damping_stiffness_ratio']
        target_damping = target_damping * DAMPING_INERTIA_PARAMS['viscous_damping_gain']
        self.current_damping = np.clip(target_damping,
                                       DAMPING_INERTIA_PARAMS['base_damping'] * 0.5,
                                       DAMPING_INERTIA_PARAMS['base_damping'] * 2.0)

        return self.current_stiffness, self.current_damping

    def monitor_precision(self):
        """精度实时监测与评估，量化精度性能"""
        # 判定是否超出允许误差
        position_error_over_limit = \
        np.where(np.abs(self.position_error) > PRECISION_MONITOR_PARAMS['max_allowed_position_error'])[0]
        trajectory_error_over_limit = \
        np.where(np.abs(self.trajectory_error) > PRECISION_MONITOR_PARAMS['max_allowed_trajectory_error'])[0]

        # 记录超限信息
        if len(position_error_over_limit) > 0:
            joint_names = [JOINT_NAMES[i] for i in position_error_over_limit]
            error_values = np.round(rad2deg(self.position_error[position_error_over_limit]), 4)
            warning_msg = f"定位误差超限：关节{joint_names}，误差：{error_values}度（最大允许：{rad2deg(PRECISION_MONITOR_PARAMS['max_allowed_position_error']):.2f}度）"
            print(f"⚠️ {warning_msg}")
            write_precision_log(warning_msg)

        if len(trajectory_error_over_limit) > 0:
            joint_names = [JOINT_NAMES[i] for i in trajectory_error_over_limit]
            error_values = np.round(rad2deg(self.trajectory_error[trajectory_error_over_limit]), 4)
            warning_msg = f"轨迹跟踪误差超限：关节{joint_names}，误差：{error_values}度（最大允许：{rad2deg(PRECISION_MONITOR_PARAMS['max_allowed_trajectory_error']):.2f}度）"
            print(f"⚠️ {warning_msg}")
            write_precision_log(warning_msg)

        # 记录精度统计信息
        precision_stats = f"精度统计：当前定位误差（度）：{np.round(rad2deg(np.abs(self.position_error)), 4)}，最大定位误差（度）：{np.round(rad2deg(self.max_position_error), 4)}；当前轨迹误差（度）：{np.round(rad2deg(np.abs(self.trajectory_error)), 4)}，最大轨迹误差（度）：{np.round(rad2deg(self.max_trajectory_error), 4)}"
        write_precision_log(precision_stats)

    def reliability_detection(self):
        """可靠性检测（兼容之前的优化，为精度提供保障）"""
        if self.data is None:
            return

        current_forces = self.get_joint_forces()
        current_vels = self.get_current_joint_velocities(use_deg=False)
        current_angles = self.get_current_joint_angles(use_deg=False)
        angle_error = np.abs(self.target_angles_rad - current_angles)
        current_time = time.time()

        # 卡死检测
        for i in range(JOINT_COUNT):
            vel_abs = abs(current_vels[i])
            force_ratio = current_forces[i] / JOINT_MAX_TORQUE[i] if JOINT_MAX_TORQUE[i] > 0 else 0

            if vel_abs < RELIABILITY_PARAMS['stall_detection_threshold'] and force_ratio > 0.9:
                self.stall_duration[i] += current_time - self.last_control_time
                if self.stall_duration[i] >= RELIABILITY_PARAMS['stall_duration_threshold']:
                    self.stall_detection_flag[i] = True
                    error_msg = f"关节{JOINT_NAMES[i]}卡死检测触发，速度：{vel_abs:.4f}，受力：{current_forces[i]:.2f}N·m"
                    print(f"⚠️ {error_msg}")
                    write_reliability_log(error_msg)
                    write_precision_log(f"卡死异常影响精度：{error_msg}")
            else:
                self.stall_duration[i] = 0.0
                self.stall_detection_flag[i] = False

            # 过载检测
            if force_ratio > 0.9:
                self.overload_duration[i] += current_time - self.last_control_time
                if self.overload_duration[i] >= RELIABILITY_PARAMS['overload_duration_threshold']:
                    self.overload_warning_flag = True
                    error_msg = f"关节{JOINT_NAMES[i]}过载持续触发，受力：{current_forces[i]:.2f}N·m，持续时间：{self.overload_duration[i]:.2f}s"
                    print(f"⚠️ {error_msg}")
                    write_reliability_log(error_msg)
                    write_precision_log(f"过载异常影响精度：{error_msg}")
            else:
                self.overload_duration[i] = 0.0

        # 大误差检测
        large_error_joints = np.where(angle_error > RELIABILITY_PARAMS['max_angle_error'])[0]
        if len(large_error_joints) > 0:
            joint_names = [JOINT_NAMES[i] for i in large_error_joints]
            error_msg = f"大角度误差触发，关节：{joint_names}，最大误差：{np.max(angle_error):.2f}rad"
            print(f"⚠️ {error_msg}")
            write_reliability_log(error_msg)
            write_precision_log(f"大误差异常：{error_msg}")

        # 自动复位
        if RELIABILITY_PARAMS['auto_reset_on_error'] and (
                np.any(self.stall_detection_flag) or self.overload_warning_flag or len(large_error_joints) > 0):
            self.auto_reset_joints()
            self.error_reset_count += 1
            write_reliability_log(f"异常自动复位触发，复位次数：{self.error_reset_count}")
            write_precision_log(f"异常复位恢复精度：复位次数{self.error_reset_count}")

    def auto_reset_joints(self):
        """自动复位异常关节（恢复安全状态，保障后续精度）"""
        print("\n🔧 执行关节自动复位，恢复零位并降低负载，保障精度...")
        self.set_end_effector_load(0.1)
        self.set_joint_angles(np.zeros(JOINT_COUNT), smooth=False, use_deg=False)
        self.plan_trajectory(np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT))
        self.overload_warning_flag = False
        self.stall_detection_flag = np.zeros(JOINT_COUNT, dtype=bool)
        self.stall_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.overload_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.current_stiffness = STIFFNESS_PARAMS['base_stiffness'].copy()
        self.current_damping = DAMPING_INERTIA_PARAMS['base_damping'].copy() * DAMPING_INERTIA_PARAMS[
            'viscous_damping_gain']
        # 重置精度相关状态
        self.position_error = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.trajectory_error = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.traj_step_idx = 0
        time.sleep(0.5)
        print("✅ 关节自动复位完成，恢复高精度安全状态")

    def set_end_effector_load(self, mass):
        """动态设置末端负载（高精度更新，兼容刚度优化）"""
        if mass < 0 or mass > LOAD_PARAMS['max_allowed_load']:
            self.overload_warning_flag = True
            warning_msg = f"末端负载超出限制（0 ~ {LOAD_PARAMS['max_allowed_load']}kg），当前设置：{mass}kg"
            print(f"⚠️ {warning_msg}")
            write_precision_log(warning_msg)
            write_reliability_log(warning_msg)
            return
        self.overload_warning_flag = False

        # 优先直接更新
        if self.model is not None and self.load_geom_id >= 0:
            try:
                self.model.geom_mass[self.load_geom_id] = mass
                self.current_end_load = mass
                LOAD_PARAMS['end_effector_mass'] = mass
                info_msg = f"末端负载更新为 {mass}kg（直接修改geom质量，不影响精度）"
                print(f"✅ {info_msg}")
                write_precision_log(info_msg)
                write_reliability_log(info_msg)
                return
            except Exception as e:
                error_msg = f"直接更新负载失败，将重新初始化模型: {e}"
                print(f"⚠️ {error_msg}")
                write_precision_log(error_msg)
                write_reliability_log(error_msg)

        # 降级方案
        try:
            LOAD_PARAMS['end_effector_mass'] = mass
            self.current_end_load = mass
            self.model = mujoco.MjModel.from_xml_string(create_arm_model_with_precision())
            self.data = mujoco.MjData(self.model)
            self.joint_ids = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_NAMES]
            self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
            self.ee_site_id = get_mujoco_id(self.model, 'site', "ee_site")
            self.load_geom_id = get_mujoco_id(self.model, 'geom', "load_geom")
            current_target = self.target_angles_rad.copy()
            self.target_angles_rad = current_target
            self.set_joint_angles(current_target, smooth=False, use_deg=False)
            self.plan_trajectory(current_target, current_target)
            info_msg = f"末端负载更新为 {mass}kg（重新初始化模型生效，精度恢复）"
            print(f"✅ {info_msg}")
            write_precision_log(info_msg)
            write_reliability_log(info_msg)
        except Exception as e:
            error_msg = f"更新末端负载失败: {e}"
            print(f"❌ {error_msg}")
            write_precision_log(error_msg)
            write_reliability_log(error_msg)

    def set_joint_angles(self, target_angles, smooth=True, use_deg=True):
        """设置关节目标角度（高精度限位，避免超程影响精度）"""
        if self.data is None:
            raise Exception("模型未初始化，无法设置关节角度")
        if len(target_angles) != JOINT_COUNT:
            raise ValueError(f"目标角度数量必须为{JOINT_COUNT}，当前为{len(target_angles)}")

        target_angles_rad = self.clamp_joint_angles(target_angles, use_deg=use_deg)

        if not smooth:
            for i, jid in enumerate(self.joint_ids):
                if jid >= 0:
                    self.data.qpos[jid] = target_angles_rad[i]
                    self.data.qvel[jid] = 0.0
            try:
                mujoco.mj_forward(self.model, self.data)
            except Exception as e:
                error_msg = f"更新模型状态失败: {e}"
                print(f"⚠️ {error_msg}")
                write_precision_log(error_msg)
                write_reliability_log(error_msg)

        # 若平滑模式，规划轨迹
        if smooth:
            start_angles = self.get_current_joint_angles(use_deg=use_deg)
            self.plan_trajectory(start_angles, target_angles, use_deg=use_deg)

        self.target_angles_rad = target_angles_rad.copy()

    def clamp_joint_angles(self, angles, use_deg=True):
        """关节高精度限位（更小余量，提升定位精度）"""
        angles = np.array(angles, dtype=np.float64)
        if use_deg:
            angles_rad = deg2rad(angles)
        else:
            angles_rad = angles.copy()
        # 极小安全余量：1%，防止关节撞击限位，保证定位精度
        limit_margin = 0.01
        limits_rad_margin = JOINT_LIMITS_RAD.copy()
        limits_rad_margin[:, 0] += limit_margin
        limits_rad_margin[:, 1] -= limit_margin
        clamped_rad = np.clip(angles_rad, limits_rad_margin[:, 0], limits_rad_margin[:, 1])
        if use_deg:
            return rad2deg(clamped_rad)
        return clamped_rad

    def print_precision_status(self):
        """打印精度与系统状态（实时监控）"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        fps = self.fps_counter / (current_time - self.last_print_time)
        joint_angles = self.get_current_joint_angles(use_deg=True)
        joint_vels = self.get_current_joint_velocities(use_deg=True)
        joint_forces = self.get_joint_forces()
        current_stiffness, current_damping = self.calculate_adaptive_stiffness()
        position_error_deg = rad2deg(self.position_error)
        trajectory_error_deg = rad2deg(self.trajectory_error)
        max_position_error_deg = rad2deg(self.max_position_error)
        max_trajectory_error_deg = rad2deg(self.max_trajectory_error)
        self.total_simulation_time = current_time - (SIMULATION_START_TIME or current_time)

        # 格式化打印
        print("-" * 150)
        print(
            f"📊 高精度仿真统计 | 耗时: {self.total_simulation_time:.2f}s | 步数: {self.step_count:,} | FPS: {fps:5.1f} | 复位次数: {self.error_reset_count}")
        print(
            f"🔧 关节状态 | 角度 (度): {np.round(joint_angles, 2)} | 速度 (度/s): {np.round(joint_vels, 3)} | 受力 (N·m): {np.round(joint_forces, 2)}")
        print(
            f"🎯 精度指标 | 当前定位误差 (度): {np.round(np.abs(position_error_deg), 4)} | 最大定位误差 (度): {np.round(max_position_error_deg, 4)}")
        print(
            f"🎯 精度指标 | 当前轨迹误差 (度): {np.round(np.abs(trajectory_error_deg), 4)} | 最大轨迹误差 (度): {np.round(max_trajectory_error_deg, 4)}")
        print(f"🔩 刚度阻尼 | 关节刚度: {np.round(current_stiffness, 1)} | 关节阻尼: {np.round(current_damping, 1)}")
        print(
            f"🏋️  负载状态 | 末端负载 (kg): {self.current_end_load:.2f} | 负载限制 (kg): {LOAD_PARAMS['max_allowed_load']}")
        if self.overload_warning_flag:
            print("⚠️  警告：关节过载，已启用输出限制，精度可能受影响！")
        if np.any(self.stall_detection_flag):
            stall_joints = [JOINT_NAMES[i] for i in range(JOINT_COUNT) if self.stall_detection_flag[i]]
            print(f"⚠️  警告：关节{stall_joints}卡死风险，即将触发自动复位，精度将临时下降！")
        print("-" * 150)

        self.last_print_time = current_time
        self.fps_counter = 0

    def preset_pose(self, pose_name):
        """预设高精度姿态（平滑切换，无超调）"""
        pose_map = {
            'zero': [0, 0, 0, 0, 0],  # 零位（高精度基准姿态）
            'up': [0, 30, 20, 10, 0],  # 抬起姿态
            'grasp': [0, 45, 30, 20, 10],  # 抓取姿态
            'precision_test': [10, 20, 15, 5, 8]  # 精度测试姿态
        }
        if pose_name not in pose_map:
            warning_msg = f"无效姿态名称，支持：{list(pose_map.keys())}"
            print(f"⚠️ {warning_msg}")
            write_precision_log(warning_msg)
            write_reliability_log(warning_msg)
            return
        self.set_joint_angles(pose_map[pose_name], smooth=True, use_deg=True)
        info_msg = f"切换到{pose_name}高精度姿态，轨迹规划与误差补偿已启用"
        print(f"✅ {info_msg}")
        write_precision_log(info_msg)
        write_reliability_log(info_msg)

    def run(self):
        """运行高精度仿真主循环"""
        global RUNNING

        if not self.init_viewer():
            RUNNING = False
            return

        # 启动信息
        print("=" * 150)
        print("🚀 机械臂关节精度性能优化控制器 - 启动成功（geom viscous属性修复完成）")
        print(f"✅ 模型信息 | 关节数量: {JOINT_COUNT} | 初始末端负载: {self.current_end_load:.2f}kg")
        print(
            f"✅ 精度配置 | 控制频率: {CONTROL_FREQUENCY}Hz | 仿真步长: {SIMULATION_TIMESTEP}s | 定位公差: {rad2deg(TRAJECTORY_PLANNING_PARAMS['position_tol']):.4f}度")
        print(
            f"✅ 刚度配置 | 基座最大刚度: {STIFFNESS_PARAMS['max_stiffness'][0]:.1f} | 末端最小刚度: {STIFFNESS_PARAMS['min_stiffness'][-1]:.1f}")
        print("📝 快捷指令:")
        print("   - 设置末端负载: controller.set_end_effector_load(1.0)")
        print("   - 单关节控制: controller.move_joint(0, 90)")
        print("   - 预设姿态: controller.preset_pose('precision_test')")
        print("   - 按 Ctrl+C 优雅退出")
        print("=" * 150)

        # 主循环
        while RUNNING:
            try:
                current_time = time.time()
                self.fps_counter += 1
                self.step_count += 1

                # 高频控制更新
                if current_time - self.last_control_time >= CONTROL_TIMESTEP:
                    self.precision_adaptive_pd_control()  # 高精度控制
                    self.monitor_precision()  # 精度监测
                    self.reliability_detection()  # 可靠性检测
                    self.last_control_time = current_time

                # 仿真步执行
                if self.model is not None and self.data is not None:
                    mujoco.mj_step(self.model, self.data)

                # 可视化同步
                if self.viewer_ready:
                    self.viewer_inst.sync()

                # 状态打印
                self.print_precision_status()

                # 动态睡眠
                time_diff = current_time - self.last_control_time
                if time_diff < SLEEP_TIME:
                    sleep_duration = max(0.00001, SLEEP_TIME - time_diff)
                    time.sleep(sleep_duration)

            except Exception as e:
                error_msg = f"仿真步异常（步数：{self.step_count}）: {e}"
                print(f"⚠️ {error_msg}")
                write_precision_log(error_msg)
                write_reliability_log(error_msg)
                continue

        # 资源清理
        self.cleanup()
        # 最终精度统计
        final_msg = f"高精度仿真结束 | 总耗时: {self.total_simulation_time:.2f}s | 总步数: {self.step_count:,} | 复位次数: {self.error_reset_count} | 最大定位误差: {np.round(rad2deg(np.max(self.max_position_error)), 4)}度 | 最大轨迹误差: {np.round(rad2deg(np.max(self.max_trajectory_error)), 4)}度"
        print("\n" + "=" * 150)
        print("✅ 控制器已优雅退出 - 关节精度性能仿真最终统计")
        print(f"📈 {final_msg}")
        print("=" * 150)
        write_precision_log(final_msg)
        write_reliability_log(final_msg)

    def init_viewer(self):
        """初始化Viewer（延迟加载，不影响精度）"""
        if self.model is None or self.data is None:
            return False
        if self.viewer_ready:
            return True
        try:
            if MUJOCO_NEW_VIEWER:
                self.viewer_inst = viewer.launch_passive(self.model, self.data)
            else:
                self.viewer_inst = viewer.Viewer(self.model, self.data)
            self.viewer_ready = True
            write_precision_log("Viewer初始化成功，可视化启用（不影响高精度控制）")
            write_reliability_log("Viewer初始化成功，可视化启用")
            print("✅ Viewer初始化成功")
            return True
        except Exception as e:
            error_msg = f"Viewer初始化失败: {e}"
            print(f"❌ {error_msg}")
            write_precision_log(error_msg)
            write_reliability_log(error_msg)
            return False

    def cleanup(self):
        """资源清理（完整释放，避免内存泄漏影响后续精度测试）"""
        if self.viewer_ready and self.viewer_inst:
            try:
                self.viewer_inst.close()
                write_precision_log("Viewer资源清理完成")
                write_reliability_log("Viewer资源清理完成")
            except Exception as e:
                error_msg = f"Viewer关闭失败: {e}"
                print(f"⚠️ {error_msg}")
                write_precision_log(error_msg)
                write_reliability_log(error_msg)
            self.viewer_inst = None
            self.viewer_ready = False
        self.model = None
        self.data = None
        global RUNNING, SIMULATION_START_TIME
        RUNNING = False
        SIMULATION_START_TIME = None
        write_precision_log("高精度控制器资源清理完成，仿真正常退出")
        write_reliability_log("高精度控制器资源清理完成，仿真正常退出")

    def move_joint(self, joint_idx, angle, smooth=True, use_deg=True):
        """单独控制单个关节（高精度平滑切换）"""
        if joint_idx < 0 or joint_idx >= JOINT_COUNT:
            raise ValueError(f"关节索引必须在0-{JOINT_COUNT - 1}之间，当前为{joint_idx}")

        current_angles = self.get_current_joint_angles(use_deg=use_deg)
        current_angles[joint_idx] = angle
        self.set_joint_angles(current_angles, smooth=smooth, use_deg=use_deg)


# ====================== 精度优化演示函数 ======================
def precision_optimization_demo(controller):
    """演示关节精度优化功能"""

    def demo():
        time.sleep(2)

        # 演示1：零位姿态（基准精度测试）
        print("\n🎬 演示1：切换到零位姿态，进行基准精度校准")
        controller.preset_pose('zero')
        time.sleep(3)

        # 演示2：精度测试姿态（多关节协同，验证轨迹精度）
        print("\n🎬 演示2：切换到精度测试姿态，验证多关节轨迹跟踪精度")
        controller.preset_pose('precision_test')
        time.sleep(4)

        # 演示3：增加负载（验证抗干扰精度维持）
        print("\n🎬 演示3：设置末端负载为1.5kg，验证负载下精度稳定性")
        controller.set_end_effector_load(1.5)
        time.sleep(4)

        # 演示4：单关节大角度运动（验证定位精度，无超调）
        print("\n🎬 演示4：关节1旋转45度，验证单关节高精度定位（无超调）")
        controller.move_joint(0, 45, smooth=True, use_deg=True)
        time.sleep(4)

        # 演示5：抓取姿态（验证全关节精度匹配）
        print("\n🎬 演示5：切换到抓取姿态，验证全关节协同精度")
        controller.preset_pose('grasp')
        time.sleep(3)

        # 演示6：降低负载（验证精度恢复能力）
        print("\n🎬 演示6：降低末端负载为0.2kg，验证精度恢复特性")
        controller.set_end_effector_load(0.2)
        time.sleep(3)

        # 演示7：复位零位（验证精度复位能力）
        print("\n🎬 演示7：切换回零位姿态，完成精度优化演示")
        controller.preset_pose('zero')
        time.sleep(2)

        # 结束演示
        global RUNNING
        RUNNING = False

    demo_thread = threading.Thread(target=demo)
    demo_thread.daemon = True
    demo_thread.start()


# ====================== 主入口 ======================
if __name__ == "__main__":
    # 补充完整：设置numpy输出格式，便于查看高精度关节数据
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    # 初始化控制器并运行
    controller = ArmJointPrecisionOptimizationController()
    precision_optimization_demo(controller)
    controller.run()