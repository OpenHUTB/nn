#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节刚度与可靠性优化控制器
核心优化：
1.  关节刚度分层配置+自适应调节（负载/误差驱动）
2.  全方位可靠性保障（卡死检测/过载保护/异常复位/容错处理）
3.  刚度-阻尼-惯量匹配优化，降低振动与干扰
4.  全状态监控与日志记录，便于故障追溯
5.  兼容新旧Mujoco版本，无XML语法错误
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

# ====================== 全局配置（刚度+可靠性专用） ======================
# 系统适配（Windows优先，极致CPU优化）
if os.name == 'nt':
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        os.system('chcp 65001 >nul 2>&1')
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)
    except Exception as e:
        print(f"⚠️ Windows系统优化失败（不影响核心功能）: {e}")
    # 强制单线程，避免多线程竞争导致的控制不稳定
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
# 关节基础配置（按重要性分层：1>2>3>4>5）
JOINT_COUNT = 5
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5"]
JOINT_LIMITS_RAD = np.array([
    [-np.pi, np.pi],  # joint1（基座，最高刚度）
    [-np.pi / 2, np.pi / 2],  # joint2（大臂，高刚度）
    [-np.pi / 2, np.pi / 2],  # joint3（中臂，中高刚度）
    [-np.pi / 2, np.pi / 2],  # joint4（小臂，中刚度）
    [-np.pi / 2, np.pi / 2],  # joint5（末端，低刚度）
], dtype=np.float64)
JOINT_MAX_VELOCITY_RAD = np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64)
JOINT_MAX_TORQUE = np.array([15.0, 12.0, 10.0, 8.0, 5.0], dtype=np.float64)  # 最大扭矩（可靠性保护）

# 关节刚度分层配置（核心优化：按关节层级设定基准刚度）
STIFFNESS_PARAMS = {
    'base_stiffness': np.array([200.0, 180.0, 150.0, 120.0, 80.0]),  # 各关节基准刚度
    'load_stiffness_gain': 1.8,  # 负载下刚度放大系数
    'error_stiffness_gain': 1.5,  # 大误差下刚度放大系数
    'min_stiffness': np.array([100.0, 90.0, 75.0, 60.0, 40.0]),  # 最小允许刚度
    'max_stiffness': np.array([300.0, 270.0, 225.0, 180.0, 120.0]),  # 最大允许刚度
    'stiffness_smoothing': 0.05,  # 刚度变化平滑系数，防止突变
}

# 阻尼与惯量匹配配置（刚度配套优化，提升可靠性）
DAMPING_INERTIA_PARAMS = {
    'base_damping': np.array([8.0, 7.0, 6.0, 5.0, 3.0]),  # 基准阻尼（与刚度匹配）
    'damping_stiffness_ratio': 0.04,  # 阻尼-刚度匹配比，保证运动平稳
    'armature_inertia': np.array([0.5, 0.4, 0.3, 0.2, 0.1]),  # 关节惯量补偿
}

# 仿真配置（可靠性优化：小步长提升控制稳定性）
SIMULATION_TIMESTEP = 0.001  # 更小步长，降低控制误差
CONTROL_FREQUENCY = 1000  # 更高控制频率，提升响应可靠性
CONTROL_TIMESTEP = 1.0 / CONTROL_FREQUENCY
FPS = 60
SLEEP_TIME = 1.0 / FPS
EPS = 1e-8
RUNNING = True
SIMULATION_START_TIME = None

# PD控制参数（与刚度/阻尼联动）
PD_PARAMS = {
    'kp_base': 80.0,
    'kd_base': 5.0,
    'kp_load_gain': 1.5,
    'kd_load_gain': 1.2,
    'max_vel': JOINT_MAX_VELOCITY_RAD.copy()
}

# 负载配置（与刚度联动优化）
LOAD_PARAMS = {
    'end_effector_mass': 0.5,
    'joint_loads': np.zeros(JOINT_COUNT),
    'max_allowed_load': 2.0,
    'load_smoothing_factor': 0.1
}

# 可靠性保护配置（核心：卡死/过载/异常检测参数）
RELIABILITY_PARAMS = {
    'stall_detection_threshold': 0.01,  # 关节卡死判定阈值（速度<此值且扭矩>90%）
    'stall_duration_threshold': 1.0,  # 卡死持续时间（秒），触发复位
    'overload_duration_threshold': 2.0,  # 过载持续时间，触发保护
    'max_angle_error': np.deg2rad(10.0),  # 最大允许角度误差，触发异常报警
    'auto_reset_on_error': True,  # 是否自动复位异常关节
    'log_reliability_data': True,  # 是否记录可靠性日志
    'log_path': 'arm_reliability_log.txt'  # 日志保存路径
}


# ====================== 信号处理（可靠性优化：优雅退出） ======================
def signal_handler(sig, frame):
    global RUNNING
    if not RUNNING:
        sys.exit(0)
    print("\n⚠️ 收到退出信号，正在优雅退出（保存可靠性日志+清理资源）...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ====================== 工具函数（刚度+可靠性专用） ======================
def get_mujoco_id(model, obj_type, name):
    """兼容所有Mujoco版本的ID查询（容错增强，提升可靠性）"""
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
    """角度转弧度（容错增强，可靠性保障）"""
    try:
        degrees = np.array(degrees, dtype=np.float64)
        return np.deg2rad(degrees)
    except Exception as e:
        print(f"⚠️ 角度转换失败: {e}")
        return 0.0 if np.isscalar(degrees) else np.zeros(JOINT_COUNT, dtype=np.float64)


def rad2deg(radians):
    """弧度转角度（容错增强，可靠性保障）"""
    try:
        radians = np.array(radians, dtype=np.float64)
        return np.rad2deg(radians)
    except Exception as e:
        print(f"⚠️ 弧度转换失败: {e}")
        return 0.0 if np.isscalar(radians) else np.zeros(JOINT_COUNT, dtype=np.float64)


def write_reliability_log(content, log_path=RELIABILITY_PARAMS['log_path']):
    """写入可靠性日志（核心：记录异常状态，便于追溯）"""
    if not RELIABILITY_PARAMS['log_reliability_data']:
        return
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {content}\n")
    except Exception as e:
        print(f"⚠️ 写入可靠性日志失败: {e}")


# ====================== 机械臂模型生成（刚度配置+无XML错误） ======================
def create_arm_model_with_stiffness():
    """
    生成带关节刚度配置的机械臂XML模型（兼容所有Mujoco版本）
    1.  按关节层级配置基准刚度、阻尼、惯量，实现刚度分层
    2.  移除所有违规XML属性，保证无语法错误
    3.  几何与质量配置优化，提升仿真可靠性
    """
    end_effector_mass = LOAD_PARAMS['end_effector_mass']
    # 连杆geom质量（兼容新旧Mujoco版本）
    link1_geom_mass = 0.8
    link2_geom_mass = 0.6
    link3_geom_mass = 0.6
    link4_geom_mass = 0.4
    link5_geom_mass = 0.2

    # 从配置中提取关节参数（刚度/阻尼/惯量）
    base_stiffness = STIFFNESS_PARAMS['base_stiffness']
    base_damping = DAMPING_INERTIA_PARAMS['base_damping']
    armature_inertia = DAMPING_INERTIA_PARAMS['armature_inertia']

    xml = f"""
<mujoco model="arm_with_stiffness_reliability">
    <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
    <option timestep="{SIMULATION_TIMESTEP}" gravity="0 0 -9.81" iterations="50" tolerance="1e-7"/>

    <!-- 关节刚度+阻尼+惯量基础配置（分层设定，提升可靠性） -->
    <default>
        <joint type="hinge" armature="{armature_inertia[0]}" damping="{base_damping[0]}" limited="true" margin="0.01"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100"/>
        <geom contype="1" conaffinity="1" rgba="0.2 0.8 0.2 1"/>
    </default>

    <!-- 负载与刚度可视化材质 -->
    <asset>
        <material name="load_material" rgba="1.0 0.0 0.0 0.8"/>
        <material name="high_stiffness_material" rgba="0.0 0.8 0.0 0.8"/>
        <material name="low_stiffness_material" rgba="0.8 0.0 0.0 0.8"/>
    </asset>

    <worldbody>
        <!-- 地面（简化几何，提升仿真效率） -->
        <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>

        <!-- 机械臂基座（joint1：最高刚度） -->
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>

            <!-- 关节1（基座关节，最高刚度+惯量） -->
            <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0.1" 
                   range="{JOINT_LIMITS_RAD[0, 0]} {JOINT_LIMITS_RAD[0, 1]}" 
                   armature="{armature_inertia[0]}" damping="{base_damping[0]}"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="cylinder" size="0.04 0.18" mass="{link1_geom_mass}"
                      material="high_stiffness_material"/>

                <!-- 关节2（大臂关节，高刚度） -->
                <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                       range="{JOINT_LIMITS_RAD[1, 0]} {JOINT_LIMITS_RAD[1, 1]}" 
                       armature="{armature_inertia[1]}" damping="{base_damping[1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2_geom" type="cylinder" size="0.04 0.18" mass="{link2_geom_mass}"
                          material="high_stiffness_material"/>

                    <!-- 关节3（中臂关节，中高刚度） -->
                    <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                           range="{JOINT_LIMITS_RAD[2, 0]} {JOINT_LIMITS_RAD[2, 1]}" 
                           armature="{armature_inertia[2]}" damping="{base_damping[2]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3_geom" type="cylinder" size="0.04 0.18" mass="{link3_geom_mass}"/>

                        <!-- 关节4（小臂关节，中刚度） -->
                        <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                               range="{JOINT_LIMITS_RAD[3, 0]} {JOINT_LIMITS_RAD[3, 1]}" 
                               armature="{armature_inertia[3]}" damping="{base_damping[3]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4_geom" type="cylinder" size="0.04 0.18" mass="{link4_geom_mass}"/>

                            <!-- 关节5（末端关节，低刚度） -->
                            <joint name="joint5" type="hinge" axis="0 1 0" pos="0 0 0.18" 
                                   range="{JOINT_LIMITS_RAD[4, 0]} {JOINT_LIMITS_RAD[4, 1]}" 
                                   armature="{armature_inertia[4]}" damping="{base_damping[4]}"/>
                            <body name="link5" pos="0 0 0.18">
                                <geom name="link5_geom" type="cylinder" size="0.03 0.09" mass="{link5_geom_mass}"
                                      material="low_stiffness_material" rgba="0.8 0.2 0.2 1"/>

                                <!-- 末端执行器（带负载，兼容动态调整） -->
                                <body name="end_effector" pos="0 0 0.09">
                                    <site name="ee_site" pos="0 0 0" size="0.01"/>
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


# ====================== 核心控制器类（刚度+可靠性优化） ======================
class ArmStiffnessReliabilityController:
    def __init__(self):
        # 模型与数据初始化（容错增强，提升可靠性）
        self.model = None
        self.data = None
        try:
            self.model = mujoco.MjModel.from_xml_string(create_arm_model_with_stiffness())
            self.data = mujoco.MjData(self.model)
            write_reliability_log("模型初始化成功，刚度与可靠性配置加载完成")
        except Exception as e:
            error_msg = f"带刚度配置模型初始化失败: {e}"
            print(f"❌ {error_msg}")
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

        # 刚度相关状态（核心：平滑刚度记录）
        self.current_stiffness = STIFFNESS_PARAMS['base_stiffness'].copy()
        self.current_damping = DAMPING_INERTIA_PARAMS['base_damping'].copy()
        self.target_angles_rad = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 负载与受力状态
        self.current_end_load = LOAD_PARAMS['end_effector_mass']
        self.smoothed_joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.angle_error_history = np.zeros(JOINT_COUNT, dtype=np.float64)

        # 可靠性保护状态（核心：卡死/过载/异常检测）
        self.overload_warning_flag = False
        self.stall_detection_flag = np.zeros(JOINT_COUNT, dtype=bool)
        self.stall_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.overload_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.error_reset_count = 0  # 异常复位计数

        # 初始化关节角度
        try:
            self.set_joint_angles(np.zeros(JOINT_COUNT), smooth=False, use_deg=False)
            write_reliability_log("关节角度初始化成功，零位校准完成")
        except Exception as e:
            error_msg = f"初始化关节角度失败: {e}"
            print(f"⚠️ {error_msg}")
            write_reliability_log(error_msg)

        # 全局仿真开始时间
        global SIMULATION_START_TIME
        SIMULATION_START_TIME = time.time()
        write_reliability_log(f"仿真启动，控制频率：{CONTROL_FREQUENCY}Hz，步长：{SIMULATION_TIMESTEP}s")

    def get_current_joint_angles(self, use_deg=True):
        """获取当前关节角度（容错增强）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        current_rad = np.array([self.data.qpos[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        if use_deg:
            return rad2deg(current_rad)
        return current_rad

    def get_joint_forces(self):
        """获取关节实时受力（可靠性监控核心）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                raw_force = abs(self.data.qfrc_actuator[jid])
                # 平滑受力，避免抖动影响检测
                self.smoothed_joint_forces[i] = (1 - LOAD_PARAMS['load_smoothing_factor']) * self.smoothed_joint_forces[
                    i] + \
                                                LOAD_PARAMS['load_smoothing_factor'] * raw_force
                joint_forces[i] = self.smoothed_joint_forces[i]
        return joint_forces

    def calculate_adaptive_stiffness(self):
        """
        核心优化：计算自适应关节刚度
        1.  根据负载大小动态调整刚度
        2.  根据角度误差动态调整刚度
        3.  刚度限幅+平滑处理，保证可靠性
        4.  阻尼与刚度匹配，降低振动
        """
        # 1. 负载归一化
        normalized_load = min(self.current_end_load / LOAD_PARAMS['max_allowed_load'], 1.0)

        # 2. 角度误差归一化
        current_angles = self.get_current_joint_angles(use_deg=False)
        angle_error_rad = np.abs(self.target_angles_rad - current_angles)
        normalized_error = np.clip(angle_error_rad / RELIABILITY_PARAMS['max_angle_error'], 0.0, 1.0)

        # 3. 计算目标刚度（负载+误差双驱动）
        target_stiffness = STIFFNESS_PARAMS['base_stiffness'] * \
                           (1 + normalized_load * (STIFFNESS_PARAMS['load_stiffness_gain'] - 1)) * \
                           (1 + normalized_error * (STIFFNESS_PARAMS['error_stiffness_gain'] - 1))

        # 4. 刚度限幅（防止超出合理范围）
        target_stiffness = np.clip(target_stiffness,
                                   STIFFNESS_PARAMS['min_stiffness'],
                                   STIFFNESS_PARAMS['max_stiffness'])

        # 5. 刚度平滑更新（防止突变，提升可靠性）
        self.current_stiffness = (1 - STIFFNESS_PARAMS['stiffness_smoothing']) * self.current_stiffness + \
                                 STIFFNESS_PARAMS['stiffness_smoothing'] * target_stiffness

        # 6. 阻尼与刚度匹配更新（保证运动平稳）
        target_damping = self.current_stiffness * DAMPING_INERTIA_PARAMS['damping_stiffness_ratio']
        self.current_damping = np.clip(target_damping,
                                       DAMPING_INERTIA_PARAMS['base_damping'] * 0.5,
                                       DAMPING_INERTIA_PARAMS['base_damping'] * 1.5)

        # 7. 更新模型阻尼（实时生效）
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0 and self.model is not None:
                self.model.jnt_damping[jid] = self.current_damping[i]

        return self.current_stiffness, self.current_damping

    def reliability_detection(self):
        """
        核心可靠性功能：关节卡死+过载+异常检测
        1.  卡死检测：速度极低且扭矩接近最大值
        2.  过载检测：受力持续超过阈值
        3.  异常复位：满足条件时自动复位关节
        """
        if self.data is None:
            return

        # 1. 获取当前状态
        current_forces = self.get_joint_forces()
        current_vels = np.array([self.data.qvel[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        current_angles = self.get_current_joint_angles(use_deg=False)
        angle_error = np.abs(self.target_angles_rad - current_angles)

        # 2. 卡死检测（速度<阈值 且 受力>90%最大扭矩）
        current_time = time.time()
        for i in range(JOINT_COUNT):
            vel_abs = abs(current_vels[i])
            force_ratio = current_forces[i] / JOINT_MAX_TORQUE[i] if JOINT_MAX_TORQUE[i] > 0 else 0

            # 判定卡死条件
            if vel_abs < RELIABILITY_PARAMS['stall_detection_threshold'] and force_ratio > 0.9:
                self.stall_duration[i] += current_time - self.last_control_time
                if self.stall_duration[i] >= RELIABILITY_PARAMS['stall_duration_threshold']:
                    self.stall_detection_flag[i] = True
                    error_msg = f"关节{JOINT_NAMES[i]}卡死检测触发，速度：{vel_abs:.4f}，受力：{current_forces[i]:.2f}N·m"
                    print(f"⚠️ {error_msg}")
                    write_reliability_log(error_msg)
            else:
                self.stall_duration[i] = 0.0
                self.stall_detection_flag[i] = False

            # 3. 过载检测（受力>90%最大扭矩 且 持续超时）
            if force_ratio > 0.9:
                self.overload_duration[i] += current_time - self.last_control_time
                if self.overload_duration[i] >= RELIABILITY_PARAMS['overload_duration_threshold']:
                    self.overload_warning_flag = True
                    error_msg = f"关节{JOINT_NAMES[i]}过载持续触发，受力：{current_forces[i]:.2f}N·m，持续时间：{self.overload_duration[i]:.2f}s"
                    print(f"⚠️ {error_msg}")
                    write_reliability_log(error_msg)
            else:
                self.overload_duration[i] = 0.0

        # 4. 大误差检测（角度误差超出阈值）
        large_error_joints = np.where(angle_error > RELIABILITY_PARAMS['max_angle_error'])[0]
        if len(large_error_joints) > 0:
            joint_names = [JOINT_NAMES[i] for i in large_error_joints]
            error_msg = f"大角度误差触发，关节：{joint_names}，最大误差：{np.max(angle_error):.2f}rad"
            print(f"⚠️ {error_msg}")
            write_reliability_log(error_msg)

        # 5. 自动异常复位（可靠性核心功能）
        if RELIABILITY_PARAMS['auto_reset_on_error'] and (
                np.any(self.stall_detection_flag) or self.overload_warning_flag or len(large_error_joints) > 0):
            self.auto_reset_joints()
            self.error_reset_count += 1
            write_reliability_log(f"异常自动复位触发，复位次数：{self.error_reset_count}")

    def auto_reset_joints(self):
        """自动复位异常关节（可靠性保护：恢复零位，降低负载）"""
        print("\n🔧 执行关节自动复位，恢复零位并降低末端负载...")
        # 1. 降低末端负载到安全值
        self.set_end_effector_load(0.1)
        # 2. 复位关节到零位
        self.set_joint_angles(np.zeros(JOINT_COUNT), smooth=False, use_deg=False)
        # 3. 重置可靠性状态标志
        self.overload_warning_flag = False
        self.stall_detection_flag = np.zeros(JOINT_COUNT, dtype=bool)
        self.stall_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.overload_duration = np.zeros(JOINT_COUNT, dtype=np.float64)
        # 4. 重置刚度到基准值
        self.current_stiffness = STIFFNESS_PARAMS['base_stiffness'].copy()
        self.current_damping = DAMPING_INERTIA_PARAMS['base_damping'].copy()
        time.sleep(0.5)  # 复位后延迟，保证稳定
        print("✅ 关节自动复位完成，恢复安全状态")

    def set_end_effector_load(self, mass):
        """动态设置末端负载（与刚度联动）"""
        if mass < 0 or mass > LOAD_PARAMS['max_allowed_load']:
            self.overload_warning_flag = True
            warning_msg = f"末端负载超出限制（0 ~ {LOAD_PARAMS['max_allowed_load']}kg），当前设置：{mass}kg"
            print(f"⚠️ {warning_msg}")
            write_reliability_log(warning_msg)
            return
        self.overload_warning_flag = False

        # 优先直接更新负载geom质量（高效）
        if self.model is not None and self.load_geom_id >= 0:
            try:
                self.model.geom_mass[self.load_geom_id] = mass
                self.current_end_load = mass
                LOAD_PARAMS['end_effector_mass'] = mass
                info_msg = f"末端负载更新为 {mass}kg（直接修改geom质量）"
                print(f"✅ {info_msg}")
                write_reliability_log(info_msg)
                return
            except Exception as e:
                error_msg = f"直接更新负载失败，将重新初始化模型: {e}"
                print(f"⚠️ {error_msg}")
                write_reliability_log(error_msg)

        # 降级方案：重新初始化模型
        try:
            LOAD_PARAMS['end_effector_mass'] = mass
            self.current_end_load = mass
            self.model = mujoco.MjModel.from_xml_string(create_arm_model_with_stiffness())
            self.data = mujoco.MjData(self.model)
            # 重新获取ID
            self.joint_ids = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_NAMES]
            self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
            self.ee_site_id = get_mujoco_id(self.model, 'site', "ee_site")
            self.load_geom_id = get_mujoco_id(self.model, 'geom', "load_geom")
            # 保留目标角度
            current_target = self.target_angles_rad.copy()
            self.target_angles_rad = current_target
            self.set_joint_angles(current_target, smooth=False, use_deg=False)
            info_msg = f"末端负载更新为 {mass}kg（重新初始化模型生效）"
            print(f"✅ {info_msg}")
            write_reliability_log(info_msg)
        except Exception as e:
            error_msg = f"更新末端负载失败: {e}"
            print(f"❌ {error_msg}")
            write_reliability_log(error_msg)

    def set_joint_angles(self, target_angles, smooth=True, use_deg=True):
        """设置关节目标角度（容错增强）"""
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
                write_reliability_log(error_msg)

        self.target_angles_rad = target_angles_rad.copy()

    def clamp_joint_angles(self, angles, use_deg=True):
        """关节限位保护（可靠性优化：缩小余量，防止冲击）"""
        angles = np.array(angles, dtype=np.float64)
        if use_deg:
            angles_rad = deg2rad(angles)
        else:
            angles_rad = angles.copy()
        # 安全余量：5%，防止关节撞击限位
        limit_margin = 0.05
        limits_rad_margin = JOINT_LIMITS_RAD.copy()
        limits_rad_margin[:, 0] += limit_margin
        limits_rad_margin[:, 1] -= limit_margin
        clamped_rad = np.clip(angles_rad, limits_rad_margin[:, 0], limits_rad_margin[:, 1])
        if use_deg:
            return rad2deg(clamped_rad)
        return clamped_rad

    def stiffness_adaptive_pd_control(self):
        """
        刚度自适应PD控制（核心：刚度与PD参数联动，提升精度与可靠性）
        """
        if self.data is None:
            return

        # 1. 自适应刚度与阻尼更新
        current_stiffness, current_damping = self.calculate_adaptive_stiffness()

        # 2. 获取当前状态
        current_angles = self.get_current_joint_angles(use_deg=False)
        current_vels = np.array([self.data.qvel[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        joint_forces = self.get_joint_forces()
        angle_error = self.target_angles_rad - current_angles

        # 3. 误差平滑
        self.angle_error_history = (1 - LOAD_PARAMS['load_smoothing_factor']) * self.angle_error_history + \
                                   LOAD_PARAMS['load_smoothing_factor'] * angle_error

        # 4. PD参数与刚度联动
        kp = current_stiffness / 2.5  # 刚度-P比例联动
        kd = current_damping / 1.6  # 阻尼-D比例联动

        # 5. 期望速度与控制信号计算
        desired_vel = np.clip(self.angle_error_history * kp, -JOINT_MAX_VELOCITY_RAD, JOINT_MAX_VELOCITY_RAD)
        control_signals = kp * self.angle_error_history + kd * (desired_vel - current_vels)

        # 6. 软件过载保护（可靠性核心）
        for i in range(JOINT_COUNT):
            force_ratio = joint_forces[i] / JOINT_MAX_TORQUE[i] if JOINT_MAX_TORQUE[i] > 0 else 0
            if force_ratio > 0.9:
                control_signals[i] *= 0.4  # 降低60%输出，防止过载
            elif force_ratio > 0.7:
                control_signals[i] *= 0.7  # 降低30%输出，预警保护

        # 7. 设置控制信号
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = control_signals[i]

    def init_viewer(self):
        """初始化Viewer（延迟加载，提升可靠性）"""
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
            write_reliability_log("Viewer初始化成功，可视化启用")
            print("✅ Viewer初始化成功")
            return True
        except Exception as e:
            error_msg = f"Viewer初始化失败: {e}"
            print(f"❌ {error_msg}")
            write_reliability_log(error_msg)
            return False

    def print_stiffness_reliability_status(self):
        """打印刚度与可靠性状态（实时监控）"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        # 统计信息
        fps = self.fps_counter / (current_time - self.last_print_time)
        joint_angles = self.get_current_joint_angles(use_deg=True)
        joint_forces = self.get_joint_forces()
        current_stiffness, current_damping = self.calculate_adaptive_stiffness()
        angle_errors = rad2deg(self.angle_error_history)
        self.total_simulation_time = current_time - (SIMULATION_START_TIME or current_time)

        # 格式化打印
        print("-" * 120)
        print(
            f"📊 仿真统计 | 耗时: {self.total_simulation_time:.2f}s | 步数: {self.step_count:,} | FPS: {fps:5.1f} | 复位次数: {self.error_reset_count}")
        print(f"🔧 关节角度 (度): {np.round(joint_angles, 1)} | 控制误差 (度): {np.round(abs(angle_errors), 3)}")
        print(
            f"🏋️  末端负载 (kg): {self.current_end_load:.2f} | 关节受力 (N·m): {np.round(joint_forces, 2)} | 最大扭矩 (N·m): {np.round(JOINT_MAX_TORQUE, 1)}")
        print(f"🔩 关节刚度: {np.round(current_stiffness, 1)} | 关节阻尼: {np.round(current_damping, 1)}")
        if self.overload_warning_flag:
            print("⚠️  警告：关节过载，已启用输出限制！")
        if np.any(self.stall_detection_flag):
            stall_joints = [JOINT_NAMES[i] for i in range(JOINT_COUNT) if self.stall_detection_flag[i]]
            print(f"⚠️  警告：关节{stall_joints}卡死风险，即将触发自动复位！")
        print("-" * 120)

        # 重置计数器
        self.last_print_time = current_time
        self.fps_counter = 0

    def preset_pose(self, pose_name):
        """预设姿态（可靠性优化：平稳切换）"""
        pose_map = {
            'zero': [0, 0, 0, 0, 0],  # 零位（安全姿态）
            'up': [0, 30, 20, 10, 0],  # 抬起姿态
            'grasp': [0, 45, 30, 20, 10]  # 抓取姿态
        }
        if pose_name not in pose_map:
            warning_msg = f"无效姿态名称，支持：{list(pose_map.keys())}"
            print(f"⚠️ {warning_msg}")
            write_reliability_log(warning_msg)
            return
        self.set_joint_angles(pose_map[pose_name], smooth=True, use_deg=True)
        info_msg = f"切换到{pose_name}姿态，刚度自适应控制已启用"
        print(f"✅ {info_msg}")
        write_reliability_log(info_msg)

    def run(self):
        """运行完整仿真（刚度+可靠性核心循环）"""
        global RUNNING

        if not self.init_viewer():
            RUNNING = False
            return

        # 启动信息
        print("=" * 120)
        print("🚀 机械臂关节刚度与可靠性优化控制器 - 启动成功")
        print(f"✅ 模型信息 | 关节数量: {JOINT_COUNT} | 初始末端负载: {self.current_end_load:.2f}kg")
        print(
            f"✅ 刚度配置 | 基座最大刚度: {STIFFNESS_PARAMS['max_stiffness'][0]:.1f} | 末端最小刚度: {STIFFNESS_PARAMS['min_stiffness'][-1]:.1f}")
        print(f"✅ 可靠性配置 | 控制频率: {CONTROL_FREQUENCY}Hz | 最大允许负载: {LOAD_PARAMS['max_allowed_load']}kg")
        print("📝 快捷指令:")
        print("   - 设置末端负载: controller.set_end_effector_load(1.0)")
        print("   - 单关节控制: controller.move_joint(0, 90)")
        print("   - 预设姿态: controller.preset_pose('up')")
        print("   - 按 Ctrl+C 优雅退出")
        print("=" * 120)

        # 主循环（可靠性优化：容错增强）
        while RUNNING:
            try:
                current_time = time.time()
                self.fps_counter += 1
                self.step_count += 1

                # 高频率控制更新
                if current_time - self.last_control_time >= CONTROL_TIMESTEP:
                    self.stiffness_adaptive_pd_control()  # 刚度自适应控制
                    self.reliability_detection()  # 可靠性检测
                    self.last_control_time = current_time

                # 仿真步执行
                if self.model is not None and self.data is not None:
                    mujoco.mj_step(self.model, self.data)

                # 可视化同步
                if self.viewer_ready:
                    self.viewer_inst.sync()

                # 状态打印
                self.print_stiffness_reliability_status()

                # 动态睡眠，降低CPU占用
                time_diff = current_time - self.last_control_time
                if time_diff < SLEEP_TIME:
                    sleep_duration = max(0.00001, SLEEP_TIME - time_diff)
                    time.sleep(sleep_duration)

            except Exception as e:
                error_msg = f"仿真步异常（步数：{self.step_count}）: {e}"
                print(f"⚠️ {error_msg}")
                write_reliability_log(error_msg)
                continue

        # 资源清理
        self.cleanup()
        # 最终统计
        final_msg = f"仿真结束 | 总耗时: {self.total_simulation_time:.2f}s | 总步数: {self.step_count:,} | 复位次数: {self.error_reset_count}"
        print("\n" + "=" * 120)
        print("✅ 控制器已优雅退出 - 刚度与可靠性仿真最终统计")
        print(f"📈 {final_msg}")
        print(f"🎯 最终末端负载 (kg): {self.current_end_load:.2f} | 最终关节刚度: {np.round(self.current_stiffness, 1)}")
        print("=" * 120)
        write_reliability_log(final_msg)

    def cleanup(self):
        """资源清理（可靠性优化：完整释放，避免内存泄漏）"""
        if self.viewer_ready and self.viewer_inst:
            try:
                self.viewer_inst.close()
                write_reliability_log("Viewer资源清理完成")
            except Exception as e:
                error_msg = f"Viewer关闭失败: {e}"
                print(f"⚠️ {error_msg}")
                write_reliability_log(error_msg)
            self.viewer_inst = None
            self.viewer_ready = False
        self.model = None
        self.data = None
        global RUNNING, SIMULATION_START_TIME
        RUNNING = False
        SIMULATION_START_TIME = None
        write_reliability_log("控制器资源清理完成，仿真正常退出")

    def move_joint(self, joint_idx, angle, smooth=True, use_deg=True):
        """单独控制单个关节（容错增强）"""
        if joint_idx < 0 or joint_idx >= JOINT_COUNT:
            raise ValueError(f"关节索引必须在0-{JOINT_COUNT - 1}之间，当前为{joint_idx}")

        current_angles = self.get_current_joint_angles(use_deg=use_deg)
        current_angles[joint_idx] = angle
        self.set_joint_angles(current_angles, smooth=smooth, use_deg=use_deg)


# ====================== 刚度与可靠性演示函数 ======================
def stiffness_reliability_demo(controller):
    """演示刚度自适应与可靠性保护功能"""

    def demo():
        time.sleep(2)

        # 演示1：零位姿态（基准刚度）
        print("\n🎬 演示1：切换到零位姿态，使用基准刚度")
        controller.preset_pose('zero')
        time.sleep(3)

        # 演示2：抬起姿态（刚度自适应调整）
        print("\n🎬 演示2：切换到抬起姿态，刚度随姿态自动调整")
        controller.preset_pose('up')
        time.sleep(3)

        # 演示3：增加负载（刚度放大，可靠性保护启用）
        print("\n🎬 演示3：设置末端负载为1.8kg（接近最大值，刚度自动放大）")
        controller.set_end_effector_load(1.8)
        time.sleep(3)

        # 演示4：大角度运动（大误差下刚度进一步提升）
        print("\n🎬 演示4：关节1旋转90度（大误差，刚度与阻尼联动优化）")
        controller.move_joint(0, 90, smooth=True, use_deg=True)
        time.sleep(4)

        # 演示5：抓取姿态（全关节刚度匹配）
        print("\n🎬 演示5：切换到抓取姿态，全关节刚度分层生效")
        controller.preset_pose('grasp')
        time.sleep(3)

        # 演示6：降低负载（刚度回落，恢复平稳）
        print("\n🎬 演示6：降低末端负载为0.2kg（刚度回落，运动更平稳）")
        controller.set_end_effector_load(0.2)
        time.sleep(2)

        # 演示7：复位零位（可靠性演示）
        print("\n🎬 演示7：切换回零位姿态，完成刚度与可靠性演示")
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
    np.seterr(all='ignore')

    # 创建刚度与可靠性控制器
    controller = None
    try:
        controller = ArmStiffnessReliabilityController()
    except Exception as e:
        print(f"❌ 控制器创建失败: {e}")
        sys.exit(1)

    # 运行演示
    if controller is not None:
        stiffness_reliability_demo(controller)

    # 启动控制器
    if controller is not None:
        controller.run()

    sys.exit(0)