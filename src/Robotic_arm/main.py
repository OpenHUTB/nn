#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节负载性能优化版（终极全XML错误修复）
核心优化：
1.  彻底修复所有XML Schema错误（body/mass/forcelimit 均移除违规配置）
2.  精准关节负载建模（末端负载/关节负载可配置）
3.  负载自适应PD控制（抗干扰、无超调、响应快）
4.  物理仿真优化（减少负载下的计算冗余，提升实时性）
5.  负载状态监控（实时显示负载大小、关节受力、控制误差）
6.  软件层面过载保护（替代forcelimit，兼容所有Mujoco版本）
7.  全Mujoco版本兼容（支持新旧版本，无任何语法隐患）
"""

import sys
import os
import time
import signal
import ctypes
import threading
import numpy as np
import mujoco

# ====================== 全局配置（负载优化专用） ======================
# 系统适配（Windows优先，极致CPU优化）
if os.name == 'nt':
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        os.system('chcp 65001 >nul 2>&1')
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 1)
    except Exception as e:
        print(f"⚠️ Windows系统优化失败（不影响核心功能）: {e}")
    # 强制单线程，避免负载下多线程竞争导致卡顿
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
# 关节基础配置
JOINT_COUNT = 5
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5"]
JOINT_LIMITS_RAD = np.array([
    [-np.pi, np.pi],  # joint1 (Z轴)
    [-np.pi / 2, np.pi / 2],  # joint2 (Y轴)
    [-np.pi / 2, np.pi / 2],  # joint3 (Y轴)
    [-np.pi / 2, np.pi / 2],  # joint4 (Y轴)
    [-np.pi / 2, np.pi / 2],  # joint5 (Y轴)
], dtype=np.float64)
JOINT_MAX_VELOCITY_RAD = np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64)
JOINT_MAX_TORQUE = np.array([15.0, 12.0, 10.0, 8.0, 5.0], dtype=np.float64)  # 关节最大扭矩（软件过载保护）

# 仿真配置（负载下实时性优化）
SIMULATION_TIMESTEP = 0.002  # 更小步长，提升负载下控制精度
CONTROL_FREQUENCY = 500  # 更高控制频率，应对负载下响应滞后
CONTROL_TIMESTEP = 1.0 / CONTROL_FREQUENCY
FPS = 60
SLEEP_TIME = 1.0 / FPS
EPS = 1e-8
RUNNING = True
SIMULATION_START_TIME = None

# PD控制参数（负载自适应配置）
PD_PARAMS = {
    'kp_base': 80.0,
    'kd_base': 5.0,
    'kp_load_gain': 1.5,  # 负载下比例增益放大系数
    'kd_load_gain': 1.2,  # 负载下微分增益放大系数
    'max_vel': JOINT_MAX_VELOCITY_RAD.copy()
}

# 负载配置（可动态调整）
LOAD_PARAMS = {
    'end_effector_mass': 0.5,  # 末端负载质量（kg），默认0.5kg
    'joint_loads': np.zeros(JOINT_COUNT),  # 各关节附加负载（N·m）
    'max_allowed_load': 2.0,  # 最大允许末端负载（过载保护）
    'load_smoothing_factor': 0.1  # 负载检测平滑系数，避免抖动
}


# ====================== 信号处理（负载下优雅退出） ======================
def signal_handler(sig, frame):
    global RUNNING
    if not RUNNING:
        sys.exit(0)
    print("\n⚠️ 收到退出信号，正在优雅退出（清理负载相关资源）...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ====================== 工具函数（负载优化专用） ======================
def get_mujoco_id(model, obj_type, name):
    """兼容所有Mujoco版本的ID查询函数（容错增强）"""
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
    """角度值（度）转弧度（容错增强）"""
    try:
        degrees = np.array(degrees, dtype=np.float64)
        return np.deg2rad(degrees)
    except Exception as e:
        print(f"⚠️ 角度转换失败: {e}")
        return 0.0 if np.isscalar(degrees) else np.zeros(JOINT_COUNT, dtype=np.float64)


def rad2deg(radians):
    """弧度转角度值（度）（容错增强）"""
    try:
        radians = np.array(radians, dtype=np.float64)
        return np.rad2deg(radians)
    except Exception as e:
        print(f"⚠️ 弧度转换失败: {e}")
        return 0.0 if np.isscalar(radians) else np.zeros(JOINT_COUNT, dtype=np.float64)


def calculate_load_adaptive_gains(current_load):
    """
    负载自适应增益计算（核心优化）
    根据当前末端负载，动态调整PD增益，抵消负载干扰
    :param current_load: 当前末端负载（kg）
    :return: 自适应kp, kd
    """
    # 负载归一化（0 ~ 1）
    normalized_load = min(current_load / LOAD_PARAMS['max_allowed_load'], 1.0)
    # 动态调整增益（负载越大，增益越高，保证响应性）
    adaptive_kp = PD_PARAMS['kp_base'] * (1 + normalized_load * (PD_PARAMS['kp_load_gain'] - 1))
    adaptive_kd = PD_PARAMS['kd_base'] * (1 + normalized_load * (PD_PARAMS['kd_load_gain'] - 1))
    return adaptive_kp, adaptive_kd


# ====================== 机械臂模型生成（终极全XML错误修复） ======================
def create_arm_model_with_load():
    """
    生成带负载建模的机械臂XML模型（终极全修复+全版本兼容）
    1.  彻底修复所有XML Schema错误：
        - 移除body的mass属性（改用geom定义质量）
        - 移除motor的forcelimit属性（改用软件层面过载保护）
    2.  末端负载通过geom的mass属性配置，保留可动态调整功能
    3.  关节添加阻尼和惯量，模拟真实负载特性
    4.  简化非核心几何，提升负载下仿真速度
    5.  依赖compiler的inertiafromgeom="true"，自动由geom质量推导body惯性属性
    6.  无任何违规配置，兼容所有Mujoco版本
    """
    end_effector_mass = LOAD_PARAMS['end_effector_mass']
    # 连杆geom质量（对应原body质量，通过geom定义，兼容所有Mujoco版本）
    link1_geom_mass = 0.8
    link2_geom_mass = 0.6
    link3_geom_mass = 0.6
    link4_geom_mass = 0.4
    link5_geom_mass = 0.2

    xml = f"""
<mujoco model="arm_with_load">
    <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
    <option timestep="{SIMULATION_TIMESTEP}" gravity="0 0 -9.81" iterations="30" tolerance="1e-6"/>

    <!-- 仅保留Mujoco基础支持的元素，无任何版本专属配置（全版本兼容） -->
    <default>
        <joint type="hinge" armature="0.2" damping="0.2" limited="true" margin="0.01"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="80"/>
        <geom contype="1" conaffinity="1" rgba="0.2 0.8 0.2 1"/>
    </default>

    <!-- 负载相关配置：末端负载可视化材质 -->
    <asset>
        <material name="load_material" rgba="1.0 0.0 0.0 0.8"/> <!-- 红色标记负载 -->
    </asset>

    <worldbody>
        <!-- 地面（简化尺寸，减少渲染开销） -->
        <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>

        <!-- 机械臂基座（无mass属性，符合Schema规范） -->
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>

            <!-- 关节1 -->
            <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0.1" range="{JOINT_LIMITS_RAD[0, 0]} {JOINT_LIMITS_RAD[0, 1]}"/>
            <!-- 连杆1：无body mass，质量通过geom定义（兼容所有Mujoco版本） -->
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="cylinder" size="0.04 0.18" mass="{link1_geom_mass}"/>

                <!-- 关节2 -->
                <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS_RAD[1, 0]} {JOINT_LIMITS_RAD[1, 1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2_geom" type="cylinder" size="0.04 0.18" mass="{link2_geom_mass}"/>

                    <!-- 关节3 -->
                    <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS_RAD[2, 0]} {JOINT_LIMITS_RAD[2, 1]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3_geom" type="cylinder" size="0.04 0.18" mass="{link3_geom_mass}"/>

                        <!-- 关节4 -->
                        <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS_RAD[3, 0]} {JOINT_LIMITS_RAD[3, 1]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4_geom" type="cylinder" size="0.04 0.18" mass="{link4_geom_mass}"/>

                            <!-- 关节5 -->
                            <joint name="joint5" type="hinge" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS_RAD[4, 0]} {JOINT_LIMITS_RAD[4, 1]}"/>
                            <body name="link5" pos="0 0 0.18">
                                <geom name="link5_geom" type="cylinder" size="0.03 0.09" mass="{link5_geom_mass}" rgba="0.8 0.2 0.2 1"/>

                                <!-- 末端执行器（负载通过geom mass定义，无body mass属性，兼容所有版本） -->
                                <body name="end_effector" pos="0 0 0.09">
                                    <site name="ee_site" pos="0 0 0" size="0.01"/>
                                    <!-- 末端负载：通过geom的mass属性配置，实现可动态调整 -->
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

    <!-- 关节电机（移除forcelimit属性，改用软件层面过载保护，兼容所有Mujoco版本） -->
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


# ====================== 核心控制器类（关节负载性能优化+软件过载保护） ======================
class ArmJointLoadOptimizedController:
    def __init__(self):
        # 初始化模型和数据（负载下容错增强）
        self.model = None
        self.data = None
        try:
            self.model = mujoco.MjModel.from_xml_string(create_arm_model_with_load())
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print(f"❌ 带负载模型初始化失败: {e}")
            global RUNNING
            RUNNING = False
            return

        # 获取ID
        self.joint_ids = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_NAMES]
        self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
        self.ee_site_id = get_mujoco_id(self.model, 'site', "ee_site")
        self.ee_body_id = get_mujoco_id(self.model, 'body', "end_effector")
        self.load_geom_id = get_mujoco_id(self.model, 'geom', "load_geom")  # 负载geom ID，用于更新质量

        # 状态变量（负载监控专用）
        self.viewer_inst = None
        self.viewer_ready = False
        self.last_control_time = time.time()
        self.last_print_time = time.time()
        self.fps_counter = 0
        self.step_count = 0
        self.total_simulation_time = 0.0

        # 负载相关状态
        self.current_end_load = LOAD_PARAMS['end_effector_mass']
        self.smoothed_joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        self.angle_error_history = np.zeros(JOINT_COUNT, dtype=np.float64)  # 控制误差监控
        self.overload_warning_flag = False

        # 初始化目标角度为零位（提前定义，避免属性不存在错误）
        self.target_angles_rad = np.zeros(JOINT_COUNT, dtype=np.float64)
        try:
            self.set_joint_angles(np.zeros(JOINT_COUNT), smooth=False, use_deg=False)
        except Exception as e:
            print(f"⚠️ 初始化关节角度失败: {e}")

        # 全局仿真开始时间
        global SIMULATION_START_TIME
        SIMULATION_START_TIME = time.time()

    def get_current_joint_angles(self, use_deg=True):
        """获取当前关节角度（负载下按需转换，减少冗余）"""
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        current_rad = np.array([self.data.qpos[jid] if jid >= 0 else 0 for jid in self.joint_ids], dtype=np.float64)
        if use_deg:
            return rad2deg(current_rad)
        return current_rad

    def get_joint_forces(self):
        """
        获取关节实时受力（负载监控核心）
        返回各关节扭矩（N·m），反映负载大小
        """
        if self.data is None:
            return np.zeros(JOINT_COUNT, dtype=np.float64)
        joint_forces = np.zeros(JOINT_COUNT, dtype=np.float64)
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                # 读取关节受力，并平滑处理，避免抖动
                raw_force = abs(self.data.qfrc_actuator[jid])
                self.smoothed_joint_forces[i] = (1 - LOAD_PARAMS['load_smoothing_factor']) * self.smoothed_joint_forces[
                    i] + \
                                                LOAD_PARAMS['load_smoothing_factor'] * raw_force
                joint_forces[i] = self.smoothed_joint_forces[i]
        return joint_forces

    def get_ee_position(self):
        """获取末端位置（容错增强）"""
        if self.data is None or self.ee_site_id < 0:
            return np.zeros(3, dtype=np.float64)
        return self.data.site_xpos[self.ee_site_id].copy()

    def clamp_joint_angles(self, angles, use_deg=True):
        """关节限位保护（负载下更严格的限位校验）"""
        angles = np.array(angles, dtype=np.float64)
        if use_deg:
            angles_rad = deg2rad(angles)
        else:
            angles_rad = angles.copy()
        # 负载下缩小限位余量，防止冲击
        limit_margin = 0.05  # 5%余量
        limits_rad_margin = JOINT_LIMITS_RAD.copy()
        limits_rad_margin[:, 0] += limit_margin
        limits_rad_margin[:, 1] -= limit_margin
        clamped_rad = np.clip(angles_rad, limits_rad_margin[:, 0], limits_rad_margin[:, 1])
        if use_deg:
            return rad2deg(clamped_rad)
        return clamped_rad

    def set_end_effector_load(self, mass):
        """
        动态设置末端负载（核心功能，兼容所有Mujoco版本）
        :param mass: 末端负载质量（kg），需≤max_allowed_load
        """
        if mass < 0 or mass > LOAD_PARAMS['max_allowed_load']:
            self.overload_warning_flag = True
            print(f"⚠️ 末端负载超出限制（0 ~ {LOAD_PARAMS['max_allowed_load']}kg），当前设置：{mass}kg")
            return
        self.overload_warning_flag = False

        # 方案1：直接更新负载geom的质量（无需重新初始化模型，更高效）
        if self.model is not None and self.load_geom_id >= 0:
            try:
                # 直接修改geom的mass属性，实时生效
                self.model.geom_mass[self.load_geom_id] = mass
                # 更新内部状态
                self.current_end_load = mass
                LOAD_PARAMS['end_effector_mass'] = mass
                print(f"✅ 末端负载已更新为 {mass}kg（直接修改geom质量，无需重启模型）")
                return
            except Exception as e:
                print(f"⚠️ 直接更新负载失败，将重新初始化模型: {e}")

        # 方案2：降级方案 - 重新初始化模型（兼容特殊场景）
        try:
            LOAD_PARAMS['end_effector_mass'] = mass
            self.current_end_load = mass
            # 重新初始化模型
            self.model = mujoco.MjModel.from_xml_string(create_arm_model_with_load())
            self.data = mujoco.MjData(self.model)
            # 重新获取所有ID
            self.joint_ids = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_NAMES]
            self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_COUNT)]
            self.ee_site_id = get_mujoco_id(self.model, 'site', "ee_site")
            self.ee_body_id = get_mujoco_id(self.model, 'body', "end_effector")
            self.load_geom_id = get_mujoco_id(self.model, 'geom', "load_geom")
            # 保留当前目标角度
            current_target = self.target_angles_rad.copy()
            self.target_angles_rad = current_target
            self.set_joint_angles(current_target, smooth=False, use_deg=False)
            print(f"✅ 末端负载已更新为 {mass}kg（重新初始化模型生效）")
        except Exception as e:
            print(f"❌ 更新末端负载失败: {e}")

    def set_joint_angles(self, target_angles, smooth=True, use_deg=True):
        """设置关节目标角度（负载下参数校验增强）"""
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
                print(f"⚠️ 更新模型状态失败: {e}")

        # 平滑控制时，记录目标角度（供PD控制使用）
        self.target_angles_rad = target_angles_rad.copy()

    def move_joint(self, joint_idx, angle, smooth=True, use_deg=True):
        """单独控制单个关节（负载下容错增强）"""
        if joint_idx < 0 or joint_idx >= JOINT_COUNT:
            raise ValueError(f"关节索引必须在0-{JOINT_COUNT - 1}之间，当前为{joint_idx}")

        current_angles = self.get_current_joint_angles(use_deg=use_deg)
        current_angles[joint_idx] = angle
        self.set_joint_angles(current_angles, smooth=smooth, use_deg=use_deg)

    def load_adaptive_pd_control(self):
        """
        负载自适应PD控制（核心性能优化）+ 软件层面过载保护（替代forcelimit）
        1.  动态调整PD增益，抵消负载干扰
        2.  软件扭矩限制，防止关节过载（兼容所有Mujoco版本）
        3.  误差反馈平滑，减少负载下抖动
        4.  过载时自动降低控制输出，保护关节
        """
        if self.data is None:
            return

        # 1. 获取当前状态
        current_angles_rad = self.get_current_joint_angles(use_deg=False)
        current_vels_rad = np.array([self.data.qvel[jid] if jid >= 0 else 0 for jid in self.joint_ids],
                                    dtype=np.float64)
        joint_forces = self.get_joint_forces()

        # 2. 计算负载自适应PD增益
        adaptive_kp, adaptive_kd = calculate_load_adaptive_gains(self.current_end_load)

        # 3. 计算控制误差（平滑处理）
        angle_error_rad = self.target_angles_rad - current_angles_rad
        self.angle_error_history = (1 - LOAD_PARAMS['load_smoothing_factor']) * self.angle_error_history + \
                                   LOAD_PARAMS['load_smoothing_factor'] * angle_error_rad

        # 4. 计算期望速度（带速度限制）
        desired_vel_rad = np.clip(self.angle_error_history * adaptive_kp, -PD_PARAMS['max_vel'], PD_PARAMS['max_vel'])

        # 5. PD控制输出计算
        control_signals = adaptive_kp * self.angle_error_history + adaptive_kd * (desired_vel_rad - current_vels_rad)

        # 6. 软件层面过载保护（替代motor的forcelimit，兼容所有Mujoco版本）
        for i in range(JOINT_COUNT):
            # 判断关节是否接近过载（受力达到90%最大扭矩阈值）
            if joint_forces[i] > JOINT_MAX_TORQUE[i] * 0.9:
                control_signals[i] *= 0.5  # 降低50%控制输出，防止过载损坏
                self.overload_warning_flag = True  # 置位过载警告标志
            else:
                # 过载解除后，清除警告标志
                if self.overload_warning_flag:
                    self.overload_warning_flag = False

        # 7. 设置控制信号
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = control_signals[i]

    def init_viewer(self):
        """初始化Viewer（负载下延迟加载，提升启动速度）"""
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
            print("✅ Viewer初始化成功")
            return True
        except Exception as e:
            print(f"❌ Viewer初始化失败: {e}")
            return False

    def print_load_status(self):
        """打印负载相关状态（核心监控功能）"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        # 计算统计信息
        fps = self.fps_counter / (current_time - self.last_print_time)
        joint_angles_deg = self.get_current_joint_angles(use_deg=True)
        ee_pos = self.get_ee_position()
        joint_forces = self.get_joint_forces()
        angle_errors_deg = rad2deg(self.angle_error_history)
        self.total_simulation_time = current_time - (SIMULATION_START_TIME or current_time)
        adaptive_kp, adaptive_kd = calculate_load_adaptive_gains(self.current_end_load)

        # 格式化打印（负载信息突出显示）
        print("-" * 100)
        print(f"📊 仿真统计 | 耗时: {self.total_simulation_time:.2f}s | 步数: {self.step_count:,} | FPS: {fps:5.1f}")
        print(f"🔧 关节角度 (度): {np.round(joint_angles_deg, 1)} | 控制误差 (度): {np.round(abs(angle_errors_deg), 3)}")
        print(f"🎯 末端位置 (m): {np.round(ee_pos, 3)} | 当前末端负载 (kg): {self.current_end_load:.2f}")
        print(f"⚙️ 关节受力 (N·m): {np.round(joint_forces, 2)} | 最大扭矩 (N·m): {np.round(JOINT_MAX_TORQUE, 1)}")
        print(f"📈 自适应PD参数 | kp: {adaptive_kp:.1f} | kd: {adaptive_kd:.1f}")
        if self.overload_warning_flag:
            print("⚠️  警告：关节接近过载状态，已自动降低控制输出！")
        print("-" * 100)

        # 重置计数器
        self.last_print_time = current_time
        self.fps_counter = 0

    def preset_pose(self, pose_name):
        """预设常用姿态（负载下更平稳的姿态切换）"""
        pose_map = {
            'zero': [0, 0, 0, 0, 0],  # 零位
            'up': [0, 30, 20, 10, 0],  # 抬起姿态
            'grasp': [0, 45, 30, 20, 10]  # 抓取姿态
        }
        if pose_name not in pose_map:
            print(f"⚠️ 无效姿态名称，支持：{list(pose_map.keys())}")
            return
        self.set_joint_angles(pose_map[pose_name], smooth=True, use_deg=True)
        print(f"✅ 切换到{pose_name}姿态（负载自适应控制已启用）")

    def run(self):
        """运行完整仿真（负载下循环逻辑优化）"""
        global RUNNING

        if not self.init_viewer():
            RUNNING = False
            return

        # 启动信息
        print("=" * 100)
        print("🚀 机械臂关节负载性能优化控制器 - 启动成功")
        print(f"✅ 模型信息 | 关节数量: {JOINT_COUNT} | 初始末端负载: {self.current_end_load:.2f}kg")
        print(f"✅ 仿真配置 | 控制频率: {CONTROL_FREQUENCY}Hz | 仿真步长: {SIMULATION_TIMESTEP:.3f}s")
        print(
            f"✅ 保护配置 | 最大末端负载: {LOAD_PARAMS['max_allowed_load']}kg | 关节最大扭矩: {np.max(JOINT_MAX_TORQUE)}N·m")
        print("📝 快捷指令:")
        print("   - 设置末端负载: controller.set_end_effector_load(1.0) （设置1kg负载）")
        print("   - 单关节控制: controller.move_joint(0, 90) （关节1旋转90度）")
        print("   - 预设姿态: controller.preset_pose('up') （切换抬起姿态）")
        print("   - 按 Ctrl+C 优雅退出")
        print("=" * 100)

        # 主循环（负载下极致效率优化）
        while RUNNING:
            try:
                current_time = time.time()
                self.fps_counter += 1
                self.step_count += 1

                # 按高控制频率执行负载自适应PD控制
                if current_time - self.last_control_time >= CONTROL_TIMESTEP:
                    self.load_adaptive_pd_control()
                    self.last_control_time = current_time

                # 执行仿真步（负载下容错增强）
                if self.model is not None and self.data is not None:
                    mujoco.mj_step(self.model, self.data)

                # 同步Viewer
                if self.viewer_ready:
                    self.viewer_inst.sync()

                # 打印负载状态
                self.print_load_status()

                # 动态睡眠优化，减少负载下CPU空转
                time_diff = current_time - self.last_control_time
                if time_diff < SLEEP_TIME:
                    sleep_duration = max(0.00001, SLEEP_TIME - time_diff)
                    time.sleep(sleep_duration)

            except Exception as e:
                print(f"⚠️ 仿真步异常（步数：{self.step_count}）: {e}")
                continue

        # 清理资源
        self.cleanup()
        # 最终统计
        print("\n" + "=" * 100)
        print("✅ 控制器已优雅退出 - 负载仿真最终统计")
        print(
            f"📈 总仿真时间: {self.total_simulation_time:.2f}s | 总步数: {self.step_count:,} | 平均FPS: {self.step_count / max(1, self.total_simulation_time):.1f}")
        print(
            f"🎯 最终末端负载 (kg): {self.current_end_load:.2f} | 最终关节受力 (N·m): {np.round(self.get_joint_forces(), 2)}")
        print(f"🎯 最终关节角度 (度): {np.round(self.get_current_joint_angles(), 1)}")
        print("=" * 100)

    def cleanup(self):
        """资源清理（负载下完整释放，避免内存泄漏）"""
        if self.viewer_ready and self.viewer_inst:
            try:
                self.viewer_inst.close()
            except Exception as e:
                print(f"⚠️ Viewer关闭失败: {e}")
            self.viewer_inst = None
            self.viewer_ready = False
        self.model = None
        self.data = None
        global RUNNING, SIMULATION_START_TIME
        RUNNING = False
        SIMULATION_START_TIME = None


# ====================== 负载演示函数（验证优化效果） ======================
def load_demo(controller):
    """负载变化演示，验证自适应控制效果"""

    def demo():
        time.sleep(2)

        # 演示1：初始零位（0.5kg负载）
        print("\n🎬 演示1：切换到零位姿态（初始负载0.5kg）")
        controller.preset_pose('zero')
        time.sleep(3)

        # 演示2：切换抬起姿态（0.5kg负载）
        print("\n🎬 演示2：切换到抬起姿态（0.5kg负载）")
        controller.preset_pose('up')
        time.sleep(3)

        # 演示3：增加末端负载到1.5kg
        print("\n🎬 演示3：设置末端负载为1.5kg（自适应PD控制自动生效）")
        controller.set_end_effector_load(1.5)
        time.sleep(2)

        # 演示4：负载下旋转关节1（90度）
        print("\n🎬 演示4：1.5kg负载下，关节1旋转90度（抗干扰控制，无超调）")
        controller.move_joint(0, 90, smooth=True, use_deg=True)
        time.sleep(3)

        # 演示5：切换抓取姿态（1.5kg负载）
        print("\n🎬 演示5：1.5kg负载下，切换到抓取姿态")
        controller.preset_pose('grasp')
        time.sleep(3)

        # 演示6：降低负载到0.2kg
        print("\n🎬 演示6：降低末端负载为0.2kg（PD增益自动回落）")
        controller.set_end_effector_load(0.2)
        time.sleep(2)

        # 演示7：回到零位
        print("\n🎬 演示7：切换回零位姿态")
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

    # 创建负载优化控制器
    controller = None
    try:
        controller = ArmJointLoadOptimizedController()
    except Exception as e:
        print(f"❌ 控制器创建失败: {e}")
        sys.exit(1)

    # 运行负载演示
    if controller is not None:
        load_demo(controller)

    # 启动控制器
    if controller is not None:
        controller.run()

    sys.exit(0)