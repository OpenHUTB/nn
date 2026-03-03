#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂关节运动性能优化控制器
核心特性：
- 梯形速度轨迹规划（无超调、高平滑性）
- 自适应刚度阻尼（负载/误差感知）
- 多维度误差补偿（间隙/摩擦/重力）
- PD+前馈控制（高精度定位）
- 完整的异常处理和资源管理
"""

import sys
import time
import signal
import threading
import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime


# ====================== 全局配置与状态 ======================
@dataclass
class JointConfig:
    """关节物理参数配置"""
    count: int = 5
    names: List[str] = field(default_factory=lambda: ["joint1", "joint2", "joint3", "joint4", "joint5"])
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
class ControlParams:
    """控制算法参数"""
    # 时间配置
    sim_dt: float = 0.0005
    ctrl_freq: int = 2000
    fps: int = 60

    # 派生参数（自动计算）
    ctrl_dt: float = field(init=False)
    sleep_dt: float = field(init=False)

    # PD控制参数
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
    comp_smoothing: float = 0.02

    # 刚度阻尼
    stiffness_base: np.ndarray = field(default_factory=lambda: np.array([200.0, 180.0, 150.0, 120.0, 80.0]))
    stiffness_load_gain: float = 1.8
    stiffness_error_gain: float = 1.5
    stiffness_min: np.ndarray = field(default_factory=lambda: np.array([100.0, 90.0, 75.0, 60.0, 40.0]))
    stiffness_max: np.ndarray = field(default_factory=lambda: np.array([300.0, 270.0, 225.0, 180.0, 120.0]))
    damping_ratio: float = 0.04

    def __post_init__(self):
        self.ctrl_dt = 1.0 / self.ctrl_freq
        self.sleep_dt = 1.0 / self.fps


# 全局实例化
JOINT_CFG = JointConfig()
CTRL_CFG = ControlParams()
RUNNING = True
LOCK = threading.Lock()


# ====================== 工具函数 ======================
def deg2rad(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """角度转弧度（带类型检查）"""
    try:
        return np.deg2rad(np.array(deg, dtype=np.float64))
    except:
        return 0.0 if np.isscalar(deg) else np.zeros(JOINT_CFG.count)


def rad2deg(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """弧度转角度（带类型检查）"""
    try:
        return np.rad2deg(np.array(rad, dtype=np.float64))
    except:
        return 0.0 if np.isscalar(rad) else np.zeros(JOINT_CFG.count)


def get_mujoco_id(model: mujoco.MjModel, obj_type: str, name: str) -> int:
    """安全获取MuJoCo对象ID"""
    type_map = {
        'joint': mujoco.mjtObj.mjOBJ_JOINT,
        'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
        'geom': mujoco.mjtObj.mjOBJ_GEOM
    }
    try:
        return mujoco.mj_name2id(model, type_map.get(obj_type, mujoco.mjtObj.mjOBJ_JOINT), name)
    except:
        return -1


def log_info(content: str):
    """线程安全的日志记录"""
    with LOCK:
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            with open("arm_controller.log", "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {content}\n")
            print(f"[{timestamp}] {content}")
        except:
            pass


# ====================== 轨迹规划器 ======================
class TrajectoryPlanner:
    """梯形速度轨迹规划器（向量化实现）"""

    @staticmethod
    def plan_single_joint(
            start: float,
            target: float,
            max_vel: float,
            max_acc: float,
            dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单关节轨迹规划"""
        delta = target - start
        if abs(delta) < 1e-5:
            return np.array([target]), np.array([0.0])

        direction = np.sign(delta)
        dist = abs(delta)

        # 计算轨迹参数
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

        # 生成时间序列
        t = np.arange(0, total_time + dt, dt)
        pos = np.zeros_like(t)
        vel = np.zeros_like(t)

        # 分段计算
        if dist <= 2 * accel_dist:
            # 加速段
            mask_acc = t <= accel_time
            vel[mask_acc] = max_acc * t[mask_acc] * direction
            pos[mask_acc] = start + 0.5 * max_acc * t[mask_acc] ** 2 * direction

            # 减速段
            mask_dec = ~mask_acc
            t_dec = t[mask_dec] - accel_time
            vel[mask_dec] = (peak_vel - max_acc * t_dec) * direction
            pos[mask_dec] = start + (peak_vel * accel_time - 0.5 * max_acc * t_dec ** 2) * direction
        else:
            # 加速段
            mask_acc = t <= accel_time
            vel[mask_acc] = max_acc * t[mask_acc] * direction
            pos[mask_acc] = start + 0.5 * max_acc * t[mask_acc] ** 2 * direction

            # 匀速段
            mask_uni = (t > accel_time) & (t <= accel_time + uniform_time)
            t_uni = t[mask_uni] - accel_time
            vel[mask_uni] = max_vel * direction
            pos[mask_uni] = start + (accel_dist + max_vel * t_uni) * direction

            # 减速段
            mask_dec = t > accel_time + uniform_time
            t_dec = t[mask_dec] - (accel_time + uniform_time)
            vel[mask_dec] = (max_vel - max_acc * t_dec) * direction
            pos[mask_dec] = start + (dist - (accel_dist - 0.5 * max_acc * t_dec ** 2)) * direction

        # 强制终点
        pos[-1] = target
        vel[-1] = 0.0

        return pos, vel

    @staticmethod
    def plan_joints(
            start_rad: np.ndarray,
            target_rad: np.ndarray,
            dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """多关节轨迹规划（向量化）"""
        # 边界检查
        start_rad = np.clip(start_rad, JOINT_CFG.limits_rad[:, 0] + 0.01, JOINT_CFG.limits_rad[:, 1] - 0.01)
        target_rad = np.clip(target_rad, JOINT_CFG.limits_rad[:, 0] + 0.01, JOINT_CFG.limits_rad[:, 1] - 0.01)

        # 批量规划
        traj_pos = []
        traj_vel = []
        max_len = 1

        for i in range(JOINT_CFG.count):
            pos, vel = TrajectoryPlanner.plan_single_joint(
                start_rad[i], target_rad[i],
                JOINT_CFG.max_vel_rad[i],
                JOINT_CFG.max_acc_rad[i],
                dt
            )
            traj_pos.append(pos)
            traj_vel.append(vel)
            max_len = max(max_len, len(pos))

        # 统一长度
        for i in range(JOINT_CFG.count):
            if len(traj_pos[i]) < max_len:
                pad_len = max_len - len(traj_pos[i])
                traj_pos[i] = np.pad(traj_pos[i], (0, pad_len), 'constant', constant_values=target_rad[i])
                traj_vel[i] = np.pad(traj_vel[i], (0, pad_len), 'constant', constant_values=0.0)

        return np.array(traj_pos).T, np.array(traj_vel).T


# ====================== 机械臂控制器核心类 ======================
class ArmController:
    def __init__(self):
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer: Optional[mujoco.viewer.Viewer] = None

        # ID缓存
        self.joint_ids: List[int] = []
        self.motor_ids: List[int] = []
        self.ee_geom_id: int = -1

        # 状态变量
        self.traj_pos: np.ndarray = np.zeros((1, JOINT_CFG.count))
        self.traj_vel: np.ndarray = np.zeros((1, JOINT_CFG.count))
        self.traj_idx: int = 0
        self.target_rad: np.ndarray = np.zeros(JOINT_CFG.count)

        self.stiffness: np.ndarray = CTRL_CFG.stiffness_base.copy()
        self.damping: np.ndarray = CTRL_CFG.stiffness_base * CTRL_CFG.damping_ratio

        self.joint_err: np.ndarray = np.zeros(JOINT_CFG.count)
        self.max_joint_err: np.ndarray = np.zeros(JOINT_CFG.count)
        self.end_load: float = 0.5

        # 性能统计
        self.step_count: int = 0
        self.last_ctrl_time: float = time.time()
        self.last_print_time: float = time.time()
        self.fps_counter: int = 0

        # 初始化
        self._init_model()
        self._init_ids()
        self.reset()
        log_info("控制器初始化完成")

    def _init_model(self):
        """初始化MuJoCo模型"""
        try:
            xml = self._generate_model_xml()
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            log_info("MuJoCo模型加载成功")
        except Exception as e:
            log_info(f"模型初始化失败: {e}")
            global RUNNING
            RUNNING = False

    def _generate_model_xml(self) -> str:
        """生成优化的模型XML"""
        link_masses = [0.8, 0.6, 0.6, 0.4, 0.2]
        friction = CTRL_CFG.friction

        return f"""
<mujoco model="optimized_arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{CTRL_CFG.sim_dt}" gravity="0 0 -9.81" iterations="100" tolerance="1e-9"/>

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
                   range="{JOINT_CFG.limits_rad[0, 0]} {JOINT_CFG.limits_rad[0, 1]}"/>

            <!-- 大臂 -->
            <body name="link1" pos="0 0 0.1">
                <geom name="link1_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[0]}"
                      material="arm_mat" friction="{friction[0]} {friction[0]} {friction[0]}"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18"
                       range="{JOINT_CFG.limits_rad[1, 0]} {JOINT_CFG.limits_rad[1, 1]}"/>

                <!-- 中臂 -->
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[1]}"
                          material="arm_mat" friction="{friction[1]} {friction[1]} {friction[1]}"/>
                    <joint name="joint3" axis="0 1 0" pos="0 0 0.18"
                           range="{JOINT_CFG.limits_rad[2, 0]} {JOINT_CFG.limits_rad[2, 1]}"/>

                    <!-- 小臂 -->
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[2]}"
                              material="arm_mat" friction="{friction[2]} {friction[2]} {friction[2]}"/>
                        <joint name="joint4" axis="0 1 0" pos="0 0 0.18"
                               range="{JOINT_CFG.limits_rad[3, 0]} {JOINT_CFG.limits_rad[3, 1]}"/>

                        <!-- 末端 -->
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4_geom" type="cylinder" size="0.04 0.18" mass="{link_masses[3]}"
                                  material="arm_mat" friction="{friction[3]} {friction[3]} {friction[3]}"/>
                            <joint name="joint5" axis="0 1 0" pos="0 0 0.18"
                                   range="{JOINT_CFG.limits_rad[4, 0]} {JOINT_CFG.limits_rad[4, 1]}"/>

                            <!-- 末端执行器 -->
                            <body name="end_effector" pos="0 0 0.18">
                                <geom name="ee_geom" type="sphere" size="0.04" mass="{self.end_load}" 
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
        """初始化MuJoCo对象ID"""
        if self.model is None:
            return

        self.joint_ids = [get_mujoco_id(self.model, 'joint', name) for name in JOINT_CFG.names]
        self.motor_ids = [get_mujoco_id(self.model, 'actuator', f"motor{i + 1}") for i in range(JOINT_CFG.count)]
        self.ee_geom_id = get_mujoco_id(self.model, 'geom', "ee_geom")

        # 初始化阻尼
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                self.model.jnt_damping[jid] = self.damping[i]

    def reset(self):
        """重置控制器状态"""
        self.target_rad = np.zeros(JOINT_CFG.count)
        self.traj_pos = np.zeros((1, JOINT_CFG.count))
        self.traj_vel = np.zeros((1, JOINT_CFG.count))
        self.traj_idx = 0
        self.joint_err = np.zeros(JOINT_CFG.count)
        self.max_joint_err = np.zeros(JOINT_CFG.count)

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取关节状态（位置/速度，弧度）"""
        if self.data is None:
            return np.zeros(JOINT_CFG.count), np.zeros(JOINT_CFG.count)

        qpos = np.array([self.data.qpos[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        return qpos, qvel

    def calculate_compensation(self, qpos: np.ndarray, qvel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算误差补偿（间隙+摩擦+重力）"""
        # 1. 间隙补偿
        vel_sign = np.sign(qvel)
        vel_zero_mask = np.abs(qvel) < 1e-4
        vel_sign[vel_zero_mask] = np.sign(self.joint_err)[vel_zero_mask]
        backlash_comp = CTRL_CFG.backlash * vel_sign

        # 2. 摩擦补偿
        friction_comp = np.where(
            np.abs(qvel) < 1e-4,
            CTRL_CFG.friction * np.sign(self.joint_err),
            0.0
        )

        # 3. 重力补偿
        gravity_comp = np.zeros(JOINT_CFG.count)
        if CTRL_CFG.gravity_comp:
            gravity_comp = 0.5 * np.sin(qpos) * self.end_load

        total_comp = backlash_comp + friction_comp + gravity_comp
        return total_comp, gravity_comp

    def update_stiffness_damping(self, qpos: np.ndarray, qvel: np.ndarray):
        """自适应更新刚度阻尼"""
        # 计算负载比例
        joint_forces = np.abs([self.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        load_ratio = np.clip(np.mean(joint_forces / JOINT_CFG.max_torque), 0.0, 1.0)

        # 计算误差比例
        err_norm = np.clip(np.abs(self.joint_err) / deg2rad(1.0), 0.0, 1.0)

        # 更新刚度
        target_stiffness = CTRL_CFG.stiffness_base * \
                           (1 + load_ratio * (CTRL_CFG.stiffness_load_gain - 1)) * \
                           (1 + err_norm * (CTRL_CFG.stiffness_error_gain - 1))
        target_stiffness = np.clip(target_stiffness, CTRL_CFG.stiffness_min, CTRL_CFG.stiffness_max)
        self.stiffness = 0.95 * self.stiffness + 0.05 * target_stiffness

        # 更新阻尼
        self.damping = self.stiffness * CTRL_CFG.damping_ratio
        self.damping = np.clip(self.damping,
                               CTRL_CFG.stiffness_min * 0.02,
                               CTRL_CFG.stiffness_max * 0.08)

        # 应用到模型
        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                self.model.jnt_damping[jid] = self.damping[i]

    def control_step(self):
        """单步控制计算"""
        current_time = time.time()
        if current_time - self.last_ctrl_time < CTRL_CFG.ctrl_dt:
            return

        # 获取当前状态
        qpos, qvel = self.get_joint_states()

        # 获取目标轨迹点
        if self.traj_idx < len(self.traj_pos):
            target_pos = self.traj_pos[self.traj_idx]
            target_vel = self.traj_vel[self.traj_idx]
            self.traj_idx += 1
        else:
            target_pos = self.target_rad
            target_vel = np.zeros(JOINT_CFG.count)

        # 计算误差
        self.joint_err = target_pos - qpos
        self.max_joint_err = np.maximum(self.max_joint_err, np.abs(self.joint_err))

        # 自适应PD参数
        load_factor = np.clip(self.end_load / 2.0, 0.0, 1.0)
        kp = CTRL_CFG.kp_base * (1 + load_factor * (CTRL_CFG.kp_load_gain - 1))
        kd = CTRL_CFG.kd_base * (1 + load_factor * (CTRL_CFG.kd_load_gain - 1))

        # PD控制
        pd_ctrl = kp * self.joint_err + kd * (target_vel - qvel)

        # 前馈控制
        ff_ctrl = CTRL_CFG.ff_vel_gain * target_vel + \
                  CTRL_CFG.ff_acc_gain * (target_vel - qvel) / CTRL_CFG.ctrl_dt

        # 误差补偿
        comp_ctrl, gravity_comp = self.calculate_compensation(qpos, qvel)

        # 总控制输出
        total_ctrl = pd_ctrl + ff_ctrl + comp_ctrl
        total_ctrl = np.clip(total_ctrl, -JOINT_CFG.max_torque, JOINT_CFG.max_torque)

        # 应用控制信号
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = total_ctrl[i]

        # 更新刚度阻尼
        self.update_stiffness_damping(qpos, qvel)

        # 更新时间戳
        self.last_ctrl_time = current_time

    def set_end_load(self, mass: float):
        """设置末端负载（0-2kg）"""
        with LOCK:
            if 0.0 <= mass <= 2.0:
                self.end_load = mass
                if self.ee_geom_id >= 0 and self.model is not None:
                    self.model.geom_mass[self.ee_geom_id] = mass
                log_info(f"末端负载更新为: {mass}kg")
            else:
                log_info(f"负载超出范围: {mass}kg (限制: 0-2kg)")

    def move_to(self, target_deg: Union[List[float], np.ndarray]):
        """移动到目标角度（度）"""
        with LOCK:
            start_rad, _ = self.get_joint_states()
            target_rad = deg2rad(target_deg)
            self.target_rad = target_rad
            self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_joints(
                start_rad, target_rad, CTRL_CFG.ctrl_dt
            )
            self.traj_idx = 0
            log_info(f"规划轨迹: {np.round(rad2deg(start_rad), 1)}° → {np.round(rad2deg(target_rad), 1)}°")

    def preset_pose(self, pose_name: str):
        """预设姿态"""
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

    def print_status(self):
        """打印运行状态（1Hz）"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        # 计算FPS
        fps = self.fps_counter / (current_time - self.last_print_time)

        # 获取状态
        qpos_deg, qvel_deg = rad2deg(self.get_joint_states())
        err_deg = rad2deg(self.joint_err)
        max_err_deg = rad2deg(self.max_joint_err)

        # 打印状态
        log_info("=" * 80)
        log_info(f"步数: {self.step_count} | FPS: {fps:.1f} | 负载: {self.end_load:.1f}kg")
        log_info(f"关节角度: {np.round(qpos_deg, 1)} °")
        log_info(f"关节速度: {np.round(qvel_deg, 2)} °/s")
        log_info(f"定位误差: {np.round(np.abs(err_deg), 3)} ° (最大: {np.round(max_err_deg, 3)} °)")
        log_info(f"刚度: {np.round(self.stiffness, 0)} | 阻尼: {np.round(self.damping, 1)}")
        log_info("=" * 80)

        # 重置计数器
        self.last_print_time = current_time
        self.fps_counter = 0

    def run(self):
        """主运行循环"""
        global RUNNING

        # 初始化Viewer
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            log_info("Viewer启动成功")
        except Exception as e:
            log_info(f"Viewer启动失败: {e}")
            RUNNING = False
            return

        # 主循环
        log_info("控制器开始运行 (按Ctrl+C退出)")
        while RUNNING and self.viewer.is_running():
            try:
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
                self.print_status()

                # 睡眠控制
                time.sleep(CTRL_CFG.sleep_dt)

            except Exception as e:
                log_info(f"运行错误: {e}")
                continue

        # 资源清理
        self.cleanup()
        log_info(
            f"控制器停止 | 总步数: {self.step_count} | 最大定位误差: {np.round(rad2deg(np.max(self.max_joint_err)), 3)}°")

    def cleanup(self):
        """清理资源"""
        if self.viewer:
            self.viewer.close()
        self.model = None
        self.data = None


# ====================== 信号处理与演示 ======================
def signal_handler(sig, frame):
    """退出信号处理"""
    global RUNNING
    if RUNNING:
        log_info("收到退出信号，正在停止控制器...")
        RUNNING = False


def demo_routine(controller: ArmController):
    """演示程序"""

    def demo_task():
        time.sleep(2)
        controller.preset_pose("zero")
        time.sleep(3)
        controller.preset_pose("test")
        time.sleep(4)
        controller.set_end_load(1.5)
        time.sleep(4)
        controller.preset_pose("grasp")
        time.sleep(3)
        controller.set_end_load(0.2)
        time.sleep(3)
        controller.preset_pose("zero")
        time.sleep(2)
        global RUNNING
        RUNNING = False

    # 启动演示线程
    demo_thread = threading.Thread(target=demo_task, daemon=True)
    demo_thread.start()


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 设置numpy打印格式
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 初始化并运行控制器
    try:
        arm_ctrl = ArmController()
        demo_routine(arm_ctrl)
        arm_ctrl.run()
    except Exception as e:
        log_info(f"程序异常退出: {e}")
        sys.exit(1)