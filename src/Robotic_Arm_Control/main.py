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
from typing import List, Tuple, Optional, Union, Dict, Final
from datetime import datetime
from contextlib import contextmanager

# ====================== 常量定义（不可变） ======================
JOINT_COUNT: Final[int] = 5
LOG_FILE: Final[str] = "arm_controller.log"
MIN_LOAD: Final[float] = 0.0
MAX_LOAD: Final[float] = 2.0
DEG_TO_RAD_SCALE: Final[float] = np.pi / 180.0
RAD_TO_DEG_SCALE: Final[float] = 180.0 / np.pi


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


@dataclass(frozen=True)
class ControlTuning:
    """控制算法参数调优"""
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
def thread_safe_logging():
    """线程安全的日志上下文管理器"""
    with GLOBAL_LOCK:
        yield


def log_info(content: str):
    """标准化日志记录"""
    try:
        with thread_safe_logging():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_msg = f"[{timestamp}] {content}"
            # 写入日志文件
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{log_msg}\n")
            # 打印到控制台
            print(log_msg)
    except Exception as e:
        print(f"日志记录失败: {e} | 内容: {content}")


# ====================== 轨迹规划器（纯算法类） ======================
class TrajectoryPlanner:
    """梯形速度轨迹规划器（完全向量化，无状态）"""

    @staticmethod
    def _plan_single(
            start: float,
            target: float,
            max_vel: float,
            max_acc: float,
            dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单关节轨迹规划（内部方法）"""
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
            dt: float = TIME_CFG.ctrl_dt
    ) -> Tuple[np.ndarray, np.ndarray]:
        """多关节轨迹规划（对外接口）"""
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

        return np.array(traj_pos).T, np.array(traj_vel).T


# ====================== 机械臂控制器（核心业务类） ======================
class ArmController:
    """机械臂控制器（面向对象，封装状态与行为）"""

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

        # 物理参数状态
        self.stiffness: np.ndarray = CONTROL_TUNING.stiffness_base.copy()
        self.damping: np.ndarray = CONTROL_TUNING.stiffness_base * CONTROL_TUNING.damping_ratio
        self.end_load: float = 0.5

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
        log_info("控制器初始化完成")

    def _init_mujoco(self):
        """初始化MuJoCo模型（异常安全）"""
        try:
            xml_str = self._generate_model_xml()
            self.model = mujoco.MjModel.from_xml_string(xml_str)
            self.data = mujoco.MjData(self.model)
            log_info("MuJoCo模型加载成功")
        except Exception as e:
            log_info(f"模型初始化失败: {e}")
            global RUNNING
            RUNNING = False

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
        self.target_rad = np.zeros(JOINT_COUNT)
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.joint_err = np.zeros(JOINT_COUNT)
        self.max_joint_err = np.zeros(JOINT_COUNT)

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

        # 3. 重力补偿
        gravity_comp = np.zeros(JOINT_COUNT)
        if CONTROL_TUNING.gravity_comp:
            gravity_comp = 0.5 * np.sin(qpos) * self.end_load

        total_comp = backlash_comp + friction_comp + gravity_comp
        return total_comp, gravity_comp

    def _update_stiffness_damping(self, qpos: np.ndarray, qvel: np.ndarray):
        """自适应更新刚度阻尼（性能优化）"""
        if self.data is None or self.model is None:
            return

        # 计算负载比例（向量化）
        joint_forces = np.abs([self.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        load_ratio = np.clip(np.mean(joint_forces / JOINT_LIMITS.max_torque), 0.0, 1.0)

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
        """单步控制计算（核心逻辑，性能优化）"""
        current_time = time.time()
        # 控制频率限流（避免高频计算）
        if current_time - self.last_ctrl_time < TIME_CFG.ctrl_dt:
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
            target_vel = np.zeros(JOINT_COUNT)

        # 计算误差（向量化）
        self.joint_err = target_pos - qpos
        self.max_joint_err = np.maximum(self.max_joint_err, np.abs(self.joint_err))

        # 自适应PD参数
        load_factor = np.clip(self.end_load / MAX_LOAD, 0.0, 1.0)
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

    def set_end_load(self, mass: float):
        """设置末端负载（带范围检查和线程安全）"""
        with thread_safe_logging():
            if not (MIN_LOAD <= mass <= MAX_LOAD):
                log_info(f"负载超出范围: {mass}kg (限制: {MIN_LOAD}-{MAX_LOAD}kg)")
                return

            self.end_load = mass
            if self.ee_geom_id >= 0 and self.model is not None:
                self.model.geom_mass[self.ee_geom_id] = mass
            log_info(f"末端负载更新为: {mass}kg")

    def move_to(self, target_deg: Union[List[float], np.ndarray]):
        """移动到目标角度（线程安全，带参数检查）"""
        with thread_safe_logging():
            # 参数检查
            target_deg_arr = np.asarray(target_deg, dtype=np.float64)
            if target_deg_arr.shape != (JOINT_COUNT,):
                log_info(f"目标角度维度错误: 期望{JOINT_COUNT}维，实际{target_deg_arr.shape}")
                return

            # 获取当前状态
            start_rad, _ = self.get_joint_states()
            target_rad = deg2rad(target_deg_arr)

            # 规划轨迹
            self.target_rad = target_rad
            self.traj_pos, self.traj_vel = TrajectoryPlanner.plan_joints(start_rad, target_rad)
            self.traj_idx = 0

            # 日志记录
            log_info(f"规划轨迹: {np.round(rad2deg(start_rad), 1)}° → {np.round(rad2deg(target_rad), 1)}°")

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
        """打印运行状态（1Hz限流）"""
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        # 计算FPS
        fps = self.fps_counter / (current_time - self.last_print_time)

        # 获取状态
        qpos_deg, qvel_deg = rad2deg(self.get_joint_states())
        err_deg = rad2deg(self.joint_err)
        max_err_deg = rad2deg(self.max_joint_err)

        # 打印状态（结构化）
        log_info("=" * 80)
        log_info(f"运行状态 | 步数: {self.step_count} | FPS: {fps:.1f} | 负载: {self.end_load:.1f}kg")
        log_info(f"关节角度: {np.round(qpos_deg, 1)} °")
        log_info(f"关节速度: {np.round(qvel_deg, 2)} °/s")
        log_info(f"定位误差: {np.round(np.abs(err_deg), 3)} ° (最大: {np.round(max_err_deg, 3)} °)")
        log_info(f"刚度: {np.round(self.stiffness, 0)} | 阻尼: {np.round(self.damping, 1)}")
        log_info("=" * 80)

        # 重置计数器
        self.last_print_time = current_time
        self.fps_counter = 0

    def run(self):
        """主运行循环（异常安全，资源自动清理）"""
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

        # 主循环（异常安全）
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

    def _cleanup(self):
        """资源清理（自动化，避免泄漏）"""
        if self.viewer:
            self.viewer.close()
        # 释放大数组内存
        self.traj_pos = np.array([])
        self.traj_vel = np.array([])
        self.model = None
        self.data = None


# ====================== 信号处理与演示 ======================
def signal_handler(sig, frame):
    """优雅退出信号处理"""
    global RUNNING
    if RUNNING:
        log_info("收到退出信号，正在停止控制器...")
        RUNNING = False


def demo_routine(controller: ArmController):
    """演示程序（解耦，易于修改）"""

    def demo_task():
        demo_steps = [
            (2, 'preset_pose', 'zero'),
            (3, 'preset_pose', 'test'),
            (4, 'set_end_load', 1.5),
            (4, 'preset_pose', 'grasp'),
            (3, 'set_end_load', 0.2),
            (3, 'preset_pose', 'zero'),
            (2, 'stop', None)
        ]

        for delay, action, param in demo_steps:
            time.sleep(delay)
            if not RUNNING:
                break

            if action == 'preset_pose':
                controller.preset_pose(param)
            elif action == 'set_end_load':
                controller.set_end_load(param)
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
