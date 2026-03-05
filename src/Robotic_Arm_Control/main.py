#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂控制器（增强优化版）
核心功能：基础控制+碰撞检测+轨迹平滑+轨迹队列+数据可视化+参数管理
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
from matplotlib.animation import FuncAnimation

# ====================== 常量定义（预计算） ======================
JOINT_COUNT = 5
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
MIN_LOAD, MAX_LOAD = 0.0, 2.0

# 关节极限
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

# 碰撞检测阈值
COLLISION_THRESHOLD = 0.01  # 1cm
COLLISION_FORCE_THRESHOLD = 5.0  # 5N


# 控制参数
@dataclass
class Cfg:
    # 基础控制
    kp_base, kd_base = 120.0, 8.0
    kp_load_gain, kd_load_gain = 1.8, 1.5
    ff_vel, ff_acc = 0.7, 0.5

    # 误差补偿
    backlash = np.array([0.001, 0.001, 0.002, 0.002, 0.003])
    friction = np.array([0.1, 0.08, 0.08, 0.06, 0.06])
    gravity_comp = True

    # 刚度阻尼
    stiffness_base = np.array([200.0, 180.0, 150.0, 120.0, 80.0])
    stiffness_load_gain = 1.8
    stiffness_error_gain = 1.5
    stiffness_min = np.array([100.0, 90.0, 75.0, 60.0, 40.0])
    stiffness_max = np.array([300.0, 270.0, 225.0, 180.0, 120.0])
    damping_ratio = 0.04

    # 轨迹平滑
    smooth_factor = 0.1  # 低通滤波系数
    jerk_limit = np.array([10.0, 8.0, 8.0, 6.0, 6.0])  # 加加速度限制


Cfg = Cfg()

# 全局状态
RUNNING = True
PAUSED = False
EMERGENCY_STOP = False
COLLISION_DETECTED = False
LOCK = threading.Lock()

# 目录初始化
for dir_name in ["trajectories", "params", "logs", "data"]:
    Path(dir_name).mkdir(exist_ok=True)


# ====================== 工具函数 ======================
@contextmanager
def lock():
    with LOCK:
        yield


def log(msg):
    """增强日志（分级+文件轮转）"""
    try:
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_msg = f"[{ts}] {msg}"
        # 主日志文件
        with open("logs/arm.log", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
        # 控制台输出
        print(log_msg)
        # 碰撞日志单独记录
        if "碰撞" in msg or "COLLISION" in msg.upper():
            with open("logs/collision.log", "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
    except:
        pass


def deg2rad(x):
    try:
        return np.asarray(x, np.float64) * DEG2RAD
    except:
        return 0.0 if np.isscalar(x) else np.zeros(JOINT_COUNT)


def rad2deg(x):
    try:
        return np.asarray(x, np.float64) * RAD2DEG
    except:
        return 0.0 if np.isscalar(x) else np.zeros(JOINT_COUNT)


def save_params(name="default"):
    """保存控制参数到文件"""
    try:
        params = {}
        for key, value in Cfg.__dict__.items():
            if isinstance(value, np.ndarray):
                params[key] = value.tolist()
            else:
                params[key] = value

        with open(f"params/{name}.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        log(f"参数已保存: params/{name}.json")
    except Exception as e:
        log(f"保存参数失败: {e}")


def load_params(name="default"):
    """从文件加载控制参数"""
    try:
        with open(f"params/{name}.json", "r", encoding="utf-8") as f:
            params = json.load(f)

        for key, value in params.items():
            if hasattr(Cfg, key):
                current = getattr(Cfg, key)
                if isinstance(current, np.ndarray):
                    setattr(Cfg, key, np.array(value, dtype=np.float64))
                else:
                    setattr(Cfg, key, value)
        log(f"参数已加载: params/{name}.json")
    except Exception as e:
        log(f"加载参数失败: {e}")


# ====================== 轨迹规划增强 ======================
TRAJ_CACHE = {}


def smooth_trajectory(traj_pos, traj_vel):
    """轨迹平滑（抑制抖动，限制加加速度）"""
    if len(traj_pos) <= 2:
        return traj_pos, traj_vel

    # 低通滤波平滑位置
    smooth_pos = np.copy(traj_pos)
    for i in range(1, len(smooth_pos)):
        smooth_pos[i] = (1 - Cfg.smooth_factor) * smooth_pos[i - 1] + Cfg.smooth_factor * traj_pos[i]

    # 重新计算速度并限制加加速度
    smooth_vel = np.zeros_like(smooth_pos)
    for i in range(1, len(smooth_pos)):
        vel = (smooth_pos[i] - smooth_pos[i - 1]) / CTRL_DT
        # 加加速度限制
        if i > 1:
            jerk = (vel - smooth_vel[i - 1]) / CTRL_DT
            jerk_clipped = np.clip(jerk, -Cfg.jerk_limit, Cfg.jerk_limit)
            vel = smooth_vel[i - 1] + jerk_clipped * CTRL_DT
        smooth_vel[i] = np.clip(vel, -MAX_VEL, MAX_VEL)

    return smooth_pos, smooth_vel


def plan_trajectory(start, target, dt=CTRL_DT, smooth=True):
    """增强轨迹规划（基础规划+平滑）"""
    # 边界裁剪
    start = np.clip(start, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)
    target = np.clip(target, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)

    # 缓存检查
    cache_key = (hash(start.tobytes()), hash(target.tobytes()), smooth)
    if cache_key in TRAJ_CACHE:
        return TRAJ_CACHE[cache_key]

    # 基础梯形规划
    traj_pos, traj_vel, max_len = [], [], 1
    for i in range(JOINT_COUNT):
        delta = target[i] - start[i]
        if abs(delta) < 1e-5:
            pos, vel = np.array([target[i]]), np.array([0.0])
        else:
            dir = np.sign(delta)
            dist = abs(delta)
            accel_dist = (MAX_VEL[i] ** 2) / (2 * MAX_ACC[i])

            if dist <= 2 * accel_dist:
                peak_vel = np.sqrt(dist * MAX_ACC[i])
                accel_time = peak_vel / MAX_ACC[i]
                total_time = 2 * accel_time
            else:
                accel_time = MAX_VEL[i] / MAX_ACC[i]
                uniform_time = (dist - 2 * accel_dist) / MAX_VEL[i]
                total_time = 2 * accel_time + uniform_time

            t = np.arange(0, total_time + dt, dt)
            pos = np.empty_like(t)
            vel = np.empty_like(t)

            mask_acc = t <= accel_time
            mask_uni = (t > accel_time) & (t <= accel_time + uniform_time) if dist > 2 * accel_dist else np.zeros_like(
                t, bool)
            mask_dec = ~(mask_acc | mask_uni)

            vel[mask_acc] = MAX_ACC[i] * t[mask_acc] * dir
            pos[mask_acc] = start[i] + 0.5 * MAX_ACC[i] * t[mask_acc] ** 2 * dir

            if dist > 2 * accel_dist:
                t_uni = t[mask_uni] - accel_time
                vel[mask_uni] = MAX_VEL[i] * dir
                pos[mask_uni] = start[i] + (accel_dist + MAX_VEL[i] * t_uni) * dir
                t_dec = t[mask_dec] - (accel_time + uniform_time)
                vel[mask_dec] = (MAX_VEL[i] - MAX_ACC[i] * t_dec) * dir
                pos[mask_dec] = start[i] + (dist - (accel_dist - 0.5 * MAX_ACC[i] * t_dec ** 2)) * dir
            else:
                t_dec = t[mask_dec] - accel_time
                vel[mask_dec] = (peak_vel - MAX_ACC[i] * t_dec) * dir
                pos[mask_dec] = start[i] + (peak_vel * accel_time - 0.5 * MAX_ACC[i] * t_dec ** 2) * dir

            pos[-1], vel[-1] = target[i], 0.0

        traj_pos.append(pos)
        traj_vel.append(vel)
        max_len = max(max_len, len(pos))

    # 统一长度
    for i in range(JOINT_COUNT):
        if len(traj_pos[i]) < max_len:
            pad = max_len - len(traj_pos[i])
            traj_pos[i] = np.pad(traj_pos[i], (0, pad), 'constant', constant_values=target[i])
            traj_vel[i] = np.pad(traj_vel[i], (0, pad), 'constant')

    # 转换为数组
    traj_pos = np.array(traj_pos).T
    traj_vel = np.array(traj_vel).T

    # 轨迹平滑
    if smooth:
        traj_pos, traj_vel = smooth_trajectory(traj_pos, traj_vel)

    # 缓存结果
    TRAJ_CACHE[cache_key] = (traj_pos, traj_vel)
    return traj_pos, traj_vel


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
        # 核心状态
        self.model, self.data = self._init_mujoco()
        self.viewer = None

        # ID缓存
        self.joint_ids = [self._get_id('joint', f'joint{i + 1}') for i in range(JOINT_COUNT)]
        self.motor_ids = [self._get_id('actuator', f'motor{i + 1}') for i in range(JOINT_COUNT)]
        self.ee_id = self._get_id('geom', 'ee_geom')
        self.link_geom_ids = [self._get_id('geom', f'link{i + 1}') for i in range(JOINT_COUNT)]

        # 控制状态
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.saved_idx = 0
        self.target = np.zeros(JOINT_COUNT)

        # 轨迹队列（新增）
        self.traj_queue = []
        self.current_queue_idx = 0

        # 物理状态
        self.stiffness = Cfg.stiffness_base.copy()
        self.damping = self.stiffness * Cfg.damping_ratio
        self.load_set = 0.5
        self.load_actual = 0.5

        # 误差状态
        self.err = np.zeros(JOINT_COUNT)
        self.max_err = np.zeros(JOINT_COUNT)

        # 数据记录（新增）
        self.data_recorder = {
            'time': [], 'qpos': [], 'qvel': [], 'err': [], 'load': [],
            'stiffness': [], 'torque': [], 'collision': []
        }
        self.record_enabled = False
        self.record_count = 0

        # 性能统计
        self.step = 0
        self.last_ctrl = time.time()
        self.last_print = time.time()
        self.fps_count = 0

    def _init_mujoco(self):
        """增强MuJoCo初始化（添加碰撞检测配置）"""
        xml = f"""
<mujoco model="arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{SIM_DT}" gravity="0 0 -9.81" collision="all"/>
    <default>
        <joint type="hinge" limited="true"/>
        <motor ctrllimited="true" ctrlrange="-1 1" gear="100"/>
        <geom contype="1" conaffinity="1" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1" contype="1" conaffinity="1"/>
        <body name="base" pos="0 0 0">
            <geom type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" axis="0 0 1" pos="0 0 0.1" range="{JOINT_LIMITS[0, 0]} {JOINT_LIMITS[0, 1]}"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1" type="cylinder" size="0.04 0.18" mass="0.8" rgba="0 0.8 0 0.8"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[1, 0]} {JOINT_LIMITS[1, 1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2" type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                    <joint name="joint3" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[2, 0]} {JOINT_LIMITS[2, 1]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3" type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                        <joint name="joint4" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[3, 0]} {JOINT_LIMITS[3, 1]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4" type="cylinder" size="0.04 0.18" mass="0.4" rgba="0 0.8 0 0.8"/>
                            <joint name="joint5" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[4, 0]} {JOINT_LIMITS[4, 1]}"/>
                            <body name="ee" pos="0 0 0.18">
                                <geom name="ee_geom" type="sphere" size="0.04" mass="{self.load_set}" rgba="0.8 0.2 0.2 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
        <!-- 障碍物（用于碰撞检测测试） -->
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
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            log("MuJoCo模型初始化成功（含碰撞检测）")
            return model, data
        except Exception as e:
            log(f"MuJoCo初始化失败: {e}")
            global RUNNING, EMERGENCY_STOP
            RUNNING = False
            EMERGENCY_STOP = True
            return None, None

    def _get_id(self, obj_type, name):
        type_map = {'joint': mujoco.mjtObj.mjOBJ_JOINT, 'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
                    'geom': mujoco.mjtObj.mjOBJ_GEOM, 'sensor': mujoco.mjtObj.mjOBJ_SENSOR}
        try:
            return mujoco.mj_name2id(self.model, type_map[obj_type], name)
        except:
            return -1

    def get_states(self):
        if self.data is None:
            return np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT)
        qpos = np.array([self.data.qpos[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        return qpos, qvel

    def _calc_load(self):
        if self.data is None:
            return 0.0
        forces = np.abs([self.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qpos, _ = self.get_states()
        load = np.sum(forces * np.sin(qpos)) / 9.81
        return np.clip(load, MIN_LOAD, MAX_LOAD)

    def _detect_collision(self):
        """碰撞检测（新增核心功能）"""
        global COLLISION_DETECTED, PAUSED

        if self.data is None:
            return False

        # 1. 距离检测
        collision = False
        ee_pos = self.data.geom_xpos[self.ee_id] if self.ee_id >= 0 else np.zeros(3)

        # 检查与障碍物的距离
        obstacle_ids = [self._get_id('geom', 'obstacle1'), self._get_id('geom', 'obstacle2')]
        for obs_id in obstacle_ids:
            if obs_id >= 0:
                obs_pos = self.data.geom_xpos[obs_id]
                dist = np.linalg.norm(ee_pos - obs_pos)
                if dist < COLLISION_THRESHOLD:
                    collision = True
                    log(f"碰撞检测: 末端执行器与障碍物距离过近 ({dist:.4f}m < {COLLISION_THRESHOLD}m)")
                    break

        # 2. 接触力检测
        contact_forces = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, 0, contact_forces)
        max_force = np.max(np.abs(contact_forces[:3]))
        if max_force > COLLISION_FORCE_THRESHOLD:
            collision = True
            log(f"碰撞检测: 接触力过大 ({max_force:.2f}N > {COLLISION_FORCE_THRESHOLD}N)")

        # 3. 自碰撞检测（连杆间）
        for i in range(len(self.link_geom_ids)):
            for j in range(i + 1, len(self.link_geom_ids)):
                if self.link_geom_ids[i] >= 0 and self.link_geom_ids[j] >= 0:
                    pos1 = self.data.geom_xpos[self.link_geom_ids[i]]
                    pos2 = self.data.geom_xpos[self.link_geom_ids[j]]
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < 0.005:  # 5mm
                        collision = True
                        log(f"碰撞检测: 连杆{i + 1}与连杆{j + 1}发生自碰撞 ({dist:.4f}m)")
                        break

        # 碰撞处理
        if collision and not COLLISION_DETECTED:
            COLLISION_DETECTED = True
            PAUSED = True
            log("⚠️ 碰撞检测触发！已暂停运动")
            # 可选：触发紧急停止
            # self.emergency_stop()
        elif not collision:
            COLLISION_DETECTED = False

        return collision

    def _record_data(self):
        """数据记录（新增）"""
        if not self.record_enabled or self.data is None:
            return

        self.record_count += 1
        # 每10步记录一次，避免数据量过大
        if self.record_count % 10 != 0:
            return

        qpos, qvel = self.get_states()
        torque = np.array([self.data.ctrl[mid] if mid >= 0 else 0.0 for mid in self.motor_ids])

        self.data_recorder['time'].append(time.time())
        self.data_recorder['qpos'].append(qpos.copy())
        self.data_recorder['qvel'].append(qvel.copy())
        self.data_recorder['err'].append(self.err.copy())
        self.data_recorder['load'].append(self.load_actual)
        self.data_recorder['stiffness'].append(self.stiffness.copy())
        self.data_recorder['torque'].append(torque.copy())
        self.data_recorder['collision'].append(COLLISION_DETECTED)

    def _save_recorded_data(self, name="run_data"):
        """保存记录的数据（新增）"""
        try:
            # 转换为numpy数组
            data = {
                'time': np.array(self.data_recorder['time']),
                'qpos': np.array(self.data_recorder['qpos']),
                'qvel': np.array(self.data_recorder['qvel']),
                'err': np.array(self.data_recorder['err']),
                'load': np.array(self.data_recorder['load']),
                'stiffness': np.array(self.data_recorder['stiffness']),
                'torque': np.array(self.data_recorder['torque']),
                'collision': np.array(self.data_recorder['collision'])
            }

            # 保存为npz文件
            np.savez(f"data/{name}.npz", **data)
            log(f"记录数据已保存: data/{name}.npz")

            # 重置记录器
            self.data_recorder = {k: [] for k in self.data_recorder.keys()}
            self.record_count = 0
        except Exception as e:
            log(f"保存记录数据失败: {e}")

    def _plot_data(self, name="run_plot"):
        """实时数据可视化（新增）"""
        try:
            if len(self.data_recorder['time']) < 10:
                log("数据量不足，无法绘图")
                return

            # 转换数据
            time = np.array(self.data_recorder['time'])
            time -= time[0]  # 相对时间
            qpos = np.array(self.data_recorder['qpos'])
            err = np.array(self.data_recorder['err'])
            load = np.array(self.data_recorder['load'])

            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('机械臂运行数据', fontsize=14)

            # 关节角度
            for i in range(JOINT_COUNT):
                ax1.plot(time, rad2deg(qpos[:, i]), label=f'关节{i + 1}')
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('角度 (°)')
            ax1.legend()
            ax1.grid(True)

            # 跟踪误差
            for i in range(JOINT_COUNT):
                ax2.plot(time, rad2deg(np.abs(err[:, i])), label=f'关节{i + 1}')
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('误差 (°)')
            ax2.legend()
            ax2.grid(True)

            # 负载变化
            ax3.plot(time, load)
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('负载 (kg)')
            ax3.grid(True)

            # 碰撞事件
            collision = np.array(self.data_recorder['collision'])
            collision_times = time[collision]
            ax4.scatter(collision_times, np.ones_like(collision_times), color='red', label='碰撞事件')
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('碰撞检测')
            ax4.legend()
            ax4.grid(True)

            # 保存图表
            plt.tight_layout()
            plt.savefig(f"data/{name}.png", dpi=150)
            plt.close()
            log(f"数据可视化图表已保存: data/{name}.png")
        except Exception as e:
            log(f"绘图失败: {e}")

    def control_step(self):
        """增强控制步（含碰撞检测+数据记录）"""
        global PAUSED, EMERGENCY_STOP

        # 急停/暂停处理
        if EMERGENCY_STOP:
            if self.data is not None:
                self.data.ctrl[:] = 0.0
            return
        if PAUSED:
            self.saved_idx = self.traj_idx
            return

        # 频率限流
        now = time.time()
        if now - self.last_ctrl < CTRL_DT:
            return

        # 碰撞检测
        self._detect_collision()

        # 状态获取
        qpos, qvel = self.get_states()
        self.load_actual = self._calc_load()

        # 轨迹队列处理（新增）
        if len(self.traj_queue) > 0 and self.traj_idx >= len(self.traj_pos):
            self.current_queue_idx += 1
            if self.current_queue_idx < len(self.traj_queue):
                self.traj_pos, self.traj_vel = self.traj_queue[self.current_queue_idx]
                self.target = self.traj_pos[-1] if len(self.traj_pos) > 0 else np.zeros(JOINT_COUNT)
                self.traj_idx = 0
                log(f"执行轨迹队列第{self.current_queue_idx + 1}/{len(self.traj_queue)}段")
            else:
                # 队列执行完毕
                self.traj_queue = []
                self.current_queue_idx = 0
                log("轨迹队列执行完毕")

        # 目标点获取
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

        # PD+前馈控制
        load_factor = np.clip(self.load_actual / MAX_LOAD, 0.0, 1.0)
        kp = Cfg.kp_base * (1 + load_factor * (Cfg.kp_load_gain - 1))
        kd = Cfg.kd_base * (1 + load_factor * (Cfg.kd_load_gain - 1))

        pd = kp * self.err + kd * (target_vel - qvel)
        ff = Cfg.ff_vel * target_vel + Cfg.ff_acc * (target_vel - qvel) / CTRL_DT

        # 误差补偿
        vel_sign = np.sign(qvel)
        vel_zero = np.abs(qvel) < 1e-4
        vel_sign[vel_zero] = np.sign(self.err)[vel_zero]
        backlash = Cfg.backlash * vel_sign
        friction = np.where(vel_zero, Cfg.friction * np.sign(self.err), 0.0)
        gravity = 0.5 * np.sin(qpos) * self.load_actual if Cfg.gravity_comp else 0.0
        comp = backlash + friction + gravity

        # 控制输出
        ctrl = pd + ff + comp
        ctrl = np.clip(ctrl, -MAX_TORQUE, MAX_TORQUE)

        # 应用控制
        for i, mid in enumerate(self.motor_ids):
            if mid >= 0:
                self.data.ctrl[mid] = ctrl[i]

        # 自适应刚度阻尼
        load_ratio = np.clip(self.load_actual / MAX_LOAD, 0.0, 1.0)
        err_norm = np.clip(np.abs(self.err) / deg2rad(1.0), 0.0, 1.0)
        target_stiff = Cfg.stiffness_base * (1 + load_ratio * (Cfg.stiffness_load_gain - 1)) * (
                    1 + err_norm * (Cfg.stiffness_error_gain - 1))
        target_stiff = np.clip(target_stiff, Cfg.stiffness_min, Cfg.stiffness_max)
        self.stiffness = 0.95 * self.stiffness + 0.05 * target_stiff
        self.damping = self.stiffness * Cfg.damping_ratio
        self.damping = np.clip(self.damping, Cfg.stiffness_min * 0.02, Cfg.stiffness_max * 0.08)

        for i, jid in enumerate(self.joint_ids):
            if jid >= 0:
                self.model.jnt_damping[jid] = self.damping[i]

        # 数据记录
        self._record_data()

        self.last_ctrl = now

    # ====================== 增强控制接口 ======================
    def move_to(self, target_deg, save=False, name="default", smooth=True):
        with lock():
            target_deg = np.asarray(target_deg, np.float64)
            if target_deg.shape != (JOINT_COUNT,):
                log(f"目标维度错误: {target_deg.shape}")
                return

            start, _ = self.get_states()
            target = deg2rad(target_deg)
            self.traj_pos, self.traj_vel = plan_trajectory(start, target, smooth=smooth)
            self.target = target
            self.traj_idx = 0

            if save:
                save_traj(self.traj_pos, self.traj_vel, name)
            log(f"规划轨迹: {np.round(rad2deg(start), 1)}° → {np.round(rad2deg(target), 1)}° (平滑: {smooth})")

    def add_to_traj_queue(self, target_deg_list, smooth=True):
        """添加多段轨迹到队列（新增）"""
        with lock():
            if not isinstance(target_deg_list, list) or len(target_deg_list) == 0:
                log("轨迹队列参数错误")
                return

            start, _ = self.get_states()
            self.traj_queue = []

            for i, target_deg in enumerate(target_deg_list):
                target_deg = np.asarray(target_deg, np.float64)
                if target_deg.shape != (JOINT_COUNT,):
                    log(f"队列第{i + 1}段轨迹参数错误，跳过")
                    continue

                target = deg2rad(target_deg)
                traj_pos, traj_vel = plan_trajectory(start, target, smooth=smooth)
                self.traj_queue.append((traj_pos, traj_vel))
                start = target  # 下一段的起点是当前段的终点

            self.current_queue_idx = 0
            # 加载第一段轨迹
            if len(self.traj_queue) > 0:
                self.traj_pos, self.traj_vel = self.traj_queue[0]
                self.target = self.traj_pos[-1]
                self.traj_idx = 0

            log(f"轨迹队列已添加 {len(self.traj_queue)} 段轨迹")

    def clear_traj_queue(self):
        """清空轨迹队列（新增）"""
        with lock():
            self.traj_queue = []
            self.current_queue_idx = 0
            log("轨迹队列已清空")

    def control_joint(self, idx, target_deg, smooth=True):
        if not (0 <= idx < JOINT_COUNT):
            log(f"无效关节索引: {idx}")
            return

        current, _ = self.get_states()
        target = current.copy()
        target[idx] = deg2rad(target_deg)
        self.traj_pos, self.traj_vel = plan_trajectory(current, target, smooth=smooth)
        self.target = target
        self.traj_idx = 0
        log(f"控制关节{idx + 1}: {np.round(rad2deg(current[idx]), 1)}° → {target_deg:.1f}° (平滑: {smooth})")

    def set_load(self, mass):
        with lock():
            if not (MIN_LOAD <= mass <= MAX_LOAD):
                log(f"负载超出范围: {mass}kg (0-2kg)")
                return

            self.load_set = mass
            if self.ee_id >= 0 and self.model is not None:
                self.model.geom_mass[self.ee_id] = mass
            log(f"负载设置为: {mass}kg")

    def pause(self):
        global PAUSED
        with lock():
            PAUSED = True
            log("轨迹暂停")

    def resume(self):
        global PAUSED
        with lock():
            PAUSED = False
            self.traj_idx = self.saved_idx
            log(f"轨迹恢复（第{self.saved_idx}步）")

    def emergency_stop(self):
        global RUNNING, PAUSED, EMERGENCY_STOP
        with lock():
            EMERGENCY_STOP = True
            PAUSED = True
            RUNNING = False
            log("⚠️ 紧急停止触发")

    def reset_collision(self):
        """重置碰撞检测状态（新增）"""
        global COLLISION_DETECTED, PAUSED
        with lock():
            COLLISION_DETECTED = False
            PAUSED = False
            log("碰撞检测状态已重置，可恢复运动")

    def adjust_param(self, param, value, idx=None):
        with lock():
            if not hasattr(Cfg, param):
                log(f"无效参数: {param}")
                return

            val = getattr(Cfg, param)
            if isinstance(val, np.ndarray):
                if idx is None:
                    setattr(Cfg, param, np.full(JOINT_COUNT, value))
                    log(f"参数 {param} 全部更新为: {value}")
                elif 0 <= idx < JOINT_COUNT:
                    val[idx] = value
                    setattr(Cfg, param, val)
                    log(f"参数 {param} 关节{idx + 1}更新为: {value}")
                else:
                    log(f"无效索引: {idx}")
            else:
                setattr(Cfg, param, value)
                log(f"参数 {param} 更新为: {value}")

    def load_trajectory(self, name, smooth=True):
        with lock():
            traj_pos, traj_vel = load_traj(name)
            if len(traj_pos) == 0:
                return

            # 可选平滑
            if smooth:
                traj_pos, traj_vel = smooth_trajectory(traj_pos, traj_vel)

            self.traj_pos = traj_pos
            self.traj_vel = traj_vel
            self.target = traj_pos[-1] if len(traj_pos) > 0 else np.zeros(JOINT_COUNT)
            self.traj_idx = 0
            log(f"加载轨迹: {name} (共{len(traj_pos)}步，平滑: {smooth})")

    def preset_pose(self, pose):
        poses = {
            'zero': [0, 0, 0, 0, 0],
            'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10],
            'test': [10, 20, 15, 5, 8],
            'avoid': [15, 25, 10, 5, 0]  # 避障姿态（新增）
        }
        if pose in poses:
            self.move_to(poses[pose])
        else:
            log(f"未知姿态: {pose} (支持: {list(poses.keys())})")

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
        """增强状态打印（含碰撞+队列信息）"""
        now = time.time()
        if now - self.last_print < 1.0:
            return

        fps = self.fps_count / (now - self.last_print)
        qpos, qvel = self.get_states()
        err = rad2deg(self.err)
        max_err = rad2deg(self.max_err)

        status = []
        if PAUSED: status.append("暂停")
        if EMERGENCY_STOP: status.append("紧急停止")
        if COLLISION_DETECTED: status.append("碰撞检测")
        status_str = " | ".join(status) if status else "运行中"

        log("=" * 70)
        log(f"状态: {status_str} | 步数: {self.step} | FPS: {fps:.1f}")
        log(f"负载: {self.load_set:.1f}kg(设定) | {self.load_actual:.1f}kg(实际)")
        log(f"角度: {np.round(rad2deg(qpos), 1)}° | 误差: {np.round(np.abs(err), 3)}°(最大:{np.round(max_err, 3)}°)")
        log(f"轨迹队列: {len(self.traj_queue)}段 | 当前段: {self.current_queue_idx + 1 if self.traj_queue else 0}")
        log("=" * 70)

        self.last_print = now
        self.fps_count = 0

    def _interactive(self):
        """增强交互线程（支持新功能命令）"""
        help_text = """
增强版命令列表：
  help          - 查看帮助
  pause/resume  - 暂停/恢复
  stop          - 紧急停止
  reset_collision - 重置碰撞检测
  pose [名称]   - 预设姿态(zero/up/grasp/test/avoid)
  joint [索引] [角度] - 控制单个关节
  load [kg]     - 设置负载(0-2kg)
  param [名] [值] [关节] - 调整参数
  save [名]     - 保存轨迹
  load_traj [名] - 加载轨迹
  queue [姿态1,姿态2,...] - 添加轨迹队列
  clear_queue   - 清空轨迹队列
  record_start  - 开始数据记录
  record_stop   - 停止数据记录并保存
  save_params [名] - 保存控制参数
  load_params [名] - 加载控制参数
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
                elif parts[0] == 'reset_collision':
                    self.reset_collision()
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
                    self.move_to(rad2deg(self.target), save=True, name=parts[1])
                elif parts[0] == 'load_traj' and len(parts) == 2:
                    self.load_trajectory(parts[1])
                elif parts[0] == 'queue' and len(parts) > 1:
                    # 解析队列参数，如: queue zero up grasp
                    pose_list = parts[1:]
                    target_list = []
                    poses = {'zero': [0, 0, 0, 0, 0], 'up': [0, 30, 20, 10, 0],
                             'grasp': [0, 45, 30, 20, 10], 'test': [10, 20, 15, 5, 8],
                             'avoid': [15, 25, 10, 5, 0]}
                    for pose in pose_list:
                        if pose in poses:
                            target_list.append(poses[pose])
                    if target_list:
                        self.add_to_traj_queue(target_list)
                    else:
                        log("无效的轨迹队列参数")
                elif parts[0] == 'clear_queue':
                    self.clear_traj_queue()
                elif parts[0] == 'record_start':
                    self.start_recording()
                elif parts[0] == 'record_stop':
                    self.stop_recording()
                elif parts[0] == 'save_params' and len(parts) >= 2:
                    save_params(parts[1])
                elif parts[0] == 'load_params' and len(parts) >= 2:
                    load_params(parts[1])
                else:
                    log("未知命令，输入help查看帮助")
            except:
                continue

    def run(self):
        """增强主运行循环"""
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
        threading.Thread(target=self._interactive, daemon=True).start()

        # 增强演示程序（含碰撞测试）
        def demo():
            # 初始参数保存
            save_params("default")

            # 开始数据记录
            self.start_recording()

            steps = [
                (2, 'pose', 'zero'),
                (3, 'pose', 'test'),
                (2, 'pause', None),
                (2, 'resume', None),
                (4, 'load', 1.5),
                (4, 'pose', 'grasp'),  # 可能触发碰撞
                (1, 'reset_collision', None),
                (1, 'joint', (0, 10)),
                (3, 'load', 0.2),
                (3, 'queue', ['zero', 'up', 'avoid', 'grasp']),  # 轨迹队列
                (8, 'record_stop', None),
                (2, 'pose', 'zero'),
                (2, 'stop', None)
            ]

            for delay, action, param in steps:
                time.sleep(delay)
                if not RUNNING:
                    break
                if action == 'pose':
                    self.preset_pose(param)
                elif action == 'load':
                    self.set_load(param)
                elif action == 'pause':
                    self.pause()
                elif action == 'resume':
                    self.resume()
                elif action == 'joint':
                    self.control_joint(*param)
                elif action == 'stop':
                    self.emergency_stop()
                elif action == 'reset_collision':
                    self.reset_collision()
                elif action == 'queue':
                    self.add_to_traj_queue(
                        [[0, 0, 0, 0, 0], [0, 30, 20, 10, 0], [15, 25, 10, 5, 0], [0, 45, 30, 20, 10]])
                elif action == 'record_stop':
                    self.stop_recording()

        threading.Thread(target=demo, daemon=True).start()

        # 主循环
        log("增强版机械臂控制器启动 (Ctrl+C退出)")
        log("新增功能：碰撞检测、轨迹平滑、轨迹队列、数据记录与可视化、参数保存/加载")
        while RUNNING and self.viewer.is_running():
            try:
                self.step += 1
                self.fps_count += 1

                self.control_step()
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                self._print_status()

                time.sleep(SLEEP_DT)
            except Exception as e:
                log(f"运行错误: {e}")
                continue

        # 资源清理
        if self.viewer:
            self.viewer.close()
        self.traj_pos = np.array([])
        self.traj_vel = np.array([])
        TRAJ_CACHE.clear()

        # 最终数据保存
        if self.record_enabled:
            self.stop_recording()

        max_err = rad2deg(np.max(self.max_err))
        log(f"控制器停止 | 总步数: {self.step} | 最大误差: {np.round(max_err, 3)}°")


# ====================== 信号处理与主函数 ======================
def signal_handler(sig, frame):
    global RUNNING, EMERGENCY_STOP
    if RUNNING:
        log("收到退出信号，正在停止...")
        RUNNING = False
        EMERGENCY_STOP = True


def main():
    np.set_printoptions(precision=3, suppress=True)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        controller = ArmController()
        controller.run()
    except Exception as e:
        log(f"程序异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()