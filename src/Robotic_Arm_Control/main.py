#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂控制器（最终精简优化版）
核心特性：极致性能 + 极简架构 + 完整功能 + 工业级鲁棒性
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
from collections import deque

# ====================== 常量定义（只读+预计算） ======================
# 硬件参数
JOINT_COUNT = 5
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
MIN_LOAD, MAX_LOAD = 0.0, 2.0

# 关节极限（预计算）
JOINT_LIMITS = np.array([
    [-np.pi, np.pi], [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2],
    [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2]
], dtype=np.float64)
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
COLLISION_THRESHOLD = 0.01
COLLISION_FORCE_THRESHOLD = 5.0

# 目录配置（预创建）
DIRS = {name: Path(name) for name in ["trajectories", "params", "logs", "data"]}
for dir_path in DIRS.values():
    dir_path.mkdir(exist_ok=True)


# ====================== 配置类（极简+校验） ======================
@dataclass
class ControlConfig:
    # 基础控制
    kp_base: float = 120.0
    kd_base: float = 8.0
    kp_load_gain: float = 1.8
    kd_load_gain: float = 1.5
    ff_vel: float = 0.7
    ff_acc: float = 0.5

    # 误差补偿
    backlash: np.ndarray = np.array([0.001, 0.001, 0.002, 0.002, 0.003])
    friction: np.ndarray = np.array([0.1, 0.08, 0.08, 0.06, 0.06])
    gravity_comp: bool = True

    # 刚度阻尼
    stiffness_base: np.ndarray = np.array([200.0, 180.0, 150.0, 120.0, 80.0])
    stiffness_load_gain: float = 1.8
    stiffness_error_gain: float = 1.5
    stiffness_min: np.ndarray = np.array([100.0, 90.0, 75.0, 60.0, 40.0])
    stiffness_max: np.ndarray = np.array([300.0, 270.0, 225.0, 180.0, 120.0])
    damping_ratio: float = 0.04

    # 轨迹平滑
    smooth_factor: float = 0.1
    jerk_limit: np.ndarray = np.array([10.0, 8.0, 8.0, 6.0, 6.0])

    def to_dict(self):
        """序列化为字典"""
        data = {}
        for k, v in self.__dict__.items():
            data[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return data

    @classmethod
    def from_dict(cls, data):
        """从字典加载并校验"""
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                if k in ['backlash', 'friction', 'stiffness_base', 'stiffness_min',
                         'stiffness_max', 'jerk_limit']:
                    setattr(cfg, k, np.array(v, dtype=np.float64))
                else:
                    setattr(cfg, k, v)
        return cfg

            # 写入特殊级别日志
            if level in log_files and level != "INFO":
                with open(log_files[level], "a", encoding="utf-8") as f:
                    f.write(log_msg + "\n")

# 全局配置（单例）
CFG = ControlConfig()

    @classmethod
    def deg2rad(cls, x):
        """角度转弧度（向量化+异常安全）"""
        try:
            return np.asarray(x, np.float64) * DEG2RAD
        except:
            return np.zeros(JOINT_COUNT) if isinstance(x, (list, np.ndarray)) else 0.0

# ====================== 工具类（极简+高效） ======================
class Utils:
    _lock = threading.Lock()
    _perf = deque(maxlen=1000)  # 性能记录

    @classmethod
    @contextmanager
    def lock(cls):
        with cls._lock:
            yield

    @classmethod
    def log(cls, msg, level="INFO"):
        """极简日志系统"""
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            log_msg = f"[{ts}] [{level}] {msg}"
            print(log_msg)

            # 分级保存
            log_file = DIRS["logs"] / f"arm_{level.lower()}.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
        except:
            pass

    @classmethod
    def deg2rad(cls, x):
        """向量化角度转弧度"""
        try:
            return np.asarray(x, np.float64) * DEG2RAD
        except:
            return np.zeros(JOINT_COUNT) if isinstance(x, (list, np.ndarray)) else 0.0

    @classmethod
    def rad2deg(cls, x):
        """向量化弧度转角度"""
        try:
            return np.asarray(x, np.float64) * RAD2DEG
        except:
            return np.zeros(JOINT_COUNT) if isinstance(x, (list, np.ndarray)) else 0.0

    @classmethod
    def perf(cls, func):
        """性能装饰器"""

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            cls._perf.append(time.time() - start)
            return result

        return wrapper


# ====================== 核心模块（极简封装） ======================
class Trajectory:
    """轨迹规划（极致优化）"""
    _cache = {}
    _cache_max = 100

    @classmethod
    @Utils.perf
    def plan(cls, start, target, smooth=True):
        """梯形轨迹规划"""
        # 边界裁剪
        start = np.clip(start, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)
        target = np.clip(target, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)

        # 缓存检查
        cache_key = (start.tobytes(), target.tobytes(), smooth)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # 预分配内存
        traj_pos = np.zeros((0, JOINT_COUNT))
        traj_vel = np.zeros((0, JOINT_COUNT))
        max_len = 1

        # 批量规划
        for i in range(JOINT_COUNT):
            delta = target[i] - start[i]
            if abs(delta) < 1e-5:
                pos, vel = np.array([target[i]]), np.array([0.0])
            else:
                dir = np.sign(delta)
                dist = abs(delta)
                accel_dist = (MAX_VEL[i] ** 2) / (2 * MAX_ACC[i])

                # 时间计算
                if dist <= 2 * accel_dist:
                    peak_vel = np.sqrt(dist * MAX_ACC[i])
                    accel_time = peak_vel / MAX_ACC[i]
                    total_time = 2 * accel_time
                else:
                    accel_time = MAX_VEL[i] / MAX_ACC[i]
                    uniform_time = (dist - 2 * accel_dist) / MAX_VEL[i]
                    total_time = 2 * accel_time + uniform_time

                # 时间序列
                t = np.arange(0, total_time + CTRL_DT, CTRL_DT)
                pos = np.empty_like(t)
                vel = np.empty_like(t)

                # 向量化分段计算
                mask_acc = t <= accel_time
                mask_uni = (t > accel_time) & (
                            t <= accel_time + uniform_time) if dist > 2 * accel_dist else np.zeros_like(t, bool)
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

            # 扩展数组
            if len(traj_pos) < len(pos):
                traj_pos = np.pad(traj_pos, ((0, len(pos) - len(traj_pos)), (0, 0)), 'constant')
                traj_vel = np.pad(traj_vel, ((0, len(pos) - len(traj_vel)), (0, 0)), 'constant')

            traj_pos[:len(pos), i] = pos
            traj_vel[:len(pos), i] = vel
            max_len = max(max_len, len(pos))

        # 统一长度
        if len(traj_pos) < max_len:
            pad = max_len - len(traj_pos)
            traj_pos = np.pad(traj_pos, ((0, pad), (0, 0)), 'constant', constant_values=target)
            traj_vel = np.pad(traj_vel, ((0, pad), (0, 0)), 'constant')

        # 轨迹平滑
        if smooth:
            traj_pos, traj_vel = cls.smooth(traj_pos, traj_vel)

        # 缓存管理
        if len(cls._cache) >= cls._cache_max:
            cls._cache.pop(next(iter(cls._cache)))
        cls._cache[cache_key] = (traj_pos, traj_vel)

        return traj_pos, traj_vel

    @classmethod
    def smooth(cls, traj_pos, traj_vel):
        """轨迹平滑（纯向量化）"""
        if len(traj_pos) <= 2:
            return traj_pos, traj_vel

        # 低通滤波
        smooth_pos = np.empty_like(traj_pos)
        smooth_pos[0] = traj_pos[0]
        alpha = 1 - CFG.smooth_factor
        smooth_pos[1:] = alpha * smooth_pos[:-1] + CFG.smooth_factor * traj_pos[1:]

        # 速度计算
        smooth_vel = np.empty_like(traj_vel)
        smooth_vel[0] = 0.0
        vel_diff = (smooth_pos[1:] - smooth_pos[:-1]) / CTRL_DT
        smooth_vel[1:] = np.clip(vel_diff, -MAX_VEL, MAX_VEL)

        # 加加速度限制
        if len(smooth_vel) > 2:
            jerk = (smooth_vel[2:] - smooth_vel[1:-1]) / CTRL_DT
            jerk_clipped = np.clip(jerk, -CFG.jerk_limit, CFG.jerk_limit)
            smooth_vel[2:] = smooth_vel[1:-1] + jerk_clipped * CTRL_DT

        return smooth_pos, smooth_vel

    @classmethod
    def save(cls, traj_pos, traj_vel, name):
        """保存轨迹"""
        try:
            header = ['step'] + [f'j{i + 1}_pos' for i in range(JOINT_COUNT)] + [f'j{i + 1}_vel' for i in
                                                                                 range(JOINT_COUNT)]
            data = np.hstack([np.arange(len(traj_pos))[:, None], traj_pos, traj_vel])
            np.savetxt(DIRS["trajectories"] / f"{name}.csv", data, delimiter=',', header=','.join(header), comments='')
            Utils.log(f"轨迹已保存: {name}.csv")
        except Exception as e:
            Utils.log(f"保存轨迹失败: {e}", "ERROR")

    @classmethod
    def load(cls, name):
        """加载轨迹"""
        try:
            data = np.genfromtxt(DIRS["trajectories"] / f"{name}.csv", delimiter=',', skip_header=1)
            if len(data) == 0:
                return np.array([]), np.array([])
            return data[:, 1:JOINT_COUNT + 1], data[:, JOINT_COUNT + 1:]
        except Exception as e:
            Utils.log(f"加载轨迹失败: {e}", "ERROR")
            return np.array([]), np.array([])


class Collision:
    """碰撞检测（极简高效）"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.detected = False

    def check(self, ee_id, link_ids):
        """检测碰撞"""
        self.detected = False

        # 末端距离检测
        if ee_id >= 0:
            ee_pos = self.data.geom_xpos[ee_id]
            for obs_name in ['obstacle1', 'obstacle2']:
                obs_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obs_name)
                if obs_id >= 0 and np.linalg.norm(ee_pos - self.data.geom_xpos[obs_id]) < COLLISION_THRESHOLD:
                    self.detected = True
                    Utils.log(f"碰撞: 末端与{obs_name}距离过近", "COLLISION")
                    return True

        # 接触力检测
        forces = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, 0, forces)
        if np.max(np.abs(forces[:3])) > COLLISION_FORCE_THRESHOLD:
            self.detected = True
            Utils.log(f"碰撞: 接触力超限", "COLLISION")
            return True

        # 自碰撞检测
        valid_links = [lid for lid in link_ids if lid >= 0]
        if len(valid_links) > 1:
            link_pos = self.data.geom_xpos[valid_links]
            dist_mat = np.linalg.norm(link_pos[:, None] - link_pos, axis=2)
            np.fill_diagonal(dist_mat, np.inf)
            if np.min(dist_mat) < 0.005:
                self.detected = True
                Utils.log(f"碰撞: 连杆自碰撞", "COLLISION")
                return True

        return self.detected


class Recorder:
    """数据记录（极简）"""

    def __init__(self):
        self.enabled = False
        self.data = {k: [] for k in ['time', 'qpos', 'qvel', 'err', 'load', 'collision']}

    def start(self):
        self.enabled = True
        self.data = {k: [] for k in self.data.keys()}
        Utils.log("开始数据记录")

    def stop(self, save=True, plot=True):
        self.enabled = False
        Utils.log("停止数据记录")
        if save and len(self.data['time']) > 10:
            self._save()
            if plot:
                self._plot()

    def record(self, qpos, qvel, err, load, collision):
        if not self.enabled:
            return
        self.data['time'].append(time.time())
        self.data['qpos'].append(qpos.copy())
        self.data['qvel'].append(qvel.copy())
        self.data['err'].append(err.copy())
        self.data['load'].append(load)
        self.data['collision'].append(collision)

    def _save(self):
        """保存数据"""
        try:
            ts = time.strftime('%Y%m%d_%H%M%S')
            data = {k: np.array(v) for k, v in self.data.items()}
            np.savez(DIRS["data"] / f"run_{ts}.npz", **data)
            Utils.log(f"数据已保存: run_{ts}.npz")
        except Exception as e:
            Utils.log(f"保存数据失败: {e}", "ERROR")

    def _plot(self):
        """极简可视化"""
        try:
            ts = time.strftime('%Y%m%d_%H%M%S')
            time_arr = np.array(self.data['time'])
            time_arr -= time_arr[0]

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

            # 关节角度
            qpos = np.array(self.data['qpos'])
            for i in range(JOINT_COUNT):
                ax1.plot(time_arr, Utils.rad2deg(qpos[:, i]), label=f'关节{i + 1}')
            ax1.set(xlabel='时间(s)', ylabel='角度(°)', title='关节角度')
            ax1.legend()

            # 跟踪误差
            err = np.array(self.data['err'])
            for i in range(JOINT_COUNT):
                ax2.plot(time_arr, Utils.rad2deg(np.abs(err[:, i])), label=f'关节{i + 1}')
            ax2.set(xlabel='时间(s)', ylabel='误差(°)', title='跟踪误差')

            # 负载
            ax3.plot(time_arr, self.data['load'])
            ax3.set(xlabel='时间(s)', ylabel='负载(kg)', title='负载变化')

            # 碰撞
            collision = np.array(self.data['collision'])
            ax4.scatter(time_arr[collision], np.ones(sum(collision)), c='r', label='碰撞')
            ax4.set(xlabel='时间(s)', ylabel='碰撞', title='碰撞事件')

            plt.tight_layout()
            plt.savefig(DIRS["data"] / f"plot_{ts}.png", dpi=120)
            plt.close()
            Utils.log(f"可视化已保存: plot_{ts}.png")
        except Exception as e:
            Utils.log(f"绘图失败: {e}", "ERROR")


# ====================== 核心控制器（终极精简） ======================
class ArmController:
    def __init__(self):
        # 状态控制
        self.running = True
        self.paused = False
        self.estop = False

        # 初始化MuJoCo
        self.model, self.data = self._init_mujoco()

        # 核心组件
        self.collision = Collision(self.model, self.data) if self.model else None
        self.recorder = Recorder()

        # ID缓存
        self._init_ids()

        # 控制状态（预分配）
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.target = np.zeros(JOINT_COUNT)
        self.traj_queue = deque()

        # 物理状态
        self.stiffness = CFG.stiffness_base.copy()
        self.damping = self.stiffness * CFG.damping_ratio
        self.load = 0.5
        self.err = np.zeros(JOINT_COUNT)
        self.max_err = np.zeros(JOINT_COUNT)

        # 性能统计
        self.step = 0
        self.last_ctrl = time.time()
        self.last_status = time.time()

        # Viewer
        self.viewer = None

        # Viewer
        self.viewer = None

    def _init_ids(self):
        """ID初始化（预计算）"""
        if self.model is None:
            self.joint_ids = [-1] * JOINT_COUNT
            self.motor_ids = [-1] * JOINT_COUNT
            self.ee_id = -1
            self.link_geom_ids = [-1] * JOINT_COUNT
            return

        # 批量获取ID
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i + 1}')
                          for i in range(JOINT_COUNT)]
        self.motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i + 1}')
                          for i in range(JOINT_COUNT)]
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ee_geom')
        self.link_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'link{i + 1}')
                              for i in range(JOINT_COUNT)]

    def _init_mujoco(self):
        """极简MuJoCo初始化"""
        xml = f"""
<mujoco model="arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{sim_dt}" gravity="0 0 -9.81" collision="all"/>
    <default>
        <joint type="hinge" limited="true"/>
        <motor ctrllimited="true" ctrlrange="-1 1" gear="100"/>
        <geom contype="1" conaffinity="1"/>
    </default>
    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="base" pos="0 0 0">
            <geom type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" axis="0 0 1" pos="0 0 0.1" range="{j1_min} {j1_max}"/>
            <body name="link1" pos="0 0 0.1">
                <geom name="link1" type="cylinder" size="0.04 0.18" mass="0.8" rgba="0 0.8 0 0.8"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18" range="{j2_min} {j2_max}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom name="link2" type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                    <joint name="joint3" axis="0 1 0" pos="0 0 0.18" range="{j3_min} {j3_max}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom name="link3" type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                        <joint name="joint4" axis="0 1 0" pos="0 0 0.18" range="{j4_min} {j4_max}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom name="link4" type="cylinder" size="0.04 0.18" mass="0.4" rgba="0 0.8 0 0.8"/>
                            <joint name="joint5" axis="0 1 0" pos="0 0 0.18" range="{j5_min} {j5_max}"/>
                            <body name="ee" pos="0 0 0.18">
                                <geom name="ee_geom" type="sphere" size="0.04" mass="{self.load}" rgba="0.8 0.2 0.2 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
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
</mujoco>
        """

        # 模板参数
        xml_params = {
            "sim_dt": SIM_DT,
            "load": self.load_set,
            "j1_min": JOINT_LIMITS[0, 0], "j1_max": JOINT_LIMITS[0, 1],
            "j2_min": JOINT_LIMITS[1, 0], "j2_max": JOINT_LIMITS[1, 1],
            "j3_min": JOINT_LIMITS[2, 0], "j3_max": JOINT_LIMITS[2, 1],
            "j4_min": JOINT_LIMITS[3, 0], "j4_max": JOINT_LIMITS[3, 1],
            "j5_min": JOINT_LIMITS[4, 0], "j5_max": JOINT_LIMITS[4, 1],
        }

        try:
            xml = xml_template.format(**xml_params)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            return model, data
        except Exception as e:
            Utils.log(f"MuJoCo初始化失败: {e}", "ERROR")
            self.estop = True
            self.running = False
            return None, None

    def _init_ids(self):
        """初始化ID"""
        if self.model is None:
            self.joint_ids = self.motor_ids = self.link_ids = [-1] * JOINT_COUNT
            self.ee_id = -1
            return

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i + 1}') for i in
                          range(JOINT_COUNT)]
        self.motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor{i + 1}') for i in
                          range(JOINT_COUNT)]
        self.link_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'link{i + 1}') for i in
                         range(JOINT_COUNT)]
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ee_geom')

    def get_states(self):
        """获取关节状态"""
        if self.data is None:
            return np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT)

        # 批量获取状态
        qpos = np.array([self.data.qpos[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])

        return qpos, qvel

    @Utils.perf
    def control_step(self):
        """核心控制步（极致优化）"""
        # 急停处理
        if self.estop:
            self.data.ctrl[:] = 0.0 if self.data else 0.0
            return

        # 暂停/频率控制
        if self.paused or (time.time() - self.last_ctrl < CTRL_DT):
            return

        # 碰撞检测
        if self.collision and self.collision.check(self.ee_id, self.link_ids) and not self.paused:
            self.paused = True
            Utils.log("碰撞触发，已暂停", "COLLISION")

        # 获取状态
        qpos, qvel = self.get_states()
        load_actual = self._calc_load()

        # 轨迹队列处理
        if len(self.traj_queue) > 0 and self.traj_idx >= len(self.traj_pos):
            self.traj_pos, self.traj_vel = self.traj_queue.popleft()
            self.target = self.traj_pos[-1]
            self.traj_idx = 0

        # 目标位置
        target_pos = self.traj_pos[self.traj_idx] if self.traj_idx < len(self.traj_pos) else self.target
        target_vel = self.traj_vel[self.traj_idx] if self.traj_idx < len(self.traj_pos) else np.zeros(JOINT_COUNT)
        self.traj_idx += 1

        # 误差计算
        self.err = target_pos - qpos
        self.max_err = np.maximum(self.max_err, np.abs(self.err))

        # PD+前馈控制（纯向量化）
        load_factor = np.clip(load_actual / MAX_LOAD, 0.0, 1.0)
        kp = CFG.kp_base * (1 + load_factor * (CFG.kp_load_gain - 1))
        kd = CFG.kd_base * (1 + load_factor * (CFG.kd_load_gain - 1))

        pd = kp * self.err + kd * (target_vel - qvel)
        ff = CFG.ff_vel * target_vel + CFG.ff_acc * (target_vel - qvel) / CTRL_DT

        # 误差补偿（向量化）
        vel_sign = np.sign(qvel)
        vel_zero = np.abs(qvel) < 1e-4
        vel_sign[vel_zero] = np.sign(self.err)[vel_zero]

        backlash = CFG.backlash * vel_sign
        friction = np.where(vel_zero, CFG.friction * np.sign(self.err), 0.0)
        gravity = 0.5 * np.sin(qpos) * load_actual if CFG.gravity_comp else 0.0
        comp = backlash + friction + gravity

        # 控制输出
        ctrl = np.clip(pd + ff + comp, -MAX_TORQUE, MAX_TORQUE)

        # 应用控制（批量）
        valid_motors = [(i, mid) for i, mid in enumerate(self.motor_ids) if mid >= 0]
        for i, mid in valid_motors:
            self.data.ctrl[mid] = ctrl[i]

        # 自适应刚度阻尼
        load_ratio = np.clip(load_actual / MAX_LOAD, 0.0, 1.0)
        err_norm = np.clip(np.abs(self.err) / Utils.deg2rad(1.0), 0.0, 1.0)

        target_stiff = CFG.stiffness_base * (1 + load_ratio * (CFG.stiffness_load_gain - 1)) * (
                    1 + err_norm * (CFG.stiffness_error_gain - 1))
        target_stiff = np.clip(target_stiff, CFG.stiffness_min, CFG.stiffness_max)

        self.stiffness = 0.95 * self.stiffness + 0.05 * target_stiff
        self.damping = np.clip(self.stiffness * CFG.damping_ratio, CFG.stiffness_min * 0.02, CFG.stiffness_max * 0.08)

        # 应用刚度阻尼（批量）
        valid_joints = [(i, jid) for i, jid in enumerate(self.joint_ids) if jid >= 0]
        for i, jid in valid_joints:
            self.model.jnt_damping[jid] = self.damping[i]

        # 数据记录
        self.recorder.record(qpos, qvel, self.err, load_actual, self.collision.detected if self.collision else False)

        self.last_ctrl = time.time()

    def _calc_load(self):
        """计算负载"""
        if self.data is None:
            return 0.0
        forces = np.abs([self.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qpos, _ = self.get_states()
        return np.clip(np.sum(forces * np.sin(qpos)) / 9.81, MIN_LOAD, MAX_LOAD)

    # ====================== 控制接口（极简） ======================
    def move_to(self, target_deg, save=False, name="default"):
        """移动到目标位置"""
        with Utils.lock():
            target = Utils.deg2rad(target_deg)
            start, _ = self.get_states()
            self.traj_pos, self.traj_vel = Trajectory.plan(start, target)
            self.target = target
            self.traj_idx = 0
            if save:
                Trajectory.save(self.traj_pos, self.traj_vel, name)
            Utils.log(f"规划轨迹: {np.round(Utils.rad2deg(start), 1)}° → {np.round(Utils.rad2deg(target), 1)}°")

    def add_queue(self, targets):
        """添加轨迹队列"""
        with Utils.lock():
            start, _ = self.get_states()
            for target_deg in targets:
                target = Utils.deg2rad(target_deg)
                traj_pos, traj_vel = Trajectory.plan(start, target)
                self.traj_queue.append((traj_pos, traj_vel))
                start = target
            Utils.log(f"添加{len(targets)}段轨迹到队列")

    def set_load(self, mass):
        """设置负载"""
        with Utils.lock():
            if not (MIN_LOAD <= mass <= MAX_LOAD):
                Utils.log(f"负载超出范围: {mass}kg", "ERROR")
                return
            self.load = mass
            if self.ee_id >= 0 and self.model:
                self.model.geom_mass[self.ee_id] = mass
            Utils.log(f"负载设置为: {mass}kg")

    def preset_pose(self, pose):
        """预设姿态"""
        poses = {
            'zero': [0, 0, 0, 0, 0],
            'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10],
            'test': [10, 20, 15, 5, 8],
            'avoid': [15, 25, 10, 5, 0]
        }

        if pose_name in poses:
            self.move_to(poses[pose_name])
        else:
            Utils.log(f"未知姿态: {pose}", "ERROR")

    def save_params(self, name="default"):
        """保存参数"""
        try:
            with open(DIRS["params"] / f"{name}.json", "w") as f:
                json.dump(CFG.to_dict(), f, indent=2)
            Utils.log(f"参数已保存: {name}.json")
        except Exception as e:
            Utils.log(f"保存参数失败: {e}", "ERROR")

    def load_params(self, name="default"):
        """加载参数"""
        try:
            with open(DIRS["params"] / f"{name}.json", "r") as f:
                global CFG
                CFG = ControlConfig.from_dict(json.load(f))
            Utils.log(f"参数已加载: {name}.json")
        except Exception as e:
            Utils.log(f"加载参数失败: {e}", "ERROR")

    # ====================== 运行控制 ======================
    def _status(self):
        """打印状态（低频）"""
        if time.time() - self.last_status < 1.0:
            return

        qpos, _ = self.get_states()
        err = Utils.rad2deg(self.err)
        max_err = Utils.rad2deg(self.max_err)

        status = []
        if self.paused: status.append("暂停")
        if self.estop: status.append("急停")
        if self.collision and self.collision.detected: status.append("碰撞")

        Utils.log("=" * 70)
        Utils.log(f"状态: {' | '.join(status) if status else '运行中'} | 步数: {self.step}")
        Utils.log(f"负载: {self.load:.1f}kg | 角度: {np.round(Utils.rad2deg(qpos), 1)}°")
        Utils.log(f"误差: {np.round(np.abs(err), 3)}° | 最大误差: {np.round(max_err, 3)}°")
        Utils.log("=" * 70)

        self.last_status = time.time()

    def _shell(self):
        """极简交互终端"""
        help_text = """
命令列表:
  help          - 查看帮助
  pause/resume  - 暂停/恢复运动
  stop          - 紧急停止
  reset_collision - 重置碰撞
  pose [名称]   - 预设姿态(zero/up/grasp/test/avoid)
  joint [索引] [角度] - 控制关节
  load [kg]     - 设置负载
  queue [姿态1,2..] - 轨迹队列
  record_start/stop - 数据记录
  save/load_params [名] - 参数保存/加载
"""
        Utils.log(help_text)

        while self.running and not self.estop:
            try:
                cmd = input("> ").strip().lower()
                if not cmd:
                    continue
                parts = cmd.split()

                # 命令映射
                if parts[0] == 'help':
                    Utils.log(help_text)
                elif parts[0] == 'pause':
                    self.paused = True
                elif parts[0] == 'resume':
                    self.paused = False
                elif parts[0] == 'stop':
                    self.estop = True
                    self.running = False
                elif parts[0] == 'reset_collision':
                    if self.collision:
                        self.collision.detected = False
                    self.paused = False
                elif parts[0] == 'pose' and len(parts) == 2:
                    self.preset_pose(parts[1])
                elif parts[0] == 'joint' and len(parts) == 3:
                    self._control_joint(int(parts[1]) - 1, float(parts[2]))
                elif parts[0] == 'load' and len(parts) == 2:
                    self.set_load(float(parts[1]))
                elif parts[0] == 'queue' and len(parts) > 1:
                    self._add_queue(parts[1:])
                elif parts[0] == 'record_start':
                    self.recorder.start()
                elif parts[0] == 'record_stop':
                    self.recorder.stop()
                elif parts[0] == 'save_params':
                    self.save_params(parts[1] if len(parts) > 1 else "default")
                elif parts[0] == 'load_params':
                    self.load_params(parts[1] if len(parts) > 1 else "default")
                else:
                    Utils.log("未知命令，输入help查看帮助")
            except:
                pass

    def _control_joint(self, idx, deg):
        """控制单个关节"""
        if not (0 <= idx < JOINT_COUNT):
            Utils.log("无效关节索引", "ERROR")
            return
        current, _ = self.get_states()
        target = current.copy()
        target[idx] = Utils.deg2rad(deg)
        self.traj_pos, self.traj_vel = Trajectory.plan(current, target)
        self.target = target
        self.traj_idx = 0

    def _add_queue(self, pose_names):
        """添加轨迹队列"""
        poses = {
            'zero': [0, 0, 0, 0, 0], 'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10], 'test': [10, 20, 15, 5, 8],
            'avoid': [15, 25, 10, 5, 0]
        }
        targets = [poses[p] for p in pose_names if p in poses]
        if targets:
            self.add_queue(targets)

    def run(self):
        """主运行循环"""
        # 启动Viewer
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if self.model else None
        except Exception as e:
            Utils.log(f"Viewer启动失败: {e}", "ERROR")
            self.running = False
            return

        # 启动交互线程
        threading.Thread(target=self._shell, daemon=True).start()

        # 演示程序
        threading.Thread(target=self._demo, daemon=True).start()

        # 主循环
        Utils.log("控制器启动成功！输入help查看命令")
        while self.running and (self.viewer.is_running() if self.viewer else True):
            try:
                self.control_step()
                if self.data:
                    mujoco.mj_step(self.model, self.data)
                    if self.viewer:
                        self.viewer.sync()
                self._status()
                self.step += 1
                time.sleep(SLEEP_DT)
            except Exception as e:
                Utils.log(f"运行错误: {e}", "ERROR")
                continue

        # 资源清理
        self._cleanup()
        Utils.log(f"控制器停止 | 总步数: {self.step} | 最大误差: {np.round(Utils.rad2deg(np.max(self.max_err)), 3)}°")

    def _demo(self):
        """极简演示"""
        time.sleep(1)
        self.save_params()
        self.recorder.start()
        self.preset_pose('zero')
        time.sleep(2)
        self.preset_pose('test')
        time.sleep(3)
        self.set_load(1.5)
        time.sleep(4)
        self.preset_pose('grasp')
        time.sleep(2)
        self.recorder.stop()
        time.sleep(2)
        self.preset_pose('zero')

    def _cleanup(self):
        """资源清理"""
        if self.viewer:
            self.viewer.close()
        if self.recorder.enabled:
            self.recorder.stop()
        self.save_params("last")


# ====================== 主函数 ======================
def signal_handler(sig, frame):
    Utils.log("收到退出信号，正在停止...")
    if 'controller' in globals():
        controller.estop = True
        controller.running = False


def main():
    """主函数"""
    # 配置numpy
    np.set_printoptions(precision=3, suppress=True)

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global controller
    controller = ArmController()
    controller.run()

    sys.exit(0)


if __name__ == "__main__":
    main()