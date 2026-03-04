#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂控制器（终极极简优化版）
核心功能：梯形轨迹规划/自适应控制/关节单独控制/轨迹管理/紧急停止
"""

import sys
import time
import signal
import threading
import numpy as np
import mujoco
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

# ====================== 常量定义（预计算，零运行时开销） ======================
JOINT_COUNT = 5
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
MIN_LOAD, MAX_LOAD = 0.0, 2.0

# 预计算关节极限（全局常量，避免运行时创建）
JOINT_LIMITS = np.array([[-np.pi, np.pi], [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2],
                         [-np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2]], dtype=np.float64)
MAX_VEL = np.array([1.0, 0.8, 0.8, 0.6, 0.6], dtype=np.float64)
MAX_ACC = np.array([2.0, 1.6, 1.6, 1.2, 1.2], dtype=np.float64)
MAX_TORQUE = np.array([15.0, 12.0, 10.0, 8.0, 5.0], dtype=np.float64)

# 时间配置（预计算）
SIM_DT = 0.0005
CTRL_FREQ = 2000
CTRL_DT = 1.0 / CTRL_FREQ
FPS = 60
SLEEP_DT = 1.0 / FPS


# 控制参数（全局单例，简化访问）
@dataclass
class Cfg:
    kp_base, kd_base = 120.0, 8.0
    kp_load_gain, kd_load_gain = 1.8, 1.5
    ff_vel, ff_acc = 0.7, 0.5
    backlash = np.array([0.001, 0.001, 0.002, 0.002, 0.003])
    friction = np.array([0.1, 0.08, 0.08, 0.06, 0.06])
    gravity_comp = True
    stiffness_base = np.array([200.0, 180.0, 150.0, 120.0, 80.0])
    stiffness_load_gain = 1.8
    stiffness_error_gain = 1.5
    stiffness_min = np.array([100.0, 90.0, 75.0, 60.0, 40.0])
    stiffness_max = np.array([300.0, 270.0, 225.0, 180.0, 120.0])
    damping_ratio = 0.04


Cfg = Cfg()

# 全局状态（原子操作，零锁竞争）
RUNNING = True
PAUSED = False
EMERGENCY_STOP = False
LOCK = threading.Lock()
Path("trajectories").mkdir(exist_ok=True)


# ====================== 极简工具函数（零冗余） ======================
@contextmanager
def lock():
    with LOCK:
        yield


def log(msg):
    """极简日志（文件+控制台）"""
    try:
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_msg = f"[{ts}] {msg}"
        with open("arm.log", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
        print(log_msg)
    except:
        pass


def deg2rad(x):
    """极简角度转弧度（向量化）"""
    try:
        return np.asarray(x, np.float64) * DEG2RAD
    except:
        return 0.0 if np.isscalar(x) else np.zeros(JOINT_COUNT)


def rad2deg(x):
    """极简弧度转角度（向量化）"""
    try:
        return np.asarray(x, np.float64) * RAD2DEG
    except:
        return 0.0 if np.isscalar(x) else np.zeros(JOINT_COUNT)


# ====================== 极简轨迹规划器（无类封装，纯函数） ======================
TRAJ_CACHE = {}


def plan_trajectory(start, target, dt=CTRL_DT):
    """极简轨迹规划（缓存+向量化）"""
    # 边界裁剪
    start = np.clip(start, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)
    target = np.clip(target, JOINT_LIMITS[:, 0] + 0.01, JOINT_LIMITS[:, 1] - 0.01)

    # 缓存检查
    cache_key = (hash(start.tobytes()), hash(target.tobytes()))
    if cache_key in TRAJ_CACHE:
        return TRAJ_CACHE[cache_key]

    # 批量规划
    traj_pos, traj_vel, max_len = [], [], 1
    for i in range(JOINT_COUNT):
        # 单关节梯形规划（纯向量化）
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

            # 时间序列
            t = np.arange(0, total_time + dt, dt)
            pos = np.empty_like(t)
            vel = np.empty_like(t)

            # 向量化分段计算
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

    # 结果缓存
    result = (np.array(traj_pos).T, np.array(traj_vel).T)
    TRAJ_CACHE[cache_key] = result
    return result


def save_traj(traj_pos, traj_vel, name):
    """极简轨迹保存（numpy批量IO）"""
    try:
        header = ['step'] + [f'j{i + 1}_pos' for i in range(JOINT_COUNT)] + [f'j{i + 1}_vel' for i in
                                                                             range(JOINT_COUNT)]
        data = np.hstack([np.arange(len(traj_pos))[:, None], traj_pos, traj_vel])
        np.savetxt(f"trajectories/{name}.csv", data, delimiter=',', header=','.join(header), comments='')
        log(f"轨迹保存: {name}.csv")
    except Exception as e:
        log(f"保存失败: {e}")


def load_traj(name):
    """极简轨迹加载"""
    try:
        data = np.genfromtxt(f"trajectories/{name}.csv", delimiter=',', skip_header=1)
        if len(data) == 0:
            return np.array([]), np.array([])
        return data[:, 1:JOINT_COUNT + 1], data[:, JOINT_COUNT + 1:]
    except Exception as e:
        log(f"加载失败: {e}")
        return np.array([]), np.array([])

    @classmethod
    def clear_cache(cls):
        """清理缓存（内存管理）"""
        cls._cache.clear()
        log("轨迹缓存已清理")


# ====================== 核心控制器（极简封装） ======================
class ArmController:
    def __init__(self):
        # 核心状态（预分配内存）
        self.model, self.data = self._init_mujoco()
        self.viewer = None

        # ID缓存
        self.joint_ids = [self._get_id('joint', f'joint{i + 1}') for i in range(JOINT_COUNT)]
        self.motor_ids = [self._get_id('actuator', f'motor{i + 1}') for i in range(JOINT_COUNT)]
        self.ee_id = self._get_id('geom', 'ee_geom')

        # 控制状态
        self.traj_pos = np.zeros((1, JOINT_COUNT))
        self.traj_vel = np.zeros((1, JOINT_COUNT))
        self.traj_idx = 0
        self.saved_idx = 0
        self.target = np.zeros(JOINT_COUNT)

        # 物理状态
        self.stiffness = Cfg.stiffness_base.copy()
        self.damping = self.stiffness * Cfg.damping_ratio
        self.load_set = 0.5
        self.load_actual = 0.5

        # 误差状态
        self.err = np.zeros(JOINT_COUNT)
        self.max_err = np.zeros(JOINT_COUNT)

        # 性能统计
        self.step = 0
        self.last_ctrl = time.time()
        self.last_print = time.time()
        self.fps_count = 0

    def _init_mujoco(self):
        """极简MuJoCo初始化"""
        xml = f"""
<mujoco model="arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{SIM_DT}" gravity="0 0 -9.81"/>
    <default>
        <joint type="hinge" limited="true"/>
        <motor ctrllimited="true" ctrlrange="-1 1" gear="100"/>
    </default>
    <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="base" pos="0 0 0">
            <geom type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
            <joint name="joint1" axis="0 0 1" pos="0 0 0.1" range="{JOINT_LIMITS[0, 0]} {JOINT_LIMITS[0, 1]}"/>
            <body name="link1" pos="0 0 0.1">
                <geom type="cylinder" size="0.04 0.18" mass="0.8" rgba="0 0.8 0 0.8"/>
                <joint name="joint2" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[1, 0]} {JOINT_LIMITS[1, 1]}"/>
                <body name="link2" pos="0 0 0.18">
                    <geom type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                    <joint name="joint3" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[2, 0]} {JOINT_LIMITS[2, 1]}"/>
                    <body name="link3" pos="0 0 0.18">
                        <geom type="cylinder" size="0.04 0.18" mass="0.6" rgba="0 0.8 0 0.8"/>
                        <joint name="joint4" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[3, 0]} {JOINT_LIMITS[3, 1]}"/>
                        <body name="link4" pos="0 0 0.18">
                            <geom type="cylinder" size="0.04 0.18" mass="0.4" rgba="0 0.8 0 0.8"/>
                            <joint name="joint5" axis="0 1 0" pos="0 0 0.18" range="{JOINT_LIMITS[4, 0]} {JOINT_LIMITS[4, 1]}"/>
                            <body name="ee" pos="0 0 0.18">
                                <geom name="ee_geom" type="sphere" size="0.04" mass="{self.load_set}" rgba="0.8 0.2 0.2 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
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
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            log("MuJoCo初始化成功")
            return model, data
        except Exception as e:
            log(f"MuJoCo初始化失败: {e}")
            global RUNNING, EMERGENCY_STOP
            RUNNING = False
            EMERGENCY_STOP = True
            return None, None

    def _get_id(self, obj_type, name):
        """极简ID获取"""
        type_map = {'joint': mujoco.mjtObj.mjOBJ_JOINT, 'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
                    'geom': mujoco.mjtObj.mjOBJ_GEOM}
        try:
            return mujoco.mj_name2id(self.model, type_map[obj_type], name)
        except:
            return -1

    def get_states(self):
        """极简状态获取"""
        if self.data is None:
            return np.zeros(JOINT_COUNT), np.zeros(JOINT_COUNT)
        qpos = np.array([self.data.qpos[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qvel = np.array([self.data.qvel[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        return qpos, qvel

    def _calc_load(self):
        """极简负载计算"""
        if self.data is None:
            return 0.0
        forces = np.abs([self.data.qfrc_actuator[jid] if jid >= 0 else 0.0 for jid in self.joint_ids])
        qpos, _ = self.get_states()
        load = np.sum(forces * np.sin(qpos)) / 9.81
        return np.clip(load, MIN_LOAD, MAX_LOAD)

    def control_step(self):
        """核心控制步（极简逻辑）"""
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

        # 状态获取
        qpos, qvel = self.get_states()
        self.load_actual = self._calc_load()

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

        # PD+前馈控制（纯向量化）
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

        self.last_ctrl = now

    # ====================== 核心控制接口（极简） ======================
    def move_to(self, target_deg, save=False, name="default"):
        """移动到目标角度"""
        with lock():
            target_deg = np.asarray(target_deg, np.float64)
            if target_deg.shape != (JOINT_COUNT,):
                log(f"目标维度错误: {target_deg.shape}")
                return

            start, _ = self.get_states()
            target = deg2rad(target_deg)
            self.traj_pos, self.traj_vel = plan_trajectory(start, target)
            self.target = target
            self.traj_idx = 0

            if save:
                save_traj(self.traj_pos, self.traj_vel, name)
            log(f"规划轨迹: {np.round(rad2deg(start), 1)}° → {np.round(rad2deg(target), 1)}°")

    def control_joint(self, idx, target_deg):
        """单独控制关节"""
        if not (0 <= idx < JOINT_COUNT):
            log(f"无效关节索引: {idx}")
            return

        current, _ = self.get_states()
        target = current.copy()
        target[idx] = deg2rad(target_deg)
        self.traj_pos, self.traj_vel = plan_trajectory(current, target)
        self.target = target
        self.traj_idx = 0
        log(f"控制关节{idx + 1}: {np.round(rad2deg(current[idx]), 1)}° → {target_deg:.1f}°")

    def set_load(self, mass):
        """设置负载"""
        with lock():
            if not (MIN_LOAD <= mass <= MAX_LOAD):
                log(f"负载超出范围: {mass}kg (0-2kg)")
                return

            self.load_set = mass
            if self.ee_id >= 0 and self.model is not None:
                self.model.geom_mass[self.ee_id] = mass
            log(f"负载设置为: {mass}kg")

    def pause(self):
        """暂停"""
        global PAUSED
        with lock():
            PAUSED = True
            log("轨迹暂停")

    def resume(self):
        """恢复"""
        global PAUSED
        with lock():
            PAUSED = False
            self.traj_idx = self.saved_idx
            log(f"轨迹恢复（第{self.saved_idx}步）")

    def emergency_stop(self):
        """紧急停止"""
        global RUNNING, PAUSED, EMERGENCY_STOP
        with lock():
            EMERGENCY_STOP = True
            PAUSED = True
            RUNNING = False
            log("⚠️ 紧急停止触发")

    def adjust_param(self, param, value, idx=None):
        """调整参数"""
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

    def load_trajectory(self, name):
        """加载轨迹"""
        with lock():
            traj_pos, traj_vel = load_traj(name)
            if len(traj_pos) == 0:
                return
            self.traj_pos = traj_pos
            self.traj_vel = traj_vel
            self.target = traj_pos[-1] if len(traj_pos) > 0 else np.zeros(JOINT_COUNT)
            self.traj_idx = 0
            log(f"加载轨迹: {name} (共{len(traj_pos)}步)")

    def preset_pose(self, pose):
        """预设姿态"""
        poses = {
            'zero': [0, 0, 0, 0, 0],
            'up': [0, 30, 20, 10, 0],
            'grasp': [0, 45, 30, 20, 10],
            'test': [10, 20, 15, 5, 8]
        }
        if pose in poses:
            self.move_to(poses[pose])
        else:
            log(f"未知姿态: {pose} (支持: {list(poses.keys())})")

    def _print_status(self):
        """极简状态打印"""
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
        status_str = " | ".join(status) if status else "运行中"

        log("=" * 60)
        log(f"状态: {status_str} | 步数: {self.step} | FPS: {fps:.1f}")
        log(f"负载: {self.load_set:.1f}kg(设定) | {self.load_actual:.1f}kg(实际)")
        log(f"角度: {np.round(rad2deg(qpos), 1)}° | 误差: {np.round(np.abs(err), 3)}°(最大:{np.round(max_err, 3)}°)")
        log("=" * 60)

        self.last_print = now
        self.fps_count = 0

    def _interactive(self):
        """极简交互线程"""
        help_text = """
命令列表：
  help          - 查看帮助
  pause/resume  - 暂停/恢复
  stop          - 紧急停止
  pose [名称]   - 预设姿态(zero/up/grasp/test)
  joint [索引] [角度] - 控制单个关节
  load [kg]     - 设置负载(0-2kg)
  param [名] [值] [关节] - 调整参数
  save [名]     - 保存轨迹
  load_traj [名] - 加载轨迹
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
                else:
                    log("未知命令，输入help查看帮助")
            except:
                continue

    def run(self):
        """主运行循环"""
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

        # 演示程序
        def demo():
            steps = [(2, 'pose', 'zero'), (3, 'pose', 'test'), (2, 'pause', None), (2, 'resume', None),
                     (4, 'load', 1.5), (4, 'pose', 'grasp'), (1, 'joint', (0, 10)), (3, 'load', 0.2),
                     (3, 'pose', 'zero'), (2, 'stop', None)]
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

        threading.Thread(target=demo, daemon=True).start()

        # 主循环
        log("控制器启动 (Ctrl+C退出)")
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