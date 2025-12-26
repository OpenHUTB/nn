#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂仿真完整单文件版本（最终完美修复版）
核心修复：
1. 统一关节数为5个（匹配XML模型）
2. 所有数组维度改为5维
3. 移除所有硬编码的6关节逻辑
4. 确保所有运算维度匹配
"""

import sys
import os
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Deque
from collections import deque

# ====================== 核心配置：统一关节数 ======================
JOINT_COUNT = 5  # 关键：改为5个关节（匹配XML模型）

# ====================== mujoco 版本兼容处理 ======================
try:
    import numpy as np
    import mujoco

    try:
        from mujoco import viewer

        MUJOCO_VIEWER_MODE = "new"
    except ImportError:
        if hasattr(mujoco, 'viewer'):
            viewer = mujoco.viewer
            MUJOCO_VIEWER_MODE = "old"
        else:
            raise ImportError("请安装最新版mujoco：pip install mujoco>=2.3.0")

    from scipy import interpolate
    from scipy.signal import filtfilt, butter
    import cvxpy as cp
except ImportError as e:
    print(f"❌ 缺少依赖库：{e.name}")
    print("🔧 请运行：pip install mujoco>=2.3.0 numpy scipy cvxpy ecos osqp")
    sys.exit(1)


# ====================== 1. 配置管理模块（改为5关节） ======================
@dataclass
class PhysicsConfig:
    # 改为5个关节的限制参数
    max_vel: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.8, 1.2, 0.9])
    max_acc: List[float] = field(default_factory=lambda: [0.5, 0.4, 0.4, 0.6, 0.5])
    max_jerk: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.2, 0.4, 0.3])
    max_torque: List[float] = field(default_factory=lambda: [15.0, 15.0, 10.0, 5.0, 5.0])
    ctrl_limit: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class ObstacleConfig:
    base_k_att: float = 0.8
    base_k_rep: float = 0.6
    rep_radius: float = 0.3
    stagnant_threshold: float = 0.01
    stagnant_time: float = 1.0
    guide_offset: float = 0.1
    obstacle_list: List[List[float]] = field(
        default_factory=lambda: [[0.6, 0.1, 0.5, 0.1], [0.55, 0.05, 0.55, 0.08], [0.4, -0.1, 0.6, 0.08]])
    safety_margin: float = 0.05


@dataclass
class EfficiencyConfig:
    time_weight: float = 0.6
    energy_weight: float = 0.4
    traj_interp_points: int = 20
    opt_horizon: float = 1.0
    smooth_factor: float = 0.2
    motor_efficiency: float = 0.85
    # 改为5个关节的摩擦系数
    joint_friction: List[float] = field(default_factory=lambda: [0.001, 0.002, 0.0015, 0.001, 0.0008])


@dataclass
class TrajectoryConfig:
    cart_waypoints: List[List[float]] = field(
        default_factory=lambda: [[0.5, 0.0, 0.6], [0.6, 0.0, 0.58], [0.8, 0.1, 0.8], [0.6, 0.0, 0.58], [0.5, 0.0, 0.6]])


@dataclass
class SimulationConfig:
    timestep: float = 0.005
    fps: int = 60
    log_level: str = "INFO"
    enable_interaction: bool = False


@dataclass
class RobotConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self):
        """校验并自动修复配置参数"""
        logger = logging.getLogger(__name__)
        if self.simulation.fps < 1 or self.simulation.fps > 120:
            logger.warning(f"⚠️ FPS {self.simulation.fps} 超出范围，自动调整为30")
            self.simulation.fps = 30
        if self.efficiency.traj_interp_points < 5 or self.efficiency.traj_interp_points > 100:
            logger.warning(f"⚠️ 插值点数 {self.efficiency.traj_interp_points} 超出范围，自动调整为20")
            self.efficiency.traj_interp_points = 20
        weight_sum = self.efficiency.time_weight + self.efficiency.energy_weight
        if not abs(weight_sum - 1.0) < 1e-6:
            logger.warning(f"⚠️ 时间+能耗权重和为 {weight_sum}（应为1），自动归一化")
            self.efficiency.time_weight /= weight_sum
            self.efficiency.energy_weight /= weight_sum


# 全局配置实例
_global_config: Optional[RobotConfig] = None


def get_config() -> RobotConfig:
    """获取全局配置（单例+参数校验）"""
    global _global_config
    if _global_config is None:
        _global_config = RobotConfig()

        # 应用命令行参数
        parser = argparse.ArgumentParser(description="机械臂仿真配置", add_help=False)
        parser.add_argument("--fps", type=int, help="仿真帧率（1-120）")
        parser.add_argument("--traj-points", type=int, dest="traj_interp_points", help="轨迹插值点数（5-100）")
        parser.add_argument("--smooth-factor", type=float, help="轨迹平滑系数（0.01-1.0）")
        parser.add_argument("--time-weight", type=float, help="时间权重（0-1）")
        parser.add_argument("--energy-weight", type=float, help="能耗权重（0-1）")
        parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
        parser.add_argument("-h", "--help", action="store_true", help="显示帮助信息")

        args, _ = parser.parse_known_args()

        # 应用参数到配置
        if args.fps:
            _global_config.simulation.fps = args.fps
        if args.traj_interp_points:
            _global_config.efficiency.traj_interp_points = args.traj_interp_points
        if args.smooth_factor:
            _global_config.efficiency.smooth_factor = args.smooth_factor
        if args.time_weight:
            _global_config.efficiency.time_weight = args.time_weight
        if args.energy_weight:
            _global_config.efficiency.energy_weight = args.energy_weight
        if args.log_level:
            _global_config.simulation.log_level = args.log_level

        # 校验配置
        _global_config.validate()

        # 显示帮助信息
        if args.help:
            print("""
🤖 机械臂仿真使用帮助：
命令行参数：
  --fps N           设置仿真帧率（1-120），默认60
  --traj-points N   设置轨迹插值点数（5-100），默认20
  --smooth-factor F 设置轨迹平滑系数（0.01-1.0），默认0.2
  --time-weight F   设置时间权重（0-1），默认0.6
  --energy-weight F 设置能耗权重（0-1），默认0.4
  --log-level LEVEL 设置日志级别（DEBUG/INFO/WARNING/ERROR），默认INFO
  -h/--help         显示此帮助信息
            """)
            sys.exit(0)

    return _global_config


# ====================== 2. 性能优化缓存 ======================
_TRAJ_CACHE = {
    "butter_coeffs": {},
    "joint_limits": None
}

_COLLISION_CACHE = {
    "link_ids": {},
    "obstacle_grid": None,
    "obstacle_array": None,
    "safety_margin": None
}

_ENERGY_CACHE = {
    "friction": None,
    "motor_eff": None
}


def init_global_caches():
    """初始化全局缓存（只执行一次）"""
    config = get_config()

    if _TRAJ_CACHE["joint_limits"] is None:
        # 改为5个关节的限制参数
        _TRAJ_CACHE["joint_limits"] = {
            "max_vel": np.array(config.physics.max_vel, dtype=np.float64),
            "max_acc": np.array(config.physics.max_acc, dtype=np.float64),
            "max_torque": np.array(config.physics.max_torque, dtype=np.float64)
        }

    if not _COLLISION_CACHE["link_ids"]:
        _COLLISION_CACHE["obstacle_array"] = np.array(config.obstacle.obstacle_list, dtype=np.float64)
        _COLLISION_CACHE["safety_margin"] = config.obstacle.safety_margin
        obs_pos = _COLLISION_CACHE["obstacle_array"][:, :3]
        min_coords = np.min(obs_pos, axis=0) - 0.5
        max_coords = np.max(obs_pos, axis=0) + 0.5
        _COLLISION_CACHE["obstacle_grid"] = (min_coords, max_coords)

    if _ENERGY_CACHE["friction"] is None:
        _ENERGY_CACHE["friction"] = np.array(config.efficiency.joint_friction, dtype=np.float64)
        _ENERGY_CACHE["motor_eff"] = config.efficiency.motor_efficiency


# ====================== 3. 核心算法模块（改为5关节） ======================
def smooth_cartesian_traj(traj_points: List[List[float]], smooth_factor: float = None) -> List[List[float]]:
    """笛卡尔轨迹平滑"""
    config = get_config()
    smooth_factor = smooth_factor or config.efficiency.smooth_factor
    traj_array = np.asarray(traj_points, dtype=np.float64)

    if traj_array.size == 0 or len(traj_array) <= 1:
        return traj_points

    key = round(smooth_factor, 3)
    if key not in _TRAJ_CACHE["butter_coeffs"]:
        b, a = butter(1, smooth_factor, btype="low")
        _TRAJ_CACHE["butter_coeffs"][key] = (b.astype(np.float64), a.astype(np.float64))
    b, a = _TRAJ_CACHE["butter_coeffs"][key]

    k = min(3, len(traj_array) - 1)
    t = np.linspace(0, 1, len(traj_array), dtype=np.float64)
    t_smooth = np.linspace(0, 1, max(10, len(traj_array) * 2), dtype=np.float64)

    try:
        spline = interpolate.make_interp_spline(t, traj_array, k=k, axis=0)
        smooth_vals = spline(t_smooth)
        smooth_vals = filtfilt(b, a, smooth_vals, axis=0)

        smoothed_traj = np.empty_like(traj_array)
        for dim in range(3):
            smoothed_traj[:, dim] = np.interp(t, t_smooth, smooth_vals[:, dim])
        return smoothed_traj.tolist()
    except Exception:
        return traj_points


def time_optimal_joint_trajectory(
        start_joint: np.ndarray,
        end_joint: np.ndarray,
        seg_time: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """时间最优关节轨迹（改为5关节）"""
    limits = _TRAJ_CACHE["joint_limits"]
    max_vel = limits["max_vel"]
    max_acc = limits["max_acc"]

    config = get_config()
    traj_points = config.efficiency.traj_interp_points

    t_steps = np.linspace(0, seg_time, traj_points, dtype=np.float64)
    # 改为5列（5个关节）
    opt_pos = np.empty((traj_points, JOINT_COUNT), dtype=np.float64)
    opt_vel = np.empty_like(opt_pos)
    opt_acc = np.empty_like(opt_pos)

    delta = end_joint - start_joint
    delta_abs = np.abs(delta)
    sign = np.sign(delta)

    t_acc = max_vel / max_acc
    s_acc = 0.5 * max_acc * t_acc ** 2
    t_joint = np.where(
        delta_abs < 2 * s_acc,
        2 * np.sqrt(delta_abs / max_acc),
        2 * t_acc + (delta_abs - 2 * s_acc) / max_vel
    )

    # 遍历5个关节
    for i, t in enumerate(t_steps):
        for j in range(JOINT_COUNT):
            if delta_abs[j] < 2 * s_acc[j]:
                if t <= t_joint[j] / 2:
                    opt_pos[i, j] = start_joint[j] + 0.5 * max_acc[j] * t ** 2 * sign[j]
                    opt_vel[i, j] = max_acc[j] * t * sign[j]
                    opt_acc[i, j] = max_acc[j] * sign[j]
                else:
                    t_rem = t_joint[j] - t
                    opt_pos[i, j] = end_joint[j] - 0.5 * max_acc[j] * t_rem ** 2 * sign[j]
                    opt_vel[i, j] = max_acc[j] * t_rem * sign[j]
                    opt_acc[i, j] = -max_acc[j] * sign[j]
            else:
                if t <= t_acc[j]:
                    opt_pos[i, j] = start_joint[j] + 0.5 * max_acc[j] * t ** 2 * sign[j]
                    opt_vel[i, j] = max_acc[j] * t * sign[j]
                    opt_acc[i, j] = max_acc[j] * sign[j]
                elif t <= t_acc[j] + (delta_abs[j] - 2 * s_acc[j]) / max_vel[j]:
                    opt_pos[i, j] = start_joint[j] + (s_acc[j] + max_vel[j] * (t - t_acc[j])) * sign[j]
                    opt_vel[i, j] = max_vel[j] * sign[j]
                    opt_acc[i, j] = 0.0
                else:
                    t_rem = t_joint[j] - t
                    opt_pos[i, j] = end_joint[j] - 0.5 * max_acc[j] * t_rem ** 2 * sign[j]
                    opt_vel[i, j] = max_acc[j] * t_rem * sign[j]
                    opt_acc[i, j] = -max_acc[j] * sign[j]

        opt_vel[i] = np.clip(opt_vel[i], -max_vel, max_vel)
        opt_acc[i] = np.clip(opt_acc[i], -max_acc, max_acc)

    return opt_pos, opt_vel, opt_acc


def full_arm_collision_check(
        model,
        data,
        return_min_dist: bool = True
) -> Tuple[bool, float] | bool:
    """全链路碰撞检测"""
    if not _COLLISION_CACHE["link_ids"]:
        # 5个关节对应的连杆
        link_names = ["link1", "link2", "link3", "link4", "link5", "end_effector"]
        for name in link_names:
            _COLLISION_CACHE["link_ids"][name] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )

    collision = False
    min_dist = float("inf")
    obstacle_array = _COLLISION_CACHE["obstacle_array"]
    safety_margin = _COLLISION_CACHE["safety_margin"]
    grid_min, grid_max = _COLLISION_CACHE["obstacle_grid"]

    for link_name, link_id in _COLLISION_CACHE["link_ids"].items():
        try:
            link_pos = data.xpos[link_id].astype(np.float64)

            if np.any(link_pos < grid_min) or np.any(link_pos > grid_max):
                continue

            obs_pos = obstacle_array[:, :3]
            obs_radius = obstacle_array[:, 3]
            distances = np.linalg.norm(link_pos - obs_pos, axis=1) - (obs_radius + safety_margin)

            if np.any(distances < 0):
                collision = True
                if not return_min_dist:
                    return True

            if return_min_dist:
                min_dist = min(min_dist, np.min(distances))
        except Exception:
            continue

    if return_min_dist:
        return collision, min_dist
    return collision


def calculate_real_energy_consumption(model, data, dt: float) -> float:
    """真实能耗计算（改为5关节）"""
    friction = _ENERGY_CACHE["friction"]
    motor_eff = _ENERGY_CACHE["motor_eff"]

    # 只取前5个关节的数据
    torques = data.qfrc_actuator[:JOINT_COUNT].astype(np.float64)
    velocities = data.qvel[:JOINT_COUNT].astype(np.float64)

    friction_loss = np.sum(friction * np.abs(velocities))
    mechanical_power = np.sum(np.abs(torques * velocities))
    total_energy = (mechanical_power + friction_loss) * dt / motor_eff

    return float(total_energy)


# ====================== 4. 可视化模块 ======================
def draw_enhanced_visualization(
        viewer_inst,
        model,
        data,
        traj_history: Deque[list],
        collision_warning: bool
):
    """增强可视化"""
    try:
        scene = viewer_inst.user_scn
        scene.ngeom = 0

        if len(traj_history) > 1:
            traj_array = np.array(traj_history, dtype=np.float64)

            for i in range(len(traj_array) - 1):
                geom = mujoco.MjvGeom()
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.array([0.003, 0, 0], dtype=np.float64),
                    traj_array[i],
                    traj_array[i + 1],
                    np.array([0, 1, 0, 0.6], dtype=np.float64)
                )
                mujoco.mjv_addGeom(scene, model, data, geom)

            def draw_sphere(pos, rgba, size):
                geom = mujoco.MjvGeom()
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([size, 0, 0], dtype=np.float64),
                    pos,
                    np.array([0, 0, 0], dtype=np.float64),
                    np.array(rgba, dtype=np.float64)
                )
                mujoco.mjv_addGeom(scene, model, data, geom)

            draw_sphere(traj_array[0], [0, 0, 1, 0.8], 0.015)
            draw_sphere(traj_array[-1], [1, 0, 0, 0.8], 0.015)

        if collision_warning:
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            ee_pos = data.site_xpos[ee_id]
            draw_sphere(ee_pos, [1, 0, 0, 0.3], 0.08)

    except Exception as e:
        logging.warning(f"可视化绘制失败：{e}")


# ====================== 5. 机械臂模型构建（5关节） ======================
def get_arm_xml_with_obstacles(config: RobotConfig) -> str:
    """生成机械臂XML模型（5关节）"""
    obstacles_xml = ""
    for i, obs in enumerate(config.obstacle.obstacle_list):
        x, y, z, r = obs
        obstacles_xml += f"""
    <body name="obstacle_{i}" pos="{x} {y} {z}">
        <geom name="obs_geom_{i}" type="sphere" size="{r}" rgba="1 0 0 0.5"/>
    </body>
        """

    xml = f"""
<mujoco model="robotic_arm">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="{config.simulation.timestep}" gravity="0 0 -9.81"/>

    <worldbody>
        <!-- 地面 -->
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>

        <!-- 机械臂基座 -->
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>

            <!-- 关节1 -->
            <body name="link1" pos="0 0 0.1">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom name="link1_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                <!-- 关节2 -->
                <body name="link2" pos="0 0 0.2">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                    <geom name="link2_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                    <!-- 关节3 -->
                    <body name="link3" pos="0 0 0.2">
                        <joint name="joint3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                        <geom name="link3_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                        <!-- 关节4 -->
                        <body name="link4" pos="0 0 0.2">
                            <joint name="joint4" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                            <geom name="link4_geom" type="cylinder" size="0.05 0.2" rgba="0.2 0.8 0.2 1"/>

                            <!-- 关节5 -->
                            <body name="link5" pos="0 0 0.2">
                                <joint name="joint5" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                                <geom name="link5_geom" type="cylinder" size="0.05 0.1" rgba="0.2 0.8 0.2 1"/>

                                <!-- 末端执行器 -->
                                <body name="end_effector" pos="0 0 0.1">
                                    <site name="ee_site" pos="0 0 0" size="0.01"/>
                                    <geom name="ee_geom" type="sphere" size="0.05" rgba="0.8 0.2 0.2 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>

        <!-- 障碍物 -->
        {obstacles_xml}
    </worldbody>

    <!-- 控制器（5个电机） -->
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


# ====================== 6. 仿真器主类（5关节） ======================
class ArmSimulator:
    def __init__(self):
        self.config = get_config()
        init_global_caches()

        # 配置日志
        self._setup_logging()

        # 初始化仿真环境
        self._init_simulation()

        # 状态管理
        self.total_motion_time = 0.0
        self.total_energy_consume = 0.0
        self.traj_history: Deque[list] = deque(maxlen=50)
        self.collision_warning = False
        self.stagnant_start_time: Optional[float] = None

        # 预计算关节起点
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.joint_waypoints = self._precompute_joint_waypoints()

        self.logger.info("✅ 机械臂仿真器初始化完成")
        self.logger.info(f"🔧 使用mujoco viewer模式：{MUJOCO_VIEWER_MODE}")
        self.logger.info(f"🔧 机械臂关节数：{JOINT_COUNT}")

    def _setup_logging(self):
        """配置日志系统"""
        log_level = getattr(logging, self.config.simulation.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger("ArmSimulator")

    def _init_simulation(self):
        """初始化仿真环境"""
        arm_xml = get_arm_xml_with_obstacles(self.config)

        # 创建临时文件
        import tempfile
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.temp_file.write(arm_xml)
        self.temp_file.close()

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(self.temp_file.name)
        self.model.opt.timestep = self.config.simulation.timestep
        self.data = mujoco.MjData(self.model)

    def _precompute_joint_waypoints(self) -> list:
        """预计算关节起点（5关节）"""
        joint_waypoints = []
        for cart_pos in self.config.trajectory.cart_waypoints:
            mujoco.mj_resetData(self.model, self.data)
            self.data.site_xpos[self.ee_site_id] = cart_pos
            mujoco.mj_inverse(self.model, self.data)
            # 只取前5个关节
            joint_waypoints.append(self.data.qpos[:JOINT_COUNT].copy())
        return joint_waypoints

    def _get_ee_cartesian_velocity(self) -> np.ndarray:
        """获取末端笛卡尔速度"""
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        ee_vel = jacp @ self.data.qvel
        return ee_vel

    def _check_local_optimum(self, ee_vel: np.ndarray, ee_pos: list, target_pos: list) -> tuple:
        """检测局部最优"""
        vel_mag = np.linalg.norm(ee_vel)
        if vel_mag < self.config.obstacle.stagnant_threshold:
            if self.stagnant_start_time is None:
                self.stagnant_start_time = time.time()
            elif time.time() - self.stagnant_start_time > self.config.obstacle.stagnant_time:
                self.logger.warning(f"检测到局部最优！末端速度={vel_mag:.4f}m/s")
                dir_to_target = np.array(target_pos) - np.array(ee_pos, dtype=np.float64)
                dir_norm = np.linalg.norm(dir_to_target)
                if dir_norm < 1e-6:
                    dir_to_target = np.array([0, 0, 0.1], dtype=np.float64)
                else:
                    dir_to_target = dir_to_target / dir_norm

                guide_target = np.array(ee_pos, dtype=np.float64) + dir_to_target * self.config.obstacle.guide_offset
                self.stagnant_start_time = None
                return True, guide_target.tolist()
        else:
            self.stagnant_start_time = None
        return False, target_pos

    def _robust_artificial_potential_field(self, ee_pos: list, target_pos: list) -> list:
        """人工势场法避障"""
        ee_pos = np.array(ee_pos, dtype=np.float64)
        target_pos = np.array(target_pos, dtype=np.float64)

        ee_vel = self._get_ee_cartesian_velocity()
        is_local_opt, guide_target = self._check_local_optimum(ee_vel, ee_pos.tolist(), target_pos.tolist())
        current_target = np.array(guide_target, dtype=np.float64) if is_local_opt else target_pos

        # 自适应参数
        obs_distances = [np.linalg.norm(ee_pos - np.array(obs[:3], dtype=np.float64))
                         for obs in self.config.obstacle.obstacle_list]
        min_dist = min(obs_distances) if obs_distances else 1.0
        k_rep = self.config.obstacle.base_k_rep if min_dist > 0.2 else self.config.obstacle.base_k_rep * 2.0
        k_att = self.config.obstacle.base_k_att if len(
            self.config.obstacle.obstacle_list) <= 2 else self.config.obstacle.base_k_att * 0.5

        # 引力+斥力
        att_force = k_att * (current_target - ee_pos)
        rep_force = np.zeros(3, dtype=np.float64)

        for obs in self.config.obstacle.obstacle_list:
            obs_pos = np.array(obs[:3], dtype=np.float64)
            obs_radius = obs[3]
            dist = np.linalg.norm(ee_pos - obs_pos)

            if dist < self.config.obstacle.rep_radius + obs_radius:
                rep_dir = (ee_pos - obs_pos) / (dist + 1e-6)
                rep_force += k_rep * (1 / (dist - obs_radius) - 1 / self.config.obstacle.rep_radius) * (
                            1 / dist ** 2) * rep_dir

        corrected_target = ee_pos + att_force + rep_force
        corrected_target = np.clip(corrected_target, [0.3, -0.4, 0.2], [0.9, 0.4, 1.0])
        return corrected_target.tolist()

    def _energy_optimal_trajectory(self, joint_waypoints: np.ndarray, seg_time: float) -> Optional[np.ndarray]:
        """能耗最优轨迹（5关节）"""
        n_joints = JOINT_COUNT
        n_points = len(joint_waypoints)
        t_step = seg_time / (n_points - 1)

        q = cp.Variable((n_joints, n_points))
        qd = cp.Variable((n_joints, n_points))
        qdd = cp.Variable((n_joints, n_points))

        energy_cost = cp.sum_squares(qdd)
        time_cost = cp.sum(cp.max(cp.abs(qd), axis=1))
        total_cost = self.config.efficiency.time_weight * time_cost + self.config.efficiency.energy_weight * energy_cost

        constraints = [
            q[:, 0] == joint_waypoints[0],
            q[:, -1] == joint_waypoints[-1],
            qd[:, 0] == 0,
            qd[:, -1] == 0
        ]

        max_vel = self.config.physics.max_vel
        max_acc = self.config.physics.max_acc
        for j in range(n_joints):
            constraints.extend([
                qd[j, :] <= max_vel[j],
                qd[j, :] >= -max_vel[j],
                qdd[j, :] <= max_acc[j],
                qdd[j, :] >= -max_acc[j]
            ])

        for i in range(n_points - 1):
            constraints.extend([
                qd[:, i + 1] == (q[:, i + 1] - q[:, i]) / t_step,
                qdd[:, i + 1] == (qd[:, i + 1] - qd[:, i]) / t_step
            ])

        prob = cp.Problem(cp.Minimize(total_cost), constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
        except:
            try:
                prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            except:
                prob.solve(verbose=False)

        if prob.status != cp.OPTIMAL:
            self.logger.warning("能耗优化求解失败，降级为时间最优轨迹")
            return None

        return q.value.T

    def _optimize_obstacle_traj_with_efficiency(self, ee_pos: list, target_pos: list) -> tuple:
        """轨迹优化主逻辑"""
        # 避障修正
        corrected_cart_target = self._robust_artificial_potential_field(ee_pos, target_pos)

        # 平滑轨迹
        corrected_cart_target = smooth_cartesian_traj([ee_pos, corrected_cart_target])[-1]

        # 逆解
        self.data.site_xpos[self.ee_site_id] = corrected_cart_target
        mujoco.mj_inverse(self.model, self.data)
        # 只取前5个关节
        end_joint = self.data.qpos[:JOINT_COUNT].copy()
        start_joint = self.data.qpos[:JOINT_COUNT].copy()

        # 时间最优轨迹
        seg_time = 2.0
        time_opt_pos, _, _ = time_optimal_joint_trajectory(start_joint, end_joint, seg_time)

        # 能耗最优
        energy_opt_pos = self._energy_optimal_trajectory(time_opt_pos, seg_time)
        final_joint_traj = energy_opt_pos if energy_opt_pos is not None else time_opt_pos

        # 能耗计算
        dt = seg_time / len(final_joint_traj)
        seg_energy = sum([calculate_real_energy_consumption(self.model, self.data, dt)
                          for _ in range(1, len(final_joint_traj))])

        # 更新状态
        self.total_motion_time += seg_time
        self.total_energy_consume += seg_energy
        self.traj_history.append(corrected_cart_target)

        return final_joint_traj[0], corrected_cart_target, seg_energy

    def _run_simulation_loop(self, viewer_inst):
        """通用仿真循环"""
        self.logger.info("🎮 机械臂仿真启动！")

        config = self.config
        fps = config.simulation.fps
        sleep_time = 1.0 / fps
        print_interval = 2.0
        waypoints = np.array(config.trajectory.cart_waypoints, dtype=np.float64)
        n_waypoints = len(waypoints)

        current_waypoint = 0
        last_print_time = 0.0
        last_step_time = time.time()

        while viewer_inst.is_running():
            # 固定步长控制
            current_time = time.time()
            if current_time - last_step_time < sleep_time:
                continue
            last_step_time = current_time

            # 获取当前状态
            t_total = self.data.time
            ee_pos = self.data.site_xpos[self.ee_site_id].tolist()

            # 切换目标点
            target_cart = waypoints[current_waypoint].tolist()
            if np.linalg.norm(np.array(ee_pos, dtype=np.float64) - np.array(target_cart, dtype=np.float64)) < 0.01:
                current_waypoint = (current_waypoint + 1) % n_waypoints
                self.logger.info(f"🔄 切换到目标点 {current_waypoint}: {np.round(target_cart, 3)}")

            try:
                # 轨迹优化
                target_joints, corrected_cart, _ = self._optimize_obstacle_traj_with_efficiency(ee_pos, target_cart)
                target_joints = np.array(target_joints, dtype=np.float64)

                # 碰撞检测
                is_collision, min_obs_dist = full_arm_collision_check(self.model, self.data)
                self.collision_warning = is_collision

                # 紧急避障
                if is_collision:
                    self.logger.warning("🆘 检测到碰撞风险，执行紧急避障！")
                    emergency_rep = np.array(ee_pos, dtype=np.float64) - np.array(config.obstacle.obstacle_list[0][:3],
                                                                                  dtype=np.float64)
                    emergency_rep = emergency_rep / np.linalg.norm(emergency_rep) * 0.05
                    corrected_cart = np.array(corrected_cart, dtype=np.float64) + emergency_rep
                    self.data.site_xpos[self.ee_site_id] = corrected_cart
                    mujoco.mj_inverse(self.model, self.data)
                    target_joints = self.data.qpos[:JOINT_COUNT].copy()

                # PD控制（5关节）
                max_torque = np.array(config.physics.max_torque, dtype=np.float64) / 100.0
                pos_error = target_joints - self.data.qpos[:JOINT_COUNT]
                vel_error = -self.data.qvel[:JOINT_COUNT]
                ctrl_signals = 8.0 * pos_error + 0.2 * vel_error
                ctrl_signals = np.clip(ctrl_signals, -max_torque, max_torque)
                self.data.ctrl[:JOINT_COUNT] = ctrl_signals

                # 打印统计信息
                if t_total - last_print_time > print_interval and t_total > 0:
                    ee_vel = self._get_ee_cartesian_velocity()
                    avg_vel = np.linalg.norm(ee_vel)
                    avg_energy = self.total_energy_consume / t_total if t_total > 0 else 0.0

                    self.logger.info(
                        f"\n⏱️ 仿真时间：{t_total:.2f}s | 累计运动时间：{self.total_motion_time:.2f}s\n"
                        f"   末端位置：{np.round(ee_pos, 3)} | 目标位置：{np.round(corrected_cart, 3)}\n"
                        f"   末端速度：{avg_vel:.4f}m/s | 最近障碍距离：{min_obs_dist:.3f}m\n"
                        f"   累计能耗：{self.total_energy_consume:.2f}J | 平均能耗：{avg_energy:.2f}J/s\n"
                        f"   碰撞风险：{'⚠️ 高' if is_collision else '✅ 低'}"
                    )
                    last_print_time = t_total

                # 可视化
                draw_enhanced_visualization(viewer_inst, self.model, self.data,
                                            self.traj_history, self.collision_warning)

            except Exception as e:
                self.logger.error(f"仿真步执行失败：{e}", exc_info=False)
                continue

            # 执行仿真步
            mujoco.mj_step(self.model, self.data)
            viewer_inst.sync()

    def run(self):
        """运行仿真主循环"""
        try:
            with viewer.launch_passive(self.model, self.data) as viewer_inst:
                self._run_simulation_loop(viewer_inst)

        except KeyboardInterrupt:
            self.logger.info("\n🛑 用户终止仿真")
        except Exception as e:
            self.logger.error(f"❌ 仿真出错：{e}", exc_info=True)
        finally:
            # 清理资源
            if hasattr(self, 'temp_file'):
                os.unlink(self.temp_file.name)
            self.logger.info(f"\n📊 仿真结束 - 最终统计")
            self.logger.info(f"   总运动时间：{self.total_motion_time:.2f}s")
            self.logger.info(f"   总能耗：{self.total_energy_consume:.2f}J")
            self.logger.info(
                f"   综合得分：{self.total_motion_time * self.config.efficiency.time_weight + self.total_energy_consume * self.config.efficiency.energy_weight:.2f}")


# ====================== 7. 主入口 ======================
def main():
    """程序主入口"""
    try:
        simulator = ArmSimulator()
        simulator.run()
    except Exception as e:
        print(f"❌ 程序运行失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()