import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time
from scipy import interpolate
import cvxpy as cp
import warnings
import argparse
from dataclasses import dataclass
from collections import deque
import logging

# ====================== 0. 初始化配置与日志 ======================
warnings.filterwarnings("ignore")

# 日志配置（结构化输出）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("robotic_arm")


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="机械臂效率+鲁棒双优化仿真（核心增强版）")
    parser.add_argument("--traj-points", type=int, default=20, help="轨迹插值点数（平衡精度与速度）")
    parser.add_argument("--time-weight", type=float, default=0.6, help="时间权重（0-1）")
    parser.add_argument("--energy-weight", type=float, default=0.4, help="能耗权重（0-1）")
    parser.add_argument("--smooth-factor", type=float, default=0.2, help="轨迹平滑系数（0-1）")
    return parser.parse_args()


args = parse_args()


# ====================== 1. 配置参数（结构化管理） ======================
@dataclass
class RobotConfig:
    # 物理约束（UR5工业级参数）
    max_vel: list = None
    max_acc: list = None
    max_jerk: list = None
    max_torque: list = None
    ctrl_limit: tuple = (-10.0, 10.0)

    # 避障鲁棒性参数
    base_k_att: float = 0.8
    base_k_rep: float = 0.6
    rep_radius: float = 0.3
    stagnant_threshold: float = 0.01
    stagnant_time: float = 1.0
    guide_offset: float = 0.1
    obstacle_list: list = None

    # 效率优化参数
    time_weight: float = args.time_weight
    energy_weight: float = args.energy_weight
    traj_interp_points: int = args.traj_points
    safety_margin: float = 0.05
    opt_horizon: float = 1.0
    smooth_factor: float = args.smooth_factor

    # 能耗计算参数（工业级电机参数）
    motor_efficiency: float = 0.85  # 电机效率
    joint_friction: list = None  # 关节摩擦系数

    # 笛卡尔轨迹关键点
    cart_waypoints: list = None


# 初始化配置
config = RobotConfig(
    max_vel=[1.0, 0.8, 0.8, 1.2, 0.9, 1.2],
    max_acc=[0.5, 0.4, 0.4, 0.6, 0.5, 0.6],
    max_jerk=[0.3, 0.2, 0.2, 0.4, 0.3, 0.4],
    max_torque=[15.0, 15.0, 10.0, 5.0, 5.0, 3.0],
    obstacle_list=[
        [0.6, 0.1, 0.5, 0.1],
        [0.55, 0.05, 0.55, 0.08],
        [0.4, -0.1, 0.6, 0.08]
    ],
    joint_friction=[0.001, 0.002, 0.0015, 0.001, 0.0008, 0.0005],
    cart_waypoints=[
        [0.5, 0.0, 0.6],
        [0.6, 0.0, 0.58],
        [0.8, 0.1, 0.8],
        [0.6, 0.0, 0.58],
        [0.5, 0.0, 0.6]
    ]
)

# 全局变量
stagnant_start_time = None
total_motion_time = 0.0
total_energy_consume = 0.0
traj_history = deque(maxlen=50)  # 轨迹历史（用于可视化）
collision_warning = False

# 预定义关节惯性参数
JOINT_INERTIA = [0.01, 0.02, 0.015, 0.01, 0.008, 0.005]
JOINT_GRAVITY = [0.5, 0.8, 0.6, 0.3, 0.2, 0.1]


# ====================== 2. 核心增强：全链路碰撞检测 ======================
def full_arm_collision_check(model, data, config, return_min_dist=True):
    """
    检测所有连杆与障碍物的碰撞（核心增强）
    :param return_min_dist: 是否返回最近距离
    :return: 是否碰撞, 最近距离（可选）
    """
    collision = False
    min_dist = float("inf")

    # 所有需要检测的连杆/末端
    link_names = ["link1", "link2", "link3", "link4", "link5", "end_effector"]

    for link_name in link_names:
        try:
            link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            link_pos = data.xpos[link_id]  # 连杆中心位置

            # 检测与每个障碍物的距离
            for obs in config.obstacle_list:
                obs_pos = np.array(obs[:3])
                obs_radius = obs[3]

                # 计算连杆到障碍物的距离（减去安全裕度）
                dist = np.linalg.norm(link_pos - obs_pos) - (obs_radius + config.safety_margin)

                if dist < 0:
                    collision = True
                    logger.warning(f"碰撞风险：{link_name} 与障碍物 {obs_pos} 距离 {dist:.4f}m")

                if return_min_dist:
                    min_dist = min(min_dist, dist)
        except Exception as e:
            logger.error(f"碰撞检测失败：{link_name} - {e}")
            continue

    global collision_warning
    collision_warning = collision

    if return_min_dist:
        return collision, min_dist
    return collision


# ====================== 3. 核心增强：轨迹平滑滤波（修复阶数不匹配） ======================
def smooth_cartesian_traj(traj_points, smooth_factor=0.2):
    """
    贝塞尔曲线平滑笛卡尔轨迹（修复点数不足问题）
    :param traj_points: 原始轨迹点列表
    :param smooth_factor: 平滑系数（0-1，越大越平滑）
    :return: 平滑后的轨迹点
    """
    traj_array = np.array(traj_points)
    n_points = len(traj_array)

    # 处理点数不足的情况：
    # - 1个点：直接返回
    # - 2-3个点：使用线性插值/低阶样条
    # - 4个及以上点：使用3阶样条
    if n_points <= 1:
        return traj_points
    elif n_points <= 3:
        k = n_points - 1  # 适配点数的阶数（2点→1阶，3点→2阶）
    else:
        k = 3  # 4点及以上用3阶样条

    # 生成插值参数
    t = np.linspace(0, 1, n_points)
    smoothed_traj = np.zeros_like(traj_array)

    for dim in range(3):  # x/y/z三个维度
        try:
            # 生成适配阶数的样条曲线（核心修复）
            spline = interpolate.make_interp_spline(t, traj_array[:, dim], k=k)
            # 生成更密集的插值点（提升平滑度）
            t_smooth = np.linspace(0, 1, max(10, n_points * 2))
            smooth_vals = spline(t_smooth)

            # 低通滤波（增加异常处理，提升稳定性）
            try:
                from scipy.signal import filtfilt, butter
                b, a = butter(2, smooth_factor, btype="low")
                smooth_vals = filtfilt(b, a, smooth_vals)
            except:
                # 滤波失败时直接使用插值结果
                pass

            # 采样回原点数
            smoothed_traj[:, dim] = np.interp(t, t_smooth, smooth_vals)
        except Exception as e:
            # 插值失败时降级为原始轨迹
            logger.warning(f"轨迹平滑失败（维度{dim}）：{e}，使用原始轨迹")
            smoothed_traj = traj_array
            break

    return smoothed_traj.tolist()


def smooth_joint_traj(joint_traj, smooth_factor=0.1):
    """
    平滑关节轨迹（修复点数不足+简化滤波）
    """
    joint_array = np.array(joint_traj)
    n_points, n_joints = joint_array.shape

    # 点数不足时直接返回
    if n_points <= 1:
        return joint_traj

    # 简化滤波逻辑，提升兼容性
    try:
        from scipy.signal import filtfilt, butter
        b, a = butter(1, smooth_factor, btype="low")  # 1阶滤波更稳定
        smoothed_joints = np.zeros_like(joint_array)

        for j in range(n_joints):
            smoothed_joints[:, j] = filtfilt(b, a, joint_array[:, j])
    except Exception as e:
        logger.warning(f"关节轨迹平滑失败：{e}，使用原始轨迹")
        smoothed_joints = joint_array

    return smoothed_joints


# ====================== 4. 核心增强：真实能耗计算 ======================
def calculate_real_energy_consumption(model, data, config, dt):
    """
    计算真实能耗（考虑电机效率、摩擦损耗）
    """
    energy = 0.0

    for joint_idx in range(6):
        # 1. 获取MuJoCo实时输出扭矩
        real_torque = data.qfrc_actuator[joint_idx]
        # 2. 关节速度
        joint_vel = data.qvel[joint_idx]
        # 3. 摩擦损耗
        friction_loss = config.joint_friction[joint_idx] * abs(joint_vel)
        # 4. 实际能耗计算
        mechanical_power = abs(real_torque * joint_vel)
        total_power = mechanical_power + friction_loss
        energy += total_power / config.motor_efficiency * dt

    return energy


# ====================== 5. 核心增强：可视化工具（适配新版MuJoCo API） ======================
def draw_enhanced_visualization(viewer, model, data, config):
    """
    增强可视化：适配新版MuJoCo API，避免MjGeom报错
    注：新版MuJoCo简化了用户绘制，这里改用更兼容的方式展示关键信息
    """
    try:
        # 方式1：使用新版Viewer的user_scn（兼容大部分版本）
        scene = viewer.user_scn
        scene.ngeom = 0  # 清空原有绘制

        # 1. 绘制轨迹历史（使用mjv_geom接口）
        if len(traj_history) > 1:
            traj_array = np.array(list(traj_history))

            # 轨迹线
            for i in range(len(traj_array) - 1):
                # 初始化几何对象（适配新版API）
                geom = mujoco.MjvGeom()
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.array([0.003, 0, 0]),  # 大小
                    traj_array[i],  # 起始点
                    traj_array[i + 1],  # 结束点
                    np.array([0, 1, 0, 0.6])  # 绿色轨迹线
                )
                # 添加到场景
                mujoco.mjv_addGeom(scene, model, data, geom)

            # 轨迹起点（蓝色）
            start_geom = mujoco.MjvGeom()
            mujoco.mjv_initGeom(
                start_geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.015, 0, 0]),
                traj_array[0],
                np.array([0, 0, 0]),
                np.array([0, 0, 1, 0.8])
            )
            mujoco.mjv_addGeom(scene, model, data, start_geom)

            # 轨迹终点（红色）
            end_geom = mujoco.MjvGeom()
            mujoco.mjv_initGeom(
                end_geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.015, 0, 0]),
                traj_array[-1],
                np.array([0, 0, 0]),
                np.array([1, 0, 0, 0.8])
            )
            mujoco.mjv_addGeom(scene, model, data, end_geom)

        # 2. 绘制碰撞警告（红色半透明球）
        if collision_warning:
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            ee_pos = data.site_xpos[ee_id]

            warn_geom = mujoco.MjvGeom()
            mujoco.mjv_initGeom(
                warn_geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.08, 0, 0]),
                ee_pos,
                np.array([0, 0, 0]),
                np.array([1, 0, 0, 0.3])
            )
            mujoco.mjv_addGeom(scene, model, data, warn_geom)

    except Exception as e:
        # 兼容最简化版本：禁用可视化绘制，避免影响核心功能
        logger.warning(f"可视化绘制失败（MuJoCo版本兼容问题）：{e}")
        logger.warning("已禁用可视化增强，核心仿真功能不受影响")


# ====================== 6. 基础工具函数 ======================
def get_ee_cartesian_velocity(model, data, ee_site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    joint_vel = data.qvel[:6]
    ee_cart_vel = jacp @ joint_vel
    return ee_cart_vel


def check_local_optimum(ee_vel, ee_pos, target_pos, config):
    global stagnant_start_time
    vel_mag = np.linalg.norm(ee_vel)

    if vel_mag < config.stagnant_threshold:
        if stagnant_start_time is None:
            stagnant_start_time = time.time()
        elif time.time() - stagnant_start_time > config.stagnant_time:
            logger.warning(f"检测到局部最优！末端速度={vel_mag:.4f}m/s")
            dir_to_target = np.array(target_pos) - np.array(ee_pos)
            dir_norm = np.linalg.norm(dir_to_target)
            if dir_norm < 1e-6:
                dir_to_target = np.array([0, 0, 0.1])
            else:
                dir_to_target = dir_to_target / dir_norm

            guide_target = np.array(ee_pos) + dir_to_target * config.guide_offset
            stagnant_start_time = None
            return True, guide_target.tolist()
    else:
        stagnant_start_time = None

    return False, target_pos


def adaptive_potential_params(ee_pos, obstacle_list, config):
    obs_distances = [np.linalg.norm(np.array(ee_pos) - np.array(obs[:3])) for obs in obstacle_list]
    min_dist = min(obs_distances) if obs_distances else 1.0

    k_rep = config.base_k_rep if min_dist > 0.2 else config.base_k_rep * 2.0
    k_att = config.base_k_att if len(obstacle_list) <= 2 else config.base_k_att * 0.5
    return k_att, k_rep


def robust_artificial_potential_field(ee_pos, ee_vel, target_pos, obstacle_list, config):
    ee_pos = np.array(ee_pos)
    target_pos = np.array(target_pos)

    # 局部最优规避
    is_local_opt, guide_target = check_local_optimum(ee_vel, ee_pos, target_pos, config)
    current_target = np.array(guide_target) if is_local_opt else target_pos

    # 自适应参数
    k_att, k_rep = adaptive_potential_params(ee_pos, obstacle_list, config)

    # 引力+斥力计算
    att_force = k_att * (current_target - ee_pos)
    rep_force = np.zeros(3)

    for obs in obstacle_list:
        obs_pos = np.array(obs[:3])
        obs_radius = obs[3]
        dist = np.linalg.norm(ee_pos - obs_pos)

        if dist < config.rep_radius + obs_radius:
            rep_dir = (ee_pos - obs_pos) / (dist + 1e-6)
            rep_force += k_rep * (1 / (dist - obs_radius) - 1 / config.rep_radius) * (1 / dist ** 2) * rep_dir

    # 修正目标并约束
    corrected_target = ee_pos + att_force + rep_force
    corrected_target = np.clip(corrected_target, [0.3, -0.4, 0.2], [0.9, 0.4, 1.0])

    return corrected_target.tolist()


# ====================== 7. 轨迹规划 ======================
def time_optimal_joint_trajectory(start_joint, end_joint, seg_time, config):
    n_joints = 6
    traj_points = config.traj_interp_points
    t_steps = np.linspace(0, seg_time, traj_points)

    opt_pos = np.zeros((traj_points, n_joints))
    opt_vel = np.zeros((traj_points, n_joints))
    opt_acc = np.zeros((traj_points, n_joints))

    for j in range(n_joints):
        delta = end_joint[j] - start_joint[j]
        max_vel = config.max_vel[j]
        max_acc = config.max_acc[j]

        t_acc = max_vel / max_acc
        s_acc = 0.5 * max_acc * t_acc ** 2

        if abs(delta) < 2 * s_acc:
            t_joint = 2 * np.sqrt(abs(delta) / max_acc)
            for i, t in enumerate(t_steps):
                if t <= t_joint / 2:
                    opt_pos[i, j] = start_joint[j] + 0.5 * max_acc * t ** 2 * np.sign(delta)
                    opt_vel[i, j] = max_acc * t * np.sign(delta)
                    opt_acc[i, j] = max_acc * np.sign(delta)
                else:
                    t_rem = t_joint - t
                    opt_pos[i, j] = end_joint[j] - 0.5 * max_acc * t_rem ** 2 * np.sign(delta)
                    opt_vel[i, j] = max_acc * t_rem * np.sign(delta)
                    opt_acc[i, j] = -max_acc * np.sign(delta)
        else:
            t_const = (abs(delta) - 2 * s_acc) / max_vel
            t_joint = 2 * t_acc + t_const
            for i, t in enumerate(t_steps):
                if t <= t_acc:
                    opt_pos[i, j] = start_joint[j] + 0.5 * max_acc * t ** 2 * np.sign(delta)
                    opt_vel[i, j] = max_acc * t * np.sign(delta)
                    opt_acc[i, j] = max_acc * np.sign(delta)
                elif t <= t_acc + t_const:
                    opt_pos[i, j] = start_joint[j] + (s_acc + max_vel * (t - t_acc)) * np.sign(delta)
                    opt_vel[i, j] = max_vel * np.sign(delta)
                    opt_acc[i, j] = 0.0
                else:
                    t_rem = t_joint - t
                    opt_pos[i, j] = end_joint[j] - 0.5 * max_acc * t_rem ** 2 * np.sign(delta)
                    opt_vel[i, j] = max_acc * t_rem * np.sign(delta)
                    opt_acc[i, j] = -max_acc * np.sign(delta)

        opt_vel[:, j] = np.clip(opt_vel[:, j], -max_vel, max_vel)
        opt_acc[:, j] = np.clip(opt_acc[:, j], -max_acc, max_acc)

    # 平滑关节轨迹（使用修复后的函数）
    opt_pos = smooth_joint_traj(opt_pos, config.smooth_factor)

    return opt_pos, opt_vel, opt_acc


def energy_optimal_trajectory(joint_waypoints, seg_time, config):
    n_joints = 6
    n_points = len(joint_waypoints)
    t_step = seg_time / (n_points - 1)

    q = cp.Variable((n_joints, n_points))
    qd = cp.Variable((n_joints, n_points))
    qdd = cp.Variable((n_joints, n_points))

    energy_cost = cp.sum_squares(qdd)
    time_cost = cp.sum(cp.max(cp.abs(qd), axis=1))
    total_cost = config.time_weight * time_cost + config.energy_weight * energy_cost

    constraints = []
    constraints.append(q[:, 0] == joint_waypoints[0])
    constraints.append(q[:, -1] == joint_waypoints[-1])
    constraints.append(qd[:, 0] == 0)
    constraints.append(qd[:, -1] == 0)

    for j in range(n_joints):
        constraints.append(qd[j, :] <= config.max_vel[j])
        constraints.append(qd[j, :] >= -config.max_vel[j])
        constraints.append(qdd[j, :] <= config.max_acc[j])
        constraints.append(qdd[j, :] >= -config.max_acc[j])

    for i in range(n_points - 1):
        constraints.append(qd[:, i + 1] == (q[:, i + 1] - q[:, i]) / t_step)
        constraints.append(qdd[:, i + 1] == (qd[:, i + 1] - qd[:, i]) / t_step)

    prob = cp.Problem(cp.Minimize(total_cost), constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except:
            prob.solve(verbose=False)

    if prob.status != cp.OPTIMAL:
        logger.warning("能耗优化求解失败，降级为时间最优轨迹")
        return None

    return q.value.T


def optimize_obstacle_traj_with_efficiency(model, data, ee_pos, target_pos, config):
    global total_motion_time, total_energy_consume

    # 1. 鲁棒避障修正目标
    ee_vel = get_ee_cartesian_velocity(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))
    corrected_cart_target = robust_artificial_potential_field(ee_pos, ee_vel, target_pos, config.obstacle_list, config)

    # 2. 平滑笛卡尔目标轨迹（使用修复后的函数）
    corrected_cart_target = smooth_cartesian_traj([ee_pos, corrected_cart_target], config.smooth_factor)[-1]

    # 3. 逆解得到关节目标
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    data.site_xpos[ee_site_id] = corrected_cart_target
    mujoco.mj_inverse(model, data)
    end_joint = data.qpos[:6].copy()
    start_joint = data.qpos[:6].copy()

    # 4. 时间最优轨迹
    seg_time = 2.0
    time_opt_pos, time_opt_vel, time_opt_acc = time_optimal_joint_trajectory(start_joint, end_joint, seg_time, config)

    # 5. 能耗最优优化
    energy_opt_pos = energy_optimal_trajectory(time_opt_pos, seg_time, config)
    if energy_opt_pos is None:
        final_joint_traj = time_opt_pos
    else:
        final_joint_traj = energy_opt_pos

    # 6. 计算真实能耗
    dt = seg_time / len(final_joint_traj)
    seg_energy = 0.0

    for traj_idx in range(len(final_joint_traj)):
        if traj_idx == 0:
            continue

        # 使用真实扭矩计算能耗
        seg_energy += calculate_real_energy_consumption(model, data, config, dt)

    # 更新全局统计
    total_motion_time += seg_time
    total_energy_consume += seg_energy

    # 记录轨迹历史
    traj_history.append(corrected_cart_target)

    return final_joint_traj[0], corrected_cart_target, seg_energy


# ====================== 8. 机械臂模型 ======================
def get_arm_xml_with_obstacles(config):
    arm_xml = """
<mujoco model="6dof_arm_efficiency_optimized">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <material name="gray" rgba="0.7 0.7 0.7 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="obstacle" rgba="1 0 0 0.5"/>
    <material name="critical_obstacle" rgba="1 0 0 0.7"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="gray"/>
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="5.0" diaginertia="0.01 0.01 0.01"/>
      <geom name="base_geom" type="cylinder" size="0.15 0.1" pos="0 0 0" material="gray"/>
      <joint name="joint0" type="hinge" axis="0 0 1" pos="0 0 0.1"/>
      <body name="link1" pos="0 0 0.1">
        <inertial pos="0 0 0.15" mass="1.2" diaginertia="0.02 0.02 0.02"/>
        <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" material="blue"/>
        <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.3"/>
        <body name="link2" pos="0 0 0.3">
          <inertial pos="0.2 0 0" mass="1.0" diaginertia="0.015 0.015 0.015"/>
          <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.4 0 0" material="blue"/>
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0.4 0 0"/>
          <body name="link3" pos="0.4 0 0">
            <inertial pos="0.175 0 0" mass="0.8" diaginertia="0.01 0.01 0.01"/>
            <geom name="link3_geom" type="capsule" size="0.04" fromto="0 0 0 0.35 0 0" material="blue"/>
            <joint name="joint3" type="hinge" axis="1 0 0" pos="0.35 0 0"/>
            <body name="link4" pos="0.35 0 0">
              <inertial pos="0 0 0.125" mass="0.6" diaginertia="0.008 0.008 0.008"/>
              <geom name="link4_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.25" material="blue"/>
              <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.25"/>
              <body name="link5" pos="0 0 0.25">
                <inertial pos="0 0 0.1" mass="0.4" diaginertia="0.008 0.008 0.008"/>
                <geom name="link5_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" material="blue"/>
                <joint name="joint5" type="hinge" axis="1 0 0" pos="0 0 0.2"/>
                <body name="end_effector" pos="0 0 0.2">
                  <inertial pos="0 0 0" mass="0.2" diaginertia="0.005 0.005 0.005"/>
                  <geom name="ee_geom" type="box" size="0.08 0.08 0.08" pos="0 0 0" material="red"/>
                  <site name="ee_site" pos="0 0 0" type="sphere" size="0.01" rgba="1 0 0 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    """

    # 添加障碍物
    for i, obs in enumerate(config.obstacle_list):
        x, y, z, r = obs
        material = "critical_obstacle" if i == 0 else "obstacle"
        arm_xml += f"""
    <geom name="obstacle_{i}" type="sphere" size="{r}" pos="{x} {y} {z}" material="{material}"/>
        """

    arm_xml += """
  </worldbody>
  <actuator>
    <motor name="motor0" joint="joint0" ctrlrange="-3.14 3.14" gear="100"/>
    <motor name="motor1" joint="joint1" ctrlrange="-1.57 1.57" gear="100"/>
    <motor name="motor2" joint="joint2" ctrlrange="-1.57 1.57" gear="100"/>
    <motor name="motor3" joint="joint3" ctrlrange="-3.14 3.14" gear="100"/>
    <motor name="motor4" joint="joint4" ctrlrange="-1.57 1.57" gear="100"/>
    <motor name="motor5" joint="joint5" ctrlrange="-3.14 3.14" gear="100"/>
  </actuator>
</mujoco>
    """
    return arm_xml


# ====================== 9. 仿真主逻辑 ======================
def run_enhanced_simulation():
    global total_motion_time, total_energy_consume

    # 生成模型XML
    arm_xml = get_arm_xml_with_obstacles(config)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        logger.info("✅ 增强版机械臂仿真模型加载成功！")
        logger.info(
            f"🔧 配置：轨迹点数={config.traj_interp_points}, 时间权重={config.time_weight}, 平滑系数={config.smooth_factor}")

        # 预计算关节起点
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        joint_waypoints = []

        for cart_pos in config.cart_waypoints:
            mujoco.mj_resetData(model, data)
            data.site_xpos[ee_site_id] = cart_pos
            mujoco.mj_inverse(model, data)
            joint_waypoints.append(data.qpos[:6].copy())

        # 启动仿真视图
        with mujoco.viewer.launch_passive(model, data) as viewer:
            logger.info("\n🎮 增强版机械臂仿真启动！")
            logger.info("💡 核心增强：全链路碰撞检测 + 轨迹平滑 + 真实能耗计算 + 可视化增强")
            logger.info("💡 按 Ctrl+C 退出\n")

            current_waypoint = 0
            last_print_time = 0.0
            pause_simulation = False

            while viewer.is_running():
                if pause_simulation:
                    time.sleep(0.1)
                    continue

                t_total = data.time
                ee_pos = data.site_xpos[ee_site_id].tolist()

                # 切换目标点
                if current_waypoint < len(config.cart_waypoints):
                    target_cart = config.cart_waypoints[current_waypoint]
                    if np.linalg.norm(np.array(ee_pos) - np.array(target_cart)) < 0.01:
                        current_waypoint = (current_waypoint + 1) % len(config.cart_waypoints)
                        logger.info(f"\n🔄 切换到目标点 {current_waypoint}: {np.round(target_cart, 3)}")
                else:
                    target_cart = config.cart_waypoints[-1]

                # 轨迹优化
                target_joints, corrected_cart, seg_energy = optimize_obstacle_traj_with_efficiency(
                    model, data, ee_pos, target_cart, config
                )

                # 全链路碰撞检测
                is_collision, min_obs_dist = full_arm_collision_check(model, data, config)

                # 紧急避障
                if is_collision:
                    logger.warning("🆘 检测到碰撞风险，执行紧急避障！")
                    emergency_rep = np.array(ee_pos) - np.array(config.obstacle_list[0][:3])
                    emergency_rep = emergency_rep / np.linalg.norm(emergency_rep) * 0.05
                    corrected_cart = np.array(corrected_cart) + emergency_rep
                    data.site_xpos[ee_site_id] = corrected_cart
                    mujoco.mj_inverse(model, data)
                    target_joints = data.qpos[:6].copy()

                # PD控制
                ctrl_signals = []
                for i in range(6):
                    k_p = 8.0
                    k_d = 0.2
                    current_pos = data.qpos[i]
                    current_vel = data.qvel[i]
                    pos_error = target_joints[i] - current_pos
                    vel_error = -current_vel
                    ctrl = k_p * pos_error + k_d * vel_error
                    max_ctrl = config.max_torque[i] / 100.0
                    ctrl = np.clip(ctrl, -max_ctrl, max_ctrl)
                    ctrl_signals.append(ctrl)

                data.ctrl[:6] = ctrl_signals

                # 打印统计信息
                if t_total - last_print_time > 2.0 and t_total > 0:
                    ee_vel = get_ee_cartesian_velocity(model, data, ee_site_id)
                    avg_vel = np.linalg.norm(ee_vel)
                    avg_energy = total_energy_consume / t_total if t_total > 0 else 0.0

                    logger.info(f"\n⏱️  仿真时间：{t_total:.2f}s | 累计运动时间：{total_motion_time:.2f}s")
                    logger.info(f"   末端位置：{np.round(ee_pos, 3)} | 目标位置：{np.round(corrected_cart, 3)}")
                    logger.info(f"   末端速度：{avg_vel:.4f}m/s | 最近障碍距离：{min_obs_dist:.3f}m")
                    logger.info(f"   累计能耗：{total_energy_consume:.2f}J | 平均能耗：{avg_energy:.2f}J/s")
                    logger.info(f"   碰撞风险：{'⚠️  高' if is_collision else '✅  低'}")
                    last_print_time = t_total

                # 增强可视化（适配新版API）
                draw_enhanced_visualization(viewer, model, data, config)

                # 运行仿真步
                mujoco.mj_step(model, data)
                viewer.sync()

                try:
                    mujoco.utils.mju_sleep(1 / 60)
                except:
                    time.sleep(1 / 60)

    except KeyboardInterrupt:
        logger.info("\n🛑 用户终止仿真")
    except Exception as e:
        logger.error(f"❌ 仿真出错：{e}")
        import traceback
        traceback.print_exc()
    finally:
        os.unlink(xml_path)
        # 最终统计
        logger.info(f"\n📊 仿真结束 - 最终统计")
        logger.info(f"   总运动时间：{total_motion_time:.2f}s")
        logger.info(f"   总能耗：{total_energy_consume:.2f}J")
        logger.info(
            f"   综合得分：{total_motion_time * config.time_weight + total_energy_consume * config.energy_weight:.2f}")


if __name__ == "__main__":
    # 安装依赖
    # pip install cvxpy scipy ecos osqp mujoco
    run_enhanced_simulation()