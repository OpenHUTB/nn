import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time
from scipy import interpolate
import cvxpy as cp  # 用于能耗最优的二次规划
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 全局配置（鲁棒+效率双优化） ======================
# 物理约束（UR5工业级参数）
CONSTRAINTS = {
    "max_vel": [1.0, 0.8, 0.8, 1.2, 0.9, 1.2],  # 关节最大速度 (rad/s)
    "max_acc": [0.5, 0.4, 0.4, 0.6, 0.5, 0.6],  # 关节最大加速度 (rad/s²)
    "max_jerk": [0.3, 0.2, 0.2, 0.4, 0.3, 0.4],  # 关节最大加加速度 (rad/s³)
    "max_torque": [15.0, 15.0, 10.0, 5.0, 5.0, 3.0],  # 关节最大扭矩 (N·m)
    "ctrl_limit": [-10.0, 10.0]
}

# 避障鲁棒性参数
OBSTACLE_CONFIG = {
    "base_k_att": 0.8,  # 基础引力系数
    "base_k_rep": 0.6,  # 基础斥力系数
    "rep_radius": 0.3,  # 斥力作用半径
    "stagnant_threshold": 0.01,  # 停滞速度阈值 (m/s)
    "stagnant_time": 1.0,  # 停滞判定时间 (s)
    "guide_offset": 0.1,  # 局部最优引导偏移量 (m)
    "obstacle_list": [  # 障碍物列表 [x,y,z,半径]
        [0.6, 0.1, 0.5, 0.1],  # 障碍1：易导致局部最优
        [0.55, 0.05, 0.55, 0.08],  # 障碍2：密集障碍
        [0.4, -0.1, 0.6, 0.08]  # 障碍3
    ]
}

# 效率优化参数（工业场景可配置）
EFFICIENCY_CONFIG = {
    "time_weight": 0.6,  # 时间权重（0-1，越大越优先时间）
    "energy_weight": 0.4,  # 能耗权重（0-1，越大越优先能耗）
    "traj_interp_points": 50,  # 轨迹插值点数
    "safety_margin": 0.05,  # 碰撞安全裕度 (m)
    "opt_horizon": 1.0  # 优化时域 (s)
}

# 笛卡尔轨迹关键点（工业典型路径）
CART_WAYPOINTS = [
    [0.5, 0.0, 0.6],  # 起点
    [0.6, 0.0, 0.58],  # 中间点（障碍夹缝）
    [0.8, 0.1, 0.8],  # 终点
    [0.6, 0.0, 0.58],  # 回中间点
    [0.5, 0.0, 0.6]  # 回起点
]

# 全局变量
stagnant_start_time = None
total_motion_time = 0.0  # 累计运动时间
total_energy_consume = 0.0  # 累计能耗

# 预定义关节惯性参数（适配所有MuJoCo版本）
JOINT_INERTIA = [0.01, 0.02, 0.015, 0.01, 0.008, 0.005]
JOINT_GRAVITY = [0.5, 0.8, 0.6, 0.3, 0.2, 0.1]


# ====================== 2. 基础工具函数（兼容所有MuJoCo版本） ======================
def get_ee_cartesian_velocity(model, data, ee_site_id):
    """计算末端笛卡尔速度（兼容所有MuJoCo版本）"""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    joint_vel = data.qvel[:6]
    ee_cart_vel = jacp @ joint_vel
    return ee_cart_vel


def calculate_joint_torque(model, data, joint_idx):
    """计算关节扭矩（能耗核心指标，适配新版MuJoCo）"""
    # 方案1：使用预定义惯性参数（兼容所有版本）
    inertia = JOINT_INERTIA[joint_idx]
    gravity_comp = JOINT_GRAVITY[joint_idx]

    # 方案2（备选）：从data中获取实时扭矩（新版MuJoCo推荐）
    # torque = data.qfrc_actuator[joint_idx]  # 实际输出扭矩

    # 计算关节加速度（数值微分）
    joint_acc = np.gradient(data.qvel[joint_idx]) if data.time > 0 else 0.0
    torque = inertia * joint_acc + gravity_comp
    torque = np.clip(torque, -CONSTRAINTS["max_torque"][joint_idx], CONSTRAINTS["max_torque"][joint_idx])
    return torque


# ====================== 3. 鲁棒性避障（保留原有核心逻辑） ======================
def check_local_optimum(ee_vel, ee_pos, target_pos):
    """检测局部最优并生成引导目标"""
    global stagnant_start_time
    vel_mag = np.linalg.norm(ee_vel)
    if vel_mag < OBSTACLE_CONFIG["stagnant_threshold"]:
        if stagnant_start_time is None:
            stagnant_start_time = time.time()
        elif time.time() - stagnant_start_time > OBSTACLE_CONFIG["stagnant_time"]:
            print(f"\n⚠️  检测到局部最优！末端速度={vel_mag:.4f}m/s")
            dir_to_target = np.array(target_pos) - np.array(ee_pos)
            dir_to_target = dir_to_target / np.linalg.norm(dir_to_target) if np.linalg.norm(
                dir_to_target) > 1e-6 else np.array([0, 0, 0.1])
            guide_target = np.array(ee_pos) + dir_to_target * OBSTACLE_CONFIG["guide_offset"]
            stagnant_start_time = None
            return True, guide_target.tolist()
    else:
        stagnant_start_time = None
    return False, target_pos


def adaptive_potential_params(ee_pos, obstacle_list):
    """自适应势场参数"""
    obs_distances = [np.linalg.norm(np.array(ee_pos) - np.array(obs[:3])) for obs in obstacle_list]
    min_dist = min(obs_distances) if obs_distances else 1.0
    k_rep = OBSTACLE_CONFIG["base_k_rep"] if min_dist > 0.2 else OBSTACLE_CONFIG["base_k_rep"] * 2.0
    k_att = OBSTACLE_CONFIG["base_k_att"] if len(obstacle_list) <= 2 else OBSTACLE_CONFIG["base_k_att"] * 0.5
    return k_att, k_rep


def robust_artificial_potential_field(ee_pos, ee_vel, target_pos, obstacle_list):
    """鲁棒版人工势场法"""
    ee_pos = np.array(ee_pos)
    target_pos = np.array(target_pos)

    # 局部最优规避
    is_local_opt, guide_target = check_local_optimum(ee_vel, ee_pos, target_pos)
    current_target = np.array(guide_target) if is_local_opt else target_pos

    # 自适应参数
    k_att, k_rep = adaptive_potential_params(ee_pos, obstacle_list)

    # 引力+斥力计算
    att_force = k_att * (current_target - ee_pos)
    rep_force = np.zeros(3)
    for obs in obstacle_list:
        obs_pos = np.array(obs[:3])
        obs_radius = obs[3]
        dist = np.linalg.norm(ee_pos - obs_pos)
        if dist < OBSTACLE_CONFIG["rep_radius"] + obs_radius:
            rep_dir = (ee_pos - obs_pos) / (dist + 1e-6)
            rep_force += k_rep * (1 / (dist - obs_radius) - 1 / OBSTACLE_CONFIG["rep_radius"]) * (
                        1 / dist ** 2) * rep_dir

    # 修正目标并约束
    corrected_target = ee_pos + att_force + rep_force
    corrected_target = np.clip(corrected_target, [0.3, -0.4, 0.2], [0.9, 0.4, 1.0])

    return corrected_target.tolist()


def collision_check_approx(ee_pos, joint_pos, obstacle_list):
    """碰撞冗余检测"""
    ee_collision = False
    min_ee_dist = 100.0
    for obs in obstacle_list:
        obs_pos = np.array(obs[:3])
        obs_radius = obs[3]
        dist = np.linalg.norm(np.array(ee_pos) - obs_pos)
        min_ee_dist = min(min_ee_dist, dist)
        if dist < obs_radius + EFFICIENCY_CONFIG["safety_margin"]:
            ee_collision = True
            break
    return ee_collision, min_ee_dist


# ====================== 4. 效率优化核心：时间最优轨迹规划 ======================
def time_optimal_joint_trajectory(start_joint, end_joint, seg_time):
    """
    时间最优关节轨迹（梯形速度曲线，满足速度/加速度约束）
    :return: 时间最优的关节位置/速度/加速度轨迹
    """
    n_joints = 6
    traj_points = EFFICIENCY_CONFIG["traj_interp_points"]
    t_steps = np.linspace(0, seg_time, traj_points)

    # 初始化轨迹数组
    opt_pos = np.zeros((traj_points, n_joints))
    opt_vel = np.zeros((traj_points, n_joints))
    opt_acc = np.zeros((traj_points, n_joints))

    for j in range(n_joints):
        delta = end_joint[j] - start_joint[j]
        max_vel = CONSTRAINTS["max_vel"][j]
        max_acc = CONSTRAINTS["max_acc"][j]

        # 计算梯形速度曲线的关键时间点
        t_acc = max_vel / max_acc  # 加速时间
        s_acc = 0.5 * max_acc * t_acc ** 2  # 加速段位移

        if abs(delta) < 2 * s_acc:
            # 三角形速度曲线（未到最大速度）
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
            # 梯形速度曲线（达到最大速度）
            t_const = (abs(delta) - 2 * s_acc) / max_vel  # 匀速时间
            t_joint = 2 * t_acc + t_const
            for i, t in enumerate(t_steps):
                if t <= t_acc:
                    # 加速段
                    opt_pos[i, j] = start_joint[j] + 0.5 * max_acc * t ** 2 * np.sign(delta)
                    opt_vel[i, j] = max_acc * t * np.sign(delta)
                    opt_acc[i, j] = max_acc * np.sign(delta)
                elif t <= t_acc + t_const:
                    # 匀速段
                    opt_pos[i, j] = start_joint[j] + (s_acc + max_vel * (t - t_acc)) * np.sign(delta)
                    opt_vel[i, j] = max_vel * np.sign(delta)
                    opt_acc[i, j] = 0.0
                else:
                    # 减速段
                    t_rem = t_joint - t
                    opt_pos[i, j] = end_joint[j] - 0.5 * max_acc * t_rem ** 2 * np.sign(delta)
                    opt_vel[i, j] = max_acc * t_rem * np.sign(delta)
                    opt_acc[i, j] = -max_acc * np.sign(delta)

        # 约束速度/加速度
        opt_vel[:, j] = np.clip(opt_vel[:, j], -max_vel, max_vel)
        opt_acc[:, j] = np.clip(opt_acc[:, j], -max_acc, max_acc)

    return opt_pos, opt_vel, opt_acc


# ====================== 5. 效率优化核心：能耗最优二次规划（兼容所有求解器） ======================
def energy_optimal_trajectory(joint_waypoints, seg_time):
    """
    能耗最优轨迹（二次规划求解，最小化扭矩平方积分）
    :return: 能耗最优的关节位置轨迹
    """
    n_joints = 6
    n_points = len(joint_waypoints)
    t_step = seg_time / (n_points - 1)

    # 定义优化变量
    q = cp.Variable((n_joints, n_points))  # 关节位置
    qd = cp.Variable((n_joints, n_points))  # 关节速度
    qdd = cp.Variable((n_joints, n_points))  # 关节加速度

    # 代价函数：最小化能耗（扭矩平方积分≈加速度平方积分）
    energy_cost = cp.sum_squares(qdd)
    time_cost = cp.sum(cp.max(cp.abs(qd), axis=1))  # 时间代价：速度越大时间越短
    total_cost = EFFICIENCY_CONFIG["time_weight"] * time_cost + EFFICIENCY_CONFIG["energy_weight"] * energy_cost

    # 约束条件
    constraints = []
    # 初始/终止条件
    constraints.append(q[:, 0] == joint_waypoints[0])
    constraints.append(q[:, -1] == joint_waypoints[-1])
    constraints.append(qd[:, 0] == 0)
    constraints.append(qd[:, -1] == 0)
    # 速度/加速度约束
    for j in range(n_joints):
        constraints.append(qd[j, :] <= CONSTRAINTS["max_vel"][j])
        constraints.append(qd[j, :] >= -CONSTRAINTS["max_vel"][j])
        constraints.append(qdd[j, :] <= CONSTRAINTS["max_acc"][j])
        constraints.append(qdd[j, :] >= -CONSTRAINTS["max_acc"][j])
    # 动力学约束（差分）
    for i in range(n_points - 1):
        constraints.append(qd[:, i + 1] == (q[:, i + 1] - q[:, i]) / t_step)
        constraints.append(qdd[:, i + 1] == (qd[:, i + 1] - qd[:, i]) / t_step)

    # 求解二次规划（自动选择可用求解器，增加容错）
    prob = cp.Problem(cp.Minimize(total_cost), constraints)
    try:
        # 优先尝试ECOS求解器
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            # 备选：OSQP求解器（CVXPY默认推荐）
            prob.solve(solver=cp.OSQP, verbose=False)
        except:
            # 最后：使用CVXPY自动选择的求解器
            prob.solve(verbose=False)

    if prob.status != cp.OPTIMAL:
        print("⚠️  能耗优化求解失败，降级为时间最优轨迹")
        return None
    return q.value.T


# ====================== 6. 效率+鲁棒融合：避障轨迹的双优优化（修复索引越界） ======================
def optimize_obstacle_traj_with_efficiency(model, data, ee_pos, target_pos, obstacle_list):
    """
    融合避障鲁棒性+时间/能耗最优的轨迹规划
    :return: 优化后的关节目标、当前段能耗
    """
    global total_motion_time, total_energy_consume

    # 步骤1：鲁棒避障修正笛卡尔目标
    ee_vel = get_ee_cartesian_velocity(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))
    corrected_cart_target = robust_artificial_potential_field(ee_pos, ee_vel, target_pos, obstacle_list)

    # 步骤2：逆解得到关节目标
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    data.site_xpos[ee_site_id] = corrected_cart_target
    mujoco.mj_inverse(model, data)
    end_joint = data.qpos[:6].copy()
    start_joint = data.qpos[:6].copy()  # 当前关节位置为起点

    # 步骤3：时间最优轨迹初值
    seg_time = 2.0  # 初始段时间
    time_opt_pos, time_opt_vel, time_opt_acc = time_optimal_joint_trajectory(start_joint, end_joint, seg_time)

    # 步骤4：能耗最优优化（二次规划）
    energy_opt_pos = energy_optimal_trajectory(time_opt_pos, seg_time)
    if energy_opt_pos is None:
        final_joint_traj = time_opt_pos
    else:
        final_joint_traj = energy_opt_pos

    # 步骤5：计算当前段能耗（扭矩平方积分，修复索引越界错误）
    seg_energy = 0.0
    # 遍历每个轨迹点
    for traj_idx in range(len(final_joint_traj)):
        # 跳过第一个点（无加速度）
        if traj_idx == 0:
            continue

        # 遍历每个关节计算扭矩
        for joint_idx in range(6):
            # 获取当前轨迹点和上一个轨迹点的关节角度
            curr_angle = final_joint_traj[traj_idx, joint_idx]
            prev_angle = final_joint_traj[traj_idx - 1, joint_idx]

            # 计算关节速度（差分）
            dt = seg_time / len(final_joint_traj)
            joint_vel = (curr_angle - prev_angle) / dt

            # 计算关节加速度（差分，使用前一个速度）
            if traj_idx == 1:
                joint_acc = joint_vel / dt
            else:
                prev_vel = (final_joint_traj[traj_idx - 1, joint_idx] - final_joint_traj[traj_idx - 2, joint_idx]) / dt
                joint_acc = (joint_vel - prev_vel) / dt

            # 计算扭矩和能耗（积分）
            torque = JOINT_INERTIA[joint_idx] * joint_acc + JOINT_GRAVITY[joint_idx]
            seg_energy += np.square(torque) * dt

    # 更新全局统计
    total_motion_time += seg_time
    total_energy_consume += seg_energy

    # 返回当前时刻的关节目标（取第一个插值点）
    return final_joint_traj[0], corrected_cart_target, seg_energy


# ====================== 7. 机械臂模型（修复XML语法错误） ======================
def get_arm_xml_with_obstacles():
    """生成带障碍的机械臂XML模型（修复inertial标签错误）"""
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
    <!-- 基座（包含inertial标签） -->
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="5.0" diaginertia="0.01 0.01 0.01"/>
      <geom name="base_geom" type="cylinder" size="0.15 0.1" pos="0 0 0" material="gray"/>
      <joint name="joint0" type="hinge" axis="0 0 1" pos="0 0 0.1"/>
      <!-- 连杆1 -->
      <body name="link1" pos="0 0 0.1">
        <inertial pos="0 0 0.15" mass="1.2" diaginertia="0.02 0.02 0.02"/>
        <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" material="blue"/>
        <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.3"/>
        <!-- 连杆2 -->
        <body name="link2" pos="0 0 0.3">
          <inertial pos="0.2 0 0" mass="1.0" diaginertia="0.015 0.015 0.015"/>
          <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.4 0 0" material="blue"/>
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0.4 0 0"/>
          <!-- 连杆3 -->
          <body name="link3" pos="0.4 0 0">
            <inertial pos="0.175 0 0" mass="0.8" diaginertia="0.01 0.01 0.01"/>
            <geom name="link3_geom" type="capsule" size="0.04" fromto="0 0 0 0.35 0 0" material="blue"/>
            <joint name="joint3" type="hinge" axis="1 0 0" pos="0.35 0 0"/>
            <!-- 连杆4 -->
            <body name="link4" pos="0.35 0 0">
              <inertial pos="0 0 0.125" mass="0.6" diaginertia="0.008 0.008 0.008"/>
              <geom name="link4_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.25" material="blue"/>
              <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.25"/>
              <!-- 连杆5 -->
              <body name="link5" pos="0 0 0.25">
                <inertial pos="0 0 0.1" mass="0.4" diaginertia="0.008 0.008 0.008"/>
                <geom name="link5_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" material="blue"/>
                <joint name="joint5" type="hinge" axis="1 0 0" pos="0 0 0.2"/>
                <!-- 末端执行器 -->
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
    <!-- 障碍物 -->
    """
    # 添加障碍
    for i, obs in enumerate(OBSTACLE_CONFIG["obstacle_list"]):
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


# ====================== 8. 仿真主逻辑 ======================
def run_efficiency_optimized_simulation():
    """运行效率+鲁棒双优化的仿真"""
    global total_motion_time, total_energy_consume
    arm_xml = get_arm_xml_with_obstacles()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ 效率+鲁棒双优化机械臂模型加载成功！")
        print(f"🔧 效率配置：时间权重={EFFICIENCY_CONFIG['time_weight']}, 能耗权重={EFFICIENCY_CONFIG['energy_weight']}")
        print(f"🔧 鲁棒配置：局部最优规避 + 自适应势场 + 碰撞冗余检测")

        # 预计算笛卡尔轨迹对应的关节起点
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        joint_waypoints = []
        for cart_pos in CART_WAYPOINTS:
            mujoco.mj_resetData(model, data)
            data.site_xpos[ee_site_id] = cart_pos
            mujoco.mj_inverse(model, data)
            joint_waypoints.append(data.qpos[:6].copy())

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\n🎮 效率+鲁棒双优化仿真启动！")
            print("💡 核心：避障安全 + 时间/能耗最优（工业级需求）")
            print("💡 按 Ctrl+C 退出\n")

            traj_idx = 0
            current_waypoint = 0
            last_print_time = 0.0

            while viewer.is_running():
                t_total = data.time
                ee_pos = data.site_xpos[ee_site_id].tolist()

                # 切换笛卡尔目标点
                if current_waypoint < len(CART_WAYPOINTS):
                    target_cart = CART_WAYPOINTS[current_waypoint]
                    # 距离目标点<0.01m时切换下一个
                    if np.linalg.norm(np.array(ee_pos) - np.array(target_cart)) < 0.01:
                        current_waypoint = (current_waypoint + 1) % len(CART_WAYPOINTS)
                        print(f"\n🔄 切换到目标点 {current_waypoint}: {np.round(target_cart, 3)}")
                else:
                    target_cart = CART_WAYPOINTS[-1]

                # 融合避障+效率优化的轨迹规划
                target_joints, corrected_cart, seg_energy = optimize_obstacle_traj_with_efficiency(
                    model, data, ee_pos, target_cart, OBSTACLE_CONFIG["obstacle_list"]
                )

                # 碰撞检测与紧急避障
                is_collision, min_obs_dist = collision_check_approx(ee_pos, target_joints,
                                                                    OBSTACLE_CONFIG["obstacle_list"])
                if is_collision:
                    emergency_rep = np.array(ee_pos) - np.array(OBSTACLE_CONFIG["obstacle_list"][0][:3])
                    emergency_rep = emergency_rep / np.linalg.norm(emergency_rep) * 0.05
                    corrected_cart = np.array(corrected_cart) + emergency_rep
                    data.site_xpos[ee_site_id] = corrected_cart
                    mujoco.mj_inverse(model, data)
                    target_joints = data.qpos[:6].copy()
                    print(f"🆘 紧急避障：修正目标={np.round(corrected_cart, 3)}")

                # 闭环PD控制（带扭矩约束）
                ctrl_signals = []
                for i in range(6):
                    k_p = 8.0
                    k_d = 0.2
                    current_pos = data.qpos[i]
                    current_vel = data.qvel[i]
                    pos_error = target_joints[i] - current_pos
                    vel_error = -current_vel
                    ctrl = k_p * pos_error + k_d * vel_error
                    # 扭矩约束（转换为控制量约束）
                    max_ctrl = CONSTRAINTS["max_torque"][i] / 100.0  # gear=100，直接计算
                    ctrl = np.clip(ctrl, -max_ctrl, max_ctrl)
                    ctrl_signals.append(ctrl)
                data.ctrl[:6] = ctrl_signals

                # 打印效率统计（每2秒）
                if t_total - last_print_time > 2.0 and t_total > 0:
                    ee_vel = get_ee_cartesian_velocity(model, data, ee_site_id)
                    avg_vel = np.linalg.norm(ee_vel)
                    avg_energy = total_energy_consume / t_total if t_total > 0 else 0.0
                    print(f"\n⏱️  时间：{t_total:.2f}s | 累计运动时间：{total_motion_time:.2f}s")
                    print(f"   末端位置：{np.round(ee_pos, 3)} | 目标位置：{np.round(corrected_cart, 3)}")
                    print(f"   末端速度：{avg_vel:.4f}m/s | 最近障碍：{min_obs_dist:.3f}m")
                    print(f"   累计能耗：{total_energy_consume:.2f}J | 平均能耗：{avg_energy:.2f}J/s")
                    print(f"   碰撞风险：{'是' if is_collision else '否'}")
                    last_print_time = t_total

                # 仿真步运行
                mujoco.mj_step(model, data)
                viewer.sync()
                try:
                    mujoco.utils.mju_sleep(1 / 60)
                except:
                    time.sleep(1 / 60)

    except Exception as e:
        print(f"❌ 仿真出错：{e}")
        import traceback
        traceback.print_exc()
    finally:
        os.unlink(xml_path)
        # 打印最终效率统计
        print(f"\n📊 仿真结束 - 效率统计")
        print(f"   总运动时间：{total_motion_time:.2f}s")
        print(f"   总能耗：{total_energy_consume:.2f}J")
        print(
            f"   时间/能耗综合得分：{total_motion_time * EFFICIENCY_CONFIG['time_weight'] + total_energy_consume * EFFICIENCY_CONFIG['energy_weight']:.2f}")


if __name__ == "__main__":
    # 安装依赖（首次运行需执行）
    # pip install cvxpy scipy ecos osqp
    run_efficiency_optimized_simulation()