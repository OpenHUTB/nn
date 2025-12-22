import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time
from scipy import interpolate

# ====================== 1. 全局配置（物理约束+避障+时间最优参数） ======================
# 机械臂物理约束（UR5参考参数）
CONSTRAINTS = {
    "max_vel": [1.0, 0.8, 0.8, 1.2, 0.9, 1.2],  # 各关节最大角速度 (rad/s)
    "max_acc": [0.5, 0.4, 0.4, 0.6, 0.5, 0.6],  # 各关节最大角加速度 (rad/s²)
    "max_jerk": [0.3, 0.2, 0.2, 0.4, 0.3, 0.4],  # 最大加加速度 (rad/s³)
    "ctrl_limit": [-10.0, 10.0]  # 电机控制量限制
}

# 避障参数
OBSTACLE_CONFIG = {
    "k_att": 0.8,  # 引力系数
    "k_rep": 0.6,  # 斥力系数
    "rep_radius": 0.3,  # 斥力作用半径
    "obstacle_list": [  # 障碍物：[x,y,z,半径]
        [0.6, 0.1, 0.5, 0.1],
        [0.4, -0.1, 0.6, 0.08]
    ]
}

# 笛卡尔轨迹关键点
CART_WAYPOINTS = [
    [0.5, 0.0, 0.6],  # 起点
    [0.7, 0.2, 0.7],  # 中间点（障碍区）
    [0.8, 0.1, 0.8],  # 终点
    [0.7, 0.2, 0.7],  # 回中间点
    [0.5, 0.0, 0.6]  # 回起点
]


# ====================== 2. 核心：时间最优轨迹生成（梯形速度曲线） ======================
def time_optimal_trajectory(start, end, joint_idx):
    """
    生成单个关节的时间最优轨迹（梯形速度曲线）
    :param start: 起点角度 (rad)
    :param end: 终点角度 (rad)
    :param joint_idx: 关节索引（0-5）
    :return: 最优运动时间 + 轨迹点数组 [时间, 位置, 速度, 加速度]
    """
    max_vel = CONSTRAINTS["max_vel"][joint_idx]
    max_acc = CONSTRAINTS["max_acc"][joint_idx]
    delta_pos = end - start  # 位置差
    sign = np.sign(delta_pos)  # 运动方向

    # 步骤1：计算达到最大速度所需的加速时间和位移
    t_acc = max_vel / max_acc  # 加速到最大速度的时间
    s_acc = 0.5 * max_acc * t_acc ** 2  # 加速阶段位移

    # 步骤2：判断是否能达到最大速度（决定是梯形/三角形速度曲线）
    if abs(delta_pos) < 2 * s_acc:
        # 位移太小，无法匀速（三角形曲线）
        t_acc = np.sqrt(abs(delta_pos) / max_acc)
        t_const = 0  # 无匀速阶段
        total_time = 2 * t_acc
    else:
        # 能达到最大速度（梯形曲线）
        t_const = (abs(delta_pos) - 2 * s_acc) / max_vel  # 匀速时间
        total_time = 2 * t_acc + t_const

    # 步骤3：生成离散轨迹点（1ms步长，保证精度）
    dt = 0.001
    time_list = np.arange(0, total_time + dt, dt)
    pos_list = []
    vel_list = []
    acc_list = []

    for t in time_list:
        if t < t_acc:
            # 加速阶段
            pos = start + sign * 0.5 * max_acc * t ** 2
            vel = sign * max_acc * t
            acc = sign * max_acc
        elif t < t_acc + t_const:
            # 匀速阶段
            pos = start + sign * (s_acc + max_vel * (t - t_acc))
            vel = sign * max_vel
            acc = 0
        else:
            # 减速阶段
            t_dec = t - (t_acc + t_const)
            pos = end - sign * 0.5 * max_acc * t_dec ** 2
            vel = sign * (max_vel - max_acc * t_dec)
            acc = -sign * max_acc

        pos_list.append(pos)
        vel_list.append(vel)
        acc_list.append(acc)

    # 封装轨迹数据
    traj_data = np.vstack((time_list, pos_list, vel_list, acc_list)).T
    return total_time, traj_data


# ====================== 3. 多关节时间最优轨迹同步 ======================
def sync_joint_trajectories(joint_waypoints):
    """
    同步所有关节的时间最优轨迹（保证同时到达目标点）
    :param joint_waypoints: 关节轨迹关键点 [[j0,j1,...j5], ...]
    :return: 全局时间最优轨迹数组 [时间, j0_pos, j1_pos, ..., j5_pos]
    """
    num_joints = 6
    segment_trajs = []  # 存储每段轨迹的各关节数据

    # 遍历每段轨迹（关键点之间的段）
    for seg_idx in range(len(joint_waypoints) - 1):
        start_wp = joint_waypoints[seg_idx]
        end_wp = joint_waypoints[seg_idx + 1]
        joint_trajs = []
        seg_max_time = 0

        # 为每个关节生成时间最优轨迹
        for j in range(num_joints):
            seg_time, traj_data = time_optimal_trajectory(start_wp[j], end_wp[j], j)
            joint_trajs.append(traj_data)
            if seg_time > seg_max_time:
                seg_max_time = seg_time  # 取最长时间作为段总时间

        # 同步所有关节轨迹（拉伸到段总时间）
        synced_seg_traj = []
        dt = 0.001
        seg_time_list = np.arange(0, seg_max_time + dt, dt)

        for t in seg_time_list:
            row = [t]
            for j in range(num_joints):
                # 找到当前时间对应的关节位置（插值补全）
                j_traj = joint_trajs[j]
                if t > j_traj[-1, 0]:
                    pos = j_traj[-1, 1]  # 已到达目标，保持位置
                else:
                    pos = np.interp(t, j_traj[:, 0], j_traj[:, 1])
                row.append(pos)
            synced_seg_traj.append(row)

        segment_trajs.append(np.array(synced_seg_traj))

    # 拼接所有段的轨迹
    global_traj = segment_trajs[0]
    for seg in segment_trajs[1:]:
        # 时间偏移（累加前序段的总时间）
        seg[:, 0] += global_traj[-1, 0]
        global_traj = np.vstack((global_traj, seg))

    return global_traj


# ====================== 4. 避障+闭环控制（复用已有逻辑） ======================
def artificial_potential_field(ee_pos, target_pos):
    ee_pos = np.array(ee_pos)
    target_pos = np.array(target_pos)
    obstacle_list = OBSTACLE_CONFIG["obstacle_list"]
    k_att = OBSTACLE_CONFIG["k_att"]
    k_rep = OBSTACLE_CONFIG["k_rep"]
    rep_radius = OBSTACLE_CONFIG["rep_radius"]

    # 引力
    att_force = k_att * (target_pos - ee_pos)

    # 斥力
    rep_force = np.zeros(3)
    for obs in obstacle_list:
        obs_pos = np.array(obs[:3])
        obs_radius = obs[3]
        dist = np.linalg.norm(ee_pos - obs_pos)

        if dist < rep_radius + obs_radius:
            if dist < 1e-6:
                dist = 1e-6
            rep_dir = (ee_pos - obs_pos) / dist
            rep_force += k_rep * (1 / (dist - obs_radius) - 1 / rep_radius) * (1 / dist ** 2) * rep_dir

    # 修正目标位置
    corrected_target = ee_pos + att_force + rep_force
    corrected_target = np.clip(corrected_target, [0.3, -0.4, 0.2], [0.9, 0.4, 1.0])

    return corrected_target.tolist()


def closed_loop_constraint_control(data, target_joints, joint_idx):
    k_p = 8.0
    k_d = 0.2

    current_pos = data.qpos[joint_idx]
    current_vel = data.qvel[joint_idx]

    pos_error = target_joints[joint_idx] - current_pos
    vel_error = -current_vel

    ctrl = k_p * pos_error + k_d * vel_error
    ctrl = np.clip(ctrl, CONSTRAINTS["ctrl_limit"][0], CONSTRAINTS["ctrl_limit"][1])

    return ctrl


# ====================== 5. 机械臂模型（带障碍物） ======================
def get_arm_xml_with_obstacles():
    arm_xml = """
<mujoco model="6dof_arm_time_optimal">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <material name="gray" rgba="0.7 0.7 0.7 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="obstacle" rgba="1 0 0 0.5"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="gray"/>
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.15 0.1" pos="0 0 0" material="gray"/>
      <joint name="joint0" type="hinge" axis="0 0 1" pos="0 0 0.1"/>
      <body name="link1" pos="0 0 0.1">
        <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" material="blue"/>
        <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.3"/>
        <body name="link2" pos="0 0 0.3">
          <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.4 0 0" material="blue"/>
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0.4 0 0"/>
          <body name="link3" pos="0.4 0 0">
            <geom name="link3_geom" type="capsule" size="0.04" fromto="0 0 0 0.35 0 0" material="blue"/>
            <joint name="joint3" type="hinge" axis="1 0 0" pos="0.35 0 0"/>
            <body name="link4" pos="0.35 0 0">
              <geom name="link4_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.25" material="blue"/>
              <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.25"/>
              <body name="link5" pos="0 0 0.25">
                <geom name="link5_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" material="blue"/>
                <joint name="joint5" type="hinge" axis="1 0 0" pos="0 0 0.2"/>
                <body name="end_effector" pos="0 0 0.2">
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

    for i, obs in enumerate(OBSTACLE_CONFIG["obstacle_list"]):
        x, y, z, r = obs
        arm_xml += f"""
    <geom name="obstacle_{i}" type="sphere" size="{r}" pos="{x} {y} {z}" material="obstacle"/>
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


# ====================== 6. 预计算关节关键点（兼容旧版MuJoCo） ======================
def precompute_joint_waypoints(model, data, cart_waypoints):
    joint_waypoints = []
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    for cart_pos in cart_waypoints:
        mujoco.mj_resetData(model, data)
        data.site_xpos[ee_site_id] = cart_pos
        mujoco.mj_inverse(model, data)
        joint_waypoints.append(data.qpos[:6].copy())

    return joint_waypoints


# ====================== 7. 主仿真逻辑（时间最优+避障+约束） ======================
def run_time_optimal_simulation():
    arm_xml = get_arm_xml_with_obstacles()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ 时间最优机械臂模型加载成功！")
        print(f"🔧 物理约束：最大速度={CONSTRAINTS['max_vel'][0]}rad/s，最大加速度={CONSTRAINTS['max_acc'][0]}rad/s²")
        print(f"🔧 避障参数：斥力半径={OBSTACLE_CONFIG['rep_radius']}m")

        # 步骤1：预计算笛卡尔对应的关节关键点
        joint_waypoints = precompute_joint_waypoints(model, data, CART_WAYPOINTS)

        # 步骤2：生成时间最优同步轨迹
        global_traj = sync_joint_trajectories(joint_waypoints)
        total_opt_time = global_traj[-1, 0]
        print(f"\n⏱️  时间最优轨迹生成完成！总运动时间：{total_opt_time:.2f}s")

        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        traj_length = len(global_traj)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\n🎮 时间最优机械臂仿真启动！")
            print("💡 核心功能：梯形速度曲线（时间最优）+ 避障 + 物理约束")
            print("💡 特征：机械臂以最短时间运动，同时避开障碍物、不超物理极限")
            print("💡 按 Ctrl+C 退出")

            while viewer.is_running():
                # 1. 获取当前仿真时间
                t_sim = data.time

                # 2. 超出总时间则循环
                if t_sim > total_opt_time:
                    mujoco.mj_resetData(model, data)
                    continue

                # 3. 查找当前时间对应的目标关节角度（插值）
                target_joints = []
                for j in range(6):
                    pos = np.interp(t_sim, global_traj[:, 0], global_traj[:, j + 1])
                    target_joints.append(pos)

                # 4. 避障修正（实时调整目标）
                ee_pos = data.site_xpos[ee_site_id].tolist()
                # 正运动学获取当前笛卡尔目标
                mujoco.mj_forward(model, data)
                raw_cart_target = data.site_xpos[ee_site_id].copy()
                # 避障修正
                corrected_cart_target = artificial_potential_field(ee_pos, raw_cart_target)
                # 修正关节目标
                data.site_xpos[ee_site_id] = corrected_cart_target
                mujoco.mj_inverse(model, data)
                corrected_joint_target = data.qpos[:6].copy()
                # 融合时间最优和避障目标（加权）
                target_joints = [0.8 * target_joints[i] + 0.2 * corrected_joint_target[i] for i in range(6)]

                # 5. 物理约束+闭环控制
                ctrl_signals = []
                for i in range(6):
                    # 约束关节角度范围
                    target_joints[i] = np.clip(target_joints[i], model.actuator_ctrlrange[i][0],
                                               model.actuator_ctrlrange[i][1])
                    # 闭环PD控制
                    ctrl = closed_loop_constraint_control(data, target_joints, i)
                    ctrl_signals.append(ctrl)

                # 6. 发送控制指令
                data.ctrl[:6] = ctrl_signals

                # 7. 打印关键状态（每0.5秒）
                if int(t_sim * 2) % 1 == 0 and t_sim > 0:
                    # 计算当前关节速度
                    joint_vel = [data.qvel[i] for i in range(6)]
                    max_vel = max([abs(v) for v in joint_vel])
                    # 计算末端与最近障碍距离
                    obs_distances = []
                    for obs in OBSTACLE_CONFIG["obstacle_list"]:
                        dist = np.linalg.norm(np.array(ee_pos) - np.array(obs[:3]))
                        obs_distances.append(dist)
                    min_obs_dist = min(obs_distances) if obs_distances else 0

                    print(f"\n⏱️  仿真时间：{t_sim:.2f}s / 最优总时间：{total_opt_time:.2f}s")
                    print(f"   最大关节速度：{max_vel:.3f}rad/s (上限：{CONSTRAINTS['max_vel'][0]})")
                    print(f"   末端与最近障碍距离：{min_obs_dist:.3f}m")
                    print(f"   末端位置：{np.round(ee_pos, 3)}")

                # 8. 运行仿真步
                mujoco.mj_step(model, data)
                viewer.sync()

                # 9. 帧率控制
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


if __name__ == "__main__":
    run_time_optimal_simulation()