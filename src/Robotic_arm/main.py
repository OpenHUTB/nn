import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time
from scipy import interpolate

# ====================== 1. 全局配置（物理约束+避障参数） ======================
CONSTRAINTS = {
    "max_vel": [1.0, 0.8, 0.8, 1.2, 0.9, 1.2],
    "max_acc": [0.5, 0.4, 0.4, 0.6, 0.5, 0.6],
    "max_jerk": [0.3, 0.2, 0.2, 0.4, 0.3, 0.4],
    "ctrl_limit": [-10.0, 10.0]
}

OBSTACLE_CONFIG = {
    "k_att": 0.8,
    "k_rep": 0.6,
    "rep_radius": 0.3,
    "obstacle_list": [
        [0.6, 0.1, 0.5, 0.1],
        [0.4, -0.1, 0.6, 0.08]
    ]
}

CART_WAYPOINTS = [
    [0.5, 0.0, 0.6],
    [0.7, 0.2, 0.7],
    [0.8, 0.1, 0.8],
    [0.7, 0.2, 0.7],
    [0.5, 0.0, 0.6]
]


# ====================== 2. 物理约束轨迹生成 ======================
def constrained_quintic_polynomial(start, end, total_time, t, joint_idx):
    s0, v0, a0 = start, 0, 0
    s1, v1, a1 = end, 0, 0

    T = total_time
    a = s0
    b = v0
    c = a0 / 2
    d = (20 * (s1 - s0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
    e = (30 * (s0 - s1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
    f = (12 * (s1 - s0) - (6 * v1 + 6 * v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)

    pos = a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5
    vel = b + 2 * c * t + 3 * d * t ** 2 + 4 * e * t ** 3 + 5 * f * t ** 4
    acc = 2 * c + 6 * d * t + 12 * e * t ** 2 + 20 * f * t ** 3

    vel = np.clip(vel, -CONSTRAINTS["max_vel"][joint_idx], CONSTRAINTS["max_vel"][joint_idx])
    acc = np.clip(acc, -CONSTRAINTS["max_acc"][joint_idx], CONSTRAINTS["max_acc"][joint_idx])

    return pos, vel, acc


# ====================== 3. 闭环PD控制 ======================
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


# ====================== 4. 人工势场法避障 ======================
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


# ====================== 5. 笛卡尔轨迹平滑 ======================
def smooth_cartesian_trajectory(waypoints, num_points=200):
    x = np.array([p[0] for p in waypoints])
    y = np.array([p[1] for p in waypoints])
    z = np.array([p[2] for p in waypoints])

    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, num_points)

    spline_x = interpolate.CubicSpline(t, x, bc_type='natural')
    spline_y = interpolate.CubicSpline(t, y, bc_type='natural')
    spline_z = interpolate.CubicSpline(t, z, bc_type='natural')

    return np.vstack((spline_x(t_new), spline_y(t_new), spline_z(t_new))).T


# ====================== 6. 替代逆运动学：关节轨迹预生成 ======================
def precompute_joint_waypoints(model, data, cart_waypoints):
    """预计算笛卡尔轨迹对应的关节角度（兼容旧版MuJoCo）"""
    joint_waypoints = []
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    for cart_pos in cart_waypoints:
        # 重置数据
        mujoco.mj_resetData(model, data)
        # 设置目标位置到data中（旧版mj_inverse的要求）
        data.site_xpos[ee_site_id] = cart_pos
        # 调用逆运动学（仅model和data参数）
        mujoco.mj_inverse(model, data)
        # 保存关节角度
        joint_waypoints.append(data.qpos[:6].copy())

    return joint_waypoints


# ====================== 7. 机械臂模型（带障碍物） ======================
def get_arm_xml_with_obstacles():
    arm_xml = """
<mujoco model="6dof_arm_with_obstacles">
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


# ====================== 8. 主仿真逻辑（修复逆运动学调用） ======================
def run_obstacle_avoidance_simulation():
    arm_xml = get_arm_xml_with_obstacles()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ 带避障的机械臂模型加载成功！")
        print(
            f"🔧 避障参数：斥力半径={OBSTACLE_CONFIG['rep_radius']}m，障碍物数量={len(OBSTACLE_CONFIG['obstacle_list'])}")

        # 预计算关节轨迹关键点（兼容旧版MuJoCo）
        joint_waypoints = precompute_joint_waypoints(model, data, CART_WAYPOINTS)
        # 平滑关节轨迹
        num_joint_points = 200
        smooth_joint_traj = []
        for joint_idx in range(6):
            joint_vals = [wp[joint_idx] for wp in joint_waypoints]
            t = np.linspace(0, 1, len(joint_vals))
            t_new = np.linspace(0, 1, num_joint_points)
            spline = interpolate.CubicSpline(t, joint_vals, bc_type='natural')
            smooth_joint_traj.append(spline(t_new))
        # 转置为 [轨迹点, 关节]
        smooth_joint_traj = np.array(smooth_joint_traj).T

        traj_length = len(smooth_joint_traj)
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        segment_time = 5.0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\n🎮 机械臂避障仿真启动！")
            print("💡 核心功能：人工势场法避障 + 物理约束 + PD闭环控制")
            print("💡 可视化：红色半透明球体为障碍物，机械臂自动绕开")
            print("💡 按 Ctrl+C 退出")

            while viewer.is_running():
                # 1. 计算当前轨迹索引
                t_total = data.time
                traj_idx = int((t_total / segment_time) * traj_length) % traj_length

                # 2. 获取末端当前位置
                ee_pos = data.site_xpos[ee_site_id].tolist()

                # 3. 原始关节目标
                raw_joint_target = smooth_joint_traj[traj_idx]

                # 4. 笛卡尔空间避障修正（核心）
                # 先通过正运动学获取原始笛卡尔目标
                mujoco.mj_forward(model, data)  # 更新正运动学
                raw_cart_target = data.site_xpos[ee_site_id].copy()
                # 避障修正笛卡尔目标
                corrected_cart_target = artificial_potential_field(ee_pos, raw_cart_target)
                # 修正后的笛卡尔目标转关节目标（兼容旧版）
                data.site_xpos[ee_site_id] = corrected_cart_target
                mujoco.mj_inverse(model, data)
                target_joints = data.qpos[:6].copy()

                # 5. 应用物理约束 + 闭环控制
                ctrl_signals = []
                for i in range(6):
                    # 约束关节角度
                    target_joints[i] = np.clip(target_joints[i], model.actuator_ctrlrange[i][0],
                                               model.actuator_ctrlrange[i][1])
                    # 闭环PD控制
                    ctrl = closed_loop_constraint_control(data, target_joints, i)
                    ctrl_signals.append(ctrl)

                # 6. 发送控制指令
                data.ctrl[:6] = ctrl_signals

                # 7. 打印状态
                if int(t_total) % 1 == 0 and int(t_total) != 0:
                    obs_distances = []
                    for obs in OBSTACLE_CONFIG["obstacle_list"]:
                        dist = np.linalg.norm(np.array(ee_pos) - np.array(obs[:3]))
                        obs_distances.append(dist)
                    min_obs_dist = min(obs_distances) if obs_distances else 0

                    print(f"\n⏱️  时间：{t_total:.2f}s")
                    print(f"   末端当前位置：{np.round(ee_pos, 3)}")
                    print(f"   修正后目标位置：{np.round(corrected_cart_target, 3)}")
                    print(f"   与最近障碍距离：{min_obs_dist:.3f}m (斥力半径：{OBSTACLE_CONFIG['rep_radius']}m)")

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
    run_obstacle_avoidance_simulation()