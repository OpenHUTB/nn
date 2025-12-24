import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time
from scipy import interpolate
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 全局配置（鲁棒性优化参数） ======================
# 物理约束（UR5参考）
CONSTRAINTS = {
    "max_vel": [1.0, 0.8, 0.8, 1.2, 0.9, 1.2],
    "max_acc": [0.5, 0.4, 0.4, 0.6, 0.5, 0.6],
    "max_jerk": [0.3, 0.2, 0.2, 0.4, 0.3, 0.4],
    "ctrl_limit": [-10.0, 10.0]
}

# 避障基础参数（鲁棒性优化版）
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

# 笛卡尔轨迹关键点（易触发局部最优的路径）
CART_WAYPOINTS = [
    [0.5, 0.0, 0.6],  # 起点
    [0.6, 0.0, 0.58],  # 中间点（障碍夹缝，易局部最优）
    [0.8, 0.1, 0.8],  # 终点
    [0.6, 0.0, 0.58],  # 回中间点
    [0.5, 0.0, 0.6]  # 回起点
]

# 全局变量：记录停滞开始时间
stagnant_start_time = None


# ====================== 2. 新增：兼容所有版本的末端速度计算 ======================
def get_ee_cartesian_velocity(model, data, ee_site_id):
    """
    计算末端执行器的笛卡尔速度（兼容所有MuJoCo版本）
    原理：通过雅可比矩阵将关节速度转换为末端笛卡尔速度
    """
    # 获取雅可比矩阵（6xN，前3行是平移速度，后3行是旋转速度）
    jacp = np.zeros((3, model.nv))  # 平移雅可比
    jacr = np.zeros((3, model.nv))  # 旋转雅可比

    # 计算末端site的雅可比矩阵
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

    # 关节速度（data.qvel）
    joint_vel = data.qvel[:6]  # 仅取前6个关节速度

    # 笛卡尔平移速度 = 雅可比 × 关节速度
    ee_cart_vel = jacp @ joint_vel

    return ee_cart_vel


# ====================== 3. 物理约束轨迹生成（原有逻辑） ======================
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
    acc = 2 * c + 6 * d * t + 12 * e * t ** 2 + 20 * f * t ** 4

    vel = np.clip(vel, -CONSTRAINTS["max_vel"][joint_idx], CONSTRAINTS["max_vel"][joint_idx])
    acc = np.clip(acc, -CONSTRAINTS["max_acc"][joint_idx], CONSTRAINTS["max_acc"][joint_idx])

    return pos, vel, acc


# ====================== 4. 闭环PD控制（原有逻辑） ======================
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


# ====================== 5. 鲁棒性优化1：局部最优检测与规避 ======================
def check_local_optimum(ee_vel, ee_pos, target_pos):
    """
    检测是否陷入局部最优，并生成引导目标跳出陷阱
    :return: is_local_opt (是否局部最优), guide_target (引导目标位置)
    """
    global stagnant_start_time

    # 计算末端合速度
    vel_mag = np.linalg.norm(ee_vel)
    threshold = OBSTACLE_CONFIG["stagnant_threshold"]
    max_stagnant_time = OBSTACLE_CONFIG["stagnant_time"]

    if vel_mag < threshold:
        if stagnant_start_time is None:
            stagnant_start_time = time.time()
        # 超过停滞时间，判定为局部最优
        elif time.time() - stagnant_start_time > max_stagnant_time:
            print(f"\n⚠️  检测到局部最优！末端速度={vel_mag:.4f}m/s < 阈值={threshold}m/s")
            # 生成引导目标：向原始目标方向偏移
            dir_to_target = np.array(target_pos) - np.array(ee_pos)
            if np.linalg.norm(dir_to_target) < 1e-6:
                dir_to_target = np.array([0.0, 0.0, 0.1])  # 避免除零
            else:
                dir_to_target = dir_to_target / np.linalg.norm(dir_to_target)

            guide_target = np.array(ee_pos) + dir_to_target * OBSTACLE_CONFIG["guide_offset"]
            print(f"📌 生成引导目标：{np.round(guide_target, 3)} (偏移{OBSTACLE_CONFIG['guide_offset']}m)")
            stagnant_start_time = None  # 重置计时器
            return True, guide_target.tolist()
    else:
        stagnant_start_time = None  # 速度正常，重置计时器

    return False, target_pos


# ====================== 6. 鲁棒性优化2：自适应势场参数 ======================
def adaptive_potential_params(ee_pos, obstacle_list):
    """
    根据障碍距离/数量自适应调整引力/斥力系数
    - 距离越近，斥力越大；障碍越多，引力越小
    """
    base_k_att = OBSTACLE_CONFIG["base_k_att"]
    base_k_rep = OBSTACLE_CONFIG["base_k_rep"]

    # 计算与最近障碍的距离
    obs_distances = [np.linalg.norm(np.array(ee_pos) - np.array(obs[:3])) for obs in obstacle_list]
    min_dist = min(obs_distances) if obs_distances else 1.0
    obs_count = len(obstacle_list)

    # 距离自适应斥力系数：距离<0.2m时，斥力翻倍
    k_rep = base_k_rep if min_dist > 0.2 else base_k_rep * 2.0
    # 数量自适应引力系数：障碍>2个时，引力降低50%
    k_att = base_k_att if obs_count <= 2 else base_k_att * 0.5

    return k_att, k_rep


# ====================== 7. 鲁棒性优化3：碰撞冗余检测 ======================
def collision_check_approx(ee_pos, joint_pos, obstacle_list, safety_margin=0.05):
    """
    近似碰撞检测（工程简化版）：检测末端+关键关节与障碍的距离
    :return: is_collision (是否碰撞), min_safe_dist (最小安全距离)
    """
    # 检测末端执行器
    ee_collision = False
    min_ee_dist = 100.0
    for obs in obstacle_list:
        obs_pos = np.array(obs[:3])
        obs_radius = obs[3]
        dist = np.linalg.norm(np.array(ee_pos) - obs_pos)
        min_ee_dist = min(min_ee_dist, dist)
        if dist < obs_radius + safety_margin:
            ee_collision = True
            break

    # 检测关键关节（简化：仅检测关节2/3/4）
    joint_collision = False
    # 仿真中通过data获取关节位置（实际场景需正运动学计算）
    # 这里简化为基于关节角度的近似检测
    joint_2_3_4_idx = [2, 3, 4]
    for idx in joint_2_3_4_idx:
        # 近似关节位置（基于机械臂模型）
        joint_pos_approx = np.array([
            0.4 + 0.35 * np.cos(joint_pos[2]),
            0.0 + 0.35 * np.sin(joint_pos[2]),
            0.5 + 0.25 * np.sin(joint_pos[3])
        ])
        for obs in obstacle_list:
            obs_pos = np.array(obs[:3])
            obs_radius = obs[3]
            dist = np.linalg.norm(joint_pos_approx - obs_pos)
            if dist < obs_radius + safety_margin:
                joint_collision = True
                break
        if joint_collision:
            break

    is_collision = ee_collision or joint_collision
    if is_collision:
        print(f"\n🚨 碰撞风险！末端与最近障碍距离={min_ee_dist:.3f}m < 安全裕度={safety_margin}m")

    return is_collision, min_ee_dist


# ====================== 8. 鲁棒性优化后的避障核心逻辑 ======================
def robust_artificial_potential_field(ee_pos, ee_vel, target_pos, obstacle_list):
    """
    鲁棒版人工势场法：局部最优规避 + 自适应参数
    """
    ee_pos = np.array(ee_pos)
    target_pos = np.array(target_pos)
    rep_radius = OBSTACLE_CONFIG["rep_radius"]

    # 步骤1：检测局部最优，生成引导目标
    is_local_opt, guide_target = check_local_optimum(ee_vel, ee_pos, target_pos)
    current_target = np.array(guide_target) if is_local_opt else target_pos

    # 步骤2：自适应调整引力/斥力系数
    k_att, k_rep = adaptive_potential_params(ee_pos, obstacle_list)
    print(
        f"\n🔧 自适应参数：k_att={k_att:.1f}, k_rep={k_rep:.1f} (最近障碍距离={min([np.linalg.norm(ee_pos - np.array(obs[:3])) for obs in obstacle_list]):.3f}m)")

    # 步骤3：计算引力（指向当前目标）
    att_force = k_att * (current_target - ee_pos)

    # 步骤4：计算斥力（远离所有障碍）
    rep_force = np.zeros(3)
    for obs in obstacle_list:
        obs_pos = np.array(obs[:3])
        obs_radius = obs[3]
        dist = np.linalg.norm(ee_pos - obs_pos)

        if dist < rep_radius + obs_radius:
            if dist < 1e-6:
                dist = 1e-6
            rep_dir = (ee_pos - obs_pos) / dist
            # 优化斥力公式：避免距离过近时斥力突变
            rep_force += k_rep * (1 / (dist - obs_radius) - 1 / rep_radius) * (1 / dist ** 2) * rep_dir

    # 步骤5：合力修正目标位置，添加边界约束
    corrected_target = ee_pos + att_force + rep_force
    corrected_target = np.clip(corrected_target, [0.3, -0.4, 0.2], [0.9, 0.4, 1.0])

    return corrected_target.tolist()


# ====================== 9. 逆运动学预计算（兼容旧版MuJoCo） ======================
def precompute_joint_waypoints(model, data, cart_waypoints):
    joint_waypoints = []
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    for cart_pos in cart_waypoints:
        mujoco.mj_resetData(model, data)
        data.site_xpos[ee_site_id] = cart_pos
        mujoco.mj_inverse(model, data)
        joint_waypoints.append(data.qpos[:6].copy())

    return joint_waypoints


# ====================== 10. 机械臂模型（带密集障碍可视化） ======================
def get_arm_xml_with_obstacles():
    arm_xml = """
<mujoco model="6dof_arm_with_obstacles_robust">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <material name="gray" rgba="0.7 0.7 0.7 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="obstacle" rgba="1 0 0 0.5"/>
    <material name="critical_obstacle" rgba="1 0 0 0.7"/>  <!-- 易导致局部最优的障碍 -->
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

    # 不同障碍用不同透明度区分（关键障碍更醒目）
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


# ====================== 11. 主仿真逻辑（鲁棒性优化核心） ======================
def run_robust_obstacle_avoidance_simulation():
    arm_xml = get_arm_xml_with_obstacles()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ 鲁棒版避障机械臂模型加载成功！")
        print(f"🔧 鲁棒性优化：局部最优规避 + 自适应参数 + 碰撞冗余检测")
        print(f"🔧 障碍数量：{len(OBSTACLE_CONFIG['obstacle_list'])} (含1个易导致局部最优的关键障碍)")

        # 预计算关节轨迹
        joint_waypoints = precompute_joint_waypoints(model, data, CART_WAYPOINTS)
        num_joint_points = 200
        smooth_joint_traj = []
        for joint_idx in range(6):
            joint_vals = [wp[joint_idx] for wp in joint_waypoints]
            t = np.linspace(0, 1, len(joint_vals))
            t_new = np.linspace(0, 1, num_joint_points)
            spline = interpolate.CubicSpline(t, joint_vals, bc_type='natural')
            smooth_joint_traj.append(spline(t_new))
        smooth_joint_traj = np.array(smooth_joint_traj).T

        traj_length = len(smooth_joint_traj)
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        segment_time = 8.0  # 增加轨迹时间，适配鲁棒性优化

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\n🎮 鲁棒版机械臂避障仿真启动！")
            print("💡 核心优化：自动规避局部最优 + 自适应势场参数 + 碰撞冗余检测")
            print("💡 可视化：深红色障碍为易导致局部最优的关键障碍")
            print("💡 按 Ctrl+C 退出")

            while viewer.is_running():
                # 1. 时间与轨迹索引
                t_total = data.time
                traj_idx = int((t_total / segment_time) * traj_length) % traj_length

                # 2. 获取末端位置 + 计算末端速度（兼容旧版MuJoCo）
                ee_pos = data.site_xpos[ee_site_id].tolist()
                ee_vel = get_ee_cartesian_velocity(model, data, ee_site_id).tolist()  # 替换site_xvel

                # 3. 原始关节目标
                raw_joint_target = smooth_joint_traj[traj_idx]

                # 4. 正运动学获取原始笛卡尔目标
                mujoco.mj_forward(model, data)
                raw_cart_target = data.site_xpos[ee_site_id].copy()

                # 5. 鲁棒版避障修正（核心！）
                corrected_cart_target = robust_artificial_potential_field(
                    ee_pos, ee_vel, raw_cart_target, OBSTACLE_CONFIG["obstacle_list"]
                )

                # 6. 逆解得到修正后的关节目标
                data.site_xpos[ee_site_id] = corrected_cart_target
                mujoco.mj_inverse(model, data)
                target_joints = data.qpos[:6].copy()

                # 7. 碰撞冗余检测（安全兜底）
                is_collision, min_safe_dist = collision_check_approx(
                    ee_pos, target_joints, OBSTACLE_CONFIG["obstacle_list"]
                )
                if is_collision:
                    # 碰撞时紧急调整：增大斥力，远离障碍
                    emergency_rep = np.array(ee_pos) - np.array(OBSTACLE_CONFIG["obstacle_list"][0][:3])
                    emergency_rep = emergency_rep / np.linalg.norm(emergency_rep) * 0.05
                    corrected_cart_target = np.array(corrected_cart_target) + emergency_rep
                    data.site_xpos[ee_site_id] = corrected_cart_target
                    mujoco.mj_inverse(model, data)
                    target_joints = data.qpos[:6].copy()
                    print(f"🆘 紧急避障：修正目标位置={np.round(corrected_cart_target, 3)}")

                # 8. 物理约束 + 闭环控制
                ctrl_signals = []
                for i in range(6):
                    target_joints[i] = np.clip(target_joints[i], model.actuator_ctrlrange[i][0],
                                               model.actuator_ctrlrange[i][1])
                    ctrl = closed_loop_constraint_control(data, target_joints, i)
                    ctrl_signals.append(ctrl)

                # 9. 发送控制指令
                data.ctrl[:6] = ctrl_signals

                # 10. 打印关键状态（每1秒）
                if int(t_total) % 1 == 0 and int(t_total) != 0:
                    obs_distances = [np.linalg.norm(np.array(ee_pos) - np.array(obs[:3])) for obs in
                                     OBSTACLE_CONFIG["obstacle_list"]]
                    min_obs_dist = min(obs_distances) if obs_distances else 0

                    print(f"\n⏱️  时间：{t_total:.2f}s")
                    print(f"   末端位置：{np.round(ee_pos, 3)}")
                    print(f"   修正目标：{np.round(corrected_cart_target, 3)}")
                    print(f"   末端速度：{np.round(np.linalg.norm(ee_vel), 4)}m/s")
                    print(f"   最近障碍：{min_obs_dist:.3f}m | 碰撞风险：{'是' if is_collision else '否'}")

                # 11. 仿真步运行
                mujoco.mj_step(model, data)
                viewer.sync()

                # 12. 帧率控制
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
    run_robust_obstacle_avoidance_simulation()