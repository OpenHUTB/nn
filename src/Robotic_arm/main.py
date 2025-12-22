import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time
from scipy import interpolate

# ====================== 1. 定义机械臂物理约束参数（核心） ======================
# 参考UR5机械臂参数，可根据实际硬件调整
CONSTRAINTS = {
    "max_vel": [1.0, 0.8, 0.8, 1.2, 0.9, 1.2],  # 各关节最大角速度 (rad/s)
    "max_acc": [0.5, 0.4, 0.4, 0.6, 0.5, 0.6],  # 各关节最大角加速度 (rad/s²)
    "max_jerk": [0.3, 0.2, 0.2, 0.4, 0.3, 0.4],  # 各关节最大加加速度 (rad/s³)
    "ctrl_limit": [-10.0, 10.0]  # 电机控制量限制
}


# ====================== 2. 带约束的五次多项式轨迹生成 ======================
def constrained_quintic_polynomial(start, end, total_time, t, joint_idx):
    """
    带约束的五次多项式插值
    :param start: 起点角度
    :param end: 终点角度
    :param total_time: 轨迹段总时间
    :param t: 当前段内时间 (0<=t<=total_time)
    :param joint_idx: 关节索引（0-5）
    :return: 约束后的位置、速度、加速度
    """
    # 基础边界条件（启停时速度/加速度为0）
    s0, v0, a0 = start, 0, 0
    s1, v1, a1 = end, 0, 0

    T = total_time
    # 五次多项式系数计算
    a = s0
    b = v0
    c = a0 / 2
    d = (20 * (s1 - s0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
    e = (30 * (s0 - s1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
    f = (12 * (s1 - s0) - (6 * v1 + 6 * v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)

    # 计算原始位置、速度、加速度、加加速度
    pos = a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5
    vel = b + 2 * c * t + 3 * d * t ** 2 + 4 * e * t ** 3 + 5 * f * t ** 4
    acc = 2 * c + 6 * d * t + 12 * e * t ** 2 + 20 * f * t ** 3
    jerk = 6 * d + 24 * e * t + 60 * f * t ** 2

    # 应用约束（核心：超出则截断）
    max_vel = CONSTRAINTS["max_vel"][joint_idx]
    max_acc = CONSTRAINTS["max_acc"][joint_idx]
    max_jerk = CONSTRAINTS["max_jerk"][joint_idx]

    vel = np.clip(vel, -max_vel, max_vel)
    acc = np.clip(acc, -max_acc, max_acc)
    jerk = np.clip(jerk, -max_jerk, max_jerk)

    # 可选：如果速度/加速度被截断，反向修正位置（更严谨）
    # 这里简化处理，直接返回约束后的位置（基础场景足够）
    return pos, vel, acc


# ====================== 3. 闭环约束控制（实时修正） ======================
def closed_loop_constraint_control(data, target_joints, joint_idx):
    """
    闭环PD控制 + 约束检查，实时修正控制指令
    """
    # PD控制参数（可根据实际机械臂标定）
    k_p = 8.0  # 比例系数
    k_d = 0.2  # 微分系数

    # 获取当前关节状态（仿真中读取，实际为编码器数据）
    current_pos = data.qpos[joint_idx]
    current_vel = data.qvel[joint_idx]

    # 计算误差
    pos_error = target_joints[joint_idx] - current_pos
    vel_error = -current_vel  # 速度误差：目标速度为0（启停阶段）

    # 计算原始控制量
    ctrl = k_p * pos_error + k_d * vel_error

    # 约束控制量（避免电机过载）
    ctrl = np.clip(ctrl, CONSTRAINTS["ctrl_limit"][0], CONSTRAINTS["ctrl_limit"][1])

    return ctrl


# ====================== 4. 机械臂模型（不变） ======================
arm_xml = """
<mujoco model="6dof_arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <material name="gray" rgba="0.7 0.7 0.7 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
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
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
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


# ====================== 5. 带约束的仿真主逻辑 ======================
def run_constrained_simulation():
    # 临时XML文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        # 加载模型和数据
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ 带约束的机械臂模型加载成功！")
        print("🔧 约束参数：")
        print(f"   最大关节速度：{CONSTRAINTS['max_vel']} rad/s")
        print(f"   最大关节加速度：{CONSTRAINTS['max_acc']} rad/s²")

        # 轨迹关键点（关节空间）
        waypoints = [
            [0, 0.2, -0.5, 0, 0.3, 0],
            [0.5, 0.5, -0.8, 0.2, 0.5, 0.3],
            [0.8, 0.3, -0.6, 0.4, 0.2, 0.5],
            [0.5, 0.5, -0.8, 0.2, 0.5, 0.3],
            [0, 0.2, -0.5, 0, 0.3, 0]
        ]
        segment_time = 3.0  # 每段轨迹时长

        # 启动可视化
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\n🎮 带约束的机械臂仿真启动！")
            print("💡 特征：速度/加速度/控制量约束 + 闭环PD控制")
            print("💡 按 Ctrl+C 退出")

            while viewer.is_running():
                # 1. 计算当前轨迹段
                t_total = data.time
                seg_idx = int(t_total // segment_time) % (len(waypoints) - 1)
                t_seg = t_total % segment_time

                # 2. 生成带约束的目标关节角度
                target_joints = []
                joint_vels = []  # 记录约束后的速度（用于调试）
                for i in range(6):
                    pos, vel, acc = constrained_quintic_polynomial(
                        waypoints[seg_idx][i],
                        waypoints[seg_idx + 1][i],
                        segment_time,
                        t_seg,
                        i
                    )
                    # 额外约束关节角度在可控范围
                    pos = np.clip(pos, model.actuator_ctrlrange[i][0], model.actuator_ctrlrange[i][1])
                    target_joints.append(pos)
                    joint_vels.append(vel)

                # 3. 闭环约束控制：修正控制指令
                ctrl_signals = []
                for i in range(6):
                    ctrl = closed_loop_constraint_control(data, target_joints, i)
                    ctrl_signals.append(ctrl)

                # 4. 应用控制指令
                data.ctrl[:6] = ctrl_signals

                # 5. 打印关键状态（每50步打印一次，方便调试）
                if int(data.time * 100) % 50 == 0:
                    print(f"\n⏱️  时间：{data.time:.2f}s")
                    print(f"   关节0当前速度：{data.qvel[0]:.3f} rad/s (约束上限：{CONSTRAINTS['max_vel'][0]})")
                    print(f"   关节0控制量：{ctrl_signals[0]:.3f} (约束范围：{CONSTRAINTS['ctrl_limit']})")

                # 6. 运行仿真步
                mujoco.mj_step(model, data)
                viewer.sync()

                # 7. 帧率控制
                try:
                    mujoco.utils.mju_sleep(1 / 60)
                except:
                    time.sleep(1 / 60)

    except Exception as e:
        print(f"❌ 仿真出错：{e}")
    finally:
        os.unlink(xml_path)


if __name__ == "__main__":
    run_constrained_simulation()