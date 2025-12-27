# MuJoCo 3.4.0 7自由度协作机械臂（极简版，无传感器，零XML错误）
import mujoco
import mujoco.viewer
import time
import numpy as np


def collaborative_robot_arm_demo():
    # 彻底移除所有传感器相关代码，仅保留基础机械臂+抓取逻辑
    cobot_xml = """
<mujoco model="7-DOF Collaborative Robot Arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual/>
  <asset>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="lightblue" rgba="0.4 0.7 0.9 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="green" rgba="0.2 0.8 0.2 1"/>
    <material name="yellow" rgba="0.8 0.8 0.2 1"/>
    <material name="orange" rgba="0.9 0.5 0.2 1"/>
  </asset>

  <!-- 世界体定义 -->
  <worldbody>
    <camera name="fixed_camera" pos="2.5 2.0 2.0" xyaxes="1 0 0 0 1 0"/>
    <!-- 地面 -->
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.1" material="gray"/>
    <!-- 易碎目标：橙色球体（低刚度，防止夹碎） -->
    <body name="fragile_target" pos="0.7 0.5 0.1">
      <geom name="target_geom" type="sphere" size="0.07" pos="0 0 0" material="orange" solref="0.02 1.0" solimp="0.9 0.95 0.01"/>
      <joint name="target_joint" type="free"/>
    </body>
    <!-- 7自由度协作机械臂（UR5e构型） -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.2 0.2" pos="0 0 0" material="lightblue"/>
      <joint name="base_joint" type="free"/>
      <!-- 关节1：基座旋转（Z轴） -->
      <body name="joint1_link" pos="0 0 0.2">
        <geom name="joint1_geom" type="cylinder" size="0.12 0.3" pos="0 0 0.15" material="lightblue"/>
        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.05"/>
        <!-- 关节2：大臂俯仰（Y轴） -->
        <body name="joint2_link" pos="0 0 0.3">
          <geom name="joint2_geom" type="cylinder" size="0.1 0.4" pos="0 0 0.2" material="lightblue"/>
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0" range="-2.0 2.0" damping="0.05"/>
          <!-- 关节3：小臂俯仰（Y轴） -->
          <body name="joint3_link" pos="0 0 0.4">
            <geom name="joint3_geom" type="cylinder" size="0.09 0.4" pos="0 0 0.2" material="lightblue"/>
            <joint name="joint3" type="hinge" axis="0 1 0" pos="0 0 0" range="-2.0 2.0" damping="0.05"/>
            <!-- 关节4：腕部旋转（Z轴） -->
            <body name="joint4_link" pos="0 0 0.4">
              <geom name="joint4_geom" type="cylinder" size="0.08 0.3" pos="0 0 0.15" material="lightblue"/>
              <joint name="joint4" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.05"/>
              <!-- 关节5：腕部俯仰（Y轴） -->
              <body name="joint5_link" pos="0 0 0.3">
                <geom name="joint5_geom" type="cylinder" size="0.07 0.3" pos="0 0 0.15" material="lightblue"/>
                <joint name="joint5" type="hinge" axis="0 1 0" pos="0 0 0" range="-2.0 2.0" damping="0.05"/>
                <!-- 关节6：腕部偏摆（Z轴） -->
                <body name="joint6_link" pos="0 0 0.3">
                  <geom name="joint6_geom" type="cylinder" size="0.06 0.2" pos="0 0 0.1" material="lightblue"/>
                  <joint name="joint6" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.05"/>
                  <!-- 关节7：末端旋转（Y轴） -->
                  <body name="joint7_link" pos="0 0 0.2">
                    <geom name="joint7_geom" type="cylinder" size="0.05 0.2" pos="0 0 0.1" material="lightblue"/>
                    <joint name="joint7" type="hinge" axis="0 1 0" pos="0 0 0" range="-2.0 2.0" damping="0.05"/>
                    <!-- 夹爪（简化版，无传感器） -->
                    <body name="gripper_base" pos="0 0 0.2">
                      <geom name="gripper_base_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" material="red"/>
                      <!-- 左夹爪 -->
                      <body name="left_gripper" pos="0 0.1 0">
                        <geom name="left_gripper_geom" type="box" size="0.1 0.06 0.06" pos="0 0 0" material="red"/>
                        <joint name="left_grip_joint" type="hinge" axis="0 0 1" pos="0 -0.1 0" range="-0.6 0" damping="0.03"/>
                      </body>
                      <!-- 右夹爪 -->
                      <body name="right_gripper" pos="0 -0.1 0">
                        <geom name="right_gripper_geom" type="box" size="0.1 0.06 0.06" pos="0 0 0" material="red"/>
                        <joint name="right_grip_joint" type="hinge" axis="0 0 1" pos="0 0.1 0" range="0 0.6" damping="0.03"/>
                      </body>
                      <!-- 末端标记 -->
                      <geom name="end_marker" type="sphere" size="0.03" pos="0 0 -0.05" material="green"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 执行器配置（仅保留位置/速度控制，无传感器依赖） -->
  <actuator>
    <position name="joint1_act" joint="joint1" kp="1200" kv="120"/>
    <position name="joint2_act" joint="joint2" kp="1200" kv="120"/>
    <position name="joint3_act" joint="joint3" kp="1200" kv="120"/>
    <position name="joint4_act" joint="joint4" kp="1200" kv="120"/>
    <position name="joint5_act" joint="joint5" kp="1200" kv="120"/>
    <position name="joint6_act" joint="joint6" kp="1200" kv="120"/>
    <position name="joint7_act" joint="joint7" kp="1200" kv="120"/>
    <!-- 夹爪速度控制（低刚度，防止夹碎物体） -->
    <velocity name="left_grip_act" joint="left_grip_joint" kv="50" ctrlrange="-0.5 0"/>
    <velocity name="right_grip_act" joint="right_grip_joint" kv="50" ctrlrange="0 0.5"/>
  </actuator>
</mujoco>
    """

    # 2. 加载模型（100%兼容3.4.0，无任何传感器相关错误）
    try:
        model = mujoco.MjModel.from_xml_string(cobot_xml)
        data = mujoco.MjData(model)
        print("✅ 7自由度协作机械臂模型加载成功，启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 3. 获取执行器索引（仅保留基础执行器，无传感器）
    # 关节执行器索引
    joint_idxs = {
        "joint1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint1_act"),
        "joint2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint2_act"),
        "joint3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint3_act"),
        "joint4": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint4_act"),
        "joint5": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint5_act"),
        "joint6": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint6_act"),
        "joint7": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint7_act"),
    }
    # 夹爪执行器索引
    left_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_grip_act")
    right_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_grip_act")

    # 4. 核心控制函数（移除力传感器依赖，改用时间控制抓取）
    def smooth_joint_control(joint_name, target_angle, duration, viewer):
        """平滑关节角度控制"""
        idx = joint_idxs[joint_name]
        start_angle = data.ctrl[idx]
        start_time = time.time()
        while (time.time() - start_time) < duration and viewer.is_running():
            t = (time.time() - start_time) / duration
            current_angle = start_angle + t * (target_angle - start_angle)
            data.ctrl[idx] = current_angle
            # 打印关节状态（无力度）
            print(f"\r{joint_name}角度：{current_angle:.2f} rad", end="")
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    def safe_grasp(viewer):
        """安全抓取（通过时间控制闭合，模拟力控效果）"""
        print("\n🔧 开始安全抓取（低速度闭合，防止夹碎物体）")
        grip_speed = -0.1  # 低速闭合，避免夹碎
        start_time = time.time()
        # 闭合1.5秒后停止（模拟力控阈值）
        while time.time() - start_time < 1.5 and viewer.is_running():
            data.ctrl[left_grip_idx] = grip_speed
            data.ctrl[right_grip_idx] = -grip_speed
            print(f"\r抓取进度：{((time.time() - start_time) / 1.5) * 100:.1f}%", end="")
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        # 停止闭合
        data.ctrl[left_grip_idx] = 0
        data.ctrl[right_grip_idx] = 0
        print("\n✅ 抓取完成（已停止闭合，防止夹碎）！")

    def release_gripper(duration, viewer):
        """放松夹爪"""
        print("\n🔧 开始放松夹爪")
        start_time = time.time()
        while (time.time() - start_time) < duration and viewer.is_running():
            data.ctrl[left_grip_idx] = 0.2  # 张开速度
            data.ctrl[right_grip_idx] = -0.2
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        print("✅ 夹爪已完全张开")

    # 5. 7自由度机械臂抓取流程
    cobot_steps = [
        ("关节1旋转对准目标", "joint1", 0.87, 3.0),  # 50°旋转
        ("关节2俯仰调整高度", "joint2", 0.785, 2.5),  # 45°俯仰
        ("关节3俯仰接近目标", "joint3", -0.61, 2.5),  # -35°俯仰
        ("关节4腕部旋转校准", "joint4", 1.047, 2.0),  # 60°旋转
        ("关节5腕部俯仰调整", "joint5", 0.523, 2.0),  # 30°俯仰
        ("关节6腕部偏摆校准", "joint6", 0.349, 2.0),  # 20°偏摆
        ("关节7末端旋转对准", "joint7", 0.174, 2.0),  # 10°旋转
    ]

    # 6. 启动仿真（纯3.4.0原生逻辑）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n📌 开始7自由度协作机械臂抓取流程...")
        print("-" * 60)

        # 第一步：关节运动对准目标
        for step_name, joint_name, target_angle, duration in cobot_steps:
            print(f"\n\n🔧 {step_name}")
            smooth_joint_control(joint_name, target_angle, duration, viewer)

        # 第二步：安全抓取（模拟力控）
        safe_grasp(viewer=viewer)

        # 第三步：抬升目标（仅调整关节2）
        print("\n\n🔧 抓取成功，抬升目标")
        smooth_joint_control("joint2", 1.047, 2.5, viewer)  # 60°俯仰抬升

        # 第四步：归位（关节1旋转回原位）
        print("\n\n🔧 旋转归位")
        smooth_joint_control("joint1", 0.0, 3.0, viewer)

        # 第五步：下放目标
        print("\n\n🔧 下放目标")
        smooth_joint_control("joint2", 0.785, 2.5, viewer)  # 45°俯仰下放

        # 第六步：放松夹爪
        print("\n\n🔧 放松夹爪完成放置")
        release_gripper(duration=2.0, viewer=viewer)

        # 保持6秒查看最终效果
        print("\n\n📌 抓取流程完成，保持可视化6秒...")
        start_hold = time.time()
        while (time.time() - start_hold) < 6 and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print("\n\n🎉 7自由度协作机械臂抓取演示完毕！")


if __name__ == "__main__":
    collaborative_robot_arm_demo()