# MuJoCo 3.4.0 轻量版2D平面机械臂抓取（无传感器，零XML错误）
import mujoco
import mujoco.viewer
import time
import numpy as np


def simple_2d_robot_arm_demo():
    # 纯2D平面模型，仅保留MuJoCo 3.4.0原生支持标签
    robot_2d_xml = """
<mujoco model="2D Simple Robot Arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="90" elevation="-90"/>  <!-- 2D平面视角 -->
  </visual>
  <asset>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="yellow" rgba="0.8 0.8 0.2 1"/>
  </asset>

  <!-- 2D世界体定义（限制在X-Y平面运动） -->
  <worldbody>
    <!-- 固定相机（2D视角） -->
    <camera name="2d_camera" pos="0 2 1" xyaxes="1 0 0 0 0 1"/>
    <!-- 地面（2D平面） -->
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.1" material="gray"/>
    <!-- 抓取目标：黄色立方体（2D平面放置） -->
    <body name="target" pos="1.2 0 0.1">
      <geom name="target_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" material="yellow"/>
      <joint name="target_joint" type="free"/>
    </body>
    <!-- 2自由度平面机械臂 -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.15 0.1" pos="0 0 0" material="blue"/>
      <joint name="base_joint" type="free"/>

      <!-- 关节1：基座旋转（Z轴，2D平面旋转） -->
      <body name="arm1" pos="0 0 0.1">
        <geom name="arm1_geom" type="cylinder" size="0.08 0.6" pos="0 0 0.3" material="blue"/>
        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.03"/>

        <!-- 关节2：大臂俯仰（Z轴，2D平面摆动） -->
        <body name="arm2" pos="0 0 0.6">
          <geom name="arm2_geom" type="cylinder" size="0.06 0.5" pos="0 0 0.25" material="blue"/>
          <joint name="joint2" type="hinge" axis="0 0 1" pos="0 0 0" range="-2.0 2.0" damping="0.03"/>

          <!-- 简易夹爪（2D平面抓取） -->
          <body name="gripper_base" pos="0 0 0.5">
            <geom name="gripper_base_geom" type="box" size="0.08 0.08 0.08" pos="0 0 0" material="red"/>

            <!-- 左夹爪 -->
            <body name="left_grip" pos="0 0.08 0">
              <geom name="left_grip_geom" type="box" size="0.06 0.04 0.06" pos="0 0 0" material="red"/>
              <joint name="left_grip_joint" type="hinge" axis="0 0 1" pos="0 -0.08 0" range="-0.5 0" damping="0.02"/>
            </body>

            <!-- 右夹爪 -->
            <body name="right_grip" pos="0 -0.08 0">
              <geom name="right_grip_geom" type="box" size="0.06 0.04 0.06" pos="0 0 0" material="red"/>
              <joint name="right_grip_joint" type="hinge" axis="0 0 1" pos="0 0.08 0" range="0 0.5" damping="0.02"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 执行器配置（MuJoCo 3.4.0原生支持） -->
  <actuator>
    <!-- 关节位置控制 -->
    <position name="joint1_act" joint="joint1" kp="1000" kv="100"/>
    <position name="joint2_act" joint="joint2" kp="1000" kv="100"/>

    <!-- 夹爪速度控制（安全低速） -->
    <velocity name="left_grip_act" joint="left_grip_joint" kv="40" ctrlrange="-0.4 0"/>
    <velocity name="right_grip_act" joint="right_grip_joint" kv="40" ctrlrange="0 0.4"/>
  </actuator>
</mujoco>
    """

    # 加载模型（确保100%兼容MuJoCo 3.4.0）
    try:
        model = mujoco.MjModel.from_xml_string(robot_2d_xml)
        data = mujoco.MjData(model)
        print("✅ 2D平面机械臂模型加载成功，启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 获取执行器索引
    joint_idxs = {
        "joint1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint1_act"),
        "joint2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint2_act")
    }
    left_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_grip_act")
    right_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_grip_act")

    # 核心控制函数
    def smooth_joint_move(joint_name, target_angle, duration, viewer):
        """平滑移动关节到目标角度"""
        idx = joint_idxs[joint_name]
        start_angle = data.ctrl[idx]
        start_time = time.time()

        while (time.time() - start_time) < duration and viewer.is_running():
            progress = (time.time() - start_time) / duration
            current_angle = start_angle + progress * (target_angle - start_angle)
            data.ctrl[idx] = current_angle

            # 打印实时状态
            print(f"\r{joint_name} 当前角度：{current_angle:.2f} rad | 目标角度：{target_angle:.2f} rad", end="")

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        print()  # 换行

    def safe_gripper_close(viewer):
        """安全闭合夹爪（低速+定时，模拟力控）"""
        print("\n🔧 开始闭合夹爪（安全低速）")
        grip_speed = -0.3
        start_time = time.time()
        close_duration = 1.2  # 闭合1.2秒后停止，防止夹碎

        while (time.time() - start_time) < close_duration and viewer.is_running():
            progress = (time.time() - start_time) / close_duration
            data.ctrl[left_grip_idx] = grip_speed
            data.ctrl[right_grip_idx] = -grip_speed

            print(f"\r夹爪闭合进度：{progress * 100:.1f}%", end="")

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        # 停止夹爪运动
        data.ctrl[left_grip_idx] = 0
        data.ctrl[right_grip_idx] = 0
        print("\n✅ 夹爪闭合完成，已锁定目标")

    def gripper_open(duration, viewer):
        """张开夹爪"""
        print("\n🔧 开始张开夹爪")
        start_time = time.time()

        while (time.time() - start_time) < duration and viewer.is_running():
            data.ctrl[left_grip_idx] = 0.3
            data.ctrl[right_grip_idx] = -0.3

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        data.ctrl[left_grip_idx] = 0
        data.ctrl[right_grip_idx] = 0
        print("✅ 夹爪已完全张开，目标放置完成")

    # 2D机械臂抓取流程
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n📌 开始2D平面机械臂抓取流程...")
        print("-" * 60)

        # 步骤1：关节1旋转对准目标
        print("\n\n🔧 步骤1：基座旋转对准目标")
        smooth_joint_move("joint1", 0.0, 2.5, viewer)

        # 步骤2：关节2俯仰接近目标
        print("\n\n🔧 步骤2：大臂俯仰接近目标")
        smooth_joint_move("joint2", -0.785, 2.5, viewer)  # -45°俯仰

        # 步骤3：安全闭合夹爪抓取目标
        safe_gripper_close(viewer)

        # 步骤4：抬升目标（关节2回正）
        print("\n\n🔧 步骤4：抬升抓取目标")
        smooth_joint_move("joint2", 0.0, 2.0, viewer)

        # 步骤5：基座旋转归位
        print("\n\n🔧 步骤5：机械臂旋转归位")
        smooth_joint_move("joint1", 1.57, 3.0, viewer)  # 90°旋转归位

        # 步骤6：下放目标（关节2再次俯仰）
        print("\n\n🔧 步骤6：下放抓取目标")
        smooth_joint_move("joint2", -0.785, 2.0, viewer)

        # 步骤7：张开夹爪完成放置
        gripper_open(1.5, viewer)

        # 保持可视化5秒
        print("\n\n📌 抓取流程全部完成，保持可视化5秒...")
        start_hold = time.time()
        while (time.time() - start_hold) < 5 and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print("\n\n🎉 2D平面机械臂抓取演示完毕！")


if __name__ == "__main__":
    simple_2d_robot_arm_demo()