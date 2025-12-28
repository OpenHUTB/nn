# MuJoCo 3.4.0 桌面分拣机械臂（无传感器，零XML错误，批量分拣演示）
import mujoco
import mujoco.viewer
import time
import numpy as np


def desktop_sorting_robot_demo():
    # 纯MuJoCo 3.4.0原生语法，仅保留支持的根标签
    sorting_robot_xml = """
<mujoco model="3-DOF Desktop Sorting Robot">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="45" elevation="-30"/>  <!-- 直观3D视角 -->
  </visual>
  <asset>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="green" rgba="0.2 0.8 0.2 1"/>
    <material name="yellow" rgba="0.8 0.8 0.2 1"/>
  </asset>

  <!-- 世界体定义（桌面分拣场景） -->
  <worldbody>
    <!-- 固定视角相机 -->
    <camera name="sorting_camera" pos="2 2 1.5" xyaxes="1 0 0 0 1 0"/>
    <!-- 桌面（灰色平面） -->
    <geom name="desktop" type="plane" size="2 2 0.1" pos="0 0 -0.05" material="gray"/>
    <!-- 分拣目标1：红色方块（待搬运） -->
    <body name="target_red" pos="1.0 0.5 0.0">
      <geom name="target_red_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" material="red"/>
      <joint name="target_red_joint" type="free"/>
    </body>
    <!-- 分拣目标2：蓝色方块（待搬运） -->
    <body name="target_blue" pos="1.0 -0.5 0.0">
      <geom name="target_blue_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" material="blue"/>
      <joint name="target_blue_joint" type="free"/>
    </body>
    <!-- 分拣区域：绿色标记框（目标放置区域） -->
    <geom name="sorting_area" type="box" size="0.3 0.3 0.01" pos="-1.0 0 0.0" material="green"/>
    <!-- 3自由度桌面机械臂 -->
    <body name="robot_base" pos="0 0 0.0">
      <geom name="base_geom" type="cylinder" size="0.2 0.1" pos="0 0 0" material="yellow"/>
      <joint name="base_joint" type="free"/>

      <!-- 关节1：基座旋转（Z轴，360°旋转） -->
      <body name="arm_main" pos="0 0 0.1">
        <geom name="arm_main_geom" type="cylinder" size="0.09 0.7" pos="0 0 0.35" material="yellow"/>
        <joint name="joint1_rotate" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.04"/>

        <!-- 关节2：大臂俯仰（Y轴，调整高度） -->
        <body name="arm_secondary" pos="0 0 0.7">
          <geom name="arm_secondary_geom" type="cylinder" size="0.07 0.6" pos="0 0 0.3" material="yellow"/>
          <joint name="joint2_pitch" type="hinge" axis="0 1 0" pos="0 0 0" range="-1.5 1.5" damping="0.04"/>

          <!-- 关节3：小臂伸缩（X轴，调整抓取距离） -->
          <body name="arm_telescope" pos="0 0 0.6">
            <geom name="arm_telescope_geom" type="cylinder" size="0.05 0.4" pos="0.2 0 0" material="yellow"/>
            <joint name="joint3_telescope" type="slide" axis="1 0 0" pos="0 0 0" range="0 0.4" damping="0.04"/>

            <!-- 分拣夹爪（简易平行夹爪） -->
            <body name="gripper_core" pos="0.4 0 0">
              <geom name="gripper_core_geom" type="box" size="0.08 0.08 0.08" pos="0 0 0" material="red"/>

              <!-- 左夹爪 -->
              <body name="gripper_left" pos="0 0.08 0">
                <geom name="gripper_left_geom" type="box" size="0.06 0.05 0.06" pos="0 0 0" material="red"/>
                <joint name="gripper_left_joint" type="hinge" axis="0 0 1" pos="0 -0.08 0" range="-0.4 0" damping="0.02"/>
              </body>

              <!-- 右夹爪 -->
              <body name="gripper_right" pos="0 -0.08 0">
                <geom name="gripper_right_geom" type="box" size="0.06 0.05 0.06" pos="0 0 0" material="red"/>
                <joint name="gripper_right_joint" type="hinge" axis="0 0 1" pos="0 0.08 0" range="0 0.4" damping="0.02"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 执行器配置（MuJoCo 3.4.0 100%兼容） -->
  <actuator>
    <!-- 3个关节的位置控制（高精度定位） -->
    <position name="joint1_act" joint="joint1_rotate" kp="1200" kv="110"/>
    <position name="joint2_act" joint="joint2_pitch" kp="1200" kv="110"/>
    <position name="joint3_act" joint="joint3_telescope" kp="1200" kv="110"/>

    <!-- 夹爪的速度控制（软接触，防止损坏目标） -->
    <velocity name="gripper_left_act" joint="gripper_left_joint" kv="45" ctrlrange="-0.35 0"/>
    <velocity name="gripper_right_act" joint="gripper_right_joint" kv="45" ctrlrange="0 0.35"/>
  </actuator>
</mujoco>
    """

    # 加载模型（确保无XML错误，适配MuJoCo 3.4.0）
    try:
        model = mujoco.MjModel.from_xml_string(sorting_robot_xml)
        data = mujoco.MjData(model)
        print("✅ 桌面分拣机械臂模型加载成功，启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 获取执行器索引（原生API，无兼容问题）
    joint_idxs = {
        "joint1": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint1_act"),
        "joint2": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint2_act"),
        "joint3": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint3_act")
    }
    left_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_left_act")
    right_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_right_act")

    # 核心控制函数（分段式动作，精准可控）
    def precise_joint_control(joint_name, target_val, duration, viewer):
        """精准控制关节移动/伸缩到目标值"""
        idx = joint_idxs[joint_name]
        start_val = data.ctrl[idx]
        start_time = time.time()

        while (time.time() - start_time) < duration and viewer.is_running():
            progress = (time.time() - start_time) / duration
            current_val = start_val + progress * (target_val - start_val)
            data.ctrl[idx] = current_val

            # 实时打印动作状态
            print(f"\r{joint_name} 当前值：{current_val:.2f} | 目标值：{target_val:.2f}", end="")

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        print()  # 换行

    def soft_gripper_close(viewer, target_name):
        """软接触闭合夹爪（低速+定时，保护目标物体）"""
        print(f"\n🔧 开始闭合夹爪，抓取{target_name}")
        grip_speed = -0.3
        close_duration = 1.0  # 短时间闭合，避免过夹
        start_time = time.time()

        while (time.time() - start_time) < close_duration and viewer.is_running():
            progress = (time.time() - start_time) / close_duration
            data.ctrl[left_grip_idx] = grip_speed
            data.ctrl[right_grip_idx] = -grip_speed

            print(f"\r夹爪闭合进度：{progress * 100:.1f}%", end="")

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        # 锁定夹爪，保持抓取状态
        data.ctrl[left_grip_idx] = 0
        data.ctrl[right_grip_idx] = 0
        print(f"\n✅ {target_name} 抓取完成，夹爪已锁定")

    def gripper_open_full(viewer, target_name):
        """完全张开夹爪，放置目标物体"""
        print(f"\n🔧 开始张开夹爪，放置{target_name}")
        open_duration = 0.8
        start_time = time.time()

        while (time.time() - start_time) < open_duration and viewer.is_running():
            data.ctrl[left_grip_idx] = 0.3
            data.ctrl[right_grip_idx] = -0.3

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        # 复位夹爪控制
        data.ctrl[left_grip_idx] = 0
        data.ctrl[right_grip_idx] = 0
        print(f"✅ {target_name} 放置完成，夹爪已复位")

    def sort_single_target(viewer, target_name, joint1_angle, joint3_extend):
        """分拣单个目标物体的完整流程"""
        # 步骤1：旋转基座对准目标
        print(f"\n\n===== 开始分拣 {target_name} =====")
        print("\n🔧 步骤1：旋转基座对准目标")
        precise_joint_control("joint1", joint1_angle, 2.5, viewer)

        # 步骤2：俯仰大臂降低高度
        print("\n🔧 步骤2：降低机械臂高度，接近目标")
        precise_joint_control("joint2", -0.6, 2.0, viewer)

        # 步骤3：伸缩小臂靠近目标
        print("\n🔧 步骤3：伸缩小臂，对准目标中心")
        precise_joint_control("joint3", joint3_extend, 2.0, viewer)

        # 步骤4：软接触闭合夹爪，抓取目标
        soft_gripper_close(viewer, target_name)

        # 步骤5：抬升机械臂，脱离桌面
        print("\n🔧 步骤5：抬升目标，脱离桌面")
        precise_joint_control("joint2", 0.0, 1.8, viewer)

        # 步骤6：旋转基座，对准分拣区域
        print("\n🔧 步骤6：旋转机械臂，对准绿色分拣区域")
        precise_joint_control("joint1", 3.14, 3.0, viewer)

        # 步骤7：降低高度，准备放置目标
        print("\n🔧 步骤7：降低目标高度，接近分拣区域")
        precise_joint_control("joint2", -0.6, 1.8, viewer)

        # 步骤8：张开夹爪，放置目标
        gripper_open_full(viewer, target_name)

        # 步骤9：复位机械臂，准备下一个目标
        print("\n🔧 步骤9：复位机械臂，准备分拣下一个目标")
        precise_joint_control("joint2", 0.0, 1.5, viewer)
        precise_joint_control("joint3", 0.0, 1.5, viewer)
        precise_joint_control("joint1", 0.0, 2.0, viewer)

    # 启动完整分拣流程
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n📌 开始桌面机械臂双目标分拣流程...")
        print("-" * 60)

        # 第一阶段：分拣红色方块
        sort_single_target(viewer, "红色方块", 0.0, 0.35)

        # 第二阶段：分拣蓝色方块（调整关节参数，对准蓝色目标）
        sort_single_target(viewer, "蓝色方块", -0.5, 0.35)

        # 保持可视化，查看分拣结果
        print("\n\n📌 双目标分拣流程全部完成，保持可视化6秒...")
        start_hold = time.time()
        while (time.time() - start_hold) < 6 and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print("\n\n🎉 桌面分拣机械臂双目标分拣演示完毕！")


if __name__ == "__main__":
    desktop_sorting_robot_demo()