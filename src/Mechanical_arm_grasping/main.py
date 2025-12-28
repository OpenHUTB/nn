# MuJoCo 3.4.0 SCARA型机械臂（末端反馈+目标跟随）演示
import mujoco
import mujoco.viewer
import time
import numpy as np


def scara_robot_arm_demo():
    # 1. 内置SCARA机械臂XML模型（工业常用构型）
    scara_xml = """
<mujoco model="SCARA Robot Arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual/>
  <asset>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="darkblue" rgba="0.1 0.1 0.6 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="green" rgba="0.2 0.8 0.2 1"/>
    <material name="yellow" rgba="0.8 0.8 0.2 1"/>
    <material name="cyan" rgba="0.2 0.8 0.8 1"/>
  </asset>
  <worldbody>
    <camera name="fixed_camera" pos="2.0 2.0 1.5" xyaxes="1 0 0 0 1 0"/>
    <!-- 地面 -->
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.1" material="gray"/>
    <!-- 动态目标点（青色小球） -->
    <body name="moving_target" pos="0.8 0.6 0.3">
      <geom name="target_geom" type="sphere" size="0.05" pos="0 0 0" material="cyan"/>
      <joint name="target_joint" type="free"/>
    </body>
    <!-- SCARA机械臂（工业构型：旋转1+旋转2+升降+旋转夹爪） -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.25 0.15" pos="0 0 0" material="darkblue"/>
      <joint name="base_joint" type="free"/>
      <!-- 关节1：水平旋转（绕Z轴，基座旋转） -->
      <body name="joint1_link" pos="0 0 0.15">
        <geom name="joint1_geom" type="cylinder" size="0.15 0.2" pos="0 0 0.1" material="darkblue"/>
        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.08"/>
        <!-- 关节2：水平旋转（绕Z轴，大臂旋转） -->
        <body name="joint2_link" pos="0.5 0 0.1">
          <geom name="joint2_geom" type="cylinder" size="0.12 0.4" pos="0.2 0 0" material="darkblue"/>
          <joint name="joint2" type="hinge" axis="0 0 1" pos="0 0 0" range="-2.0 2.0" damping="0.08"/>
          <!-- 关节3：垂直升降（Z轴，小臂升降） -->
          <body name="joint3_link" pos="0.4 0 0">
            <geom name="joint3_geom" type="cylinder" size="0.1 0.3" pos="0 0 0.15" material="darkblue"/>
            <joint name="joint3" type="slide" axis="0 0 1" pos="0 0 0" range="0 0.8" damping="0.08"/>
            <!-- 关节4：夹爪旋转（绕Z轴，末端旋转） -->
            <body name="joint4_link" pos="0 0 0.15">
              <geom name="joint4_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" material="darkblue"/>
              <joint name="joint4" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14" damping="0.05"/>
              <!-- 末端夹爪 -->
              <body name="gripper_base" pos="0 0 0">
                <geom name="gripper_base_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" material="red"/>
                <!-- 左夹爪 -->
                <body name="left_gripper" pos="0 0.1 0">
                  <geom name="left_gripper_geom" type="box" size="0.1 0.05 0.05" pos="0 0 0" material="red"/>
                  <joint name="left_grip_joint" type="hinge" axis="0 0 1" pos="0 -0.1 0" range="-0.5 0" damping="0.05"/>
                </body>
                <!-- 右夹爪 -->
                <body name="right_gripper" pos="0 -0.1 0">
                  <geom name="right_gripper_geom" type="box" size="0.1 0.05 0.05" pos="0 0 0" material="red"/>
                  <joint name="right_grip_joint" type="hinge" axis="0 0 1" pos="0 0.1 0" range="0 0.5" damping="0.05"/>
                </body>
                <!-- 末端位置标记（绿色小球，用于反馈） -->
                <geom name="end_effector_marker" type="sphere" size="0.03" pos="0 0 -0.05" material="green"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <!-- 执行器配置（高精度位置控制） -->
  <actuator>
    <position name="joint1_act" joint="joint1" kp="1500" kv="150"/>
    <position name="joint2_act" joint="joint2" kp="1500" kv="150"/>
    <position name="joint3_act" joint="joint3" kp="1500" kv="150"/>
    <position name="joint4_act" joint="joint4" kp="1500" kv="150"/>
    <position name="left_grip_act" joint="left_grip_joint" kp="800" kv="80"/>
    <position name="right_grip_act" joint="right_grip_joint" kp="800" kv="80"/>
  </actuator>
</mujoco>
    """

    # 2. 加载模型
    try:
        model = mujoco.MjModel.from_xml_string(scara_xml)
        data = mujoco.MjData(model)
        print("✅ SCARA机械臂模型加载成功，启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 3. 获取执行器索引
    joint1_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint1_act")
    joint2_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint2_act")
    joint3_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint3_act")
    joint4_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint4_act")
    left_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_grip_act")
    right_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_grip_act")

    # 4. 获取末端执行器（绿色标记）的ID（用于位置反馈）
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "end_effector_marker")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")

    # 5. 控制函数（平滑控制+末端反馈）
    def smooth_set_joint(joint_idx, target_val, duration, viewer):
        start_val = data.ctrl[joint_idx]
        start_time = time.time()
        while (time.time() - start_time) < duration and viewer.is_running():
            t = (time.time() - start_time) / duration
            current_val = start_val + t * (target_val - start_val)
            data.ctrl[joint_idx] = current_val
            # 实时打印末端位置
            print_end_effector_position(data, end_effector_id, target_id)
            # 步进仿真
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    def smooth_set_gripper(target, duration, viewer):
        start_left = data.ctrl[left_grip_idx]
        start_right = data.ctrl[right_grip_idx]
        target_right = -target
        start_time = time.time()
        while (time.time() - start_time) < duration and viewer.is_running():
            t = (time.time() - start_time) / duration
            data.ctrl[left_grip_idx] = start_left + t * (target - start_left)
            data.ctrl[right_grip_idx] = start_right + t * (target_right - start_right)
            print_end_effector_position(data, end_effector_id, target_id)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    def print_end_effector_position(data, ee_id, tar_id):
        # 获取末端和目标的位置
        ee_pos = data.geom_xpos[ee_id]
        tar_pos = data.geom_xpos[tar_id]
        # 计算距离
        distance = np.linalg.norm(ee_pos - tar_pos)
        # 实时刷新打印（不换行）
        print(
            f"\r末端位置(X:{ee_pos[0]:.2f}, Y:{ee_pos[1]:.2f}, Z:{ee_pos[2]:.2f}) | 目标位置(X:{tar_pos[0]:.2f}, Y:{tar_pos[1]:.2f}, Z:{tar_pos[2]:.2f}) | 距离:{distance:.3f} m",
            end="")

    # 6. SCARA机械臂目标跟随流程
    scara_steps = [
        ("关节1旋转对准目标", joint1_idx, 0.785, 2.5),  # 45°旋转
        ("关节2旋转调整姿态", joint2_idx, -0.523, 2.0),  # -30°旋转
        ("关节3升降接近目标", joint3_idx, 0.3, 1.8),  # 下降接近目标
        ("关节4旋转校准方向", joint4_idx, 1.047, 2.0),  # 60°旋转校准
        ("夹紧夹爪模拟抓取", "gripper", -0.4, 1.2),  # 夹紧夹爪
        ("关节3升降抬升目标", joint3_idx, 0.6, 1.8),  # 抬升
        ("关节1反向旋转归位", joint1_idx, 0.0, 2.5),  # 归位旋转
        ("关节2反向旋转归位", joint2_idx, 0.0, 2.0),  # 归位旋转
        ("关节3下降放置目标", joint3_idx, 0.3, 1.8),  # 下降放置
        ("放松夹爪完成操作", "gripper", 0.0, 1.2),  # 放松夹爪
        ("关节3升降归位", joint3_idx, 0.0, 1.8),  # 最终归位
        ("关节4旋转归位", joint4_idx, 0.0, 2.0),  # 最终归位
    ]

    # 7. 启动仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n📌 开始SCARA机械臂目标跟随流程...")
        print("-" * 60)

        for step_name, joint_or_grip, target, duration in scara_steps:
            print(f"\n\n🔧 {step_name}")
            if joint_or_grip == "gripper":
                smooth_set_gripper(target, duration, viewer)
            else:
                smooth_set_joint(joint_or_grip, target, duration, viewer)

        # 保持5秒查看最终效果
        print("\n\n\n📌 SCARA机械臂操作完成，保持可视化5秒...")
        start_hold = time.time()
        while (time.time() - start_hold) < 5 and viewer.is_running():
            print_end_effector_position(data, end_effector_id, target_id)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print("\n\n🎉 SCARA机械臂末端反馈+目标跟随演示完毕！")


if __name__ == "__main__":
    scara_robot_arm_demo()