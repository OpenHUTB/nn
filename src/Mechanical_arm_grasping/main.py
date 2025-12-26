# MuJoCo 3.4.0 多自由度旋转机械臂演示（全新版本）
import mujoco
import mujoco.viewer
import time


def multi_dof_robot_arm_demo():
    # 1. 内置多自由度机械臂XML模型（含旋转+升降+伸展+夹爪）
    multi_dof_xml = """
<mujoco model="Multi-DOF Robot Arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual/>
  <asset>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="blue" rgba="0.2 0.2 0.8 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="yellow" rgba="0.8 0.8 0.2 1"/>
  </asset>
  <worldbody>
    <camera name="fixed_camera" pos="2.0 1.0 1.2" xyaxes="1 0 0 0 1 0"/>
    <!-- 地面 -->
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.1" material="gray"/>
    <!-- 目标标记点 -->
    <geom name="target_marker" type="sphere" size="0.05" pos="0.6 0.6 0.1" material="yellow"/>
    <!-- 多自由度机械臂 -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.2 0.1" pos="0 0 0" material="blue"/>
      <joint name="base_joint" type="free"/>
      <!-- 1. 水平旋转关节（绕Z轴旋转，调整机械臂朝向） -->
      <body name="rotate_link" pos="0 0 0.1">
        <geom name="rotate_geom" type="cylinder" size="0.12 0.2" pos="0 0 0.1" material="blue"/>
        <joint name="rotate_joint" type="hinge" axis="0 0 1" pos="0 0 0" range="-1.57 1.57" damping="0.1"/>
        <!-- 2. 升降关节 -->
        <body name="lift_link" pos="0 0 0.3">
          <geom name="lift_geom" type="cylinder" size="0.1 0.3" pos="0 0 0.3" material="blue"/>
          <joint name="lift_joint" type="slide" axis="0 0 1" pos="0 0 0" range="0 1.0" damping="0.1"/>
          <!-- 3. 伸展关节 -->
          <body name="extend_link" pos="0 0 0.6">
            <geom name="extend_geom" type="cylinder" size="0.08 0.4" pos="0.4 0 0" material="blue"/>
            <joint name="extend_joint" type="slide" axis="1 0 0" pos="0 0 0" range="0 0.8" damping="0.1"/>
            <!-- 4. 夹爪 -->
            <body name="gripper_base" pos="0.8 0 0">
              <geom name="gripper_base_geom" type="box" size="0.1 0.1 0.1" pos="0 0 0" material="red"/>
              <body name="left_gripper" pos="0 0.1 0">
                <geom name="left_gripper_geom" type="box" size="0.1 0.05 0.05" pos="0 0 0" material="red"/>
                <joint name="left_gripper_joint" type="hinge" axis="0 0 1" pos="0 -0.1 0" range="-0.5 0" damping="0.05"/>
              </body>
              <body name="right_gripper" pos="0 -0.1 0">
                <geom name="right_gripper_geom" type="box" size="0.1 0.05 0.05" pos="0 0 0" material="red"/>
                <joint name="right_gripper_joint" type="hinge" axis="0 0 1" pos="0 0.1 0" range="0 0.5" damping="0.05"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <!-- 执行器配置（新增旋转关节执行器） -->
  <actuator>
    <position name="rotate_actuator" joint="rotate_joint" kp="1000" kv="100"/>
    <position name="lift_actuator" joint="lift_joint" kp="1000" kv="100"/>
    <position name="extend_actuator" joint="extend_joint" kp="1000" kv="100"/>
    <position name="left_gripper_actuator" joint="left_gripper_joint" kp="500" kv="50"/>
    <position name="right_gripper_actuator" joint="right_gripper_joint" kp="500" kv="50"/>
  </actuator>
</mujoco>
    """

    # 2. 加载模型
    try:
        model = mujoco.MjModel.from_xml_string(multi_dof_xml)
        data = mujoco.MjData(model)
        print("✅ 多自由度机械臂模型加载成功，启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 3. 获取所有执行器索引
    rotate_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotate_actuator")
    lift_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_actuator")
    extend_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "extend_actuator")
    left_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper_actuator")
    right_grip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_gripper_actuator")

    # 4. 控制函数（新增旋转控制）
    def control_rotate(val):
        data.ctrl[rotate_idx] = val

    def control_lift(val):
        data.ctrl[lift_idx] = val

    def control_extend(val):
        data.ctrl[extend_idx] = val

    def control_gripper(val):
        data.ctrl[left_grip_idx] = val
        data.ctrl[right_grip_idx] = -val

    # 5. 多自由度动作流程（含旋转调整朝向）
    action_sequence = [
        ("旋转调整朝向", control_rotate, 1.0, 2.0),  # 水平旋转（朝向黄色标记点）
        ("上升准备", control_lift, 0.6, 1.5),
        ("伸展接近目标", control_extend, 0.7, 2.0),
        ("下降到位", control_lift, 0.2, 1.5),
        ("夹紧夹爪", control_gripper, -0.4, 1.0),
        ("抓取上升", control_lift, 0.7, 1.5),
        ("反向旋转归位", control_rotate, 0.0, 2.0),
        ("收缩机械臂", control_extend, 0.0, 2.0),
        ("下降放置", control_lift, 0.2, 1.5),
        ("放松夹爪", control_gripper, 0.0, 1.0),
        ("最终归位", control_lift, 0.5, 1.5),
    ]

    # 6. 启动仿真并执行动作（新增关节状态打印）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 打印状态表头
        print("\n📊 实时关节状态（旋转角度/升降高度/伸展长度）")
        print("-" * 50)

        for action_name, func, target, duration in action_sequence:
            print(f"\n🔧 正在执行：{action_name}")
            start_time = time.time()
            while (time.time() - start_time) < duration and viewer.is_running():
                func(target)
                mujoco.mj_step(model, data)

                # 实时打印关键关节状态
                rotate_angle = data.joint("rotate_joint").qpos[0]
                lift_height = data.joint("lift_joint").qpos[0]
                extend_length = data.joint("extend_joint").qpos[0]
                print(
                    f"\r旋转角度：{rotate_angle:.2f} rad | 升降高度：{lift_height:.2f} m | 伸展长度：{extend_length:.2f} m",
                    end="")

                viewer.sync()
                time.sleep(0.001)

        # 最后保持4秒查看效果
        print("\n\n📌 动作流程完成，保持可视化4秒...")
        start = time.time()
        while (time.time() - start) < 4 and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    print("\n🎉 多自由度机械臂演示完毕！")


if __name__ == "__main__":
    multi_dof_robot_arm_demo()