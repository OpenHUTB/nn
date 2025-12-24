# 机械臂MuJoCo 3.4.0 原生语法稳定版（100%兼容，无API报错）
import mujoco
import mujoco.viewer
import time


def robot_arm_final_stable_demo():
    # 1. 内置机械臂XML模型（slide/hinge关节，3.4.0原生兼容）
    robot_xml = """
<mujoco model="Simple Robot Arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual/>
  <asset>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="blue" rgba="0.2 0.2 0.8 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
  </asset>
  <worldbody>
    <camera name="fixed_camera" pos="1.5 1.5 1.0" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.1" material="gray"/>
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.2 0.1" pos="0 0 0" material="blue"/>
      <joint name="base_joint" type="free"/>
      <body name="lift_link" pos="0 0 0.1">
        <geom name="lift_geom" type="cylinder" size="0.15 0.3" pos="0 0 0.3" material="blue"/>
        <joint name="lift_joint" type="slide" axis="0 0 1" pos="0 0 0" range="0 1.0" damping="0.1"/>
        <body name="extend_link" pos="0 0 0.6">
          <geom name="extend_geom" type="cylinder" size="0.1 0.4" pos="0.4 0 0" material="blue"/>
          <joint name="extend_joint" type="slide" axis="1 0 0" pos="0 0 0" range="0 0.8" damping="0.1"/>
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
  </worldbody>
  <actuator>
    <position name="lift_actuator" joint="lift_joint" kp="1000" kv="100"/>
    <position name="extend_actuator" joint="extend_joint" kp="1000" kv="100"/>
    <position name="left_gripper_actuator" joint="left_gripper_joint" kp="500" kv="50"/>
    <position name="right_gripper_actuator" joint="right_gripper_joint" kp="500" kv="50"/>
  </actuator>
</mujoco>
    """

    # 2. 加载模型
    try:
        model = mujoco.MjModel.from_xml_string(robot_xml)
        data = mujoco.MjData(model)
        print("✅ 机械臂模型加载成功，启动仿真...")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 3. 获取执行器索引（对应data.ctrl数组的下标，3.4.0原生支持）
    lift_act_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_actuator")
    extend_act_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "extend_actuator")
    left_gripper_act_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper_actuator")
    right_gripper_act_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_gripper_actuator")

    # 4. 动作控制逻辑（直接操作data.ctrl，3.4.0原生语法）
    def control_lift(target):
        data.ctrl[lift_act_idx] = target  # 直接给执行器对应下标赋值

    def control_extend(target):
        data.ctrl[extend_act_idx] = target

    def control_gripper(target_left):
        target_right = -target_left
        data.ctrl[left_gripper_act_idx] = target_left
        data.ctrl[right_gripper_act_idx] = target_right

    # 5. 预设动作流程
    action_list = [
        ("上升", "lift", 0.8, 2.0),
        ("伸展", "extend", 0.6, 2.0),
        ("夹紧", "gripper", -0.4, 1.0),
        ("保持", "none", None, 1.5),
        ("放松", "gripper", 0, 1.0),
        ("收缩", "extend", 0, 2.0),
        ("下降", "lift", 0, 2.0),
    ]

    # 6. 启动可视化并执行动作
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for action_name, action_type, target, dur in action_list:
            print(f"🔧 正在执行：{action_name}")
            start_time = time.time()
            while (time.time() - start_time) < dur and viewer.is_running():
                # 执行对应动作
                if action_type == "lift":
                    control_lift(target)
                elif action_type == "extend":
                    control_extend(target)
                elif action_type == "gripper":
                    control_gripper(target)

                # 步进仿真+同步可视化
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)

    print("🎉 机械臂动作执行完毕！")


if __name__ == "__main__":
    robot_arm_final_stable_demo()