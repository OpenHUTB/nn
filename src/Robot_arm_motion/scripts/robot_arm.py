"""
MuJoCo机械臂仿真v0.2
功能：基础2关节机械臂可视化+关节角度控制+夹爪开合控制
项目根目录：Robot_arm_motion
代码目录：scripts
作者：邓卓
日期：2025-12-22
版本：v0.2（添加夹爪开合控制，修复XML Schema错误）
"""
import mujoco
import mujoco.viewer as viewer
import numpy as np

def create_simple_arm_model():
    """创建简易2关节机械臂+立方体模型的XML字符串（修复ctrlrange属性错误）"""
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81" timestep="0.001"/>
      <worldbody>
        <!-- 地面 -->
        <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>
        <!-- 机械臂基座 -->
        <body name="base" pos="0 0 0">
          <geom type="cylinder" size="0.1 0.1" rgba="0.2 0.2 0.8 1"/>
          <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.1"/>
          <body name="link1" pos="0 0 0.1">
            <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.5" rgba="0.2 0.8 0.2 1"/>
            <joint name="joint2" type="hinge" axis="1 0 0" pos="0 0 0.5"/>
            <body name="link2" pos="0 0 0.5">
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.4" rgba="0.2 0.8 0.2 1"/>
              <!-- 夹爪基座 -->
              <body name="gripper_base" pos="0 0 0.4">
                <geom type="box" size="0.02 0.08 0.02" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>
                <!-- 左夹爪（移除joint的ctrlrange属性） -->
                <body name="left_gripper" pos="-0.04 0 0">
                  <joint name="left_gripper_joint" type="slide" axis="1 0 0" pos="0 0 0"/>
                  <geom name="left_gripper_geom" type="box" size="0.04 0.02 0.02" pos="0 0 0" rgba="0.8 0.2 0.2 1"/>
                </body>
                <!-- 右夹爪（移除joint的ctrlrange属性） -->
                <body name="right_gripper" pos="0.04 0 0">
                  <joint name="right_gripper_joint" type="slide" axis="1 0 0" pos="0 0 0"/>
                  <geom name="right_gripper_geom" type="box" size="0.04 0.02 0.02" pos="0 0 0" rgba="0.8 0.2 0.2 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
        <!-- 抓取目标：立方体 -->
        <body name="cube" pos="0.3 0 0.5">
          <joint type="free"/>
          <geom type="box" size="0.05 0.05 0.05" rgba="0.8 0.8 0.2 1"/>
        </body>
      </worldbody>
      <actuator>
        <!-- 机械臂关节电机 -->
        <motor joint="joint1" ctrlrange="-3.14 3.14" gear="100"/>
        <motor joint="joint2" ctrlrange="-3.14 3.14" gear="100"/>
        <!-- 夹爪关节电机（将ctrlrange移到此处） -->
        <motor joint="left_gripper_joint" ctrlrange="0 0.06" gear="50"/>
        <motor joint="right_gripper_joint" ctrlrange="-0.06 0" gear="50"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)

def main():
    """主函数：加载模型+启动仿真（新增夹爪控制）"""
    # 加载模型
    model = create_simple_arm_model()
    data = mujoco.MjData(model)

    # 设置初始关节角度
    joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    joint2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    data.ctrl[joint1_id] = np.pi/4  # 关节1转45°
    data.ctrl[joint2_id] = -np.pi/6 # 关节2转-30°

    # 新增：设置夹爪初始开合角度（半开状态）
    left_gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_gripper_joint")
    right_gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_gripper_joint")
    data.ctrl[left_gripper_id] = 0.03    # 左夹爪张开3mm
    data.ctrl[right_gripper_id] = -0.03  # 右夹爪张开3mm

    # 启动可视化
    print("=== Robot_arm_motion v0.2 仿真启动（新增夹爪开合控制） ===")
    print("当前夹爪状态：半开（3mm）")
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == "__main__":
    main()