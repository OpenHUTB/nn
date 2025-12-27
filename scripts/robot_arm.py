"""
MuJoCo机械臂仿真初始版本
功能：基础2关节机械臂可视化+关节角度控制
项目根目录：Robot_arm_motion
代码目录：scripts
作者：邓卓
日期：2025-12-22
"""
import mujoco
import mujoco.viewer as viewer
import numpy as np

def create_simple_arm_model():
    """创建简易2关节机械臂+立方体模型的XML字符串"""
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
              <geom name="gripper" type="box" size="0.08 0.08 0.02" pos="0 0 0.4" rgba="0.8 0.2 0.2 1"/>
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
        <motor joint="joint1" ctrlrange="-3.14 3.14" gear="100"/>
        <motor joint="joint2" ctrlrange="-3.14 3.14" gear="100"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)

def main():
    """主函数：加载模型+启动仿真"""
    # 加载模型
    model = create_simple_arm_model()
    data = mujoco.MjData(model)

    # 设置初始关节角度
    joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    joint2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    data.ctrl[joint1_id] = np.pi/4  # 关节1转45°
    data.ctrl[joint2_id] = -np.pi/6 # 关节2转-30°

    # 启动可视化
    print("=== Robot_arm_motion 初始版本仿真启动 ===")
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == "__main__":
    main()