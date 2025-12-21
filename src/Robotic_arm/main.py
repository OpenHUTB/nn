import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time  # 新增：备用的睡眠函数

# ====================== 1. 定义机械臂 XML 模型 ======================
# 6自由度机械臂的 MuJoCo XML 描述
arm_xml = """
<mujoco model="6dof_arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <!-- 视觉和物理材质 -->
  <asset>
    <material name="gray" rgba="0.7 0.7 0.7 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
  </asset>

  <!-- 世界体 -->
  <worldbody>
    <!-- 地面 -->
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="gray"/>

    <!-- 机械臂基座 -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.15 0.1" pos="0 0 0" material="gray"/>
      <joint name="joint0" type="hinge" axis="0 0 1" pos="0 0 0.1"/>

      <!-- 连杆1 (肩部旋转) -->
      <body name="link1" pos="0 0 0.1">
        <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" material="blue"/>
        <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.3"/>

        <!-- 连杆2 (肘部旋转) -->
        <body name="link2" pos="0 0 0.3">
          <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.4 0 0" material="blue"/>
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0.4 0 0"/>

          <!-- 连杆3 (前臂) -->
          <body name="link3" pos="0.4 0 0">
            <geom name="link3_geom" type="capsule" size="0.04" fromto="0 0 0 0.35 0 0" material="blue"/>
            <joint name="joint3" type="hinge" axis="1 0 0" pos="0.35 0 0"/>

            <!-- 连杆4 (腕部旋转1) -->
            <body name="link4" pos="0.35 0 0">
              <geom name="link4_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.25" material="blue"/>
              <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.25"/>

              <!-- 连杆5 (腕部旋转2) -->
              <body name="link5" pos="0 0 0.25">
                <geom name="link5_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" material="blue"/>
                <joint name="joint5" type="hinge" axis="1 0 0" pos="0 0 0.2"/>

                <!-- 末端执行器 -->
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

  <!-- 关节控制器 -->
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


# ====================== 2. 模型加载和仿真控制 ======================
def create_arm_simulation():
    """创建并运行机械臂仿真"""
    # 将XML字符串写入临时文件（MuJoCo需要文件路径加载）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        # 加载模型和数据
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        print("✅ 机械臂模型加载成功！")
        print(f"🔧 关节数量：{model.njnt}")
        print(f"🔧 执行器数量：{model.nu}")

        # 设置初始关节角度
        initial_joint_angles = [0, 0.2, -0.5, 0, 0.3, 0]
        data.qpos[:6] = initial_joint_angles

        # 启动可视化界面
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\n🎮 仿真已启动！按 Ctrl+C 退出")
            print("💡 机械臂会自动缓慢运动，展示关节控制效果")

            # 仿真循环
            step = 0
            while viewer.is_running():
                # 控制频率：每20步更新一次关节目标
                if step % 20 == 0:
                    # 生成周期性的关节控制指令（让机械臂缓慢摆动）
                    t = data.time
                    target_angles = [
                        0.2 * np.sin(t * 0.5),  # joint0: 基座旋转
                        0.3 + 0.2 * np.sin(t),  # joint1: 肩部
                        -0.6 + 0.2 * np.cos(t),  # joint2: 肘部
                        0.1 * np.sin(t * 1.2),  # joint3: 前臂
                        0.2 * np.cos(t * 0.8),  # joint4: 腕部1
                        0.1 * np.sin(t * 1.5)  # joint5: 腕部2
                    ]
                    # 设置控制指令
                    data.ctrl[:6] = target_angles

                # 运行一步仿真
                mujoco.mj_step(model, data)

                # 更新可视化
                viewer.sync()

                # 修复：兼容不同版本的睡眠函数
                try:
                    # 尝试调用新版 MuJoCo 的 sleep 函数（归属到 utils）
                    mujoco.utils.mju_sleep(1 / 60)
                except AttributeError:
                    try:
                        # 尝试调用旧版 MuJoCo 的 sleep 函数（主模块）
                        mujoco.mju_sleep(1 / 60)
                    except AttributeError:
                        # 终极备用：使用 Python 内置的 time.sleep
                        time.sleep(1 / 60)

                step += 1

    except Exception as e:
        print(f"❌ 仿真出错：{e}")
    finally:
        # 删除临时XML文件
        os.unlink(xml_path)


if __name__ == "__main__":
    # 检查MuJoCo版本
    print(f"🔍 MuJoCo 版本：{mujoco.__version__}")

    # 启动仿真
    create_arm_simulation()