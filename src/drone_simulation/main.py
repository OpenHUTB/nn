# simple_visible_drone.py
import mujoco
import mujoco.viewer
import numpy as np
import time

# 最小化但可见的模型
MJCF_MODEL = """
<mujoco>
  <visual>
    <global azimuth="45" elevation="-30"/>
  </visual>

  <worldbody>
    <!-- 明亮的背景 -->
    <light name="top" pos="0 0 10" dir="0 0 -1" directional="true" diffuse="1 1 1"/>

    <!-- 彩色地面 -->
    <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.6 0.8 0.6 1"/>

    <!-- 大号彩色无人机 -->
    <body name="drone" pos="0 0 1">
      <freejoint/>

      <!-- 中心大立方体 -->
      <geom name="body" type="box" size="0.25 0.25 0.05" rgba="1 0.5 0 1" mass="1.0"/>

      <!-- 四个大旋翼（更容易看到） -->
      <geom name="rotor1" type="cylinder" pos="0.5 0.5 0" size="0.2 0.02" rgba="1 0 0 1" mass="0.2"/>
      <geom name="rotor2" type="cylinder" pos="0.5 -0.5 0" size="0.2 0.02" rgba="0 1 0 1" mass="0.2"/>
      <geom name="rotor3" type="cylinder" pos="-0.5 -0.5 0" size="0.2 0.02" rgba="0 0 1 1" mass="0.2"/>
      <geom name="rotor4" type="cylinder" pos="-0.5 0.5 0" size="0.2 0.02" rgba="1 1 0 1" mass="0.2"/>

      <!-- 连接臂 -->
      <geom name="arm1" type="capsule" fromto="0 0 0 0.5 0.5 0" size="0.03" rgba="0 0 0 1"/>
      <geom name="arm2" type="capsule" fromto="0 0 0 0.5 -0.5 0" size="0.03" rgba="0 0 0 1"/>
      <geom name="arm3" type="capsule" fromto="0 0 0 -0.5 -0.5 0" size="0.03" rgba="0 0 0 1"/>
      <geom name="arm4" type="capsule" fromto="0 0 0 -0.5 0.5 0" size="0.03" rgba="0 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    print("正在启动无人机仿真...")
    print("按ESC退出窗口")
    print("等待3秒...")

    # 直接从字符串加载模型
    model = mujoco.MjModel.from_xml_string(MJCF_MODEL)
    data = mujoco.MjData(model)

    # 等待3秒
    time.sleep(3)

    print("开始飞行演示...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角
        viewer.cam.lookat[:] = [0, 0, 1]
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30

        t = 0
        while viewer.is_running() and t < 20:  # 运行20秒
            # 简单的上下浮动
            force_z = 20 * np.sin(t * 2) + 50

            # 应用力
            data.qfrc_applied[2] = force_z

            # 缓慢旋转
            data.qfrc_applied[5] = 5 * np.sin(t * 0.5)

            mujoco.mj_step(model, data)
            viewer.sync()

            t += 0.01
            time.sleep(0.01)

    print("仿真结束！")


if __name__ == "__main__":
    main()