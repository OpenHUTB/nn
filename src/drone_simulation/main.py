# simple_visible_drone.py
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import glob

# 定义日志文件路径（可根据实际情况修改）
LOG_FILE_PATHS = [
    "mujoco.log",
    "drone_simulation.log",
    "./logs/*.log"
]


def delete_log_files():
    """删除指定的日志文件（无控制台输出）"""
    deleted_count = 0
    for path in LOG_FILE_PATHS:
        if "*" in path:
            for file in glob.glob(path):
                try:
                    os.remove(file)
                    deleted_count += 1
                except (FileNotFoundError, PermissionError):
                    continue
        else:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    deleted_count += 1
                except (FileNotFoundError, PermissionError):
                    continue


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
    # 删除日志文件（无输出）
    delete_log_files()

    # 加载模型和数据
    model = mujoco.MjModel.from_xml_string(MJCF_MODEL)
    data = mujoco.MjData(model)

    # 等待3秒（无提示）
    time.sleep(3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角
        viewer.cam.lookat[:] = [0, 0, 1]
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30

        t = 0
        while viewer.is_running() and t < 20:
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


if __name__ == "__main__":
    main()
