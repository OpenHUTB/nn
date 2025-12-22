import mujoco
import mujoco.viewer
import numpy as np
import os
import tempfile
import time
from scipy import interpolate


# ====================== 轨迹优化核心（纯原生） ======================
def quintic_polynomial(start, end, T, t):
    """五次多项式插值：保证位置/速度/加速度连续"""
    s0, v0, a0 = start, 0, 0
    s1, v1, a1 = end, 0, 0

    a = s0
    b = v0
    c = a0 / 2
    d = (20 * (s1 - s0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T ** 2) / (2 * T ** 3)
    e = (30 * (s0 - s1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T ** 2) / (2 * T ** 4)
    f = (12 * (s1 - s0) - (6 * v1 + 6 * v0) * T - (a0 - a1) * T ** 2) / (2 * T ** 5)

    return a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5


# ====================== 机械臂模型 ======================
arm_xml = """
<mujoco model="6dof_arm">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <material name="gray" rgba="0.7 0.7 0.7 1"/>
    <material name="blue" rgba="0.2 0.4 0.8 1"/>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="gray"/>
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.15 0.1" pos="0 0 0" material="gray"/>
      <joint name="joint0" type="hinge" axis="0 0 1" pos="0 0 0.1"/>
      <body name="link1" pos="0 0 0.1">
        <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" material="blue"/>
        <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0.3"/>
        <body name="link2" pos="0 0 0.3">
          <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.4 0 0" material="blue"/>
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0.4 0 0"/>
          <body name="link3" pos="0.4 0 0">
            <geom name="link3_geom" type="capsule" size="0.04" fromto="0 0 0 0.35 0 0" material="blue"/>
            <joint name="joint3" type="hinge" axis="1 0 0" pos="0.35 0 0"/>
            <body name="link4" pos="0.35 0 0">
              <geom name="link4_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.25" material="blue"/>
              <joint name="joint4" type="hinge" axis="0 1 0" pos="0 0 0.25"/>
              <body name="link5" pos="0 0 0.25">
                <geom name="link5_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" material="blue"/>
                <joint name="joint5" type="hinge" axis="1 0 0" pos="0 0 0.2"/>
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


# ====================== 仿真运行 ======================
def run_simulation():
    # 临时XML文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(arm_xml)
        xml_path = f.name

    try:
        # 加载模型
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ 模型加载成功！")

        # 轨迹关键点（关节空间）
        waypoints = [
            [0, 0.2, -0.5, 0, 0.3, 0],
            [0.5, 0.5, -0.8, 0.2, 0.5, 0.3],
            [0.8, 0.3, -0.6, 0.4, 0.2, 0.5],
            [0.5, 0.5, -0.8, 0.2, 0.5, 0.3],
            [0, 0.2, -0.5, 0, 0.3, 0]
        ]
        segment_time = 3.0  # 每段轨迹时长

        # 启动可视化
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🎮 仿真启动！机械臂沿平滑轨迹运动（按Ctrl+C退出）")
            while viewer.is_running():
                # 计算当前轨迹段
                t_total = data.time
                seg_idx = int(t_total // segment_time) % (len(waypoints) - 1)
                t_seg = t_total % segment_time

                # 生成平滑轨迹
                target = []
                for i in range(6):
                    pos = quintic_polynomial(
                        waypoints[seg_idx][i],
                        waypoints[seg_idx + 1][i],
                        segment_time,
                        t_seg
                    )
                    # 限制关节范围
                    pos = np.clip(pos, model.actuator_ctrlrange[i][0], model.actuator_ctrlrange[i][1])
                    target.append(pos)

                # 控制机械臂
                data.ctrl[:6] = target
                mujoco.mj_step(model, data)
                viewer.sync()

                # 帧率控制
                try:
                    mujoco.utils.mju_sleep(1 / 60)
                except:
                    time.sleep(1 / 60)

    except Exception as e:
        print(f"❌ 错误：{e}")
    finally:
        os.unlink(xml_path)


if __name__ == "__main__":
    run_simulation()  # 修复：删掉了多余的 + 号