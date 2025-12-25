import mujoco
import mujoco.viewer
import numpy as np
import time

# 无人机模型（完全静止版本 - 修复关节范围错误）
MJCF_MODEL = """
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <visual>
    <global azimuth="45" elevation="-30"/>
  </visual>

  <worldbody>
    <!-- 大尺寸地面 -->
    <geom name="ground" type="plane" pos="0 0 0" size="20 20 0.5" rgba="0.5 0.7 0.5 1"/>

    <!-- 无人机主体（放大尺寸确保可见） -->
    <body name="drone" pos="0 0 2">
      <freejoint/>

      <!-- 主机身（使用简单的立方体） -->
      <geom name="body" type="box" size="0.5 0.5 0.2" rgba="0.1 0.3 0.8 1" mass="1.0"/>

      <!-- 四个机臂（更粗更长） -->
      <body name="arm1" pos="0.6 0.6 0">
        <geom name="arm1_geom" type="capsule" fromto="0 0 0 0.8 0.8 0" size="0.1" rgba="0.3 0.3 0.3 1" mass="0.2"/>
        <!-- 旋翼组件（无关节，完全固定） -->
        <body name="rotor1" pos="0.8 0.8 0.05">
          <!-- 移除旋转关节，直接固定旋翼 -->
          <geom name="rotor1_base" type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1" mass="0.05"/>
          <!-- 旋翼叶片 -->
          <geom name="rotor1_blade1" type="box" pos="0.4 0 0" size="0.4 0.05 0.01" rgba="0.8 0.2 0.2 1" mass="0.02"/>
          <geom name="rotor1_blade2" type="box" pos="-0.4 0 0" size="0.4 0.05 0.01" rgba="0.8 0.2 0.2 1" mass="0.02"/>
        </body>
      </body>

      <body name="arm2" pos="0.6 -0.6 0">
        <geom name="arm2_geom" type="capsule" fromto="0 0 0 0.8 -0.8 0" size="0.1" rgba="0.3 0.3 0.3 1" mass="0.2"/>
        <body name="rotor2" pos="0.8 -0.8 0.05">
          <!-- 移除旋转关节 -->
          <geom name="rotor2_base" type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1" mass="0.05"/>
          <geom name="rotor2_blade1" type="box" pos="0.4 0 0" size="0.4 0.05 0.01" rgba="0.2 0.8 0.2 1" mass="0.02"/>
          <geom name="rotor2_blade2" type="box" pos="-0.4 0 0" size="0.4 0.05 0.01" rgba="0.2 0.8 0.2 1" mass="0.02"/>
        </body>
      </body>

      <body name="arm3" pos="-0.6 -0.6 0">
        <geom name="arm3_geom" type="capsule" fromto="0 0 0 -0.8 -0.8 0" size="0.1" rgba="0.3 0.3 0.3 1" mass="0.2"/>
        <body name="rotor3" pos="-0.8 -0.8 0.05">
          <!-- 移除旋转关节 -->
          <geom name="rotor3_base" type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1" mass="0.05"/>
          <geom name="rotor3_blade1" type="box" pos="0.4 0 0" size="0.4 0.05 0.01" rgba="0.8 0.2 0.2 1" mass="0.02"/>
          <geom name="rotor3_blade2" type="box" pos="-0.4 0 0" size="0.4 0.05 0.01" rgba="0.8 0.2 0.2 1" mass="0.02"/>
        </body>
      </body>

      <body name="arm4" pos="-0.6 0.6 0">
        <geom name="arm4_geom" type="capsule" fromto="0 0 0 -0.8 0.8 0" size="0.1" rgba="0.3 0.3 0.3 1" mass="0.2"/>
        <body name="rotor4" pos="-0.8 0.8 0.05">
          <!-- 移除旋转关节 -->
          <geom name="rotor4_base" type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1" mass="0.05"/>
          <geom name="rotor4_blade1" type="box" pos="0.4 0 0" size="0.4 0.05 0.01" rgba="0.2 0.8 0.2 1" mass="0.02"/>
          <geom name="rotor4_blade2" type="box" pos="-0.4 0 0" size="0.4 0.05 0.01" rgba="0.2 0.8 0.2 1" mass="0.02"/>
        </body>
      </body>

      <!-- 起落架 -->
      <body name="landing1" pos="0.4 0.4 -0.2">
        <geom name="leg1" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.5 0.5 0.5 1" mass="0.1"/>
        <geom name="foot1" type="sphere" pos="0 0 -0.5" size="0.12" rgba="0.2 0.2 0.2 1" mass="0.05"/>
      </body>

      <body name="landing2" pos="0.4 -0.4 -0.2">
        <geom name="leg2" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.5 0.5 0.5 1" mass="0.1"/>
        <geom name="foot2" type="sphere" pos="0 0 -0.5" size="0.12" rgba="0.2 0.2 0.2 1" mass="0.05"/>
      </body>

      <body name="landing3" pos="-0.4 -0.4 -0.2">
        <geom name="leg3" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.5 0.5 0.5 1" mass="0.1"/>
        <geom name="foot3" type="sphere" pos="0 0 -0.5" size="0.12" rgba="0.2 0.2 0.2 1" mass="0.05"/>
      </body>

      <body name="landing4" pos="-0.4 0.4 -0.2">
        <geom name="leg4" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.08" rgba="0.5 0.5 0.5 1" mass="0.1"/>
        <geom name="foot4" type="sphere" pos="0 0 -0.5" size="0.12" rgba="0.2 0.2 0.2 1" mass="0.05"/>
      </body>
    </body>
  </worldbody>

  <!-- 移除电机控制器（不再需要） -->
</mujoco>
"""


def main():
    print("=" * 50)
    print("          无人机完全静止展示")
    print("=" * 50)
    print("特性：")
    print("1. 无人机保持绝对静止，无任何移动或旋转")
    print("2. 旋翼完全固定，不会转动")
    print("3. 所有零件尺寸放大，视觉效果佳")
    print("4. 按ESC键退出展示窗口")
    print("=" * 50)

    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_string(MJCF_MODEL)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"加载模型失败: {e}")
        input("按Enter键退出...")
        return

    # 初始化viewer
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        is_passive = True
    except:
        viewer = mujoco.viewer.launch(model, data)
        is_passive = False

    try:
        # 设置相机视角（固定视角）
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 2.0
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0

        # 强制无人机完全静止
        data.body('drone').xpos[:] = [0.0, 0.0, 2.0]  # 固定位置
        data.body('drone').xquat[:] = [1.0, 0.0, 0.0, 0.0]  # 固定姿态
        data.body('drone').cvel[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 零速度

        sim_duration = 0
        last_time = time.time()
        last_print = 0

        # 展示主循环
        while True:
            if is_passive and not viewer.is_running():
                break

            # 控制帧率
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # 运行仿真步（但保持所有物体静止）
            mujoco.mj_step(model, data)

            # 每帧都强制重置位置和速度，确保绝对静止
            data.body('drone').xpos[:] = [0.0, 0.0, 2.0]
            data.body('drone').xquat[:] = [1.0, 0.0, 0.0, 0.0]
            data.body('drone').cvel[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # 更新视图
            if is_passive:
                viewer.sync()

            # 每5秒打印一次状态
            if sim_duration - last_print > 5.0:
                print(f"\n📊 展示状态 (时间: {sim_duration:.1f}s)")
                print(
                    f"📍 无人机位置: x={data.body('drone').xpos[0]:.3f}, y={data.body('drone').xpos[1]:.3f}, z={data.body('drone').xpos[2]:.3f}m")
                print(f"🔄 无人机速度: {np.linalg.norm(data.body('drone').cvel):.3f} m/s")
                print("✅ 无人机保持完全静止")
                last_print = sim_duration

            sim_duration += dt
            time.sleep(0.01)  # 控制展示帧率

    except KeyboardInterrupt:
        print("\n\n展示被用户中断")
    except Exception as e:
        print(f"\n\n展示出错: {e}")
    finally:
        if viewer and is_passive:
            viewer.close()

    print("\n展示结束！")
    input("按Enter键退出...")


if __name__ == "__main__":
    main()