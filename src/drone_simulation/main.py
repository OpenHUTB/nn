"""
MuJoCo 四旋翼无人机仿真 - 完全匹配原代码旋转方式版
✅ 无人机绕世界坐标系Z轴公转（不是机身自旋），与原代码100%一致
✅ 原地圆周运动，高度固定，无位置漂移、无闪烁
✅ 保留所有原代码核心特征，参数完全对齐
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math


class QuadrotorSimulation:
    def __init__(self):
        """初始化：完全复刻原代码旋转逻辑"""
        xml_string = self.create_quadrotor_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        print("✓ 模型加载成功")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 原代码悬停推力参数
        hover_thrust = 600
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

        # ========== 完全匹配原代码的旋转参数 ==========
        self.rotate_radius = 1.0  # 公转半径（原代码核心参数）
        self.rotate_speed = 1.0    # 公转角速度（rad/s），与原代码一致
        self.hover_height = 0.8    # 固定高度，与原代码一致
        self.rotate_angle = 0.0    # 公转角度累计
        self.rotor_visual_speed = 8.0  # 旋翼旋转速度，匹配原代码

    def create_quadrotor_xml(self):
        """完全复刻原代码的XML结构，无任何修改"""
        xml_string = """<?xml version="1.0" ?>
<mujoco model="quadrotor">
  <option timestep="0.005" iterations="100" tolerance="1e-10">
    <flag contact="enable" energy="enable"/>
  </option>
  <size nconmax="100" njmax="200"/>
  <default>
    <joint damping="0.001" frictionloss="0.001"/>
    <geom solref="0.02 1" solimp="0.9 0.95 0.01"/>
  </default>
  
  <asset>
    <material name="ground_mat" rgba="0.8 0.9 0.8 1"/>
    <material name="body_mat" rgba="0.3 0.3 0.3 1"/>
    <material name="arm_mat" rgba="0.1 0.1 0.1 1"/>
    <material name="motor_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="propeller_red" rgba="0.8 0.2 0.2 1.0"/>
    <material name="propeller_green" rgba="0.2 0.8 0.2 1.0"/>
    <material name="obs_cube_mat" rgba="0.6 0.2 0.8 0.9"/>
    <material name="obs_cyl_mat" rgba="0.2 0.6 0.8 0.9"/>
    <material name="obs_sphere_mat" rgba="0.8 0.6 0.2 0.9"/>
  </asset>
  
  <worldbody>
    <light name="ambient_light" pos="0 0 10" dir="0 0 -1" ambient="0.6 0.6 0.6" diffuse="0.8 0.8 0.8"/>
    <light name="directional_light" pos="5 5 8" dir="-1 -1 -1" directional="true"/>

    <!-- 地面 -->
    <geom name="ground" type="plane" pos="0 0 0" size="20 20 0.1" material="ground_mat" 
          condim="3" friction="0.8 0.005 0.0001"/>
    <!-- 参考坐标系 -->
    <geom name="origin_x" type="cylinder" fromto="0 0 0.1 1 0 0.1" size="0.01" rgba="1 0 0 1"/>
    <geom name="origin_y" type="cylinder" fromto="0 0 0.1 0 1 0.1" size="0.01" rgba="0 1 0 1"/>
    <geom name="origin_z" type="cylinder" fromto="0 0 0.1 0 0 1.1" size="0.01" rgba="0 0 1 1"/>
    
    <!-- 无人机：原代码初始位置 -->
    <body name="quadrotor" pos="0 0 0.8" euler="0 0 0">
      <joint name="quad_free_joint" type="free" damping="0.001"/>
      
      <!-- 无人机主体 -->
      <geom name="center_body" type="cylinder" size="0.1 0.03" material="body_mat" mass="0.4"/>
      
      <!-- 机臂 -->
      <geom name="arm_front_right" type="capsule" fromto="0 0 0 0.25 0.25 0" size="0.01" material="arm_mat" mass="0.04"/>
      <geom name="arm_front_left" type="capsule" fromto="0 0 0 0.25 -0.25 0" size="0.01" material="arm_mat" mass="0.04"/>
      <geom name="arm_back_left" type="capsule" fromto="0 0 0 -0.25 -0.25 0" size="0.01" material="arm_mat" mass="0.04"/>
      <geom name="arm_back_right" type="capsule" fromto="0 0 0 -0.25 0.25 0" size="0.01" material="arm_mat" mass="0.04"/>
      
      <!-- 电机和旋翼 -->
      <body name="motor_front_right" pos="0.25 0.25 0">
        <geom name="motor_housing_front_right" type="cylinder" size="0.03 0.03" material="motor_mat" mass="0.04"/>
        <body name="rotor_front_right" pos="0 0 0.05">
          <joint name="rotor_front_right_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_front_right" type="cylinder" size="0.12 0.008" material="propeller_red" mass="0.01"/>
        </body>
      </body>
      
      <body name="motor_front_left" pos="0.25 -0.25 0">
        <geom name="motor_housing_front_left" type="cylinder" size="0.03 0.03" material="motor_mat" mass="0.04"/>
        <body name="rotor_front_left" pos="0 0 0.05">
          <joint name="rotor_front_left_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_front_left" type="cylinder" size="0.12 0.008" material="propeller_green" mass="0.01"/>
        </body>
      </body>
      
      <body name="motor_back_left" pos="-0.25 -0.25 0">
        <geom name="motor_housing_back_left" type="cylinder" size="0.03 0.03" material="motor_mat" mass="0.04"/>
        <body name="rotor_back_left" pos="0 0 0.05">
          <joint name="rotor_back_left_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_back_left" type="cylinder" size="0.12 0.008" material="propeller_red" mass="0.01"/>
        </body>
      </body>
      
      <body name="motor_back_right" pos="-0.25 0.25 0">
        <geom name="motor_housing_back_right" type="cylinder" size="0.03 0.03" material="motor_mat" mass="0.04"/>
        <body name="rotor_back_right" pos="0 0 0.05">
          <joint name="rotor_back_right_joint" type="hinge" axis="0 0 1" damping="0.001"/>
          <geom name="propeller_back_right" type="cylinder" size="0.12 0.008" material="propeller_green" mass="0.01"/>
        </body>
      </body>

      <!-- 起落架 -->
      <geom name="landing_gear_front" type="cylinder" pos="0.15 0 0" size="0.008 0.05" rgba="0.5 0.5 0.5 1" mass="0.01"/>
      <geom name="landing_gear_back" type="cylinder" pos="-0.15 0 0" size="0.008 0.05" rgba="0.5 0.5 0.5 1" mass="0.01"/>

      <!-- 视觉标记 -->
      <geom name="front_marker" type="sphere" pos="0.15 0 0.02" size="0.02" rgba="1 1 0 1"/>
      <geom name="rear_marker" type="sphere" pos="-0.15 0 0.02" size="0.02" rgba="0 1 1 1"/>
    </body>

    <!-- 障碍物 -->
    <geom name="obstacle_cube" type="box" pos="2 0 0.75" size="0.25 0.25 0.75" material="obs_cube_mat" 
          friction="0.5 0.01 0.001" mass="5"/>
    <geom name="obstacle_cylinder" type="cylinder" pos="-1 1 0.5" size="0.3 0.5" material="obs_cyl_mat" 
          friction="0.5 0.01 0.001" mass="5"/>
    <geom name="obstacle_sphere" type="sphere" pos="0 -2 1.0" size="0.4" material="obs_sphere_mat" 
          friction="0.5 0.01 0.001" mass="5"/>
  </worldbody>

  <actuator>
    <motor name="motor_front_right" joint="rotor_front_right_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
    <motor name="motor_front_left" joint="rotor_front_left_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
    <motor name="motor_back_left" joint="rotor_back_left_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
    <motor name="motor_back_right" joint="rotor_back_right_joint" gear="80" ctrllimited="true" ctrlrange="0 1000"/>
  </actuator>
</mujoco>"""
        return xml_string

    def simulation_loop(self, viewer, duration):
        """核心：完全复刻原代码的旋转逻辑（公转而非自转）"""
        start_time = time.time()
        last_print_time = time.time()

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # ========== 原代码核心旋转逻辑：绕世界Z轴公转 ==========
            # 1. 更新公转角度
            self.rotate_angle += self.rotate_speed * self.model.opt.timestep
            # 2. 计算公转位置（原代码核心公式）
            target_x = self.rotate_radius * math.cos(self.rotate_angle)
            target_y = self.rotate_radius * math.sin(self.rotate_angle)
            target_z = self.hover_height
            # 3. 强制设置无人机位置（公转，机身姿态不变）
            self.data.qpos[0] = target_x  # X轴随角度变化（公转）
            self.data.qpos[1] = target_y  # Y轴随角度变化（公转）
            self.data.qpos[2] = target_z  # Z轴固定（悬停）
            # 4. 机身姿态保持不变（原代码逻辑：只有位置公转，机身不自旋）
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 姿态固定为初始值

            # 5. 旋翼旋转（完全匹配原代码逻辑，无闪烁）
            rotor_speed = self.rotor_visual_speed
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep * (i % 2 * 2 - 1)

            if viewer:
                viewer.sync()

            # 打印原代码风格的状态信息
            if time.time() - last_print_time > 1.0:
                current_time = self.data.time
                current_pos = self.data.qpos[0:3].copy()
                print(f"\n时间: {current_time:.1f}s | 公转角度: {self.rotate_angle:.2f}rad")
                print(f"当前位置: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] m")
                print(f"公转半径: {self.rotate_radius}m | 旋转速度: {self.rotate_speed}rad/s")
                last_print_time = time.time()

            # 控制仿真速率（原代码逻辑）
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真：完全匹配原代码的执行流程"""
        print(f"\n▶ 开始仿真（完全匹配原代码旋转方式），时长: {duration}秒")
        print(f"▶ 公转半径: {self.rotate_radius}m | 旋转速度: {self.rotate_speed}rad/s")
        print(f"▶ 悬停高度: {self.hover_height}m | 旋翼速度: {self.rotor_visual_speed}rad/s")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 原代码默认相机视角
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 15
                    viewer.cam.distance = 7.0
                    viewer.cam.lookat[:] = [0.0, 0.0, self.hover_height]
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")

        print("\n✅ 仿真结束（旋转方式与原代码完全一致）")


def main():
    print("🚁 MuJoCo 四旋翼无人机仿真 - 完全匹配原代码旋转方式")
    print("=" * 50)

    try:
        sim = QuadrotorSimulation()

        # ========== 可微调原代码参数（如需） ==========
        sim.rotate_radius = 1.0   # 公转半径（原代码核心，默认1.0m）
        sim.rotate_speed = 1.0    # 公转速度（原代码默认1.0rad/s）
        sim.hover_height = 0.8    # 悬停高度（原代码默认0.8m）

        print("✅ 初始化完成（参数与原代码100%对齐）")
        sim.run_simulation(
            duration=60.0,
            use_viewer=True
        )

    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()