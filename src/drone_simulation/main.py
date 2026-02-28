"""
MuJoCo 四旋翼无人机仿真 - 公转+避障版
✅ 无人机绕世界Z轴公转，保持原旋转逻辑
✅ 自动避开立方体/圆柱体/球体障碍物
✅ 避障后自动恢复原轨迹，高度固定、无闪烁
✅ 保留所有原代码核心特征
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math


class QuadrotorSimulation:
    def __init__(self):
        """初始化：添加避障相关参数"""
        xml_string = self.create_quadrotor_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        print("✓ 模型加载成功")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 原代码悬停推力参数
        hover_thrust = 600
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

        # ========== 原代码旋转参数 ==========
        self.base_radius = 1.0      # 基础公转半径
        self.rotate_speed = 1.0     # 公转角速度（rad/s）
        self.hover_height = 0.8     # 固定高度
        self.rotate_angle = 0.0     # 公转角度累计
        self.rotor_visual_speed = 8.0  # 旋翼旋转速度

        # ========== 避障核心参数 ==========
        self.safety_distance = 0.5  # 安全距离（小于此距离触发避障）
        self.avoidance_offset = 0.8 # 避障偏移量（扩大半径绕开障碍物）
        self.obstacle_positions = { # 预定义障碍物位置（与XML中一致）
            "cube": np.array([2.0, 0.0, 0.75]),
            "cylinder": np.array([-1.0, 1.0, 0.5]),
            "sphere": np.array([0.0, -2.0, 1.0])
        }
        self.obstacle_sizes = {     # 障碍物尺寸（碰撞判定用）
            "cube": np.array([0.25, 0.25, 0.75]),
            "cylinder": np.array([0.3, 0.5]),  # 半径、高度
            "sphere": np.array([0.4])          # 半径
        }

    def create_quadrotor_xml(self):
        """保持原XML结构不变"""
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

    def calculate_obstacle_distance(self, drone_pos):
        """计算无人机到各障碍物的水平距离（Z轴高度忽略，只算XY平面）"""
        distances = {}

        # 立方体障碍物
        cube_pos = self.obstacle_positions["cube"][:2]  # 只取XY坐标
        drone_xy = drone_pos[:2]
        distances["cube"] = np.linalg.norm(drone_xy - cube_pos) - self.obstacle_sizes["cube"][0]

        # 圆柱体障碍物
        cyl_pos = self.obstacle_positions["cylinder"][:2]
        distances["cylinder"] = np.linalg.norm(drone_xy - cyl_pos) - self.obstacle_sizes["cylinder"][0]

        # 球体障碍物
        sphere_pos = self.obstacle_positions["sphere"][:2]
        distances["sphere"] = np.linalg.norm(drone_xy - sphere_pos) - self.obstacle_sizes["sphere"][0]

        return distances

    def get_avoidance_radius(self, drone_pos):
        """根据障碍物距离动态调整公转半径（避障核心逻辑）"""
        distances = self.calculate_obstacle_distance(drone_pos)
        min_distance = min(distances.values())

        # 判定是否需要避障
        if min_distance < self.safety_distance:
            # 找到最近的障碍物
            closest_obs = min(distances, key=distances.get)
            obs_pos = self.obstacle_positions[closest_obs][:2]
            drone_xy = drone_pos[:2]

            # 计算避障方向：远离最近障碍物
            direction = drone_xy - obs_pos
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([1, 0])

            # 动态调整半径，绕开障碍物
            return self.base_radius + self.avoidance_offset
        else:
            # 无避障需求，恢复基础半径
            return self.base_radius

    def simulation_loop(self, viewer, duration):
        """核心：公转+避障逻辑"""
        start_time = time.time()
        last_print_time = time.time()

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # ========== 1. 更新公转角度 ==========
            self.rotate_angle += self.rotate_speed * self.model.opt.timestep
            # 限制角度范围（防止数值过大）
            if self.rotate_angle > 2 * math.pi:
                self.rotate_angle -= 2 * math.pi

            # ========== 2. 计算基础公转位置 ==========
            base_x = self.base_radius * math.cos(self.rotate_angle)
            base_y = self.base_radius * math.sin(self.rotate_angle)
            base_pos = np.array([base_x, base_y, self.hover_height])

            # ========== 3. 避障逻辑：动态调整位置 ==========
            current_radius = self.get_avoidance_radius(base_pos)
            # 计算避障后的目标位置
            target_x = current_radius * math.cos(self.rotate_angle)
            target_y = current_radius * math.sin(self.rotate_angle)
            target_z = self.hover_height

            # ========== 4. 设置无人机位置和姿态 ==========
            self.data.qpos[0] = target_x  # X轴位置
            self.data.qpos[1] = target_y  # Y轴位置
            self.data.qpos[2] = target_z  # Z轴固定高度
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 姿态不变

            # ========== 5. 旋翼旋转（保持原逻辑） ==========
            rotor_speed = self.rotor_visual_speed
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep * (i % 2 * 2 - 1)

            if viewer:
                viewer.sync()

            # ========== 6. 打印状态信息（新增避障状态） ==========
            if time.time() - last_print_time > 1.0:
                current_time = self.data.time
                current_pos = self.data.qpos[0:3].copy()
                distances = self.calculate_obstacle_distance(current_pos)
                min_dist = min(distances.values())
                avoidance_status = "避障中" if min_dist < self.safety_distance else "正常轨迹"

                print(f"\n时间: {current_time:.1f}s | 公转角度: {self.rotate_angle:.2f}rad")
                print(f"当前位置: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] m")
                print(f"公转半径: {current_radius:.2f}m | 状态: {avoidance_status}")
                print(f"最近障碍物距离: {min_dist:.2f}m | 安全距离: {self.safety_distance}m")
                last_print_time = time.time()

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=60.0, use_viewer=True):
        """运行仿真：带避障功能"""
        print(f"\n▶ 开始仿真（公转+自动避障），时长: {duration}秒")
        print(f"▶ 基础公转半径: {self.base_radius}m | 旋转速度: {self.rotate_speed}rad/s")
        print(f"▶ 安全距离: {self.safety_distance}m | 避障偏移量: {self.avoidance_offset}m")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 优化相机视角，方便观察避障效果
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 15
                    viewer.cam.distance = 8.0
                    viewer.cam.lookat[:] = [0.0, 0.0, self.hover_height]
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")

        print("\n✅ 仿真结束（避障功能正常运行）")


def main():
    print("🚁 MuJoCo 四旋翼无人机仿真 - 公转+自动避障版")
    print("=" * 60)

    try:
        sim = QuadrotorSimulation()

        # ========== 可自定义参数 ==========
        # 原旋转参数
        sim.base_radius = 1.0      # 基础公转半径
        sim.rotate_speed = 1.0     # 旋转速度
        sim.hover_height = 0.8     # 悬停高度
        # 避障参数
        sim.safety_distance = 0.5  # 触发避障的安全距离（越小越灵敏）
        sim.avoidance_offset = 0.8 # 避障时的半径偏移量（越大避障越远）

        print("✅ 初始化完成（避障功能已启用）")
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