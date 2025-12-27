"""
MuJoCo 四旋翼无人机仿真 - 默认设置版本
直接运行，无需用户选择
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math


class QuadrotorSimulation:
    def __init__(self):
        """初始化四旋翼无人机仿真"""
        # 使用简化的XML字符串，避免纹理问题
        xml_string = self.create_minimal_quadrotor_xml()

        # 从XML字符串加载模型
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        print("✓ 模型加载成功")

        # 创建仿真数据
        self.data = mujoco.MjData(self.model)

        # 获取执行器数量
        self.n_actuators = self.model.nu
        print(f"✓ 执行器数量: {self.n_actuators}")

        # 设置初始控制输入
        self.set_initial_control()

    def create_minimal_quadrotor_xml(self):
        """创建最简单的四旋翼无人机XML配置"""
        xml_string = """<?xml version="1.0" ?>
<mujoco model="quadrotor">

  <!-- 仿真选项 -->
  <option timestep="0.005" iterations="50" tolerance="1e-10">
    <flag contact="enable" energy="enable"/>
  </option>

  <!-- 物理参数 -->
  <size nconmax="100" njmax="200"/>

  <!-- 资产定义 - 使用最简单的材质 -->
  <asset>
    <material name="ground_mat" rgba="0.8 0.9 0.8 1"/>
    <material name="body_mat" rgba="0.3 0.3 0.3 1"/>
    <material name="arm_mat" rgba="0.1 0.1 0.1 1"/>
    <material name="motor_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="propeller_red" rgba="0.8 0.2 0.2 0.8"/>
    <material name="propeller_green" rgba="0.2 0.8 0.2 0.8"/>
    <material name="target_mat" rgba="1 0 0 0.5"/>
  </asset>

  <!-- 世界定义 -->
  <worldbody>
    <!-- 光源 -->
    <light name="top_light" pos="0 0 10" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <light name="front_light" pos="5 0 5" dir="-1 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>

    <!-- 地面 -->
    <geom name="ground" type="plane" pos="0 0 0" size="20 20 0.1" material="ground_mat" condim="3" friction="1 0.005 0.0001"/>

    <!-- 参考坐标系 -->
    <geom name="origin_x" type="cylinder" fromto="0 0 0.1 1 0 0.1" size="0.01" rgba="1 0 0 1"/>
    <geom name="origin_y" type="cylinder" fromto="0 0 0.1 0 1 0.1" size="0.01" rgba="0 1 0 1"/>
    <geom name="origin_z" type="cylinder" fromto="0 0 0.1 0 0 1.1" size="0.01" rgba="0 0 1 1"/>

    <!-- 四旋翼无人机主体 -->
    <body name="quadrotor" pos="0 0 1.5" euler="0 0 0">
      <!-- 自由关节 (6自由度) -->
      <freejoint name="quad_free_joint"/>

      <!-- 主体框架 -->
      <geom name="center_body" type="cylinder" size="0.1 0.02" material="body_mat" mass="0.5"/>

      <!-- 机臂 -->
      <geom name="arm_front_right" type="capsule" fromto="0 0 0 0.25 0.25 0" size="0.008" material="arm_mat" mass="0.05"/>
      <geom name="arm_front_left" type="capsule" fromto="0 0 0 0.25 -0.25 0" size="0.008" material="arm_mat" mass="0.05"/>
      <geom name="arm_back_left" type="capsule" fromto="0 0 0 -0.25 -0.25 0" size="0.008" material="arm_mat" mass="0.05"/>
      <geom name="arm_back_right" type="capsule" fromto="0 0 0 -0.25 0.25 0" size="0.008" material="arm_mat" mass="0.05"/>

      <!-- 电机和旋翼 (前右) -->
      <body name="motor_front_right" pos="0.25 0.25 0">
        <geom name="motor_housing_front_right" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>

        <body name="rotor_front_right" pos="0 0 0.05">
          <joint name="rotor_front_right_joint" type="hinge" axis="0 0 1"/>
          <geom name="propeller_front_right" type="cylinder" size="0.12 0.005" material="propeller_red" mass="0.02"/>
        </body>
      </body>

      <!-- 电机和旋翼 (前左) -->
      <body name="motor_front_left" pos="0.25 -0.25 0">
        <geom name="motor_housing_front_left" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>

        <body name="rotor_front_left" pos="0 0 0.05">
          <joint name="rotor_front_left_joint" type="hinge" axis="0 0 1"/>
          <geom name="propeller_front_left" type="cylinder" size="0.12 0.005" material="propeller_green" mass="0.02"/>
        </body>
      </body>

      <!-- 电机和旋翼 (后左) -->
      <body name="motor_back_left" pos="-0.25 -0.25 0">
        <geom name="motor_housing_back_left" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>

        <body name="rotor_back_left" pos="0 0 0.05">
          <joint name="rotor_back_left_joint" type="hinge" axis="0 0 1"/>
          <geom name="propeller_back_left" type="cylinder" size="0.12 0.005" material="propeller_red" mass="0.02"/>
        </body>
      </body>

      <!-- 电机和旋翼 (后右) -->
      <body name="motor_back_right" pos="-0.25 0.25 0">
        <geom name="motor_housing_back_right" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>

        <body name="rotor_back_right" pos="0 0 0.05">
          <joint name="rotor_back_right_joint" type="hinge" axis="0 0 1"/>
          <geom name="propeller_back_right" type="cylinder" size="0.12 0.005" material="propeller_green" mass="0.02"/>
        </body>
      </body>

      <!-- 起落架 -->
      <geom name="landing_gear_front" type="cylinder" pos="0.15 0 0" size="0.005 0.05" rgba="0.5 0.5 0.5 1" mass="0.01"/>
      <geom name="landing_gear_back" type="cylinder" pos="-0.15 0 0" size="0.005 0.05" rgba="0.5 0.5 0.5 1" mass="0.01"/>

      <!-- 视觉标记 -->
      <geom name="front_marker" type="sphere" pos="0.15 0 0.02" size="0.015" rgba="1 1 0 1"/>
      <geom name="rear_marker" type="sphere" pos="-0.15 0 0.02" size="0.015" rgba="0 1 1 1"/>
    </body>

    <!-- 目标点 -->
    <body name="target" pos="0 3 2">
      <geom name="target_sphere" type="sphere" size="0.1" material="target_mat" contype="0" conaffinity="0"/>
    </body>

  </worldbody>

  <!-- 执行器定义 -->
  <actuator>
    <!-- 电机控制 -->
    <motor name="motor_front_right" joint="rotor_front_right_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
    <motor name="motor_front_left" joint="rotor_front_left_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
    <motor name="motor_back_left" joint="rotor_back_left_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
    <motor name="motor_back_right" joint="rotor_back_right_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
  </actuator>

</mujoco>"""
        return xml_string

    def set_initial_control(self):
        """设置初始控制输入"""
        # 设置初始推力
        hover_thrust = 500  # 悬停推力值
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

    def get_state(self):
        """获取无人机状态"""
        state = {
            'position': self.data.qpos[0:3].copy(),
            'orientation': self.data.qpos[3:7].copy(),
            'linear_velocity': self.data.qvel[0:3].copy(),
            'angular_velocity': self.data.qvel[3:6].copy(),
            'rotor_angles': self.data.qpos[7:11].copy(),
            'rotor_velocities': self.data.qvel[6:10].copy()
        }
        return state

    def print_state(self):
        """打印无人机状态"""
        state = self.get_state()

        print("\n" + "=" * 50)
        print("四旋翼无人机状态:")
        print("=" * 50)
        print(f"位置: [{state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f}] m")
        print(f"姿态四元数: [{state['orientation'][0]:.3f}, {state['orientation'][1]:.3f}, "
              f"{state['orientation'][2]:.3f}, {state['orientation'][3]:.3f}]")
        print(f"线速度: [{state['linear_velocity'][0]:.3f}, {state['linear_velocity'][1]:.3f}, "
              f"{state['linear_velocity'][2]:.3f}] m/s")
        print(f"角速度: [{state['angular_velocity'][0]:.3f}, {state['angular_velocity'][1]:.3f}, "
              f"{state['angular_velocity'][2]:.3f}] rad/s")
        print("=" * 50)

    def apply_control(self, ctrl_values):
        """应用控制输入"""
        if len(ctrl_values) != self.n_actuators:
            print(f"⚠ 警告：控制值数量应为{self.n_actuators}，使用默认值500")
            ctrl_values = [500] * self.n_actuators

        # 应用控制值
        self.data.ctrl[:] = ctrl_values

    def altitude_controller(self, target_z=1.5):
        """高度控制器"""
        # PID参数
        Kp = 200.0  # 比例增益
        Kd = 50.0  # 微分增益

        # 获取当前状态
        current_z = self.data.qpos[2]
        current_vz = self.data.qvel[2]

        # 计算误差
        error_z = target_z - current_z
        error_vz = 0 - current_vz

        # PID控制
        control_input = Kp * error_z + Kd * error_vz

        # 基础推力
        base_thrust = 500

        # 计算推力
        thrust = base_thrust + control_input

        # 限制推力范围
        thrust = np.clip(thrust, 400, 600)

        # 应用到所有电机
        ctrl_values = [thrust] * self.n_actuators
        self.apply_control(ctrl_values)

        return error_z, thrust

    def position_controller(self, target_pos=[0, 0, 1.5]):
        """位置控制器"""
        # PID参数
        Kp_pos = np.array([100.0, 100.0, 200.0])
        Kd_pos = np.array([30.0, 30.0, 50.0])

        # 获取当前状态
        current_pos = self.data.qpos[0:3]
        current_vel = self.data.qvel[0:3]

        # 计算误差
        pos_error = np.array(target_pos) - current_pos
        vel_error = -current_vel

        # 位置控制
        pos_control = Kp_pos * pos_error + Kd_pos * vel_error

        # 基础推力
        base_thrust = 500

        # 总推力
        total_thrust = base_thrust + pos_control[2]

        # 姿态控制
        roll_control = -pos_control[1] * 0.02
        pitch_control = pos_control[0] * 0.02

        # 四旋翼混控
        ctrl_values = [
            total_thrust - pitch_control - roll_control,  # 前右
            total_thrust - pitch_control + roll_control,  # 前左
            total_thrust + pitch_control + roll_control,  # 后左
            total_thrust + pitch_control - roll_control  # 后右
        ]

        # 限制推力范围
        ctrl_values = np.clip(ctrl_values, 400, 600)

        self.apply_control(ctrl_values)

        return pos_error, ctrl_values

    def run_simulation(self, duration=10.0, use_viewer=True, controller_type="altitude"):
        """运行仿真"""
        print(f"\n▶ 开始仿真，时长: {duration}秒")
        print(f"▶ 控制器类型: {controller_type}")

        if use_viewer:
            print("▶ 使用可视化查看器 (按ESC退出)")
        else:
            print("▶ 无可视化模式")

        # 记录数据
        time_history = []
        height_history = []
        thrust_history = []

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 设置相机
                    viewer.cam.azimuth = 180
                    viewer.cam.elevation = -20
                    viewer.cam.distance = 5.0
                    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

                    self.simulation_loop(viewer, duration, controller_type,
                                         time_history, height_history, thrust_history)
            else:
                self.simulation_loop(None, duration, controller_type,
                                     time_history, height_history, thrust_history)

        except Exception as e:
            print(f"⚠ 仿真错误: {e}")

        # 分析数据
        if time_history:
            self.analyze_data(time_history, height_history, thrust_history)

    def simulation_loop(self, viewer, duration, controller_type,
                        time_history, height_history, thrust_history):
        """仿真循环"""
        start_time = time.time()
        last_print_time = time.time()
        step_count = 0

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()
            step_count += 1

            # 应用控制器
            if controller_type == "position":
                # 移动目标点
                t = self.data.time
                target_x = 1.0 * math.sin(t * 0.5)
                target_y = 1.0 * math.cos(t * 0.5)
                target_z = 1.5 + 0.3 * math.sin(t * 0.3)

                pos_error, thrusts = self.position_controller([target_x, target_y, target_z])
                control_info = f"位置误差: [{pos_error[0]:.2f}, {pos_error[1]:.2f}, {pos_error[2]:.2f}] m"
            else:
                error_z, thrust = self.altitude_controller(1.5)
                thrusts = [thrust] * 4
                control_info = f"高度误差: {error_z:.2f} m"

            # 记录数据
            current_time = self.data.time
            current_height = self.data.qpos[2]
            time_history.append(current_time)
            height_history.append(current_height)
            thrust_history.append(np.mean(thrusts))

            # 执行仿真步
            mujoco.mj_step(self.model, self.data)

            # 更新螺旋桨旋转（视觉效果）
            rotor_speed = 80.0
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep

            # 更新查看器
            if viewer:
                viewer.sync()

            # 打印状态信息
            if time.time() - last_print_time > 1.0:
                print(f"\n时间: {current_time:.1f}s | 高度: {current_height:.2f}m")
                print(f"推力: {np.mean(thrusts):.0f} | {control_info}")
                print(f"步数: {step_count}")
                last_print_time = time.time()

            # 控制仿真速度
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def analyze_data(self, time_data, height_data, thrust_data):
        """分析仿真数据"""
        print("\n" + "=" * 50)
        print("📊 仿真数据分析:")
        print("=" * 50)

        if not time_data:
            print("无数据")
            return

        time_array = np.array(time_data)
        height_array = np.array(height_data)
        thrust_array = np.array(thrust_data)

        print(f"总步数: {len(time_array)}")
        print(f"仿真时长: {time_array[-1]:.2f} 秒")
        print(f"平均高度: {np.mean(height_array):.3f} m")
        print(f"高度稳定性: ±{np.std(height_array):.3f} m")
        print(f"高度范围: [{np.min(height_array):.3f}, {np.max(height_array):.3f}] m")
        print(f"平均推力: {np.mean(thrust_array):.0f}")
        print(f"推力范围: [{np.min(thrust_array):.0f}, {np.max(thrust_array):.0f}]")

        # 询问是否绘图
        try:
            plot = input("\n是否绘制图表? (y/n): ").strip().lower()
            if plot == 'y':
                self.plot_results(time_array, height_array, thrust_array)
        except:
            pass

    def plot_results(self, time_data, height_data, thrust_data):
        """绘制结果图表"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 高度图
            ax1.plot(time_data, height_data, 'b-', linewidth=2, label='实际高度')
            ax1.axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='目标高度')
            ax1.fill_between(time_data, 1.45, 1.55, color='r', alpha=0.1)
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('高度 (米)')
            ax1.set_title('四旋翼无人机高度控制')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 推力图
            ax2.plot(time_data, thrust_data, 'g-', linewidth=2, label='平均推力')
            ax2.axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='悬停推力')
            ax2.set_xlabel('时间 (秒)')
            ax2.set_ylabel('推力')
            ax2.set_title('电机推力变化')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("⚠ 需要安装matplotlib: pip install matplotlib")
        except Exception as e:
            print(f"⚠ 绘图错误: {e}")


def main():
    """主函数 - 使用默认设置"""
    print("🚁 MuJoCo 四旋翼无人机仿真系统")
    print("=" * 50)

    try:
        # 创建仿真实例
        print("正在初始化...")
        sim = QuadrotorSimulation()
        print("✅ 初始化完成")

        # 使用默认设置
        controller_type = "position"  # 默认使用位置控制器
        duration = 15.0  # 默认仿真15秒
        use_viewer = True  # 默认使用可视化

        print(f"\n📋 默认设置:")
        print(f"  控制器类型: {controller_type}")
        print(f"  仿真时长: {duration}秒")
        print(f"  可视化: {'是' if use_viewer else '否'}")

        # 运行仿真
        sim.run_simulation(
            duration=duration,
            use_viewer=use_viewer,
            controller_type=controller_type
        )

    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 直接运行，无需用户输入
    main()