"""
MuJoCo 四旋翼无人机仿真 - 默认设置版本
直接运行，无需用户选择

核心功能说明：
1. 基于MuJoCo物理引擎构建四旋翼无人机仿真环境
2. 实现两种控制器：高度控制器(PID)和位置控制器(PID)
3. 支持可视化仿真和数据记录分析
4. 包含完整的物理参数定义和控制逻辑

环境依赖：
- mujoco: 物理仿真引擎
- numpy: 数值计算
- matplotlib: 可选，用于数据可视化
- math/time: 基础工具库
"""

import mujoco              # MuJoCo物理仿真引擎核心库
import mujoco.viewer       # MuJoCo可视化查看器
import numpy as np         # 数值计算库，用于矩阵/数组操作
import time                # 时间控制库，用于仿真时序管理
import math                # 数学库，用于三角函数等计算


class QuadrotorSimulation:
    """
    四旋翼无人机仿真类
    封装了仿真环境初始化、控制器实现、仿真运行和数据分析的完整流程
    """

    def __init__(self):
        """初始化四旋翼无人机仿真环境

        核心步骤：
        1. 创建简化的四旋翼XML配置（避免外部文件依赖和纹理问题）
        2. 加载MuJoCo模型和仿真数据结构
        3. 初始化控制输入参数
        """
        # 使用简化的XML字符串定义四旋翼模型，避免外部文件依赖
        xml_string = self.create_minimal_quadrotor_xml()

        # 从XML字符串加载MuJoCo模型（核心数据结构，存储仿真的物理参数）
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        print("✓ 模型加载成功")

        # 创建仿真数据实例（存储仿真过程中的动态数据：位置、速度、力等）
        self.data = mujoco.MjData(self.model)

        # 获取执行器数量（四旋翼有4个电机执行器）
        self.n_actuators = self.model.nu
        print(f"✓ 执行器数量: {self.n_actuators}")

        # 设置初始控制输入（所有电机初始推力）
        self.set_initial_control()

    def create_minimal_quadrotor_xml(self):
        """
        创建最简版四旋翼无人机XML配置字符串
        XML是MuJoCo的模型定义格式，包含：
        - 仿真参数（时间步、迭代次数等）
        - 物理属性（接触参数、关节限制等）
        - 视觉资产（材质、颜色等）
        - 世界体（地面、光源、参考坐标系）
        - 四旋翼主体（机身、机臂、电机、旋翼）
        - 执行器（电机控制接口）
        """
        xml_string = """<?xml version="1.0" ?>
<mujoco model="quadrotor">

  <!-- 仿真选项配置 -->
  <option timestep="0.005" iterations="50" tolerance="1e-10">
    <flag contact="enable" energy="enable"/>  <!-- 启用接触检测和能量计算 -->
  </option>

  <!-- 物理参数配置 -->
  <size nconmax="100" njmax="200"/>  <!-- 最大接触数和关节数限制 -->

  <!-- 资产定义 - 定义材质和颜色（避免纹理文件依赖） -->
  <asset>
    <material name="ground_mat" rgba="0.8 0.9 0.8 1"/>    <!-- 地面材质（浅绿色） -->
    <material name="body_mat" rgba="0.3 0.3 0.3 1"/>      <!-- 机身材质（深灰色） -->
    <material name="arm_mat" rgba="0.1 0.1 0.1 1"/>       <!-- 机臂材质（黑色） -->
    <material name="motor_mat" rgba="0.2 0.2 0.2 1"/>      <!-- 电机材质（深灰色） -->
    <material name="propeller_red" rgba="0.8 0.2 0.2 0.8"/>  <!-- 红色旋翼（半透明） -->
    <material name="propeller_green" rgba="0.2 0.8 0.2 0.8"/> <!-- 绿色旋翼（半透明） -->
    <material name="target_mat" rgba="1 0 0 0.5"/>         <!-- 目标点材质（红色半透明） -->
  </asset>

  <!-- 世界体定义 -->
  <worldbody>
    <!-- 光源配置 - 提供可视化照明 -->
    <light name="top_light" pos="0 0 10" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <light name="front_light" pos="5 0 5" dir="-1 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>

    <!-- 地面 - 无限平面，提供支撑 -->
    <geom name="ground" type="plane" pos="0 0 0" size="20 20 0.1" material="ground_mat" condim="3" friction="1 0.005 0.0001"/>

    <!-- 参考坐标系 - X(红)/Y(绿)/Z(蓝)轴，方便观察无人机姿态 -->
    <geom name="origin_x" type="cylinder" fromto="0 0 0.1 1 0 0.1" size="0.01" rgba="1 0 0 1"/>
    <geom name="origin_y" type="cylinder" fromto="0 0 0.1 0 1 0.1" size="0.01" rgba="0 1 0 1"/>
    <geom name="origin_z" type="cylinder" fromto="0 0 0.1 0 0 1.1" size="0.01" rgba="0 0 1 1"/>

    <!-- 四旋翼无人机主体 - 初始位置(0,0,1.5)，初始姿态(0,0,0) -->
    <body name="quadrotor" pos="0 0 1.5" euler="0 0 0">
      <!-- 自由关节 - 6自由度（3平移+3旋转），无人机核心运动关节 -->
      <freejoint name="quad_free_joint"/>

      <!-- 主体框架 - 圆柱形机身 -->
      <geom name="center_body" type="cylinder" size="0.1 0.02" material="body_mat" mass="0.5"/>

      <!-- 机臂 - 四个胶囊形机臂，连接机身和电机 -->
      <geom name="arm_front_right" type="capsule" fromto="0 0 0 0.25 0.25 0" size="0.008" material="arm_mat" mass="0.05"/>
      <geom name="arm_front_left" type="capsule" fromto="0 0 0 0.25 -0.25 0" size="0.008" material="arm_mat" mass="0.05"/>
      <geom name="arm_back_left" type="capsule" fromto="0 0 0 -0.25 -0.25 0" size="0.008" material="arm_mat" mass="0.05"/>
      <geom name="arm_back_right" type="capsule" fromto="0 0 0 -0.25 0.25 0" size="0.008" material="arm_mat" mass="0.05"/>

      <!-- 电机和旋翼 - 前右电机/旋翼组件 -->
      <body name="motor_front_right" pos="0.25 0.25 0">
        <geom name="motor_housing_front_right" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>
        <body name="rotor_front_right" pos="0 0 0.05">
          <joint name="rotor_front_right_joint" type="hinge" axis="0 0 1"/>  <!-- 旋翼旋转关节（Z轴） -->
          <geom name="propeller_front_right" type="cylinder" size="0.12 0.005" material="propeller_red" mass="0.02"/>
        </body>
      </body>

      <!-- 电机和旋翼 - 前左电机/旋翼组件 -->
      <body name="motor_front_left" pos="0.25 -0.25 0">
        <geom name="motor_housing_front_left" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>
        <body name="rotor_front_left" pos="0 0 0.05">
          <joint name="rotor_front_left_joint" type="hinge" axis="0 0 1"/>
          <geom name="propeller_front_left" type="cylinder" size="0.12 0.005" material="propeller_green" mass="0.02"/>
        </body>
      </body>

      <!-- 电机和旋翼 - 后左电机/旋翼组件 -->
      <body name="motor_back_left" pos="-0.25 -0.25 0">
        <geom name="motor_housing_back_left" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>
        <body name="rotor_back_left" pos="0 0 0.05">
          <joint name="rotor_back_left_joint" type="hinge" axis="0 0 1"/>
          <geom name="propeller_back_left" type="cylinder" size="0.12 0.005" material="propeller_red" mass="0.02"/>
        </body>
      </body>

      <!-- 电机和旋翼 - 后右电机/旋翼组件 -->
      <body name="motor_back_right" pos="-0.25 0.25 0">
        <geom name="motor_housing_back_right" type="cylinder" size="0.025 0.03" material="motor_mat" mass="0.05"/>
        <body name="rotor_back_right" pos="0 0 0.05">
          <joint name="rotor_back_right_joint" type="hinge" axis="0 0 1"/>
          <geom name="propeller_back_right" type="cylinder" size="0.12 0.005" material="propeller_green" mass="0.02"/>
        </body>
      </body>

      <!-- 起落架 - 简单圆柱结构，防止无人机倾倒 -->
      <geom name="landing_gear_front" type="cylinder" pos="0.15 0 0" size="0.005 0.05" rgba="0.5 0.5 0.5 1" mass="0.01"/>
      <geom name="landing_gear_back" type="cylinder" pos="-0.15 0 0" size="0.005 0.05" rgba="0.5 0.5 0.5 1" mass="0.01"/>

      <!-- 视觉标记 - 前后方向标记，方便观察无人机朝向 -->
      <geom name="front_marker" type="sphere" pos="0.15 0 0.02" size="0.015" rgba="1 1 0 1"/>
      <geom name="rear_marker" type="sphere" pos="-0.15 0 0.02" size="0.015" rgba="0 1 1 1"/>
    </body>

    <!-- 目标点 - 红色半透明球体，用于位置控制参考 -->
    <body name="target" pos="0 3 2">
      <geom name="target_sphere" type="sphere" size="0.1" material="target_mat" contype="0" conaffinity="0"/>
    </body>

  </worldbody>

  <!-- 执行器定义 - 电机控制接口 -->
  <actuator>
    <!-- 每个电机对应一个执行器，控制旋翼旋转速度 -->
    <motor name="motor_front_right" joint="rotor_front_right_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
    <motor name="motor_front_left" joint="rotor_front_left_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
    <motor name="motor_back_left" joint="rotor_back_left_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
    <motor name="motor_back_right" joint="rotor_back_right_joint" gear="50" ctrllimited="true" ctrlrange="0 800"/>
  </actuator>

</mujoco>"""
        return xml_string

    def set_initial_control(self):
        """
        设置初始控制输入
        MuJoCo通过data.ctrl数组控制执行器，这里为所有电机设置初始悬停推力
        """
        # 悬停推力值（经验值，使无人机保持悬停的基础推力）
        hover_thrust = 500
        # 将初始推力应用到所有执行器（4个电机）
        self.data.ctrl[:] = [hover_thrust] * self.n_actuators

    def get_state(self):
        """
        获取无人机完整状态信息
        返回字典格式，包含：
        - position: 位置坐标 (x,y,z)
        - orientation: 姿态四元数 (w,x,y,z)
        - linear_velocity: 线速度 (vx,vy,vz)
        - angular_velocity: 角速度 (wx,wy,wz)
        - rotor_angles: 旋翼旋转角度
        - rotor_velocities: 旋翼旋转速度

        Returns:
            dict: 无人机状态字典
        """
        state = {
            # qpos[0:3] 存储自由关节的位置坐标
            'position': self.data.qpos[0:3].copy(),
            # qpos[3:7] 存储自由关节的姿态四元数
            'orientation': self.data.qpos[3:7].copy(),
            # qvel[0:3] 存储自由关节的线速度
            'linear_velocity': self.data.qvel[0:3].copy(),
            # qvel[3:6] 存储自由关节的角速度
            'angular_velocity': self.data.qvel[3:6].copy(),
            # qpos[7:11] 存储四个旋翼的旋转角度
            'rotor_angles': self.data.qpos[7:11].copy(),
            # qvel[6:10] 存储四个旋翼的旋转速度
            'rotor_velocities': self.data.qvel[6:10].copy()
        }
        return state

    def print_state(self):
        """打印无人机状态信息（调试用）"""
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
        """
        应用控制输入到执行器
        安全检查控制值数量是否匹配执行器数量，并应用到data.ctrl数组

        Args:
            ctrl_values (list): 每个电机的控制值列表（长度应等于执行器数量）
        """
        # 安全检查：控制值数量必须匹配执行器数量
        if len(ctrl_values) != self.n_actuators:
            print(f"⚠ 警告：控制值数量应为{self.n_actuators}，使用默认值500")
            ctrl_values = [500] * self.n_actuators

        # 将控制值应用到MuJoCo的控制数组（核心操作，控制执行器输出）
        self.data.ctrl[:] = ctrl_values

    def altitude_controller(self, target_z=1.5):
        """
        高度控制器（PD控制器，简化版PID）
        通过调整电机总推力，使无人机保持在目标高度

        控制原理：
        1. 计算当前高度与目标高度的误差
        2. 计算当前垂直速度（微分项）
        3. PD控制律：控制输入 = Kp*位置误差 + Kd*速度误差
        4. 限制推力范围，防止执行器饱和

        Args:
            target_z (float): 目标高度（默认1.5米）

        Returns:
            tuple: (高度误差, 最终推力值)
        """
        # PID参数（仅使用P和D项，I项容易积分饱和）
        Kp = 200.0  # 比例增益 - 直接响应位置误差
        Kd = 50.0   # 微分增益 - 抑制速度，减少超调

        # 获取当前状态
        current_z = self.data.qpos[2]    # 当前高度
        current_vz = self.data.qvel[2]    # 当前垂直速度

        # 计算误差
        error_z = target_z - current_z    # 位置误差（目标-当前）
        error_vz = 0 - current_vz         # 速度误差（期望速度为0）

        # PD控制律计算控制输入
        control_input = Kp * error_z + Kd * error_vz

        # 基础悬停推力
        base_thrust = 500

        # 总推力 = 基础推力 + 控制输入
        thrust = base_thrust + control_input

        # 限制推力范围（防止超出执行器控制范围）
        thrust = np.clip(thrust, 400, 600)

        # 将相同推力应用到所有电机（仅控制高度，不控制水平位置）
        ctrl_values = [thrust] * self.n_actuators
        self.apply_control(ctrl_values)

        return error_z, thrust

    def position_controller(self, target_pos=[0, 0, 1.5]):
        """
        位置控制器（PID位置+姿态混控）
        同时控制无人机的x/y/z位置，通过调整四个电机的推力差实现

        控制原理：
        1. 位置PID控制：计算x/y/z方向的控制输入
        2. 姿态转换：将水平位置误差转换为滚转/俯仰指令
        3. 四旋翼混控：将总推力和姿态指令分配到四个电机
        4. 限制推力范围，确保执行器安全

        Args:
            target_pos (list): 目标位置 [x,y,z]（默认[0,0,1.5]）

        Returns:
            tuple: (位置误差数组, 四个电机的控制值列表)
        """
        # PID参数（位置比例/微分增益）
        Kp_pos = np.array([100.0, 100.0, 200.0])  # x/y/z轴比例增益
        Kd_pos = np.array([30.0, 30.0, 50.0])     # x/y/z轴微分增益

        # 获取当前状态
        current_pos = self.data.qpos[0:3]   # 当前位置 [x,y,z]
        current_vel = self.data.qvel[0:3]   # 当前速度 [vx,vy,vz]

        # 计算误差
        pos_error = np.array(target_pos) - current_pos  # 位置误差（目标-当前）
        vel_error = -current_vel                        # 速度误差（期望速度为0）

        # 位置PD控制律计算控制输入
        pos_control = Kp_pos * pos_error + Kd_pos * vel_error

        # 基础悬停推力
        base_thrust = 500

        # Z轴控制：总推力 = 基础推力 + z轴控制输入
        total_thrust = base_thrust + pos_control[2]

        # 姿态控制：将x/y位置误差转换为滚转/俯仰指令
        roll_control = -pos_control[1] * 0.02    # y误差→滚转（负号为了方向匹配）
        pitch_control = pos_control[0] * 0.02    # x误差→俯仰

        # 四旋翼混控矩阵（核心！将总推力和姿态指令分配到四个电机）
        # 前右/前左/后左/后右电机推力分配
        ctrl_values = [
            total_thrust - pitch_control - roll_control,  # 前右电机
            total_thrust - pitch_control + roll_control,  # 前左电机
            total_thrust + pitch_control + roll_control,  # 后左电机
            total_thrust + pitch_control - roll_control   # 后右电机
        ]

        # 限制每个电机的推力范围（防止执行器饱和）
        ctrl_values = np.clip(ctrl_values, 400, 600)

        # 应用控制值到执行器
        self.apply_control(ctrl_values)

        return pos_error, ctrl_values

    def run_simulation(self, duration=10.0, use_viewer=True, controller_type="altitude"):
        """
        运行完整的仿真流程
        包含可视化启动、仿真循环、数据记录和结果分析

        Args:
            duration (float): 仿真时长（秒），默认10秒
            use_viewer (bool): 是否启用可视化查看器，默认True
            controller_type (str): 控制器类型，"altitude"或"position"，默认"altitude"
        """
        print(f"\n▶ 开始仿真，时长: {duration}秒")
        print(f"▶ 控制器类型: {controller_type}")

        if use_viewer:
            print("▶ 使用可视化查看器 (按ESC退出)")
        else:
            print("▶ 无可视化模式")

        # 初始化数据记录列表
        time_history = []      # 时间序列
        height_history = []    # 高度序列
        thrust_history = []    # 推力序列

        try:
            # 启用可视化查看器
            if use_viewer:
                # 启动被动模式查看器（由用户代码控制仿真步）
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    # 初始化相机视角（方便观察无人机）
                    viewer.cam.azimuth = 180    # 方位角
                    viewer.cam.elevation = -20  # 仰角
                    viewer.cam.distance = 5.0   # 距离
                    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]  # 看向坐标

                    # 运行仿真循环
                    self.simulation_loop(viewer, duration, controller_type,
                                         time_history, height_history, thrust_history)
            else:
                # 无可视化模式运行仿真循环
                self.simulation_loop(None, duration, controller_type,
                                     time_history, height_history, thrust_history)

        except Exception as e:
            print(f"⚠ 仿真错误: {e}")

        # 仿真结束后分析数据
        if time_history:
            self.analyze_data(time_history, height_history, thrust_history)

    def simulation_loop(self, viewer, duration, controller_type,
                        time_history, height_history, thrust_history):
        """
        核心仿真循环
        每一步执行：
        1. 控制器计算
        2. 数据记录
        3. 仿真步推进
        4. 可视化更新
        5. 速度控制

        Args:
            viewer: MuJoCo查看器实例（None表示无可视化）
            duration: 仿真时长
            controller_type: 控制器类型
            time_history: 时间记录列表
            height_history: 高度记录列表
            thrust_history: 推力记录列表
        """
        start_time = time.time()         # 仿真开始时间（真实时间）
        last_print_time = time.time()    # 上次打印状态的时间
        step_count = 0                   # 仿真步数计数器

        # 仿真循环条件：查看器运行中 且 未达到仿真时长
        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()     # 记录当前步开始时间
            step_count += 1              # 步数+1

            # 根据控制器类型应用控制
            if controller_type == "position":
                # 动态目标点（正弦运动）
                t = self.data.time                          # 仿真时间（不是真实时间）
                target_x = 1.0 * math.sin(t * 0.5)          # x方向正弦运动
                target_y = 1.0 * math.cos(t * 0.5)          # y方向余弦运动
                target_z = 1.5 + 0.3 * math.sin(t * 0.3)    # z方向小幅波动

                # 应用位置控制器
                pos_error, thrusts = self.position_controller([target_x, target_y, target_z])
                control_info = f"位置误差: [{pos_error[0]:.2f}, {pos_error[1]:.2f}, {pos_error[2]:.2f}] m"
            else:
                # 应用高度控制器（固定目标高度1.5米）
                error_z, thrust = self.altitude_controller(1.5)
                thrusts = [thrust] * 4
                control_info = f"高度误差: {error_z:.2f} m"

            # 记录当前状态数据
            current_time = self.data.time
            current_height = self.data.qpos[2]
            time_history.append(current_time)
            height_history.append(current_height)
            thrust_history.append(np.mean(thrusts))

            # 执行MuJoCo仿真步（核心！推进物理仿真）
            mujoco.mj_step(self.model, self.data)

            # 更新螺旋桨旋转角度（纯视觉效果，不影响物理）
            rotor_speed = 80.0
            for i in range(4):
                self.data.qpos[7 + i] += rotor_speed * self.model.opt.timestep

            # 更新可视化查看器
            if viewer:
                viewer.sync()

            # 每秒打印一次状态信息（避免输出刷屏）
            if time.time() - last_print_time > 1.0:
                print(f"\n时间: {current_time:.1f}s | 高度: {current_height:.2f}m")
                print(f"推力: {np.mean(thrusts):.0f} | {control_info}")
                print(f"步数: {step_count}")
                last_print_time = time.time()

            # 控制仿真速度（使仿真步频匹配真实时间）
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def analyze_data(self, time_data, height_data, thrust_data):
        """
        分析仿真数据，输出关键统计信息
        包括：总步数、仿真时长、平均高度、高度稳定性、推力范围等

        Args:
            time_data: 时间序列数据
            height_data: 高度序列数据
            thrust_data: 推力序列数据
        """
        print("\n" + "=" * 50)
        print("📊 仿真数据分析:")
        print("=" * 50)

        if not time_data:
            print("无数据")
            return

        # 转换为numpy数组方便计算
        time_array = np.array(time_data)
        height_array = np.array(height_data)
        thrust_array = np.array(thrust_data)

        # 输出关键统计信息
        print(f"总步数: {len(time_array)}")
        print(f"仿真时长: {time_array[-1]:.2f} 秒")
        print(f"平均高度: {np.mean(height_array):.3f} m")
        print(f"高度稳定性: ±{np.std(height_array):.3f} m")  # 标准差越小越稳定
        print(f"高度范围: [{np.min(height_array):.3f}, {np.max(height_array):.3f}] m")
        print(f"平均推力: {np.mean(thrust_array):.0f}")
        print(f"推力范围: [{np.min(thrust_array):.0f}, {np.max(thrust_array):.0f}]")

        # 询问用户是否绘制结果图表
        try:
            plot = input("\n是否绘制图表? (y/n): ").strip().lower()
            if plot == 'y':
                self.plot_results(time_array, height_array, thrust_array)
        except:
            pass

    def plot_results(self, time_data, height_data, thrust_data):
        """
        绘制仿真结果图表
        包括高度随时间变化和推力随时间变化两个子图

        Args:
            time_data: 时间序列数据
            height_data: 高度序列数据
            thrust_data: 推力序列数据
        """
        try:
            # 延迟导入matplotlib（避免未安装时影响核心功能）
            import matplotlib.pyplot as plt

            # 创建2行1列的子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 高度变化图
            ax1.plot(time_data, height_data, 'b-', linewidth=2, label='实际高度')
            ax1.axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='目标高度')
            ax1.fill_between(time_data, 1.45, 1.55, color='r', alpha=0.1)  # 目标高度±0.05米区域
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('高度 (米)')
            ax1.set_title('四旋翼无人机高度控制')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 推力变化图
            ax2.plot(time_data, thrust_data, 'g-', linewidth=2, label='平均推力')
            ax2.axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='悬停推力')
            ax2.set_xlabel('时间 (秒)')
            ax2.set_ylabel('推力')
            ax2.set_title('电机推力变化')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 调整子图间距
            plt.tight_layout()
            # 显示图表
            plt.show()

        except ImportError:
            print("⚠ 需要安装matplotlib: pip install matplotlib")
        except Exception as e:
            print(f"⚠ 绘图错误: {e}")


def main():
    """
    主函数 - 使用默认设置运行仿真
    流程：
    1. 初始化仿真环境
    2. 设置默认仿真参数
    3. 运行仿真
    4. 处理异常（用户中断、运行错误）
    """
    print("🚁 MuJoCo 四旋翼无人机仿真系统")
    print("=" * 50)

    try:
        # 创建仿真实例
        print("正在初始化...")
        sim = QuadrotorSimulation()
        print("✅ 初始化完成")

        # 默认仿真设置
        controller_type = "position"  # 默认使用位置控制器（更丰富的运动效果）
        duration = 15.0               # 默认仿真时长15秒
        use_viewer = True             # 默认启用可视化

        # 打印默认设置
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
        # 处理用户Ctrl+C中断
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        # 处理其他错误，打印详细堆栈信息
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


# 程序入口
if __name__ == "__main__":
    # 直接运行，无需用户输入
    main()