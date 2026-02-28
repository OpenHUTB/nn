
"""
无人小车自主行驶与避让模拟
基于MuJoCo和Python实现
运行环境：PyCharm + MuJoCo
"""

import mujoco
import mujoco.viewer
import numpy as np
import glfw
import time
import math
import os


class AutonomousCar:
    def __init__(self, model_path=None):
        """初始化无人小车模拟器
        
        Args:
            model_path (str, optional): MuJoCo模型文件路径. 默认为None，使用内置模型
        """
        # 如果没有提供模型文件，使用内置的XML模型
        if model_path is None:
            self.xml = """
            <mujoco>
                <!-- 仿真参数设置 -->
                <option timestep="0.01" gravity="0 0 -9.81"/>

                <!-- 材质定义 -->
                <asset>
                    <material name="grid" rgba=".2 .3 .4 1"/>  <!-- 网格材质 -->
                    <material name="car_body" rgba="0.2 0.6 0.8 1"/>  <!-- 车身材质 -->
                    <material name="car_detail" rgba="0.8 0.6 0.2 1"/>  <!-- 车辆细节材质 -->
                    <material name="wheel" rgba="0.1 0.1 0.1 1"/>  <!-- 车轮材质 -->
                    <material name="rim" rgba="0.9 0.9 0.9 1"/>  <!-- 轮毂材质 -->
                    <material name="obstacle" rgba="0.8 0.2 0.2 1"/>  <!-- 障碍物材质 -->
                    <material name="target" rgba="0.2 0.8 0.2 1"/>  <!-- 目标点材质 -->
                    <material name="floor" rgba="0.9 0.9 0.9 1"/>  <!-- 地面材质 -->
                </asset>

                <!-- 世界主体定义 -->
                <worldbody>
                    <!-- 地面 -->
                    <geom name="floor" type="plane" size="10 10 0.1" material="floor" pos="0 0 -0.1"/>

                    <!-- 无人小车 -->
                    <body name="car" pos="0 0 0.3">  <!-- 小车起始位置 -->
                        <joint name="car_rot" type="free"/>  <!-- 自由度关节，允许6自由度运动 -->
                        <!-- 车身主体 -->
                        <geom name="car_body_main" type="box" size="0.3 0.6 0.15" material="car_body" pos="0 0 0"/>  <!-- 主车身体 -->
                        <geom name="car_front" type="box" size="0.25 0.15 0.1" pos="0 0.5 0" material="car_detail"/>  <!-- 前部装饰 -->
                        <geom name="car_back" type="box" size="0.25 0.15 0.1" pos="0 -0.5 0" material="car_detail"/>  <!-- 后部装饰 -->
                        <geom name="car_top" type="box" size="0.28 0.58 0.05" pos="0 0 0.15" material="car_body"/>  <!-- 车顶 -->

                        <!-- 车轮组件 -->
                        <body name="front_left_wheel" pos="0.25 0.4 0">  <!-- 前左轮 -->
                            <joint name="front_left_steer" type="hinge" axis="0 0 1" range="-30 30"/>  <!-- 转向关节 -->
                            <joint name="front_left_roll" type="hinge" axis="0 1 0"/>  <!-- 滚动关节 -->
                            <geom name="wheel_fl_rim" type="cylinder" size="0.09 0.06" material="rim" pos="0 0 -0.01"/>  <!-- 轮毂 -->
                            <geom name="wheel_fl_tire" type="cylinder" size="0.1 0.05" material="wheel" pos="0 0 0.01"/>  <!-- 轮胎 -->
                        </body>

                        <body name="front_right_wheel" pos="-0.25 0.4 0">
                            <joint name="front_right_steer" type="hinge" axis="0 0 1" range="-30 30"/>
                            <joint name="front_right_roll" type="hinge" axis="0 1 0"/>
                            <geom name="wheel_fr_rim" type="cylinder" size="0.09 0.06" material="rim" pos="0 0 -0.01"/>
                            <geom name="wheel_fr_tire" type="cylinder" size="0.1 0.05" material="wheel" pos="0 0 0.01"/>
                        </body>

                        <body name="rear_left_wheel" pos="0.25 -0.4 0">
                            <joint name="rear_left_roll" type="hinge" axis="0 1 0"/>
                            <geom name="wheel_rl_rim" type="cylinder" size="0.09 0.06" material="rim" pos="0 0 -0.01"/>
                            <geom name="wheel_rl_tire" type="cylinder" size="0.1 0.05" material="wheel" pos="0 0 0.01"/>
                        </body>

                        <body name="rear_right_wheel" pos="-0.25 -0.4 0">
                            <joint name="rear_right_roll" type="hinge" axis="0 1 0"/>
                            <geom name="wheel_rr_rim" type="cylinder" size="0.09 0.06" material="rim" pos="0 0 -0.01"/>
                            <geom name="wheel_rr_tire" type="cylinder" size="0.1 0.05" material="wheel" pos="0 0 0.01"/>
                        </body>

                        <!-- 传感器位置 -->
                        <site name="front_sensor" pos="0 0.7 0.1" size="0.05"/>  <!-- 前方传感器 -->
                        <site name="left_sensor" pos="0.4 0 0.1" size="0.05"/>  <!-- 左方传感器 -->
                        <site name="right_sensor" pos="-0.4 0 0.1" size="0.05"/>  <!-- 右方传感器 -->
                    </body>

                    <!-- 目标点 -->
                    <body name="target" pos="8 0 0.5">  <!-- 目标位置 (x=8, y=0, z=0.5) -->
                        <geom name="target_geom" type="sphere" size="0.3" material="target"/>  <!-- 球形目标 -->
                        <site name="target_site" pos="0 0 0" size="0.1"/>  <!-- 目标位置标记 -->
                    </body>

                    <!-- 障碍物 - 更加立体逼真的设计 -->
                    <body name="obstacle1" pos="3 2 0.5">
                        <geom name="obs1" type="cylinder" size="0.4 1.0" material="obstacle" euler="0 0 0"/>
                        <geom name="obs1_top" type="sphere" size="0.4" material="obstacle" pos="0 0 1.0"/>
                    </body>

                    <body name="obstacle2" pos="5 -1.5 0.5">
                        <geom name="obs2_base" type="box" size="0.7 0.4 0.8" material="obstacle"/>
                        <geom name="obs2_top" type="box" size="0.5 0.2 0.3" pos="0 0 0.8" material="obstacle"/>
                    </body>

                    <body name="obstacle3" pos="2 -2 0.5">
                        <geom name="obs3" type="ellipsoid" size="0.6 0.4 0.5" material="obstacle"/>
                    </body>

                    <body name="obstacle4" pos="6 2 0.5">
                        <geom name="obs4" type="capsule" size="0.3 1.0" material="obstacle" euler="1.57 0 0"/>
                    </body>

                    <!-- 灯光 -->
                    <light name="top" pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
                    <light name="car_light" pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>

                    <!-- 相机视角 -->
                    <camera name="fixed" pos="12 0 4" xyaxes="1 0 0 0 0.7 0.7"/>
                    <camera name="follow" mode="targetbody" target="car" pos="0 -8 4"/>
                </worldbody>

                <actuator>
                    <!-- 驱动电机 -->
                    <motor name="front_left_drive" joint="front_left_roll" gear="50"/>
                    <motor name="front_right_drive" joint="front_right_roll" gear="50"/>
                    <motor name="rear_left_drive" joint="rear_left_roll" gear="50"/>
                    <motor name="rear_right_drive" joint="rear_right_roll" gear="50"/>

                    <!-- 转向电机 -->
                    <position name="front_left_steer" joint="front_left_steer" kp="100"/>
                    <position name="front_right_steer" joint="front_right_steer" kp="100"/>
                </actuator>

                <sensor>
                    <!-- 位置传感器 -->
                    <framepos objtype="body" objname="car"/>
                    <framepos objtype="body" objname="target"/>
                </sensor>
            </mujoco>
            """

            # 保存XML到临时文件
            self.temp_xml_path = "temp_car_model.xml"
            with open(self.temp_xml_path, 'w') as f:
                f.write(self.xml)

            try:
                self.model = mujoco.MjModel.from_xml_path(self.temp_xml_path)
            except Exception as e:
                print(f"XML解析错误: {e}")
                # 尝试简化版本
                self.create_simple_model()
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)

        self.data = mujoco.MjData(self.model)

        # 控制参数
        self.target_speed = 6.0  # 目标速度 (m/s)
        self.max_steering_angle = 0.5  # 最大转向角度（弧度）
        self.avoidance_distance = 2.5  # 避障检测距离 (m)
        self.avoidance_strength = 2.5  # 避障强度

        # 状态变量
        self.current_speed = 0.0  # 当前速度 (m/s)
        self.steering_angle = 0.0  # 当前转向角度 (弧度)
        self.obstacle_detected = False  # 是否检测到障碍物
        self.simulation_time = 0.0  # 模拟时间 (s)
        self.target_reached = False  # 是否到达目标
        self.path_history = []  # 路径历史，用于轨迹记录

        # PID控制器参数 (当前未在代码中使用，但为后续扩展预留)
        self.speed_Kp = 4.0  # 速度控制器比例参数
        self.speed_Ki = 0.1  # 速度控制器积分参数
        self.speed_Kd = 0.3  # 速度控制器微分参数
        self.speed_integral = 0.0  # 速度控制器积分项
        self.speed_prev_error = 0.0  # 速度控制器前一误差

        self.steering_Kp = 6.0  # 转向控制器比例参数
        self.steering_Ki = 0.05  # 转向控制器积分参数
        self.steering_Kd = 0.2  # 转向控制器微分参数
        self.steering_integral = 0.0  # 转向控制器积分项
        self.steering_prev_error = 0.0  # 转向控制器前一误差

    def create_simple_model(self):
        """创建简化模型（如果完整模型有问题）
        
        当完整模型因XML格式错误或其他问题无法加载时，
        使用此方法创建一个简化的模型用于模拟
        """
        print("使用简化模型...")
        simple_xml = """
        <mujoco>
            <option timestep="0.01" gravity="0 0 -9.81"/>

            <worldbody>
                <!-- 地面 -->
                <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 -0.1" rgba="0.9 0.9 0.9 1"/>

                <!-- 无人小车 -->
                <body name="car" pos="0 0 0.3">
                    <joint name="car_rot" type="free"/>
                    <geom name="car_body" type="box" size="0.3 0.5 0.2" rgba="0.2 0.6 0.8 1"/>

                    <!-- 轮子 -->
                    <geom name="wheel_fl" type="cylinder" size="0.08 0.05" pos="0.25 0.4 0" rgba="0.1 0.1 0.1 1"/>
                    <geom name="wheel_fr" type="cylinder" size="0.08 0.05" pos="-0.25 0.4 0" rgba="0.1 0.1 0.1 1"/>
                    <geom name="wheel_rl" type="cylinder" size="0.08 0.05" pos="0.25 -0.4 0" rgba="0.1 0.1 0.1 1"/>
                    <geom name="wheel_rr" type="cylinder" size="0.08 0.05" pos="-0.25 -0.4 0" rgba="0.1 0.1 0.1 1"/>
                </body>

                <!-- 目标点 -->
                <geom name="target" type="sphere" size="0.3" pos="8 0 0.5" rgba="0.2 0.8 0.2 1"/>

                <!-- 障碍物 -->
                <geom name="obstacle1" type="cylinder" size="0.4 0.8" pos="3 2 0.5" rgba="0.8 0.2 0.2 1"/>
                <geom name="obstacle2" type="box" size="0.6 0.3 0.8" pos="5 -1.5 0.5" rgba="0.8 0.2 0.2 1"/>
                <geom name="obstacle3" type="sphere" size="0.5" pos="2 -2 0.5" rgba="0.8 0.2 0.2 1"/>
            </worldbody>

            <actuator>
                <motor name="drive" joint="car_rot" ctrlrange="-10 10" gear="100"/>
            </actuator>
        </mujoco>
        """

        with open(self.temp_xml_path, 'w') as f:
            f.write(simple_xml)

        self.model = mujoco.MjModel.from_xml_path(self.temp_xml_path)

    def __del__(self):
        """清理临时文件
        
        析构函数，确保在对象销毁时删除临时创建的XML模型文件
        避免临时文件残留
        """
        if hasattr(self, 'temp_xml_path') and os.path.exists(self.temp_xml_path):
            try:
                os.remove(self.temp_xml_path)
            except:
                pass

    def get_sensor_readings(self):
        """获取传感器读数
        
        通过模拟传感器检测周围环境中的障碍物
        返回包含前后左右距离和障碍物检测状态的字典
        
        Returns:
            dict: 包含传感器读数的字典
                - front_distance: 前方距离
                - left_distance: 左方距离
                - right_distance: 右方距离
                - front_obstacle: 前方障碍物状态
                - left_obstacle: 左方障碍物状态
                - right_obstacle: 右方障碍物状态
        """
        readings = {
            'front_distance': 10.0,
            'left_distance': 10.0,
            'right_distance': 10.0,
            'front_obstacle': False,
            'left_obstacle': False,
            'right_obstacle': False
        }

        # 获取小车位置和方向
        car_pos = self.data.body('car').xpos  # 小车当前位置 (x, y, z)
        car_orientation = self.data.body('car').xmat.reshape(3, 3)  # 小车方向矩阵
        car_forward = car_orientation @ np.array([0, 1, 0])  # 小车前进方向向量
        car_left = car_orientation @ np.array([1, 0, 0])  # 小车左侧方向向量

        # 检查所有障碍物
        obstacle_positions = [  # 预定义的障碍物位置
            np.array([3, 2, 0.5]),  # obstacle1
            np.array([5, -1.5, 0.5]),  # obstacle2
            np.array([2, -2, 0.5]),  # obstacle3
            np.array([6, 2, 0.5])  # obstacle4
        ]

        obstacle_sizes = [  # 障碍物尺寸参数，用于碰撞检测
            1.2,  # obstacle1 半径+高度
            1.4,  # obstacle2 尺寸
            1.0,  # obstacle3 直径
            1.3  # obstacle4 半径+高度
        ]

        for i, obs_pos in enumerate(obstacle_positions):
            # 计算障碍物到小车的向量
            obs_vector = obs_pos - car_pos  # 从小车指向障碍物的向量
            distance = np.linalg.norm(obs_vector[:2])  # 只考虑平面距离 (x, y方向)

            if distance < self.avoidance_distance + obstacle_sizes[i]:
                # 计算障碍物相对于小车的方向
                obs_direction = obs_vector[:2] / distance if distance > 0 else np.array([0, 0])

                # 计算与前进方向的夹角
                forward_2d = car_forward[:2]
                angle = math.atan2(
                    obs_direction[1] * forward_2d[0] - obs_direction[0] * forward_2d[1],
                    obs_direction[0] * forward_2d[0] + obs_direction[1] * forward_2d[1]
                )

                angle_deg = math.degrees(angle)

                # 更新传感器读数
                if -45 < angle_deg < 45:  # 前方
                    readings['front_distance'] = min(readings['front_distance'], distance)
                    if distance < 2.0:
                        readings['front_obstacle'] = True

                elif 45 <= angle_deg < 135:  # 左侧
                    readings['left_distance'] = min(readings['left_distance'], distance)
                    if distance < 1.5:
                        readings['left_obstacle'] = True

                elif -135 < angle_deg <= -45:  # 右侧
                    readings['right_distance'] = min(readings['right_distance'], distance)
                    if distance < 1.5:
                        readings['right_obstacle'] = True

        return readings

    def autonomous_driving(self, dt):
        """自主驾驶算法
        
        实现无人小车的自主导航和避障功能
        包括目标导向导航、障碍物检测和避让策略
        
        Args:
            dt (float): 时间步长
        
        Returns:
            numpy.ndarray: 控制信号数组
        """
        if dt <= 0:
            dt = 0.01

        # 获取传感器数据
        sensor_data = self.get_sensor_readings()

        # 获取目标位置
        target_pos = np.array([8, 0, 0.5])  # 目标位置 (x, y, z)
        car_pos = self.data.body('car').xpos  # 小车当前位置

        # 计算到目标的距离和方向
        target_vector = target_pos - car_pos
        target_distance = np.linalg.norm(target_vector[:2])

        # 检查是否到达目标
        if target_distance < 0.5:
            self.target_reached = True
            return np.zeros(self.model.nu)

        # 计算目标方向（归一化）
        if target_distance > 0:
            target_direction = target_vector[:2] / target_distance  # 归一化目标方向向量
        else:
            target_direction = np.array([0, 1])  # 默认朝y轴正方向

        # 获取小车当前方向
        car_orientation = self.data.body('car').xmat.reshape(3, 3)  # 获取小车方向矩阵
        car_direction = car_orientation @ np.array([0, 1, 0])  # 计算小车实际前进方向
        car_direction_2d = car_direction[:2]  # 取x,y方向分量

        if np.linalg.norm(car_direction_2d) > 0:
            car_direction_2d = car_direction_2d / np.linalg.norm(car_direction_2d)  # 归一化方向向量

        # 计算转向误差 (使用向量叉积和点积计算角度差)
        steering_error = math.atan2(
            target_direction[1] * car_direction_2d[0] - target_direction[0] * car_direction_2d[1],  # 叉积计算转向方向
            target_direction[0] * car_direction_2d[0] + target_direction[1] * car_direction_2d[1]   # 点积计算转向大小
        )

        # 避障逻辑 - 根据传感器数据调整转向
        avoidance_steering = 0.0  # 初始化避障转向量
        self.obstacle_detected = False  # 重置障碍物检测标志

        if sensor_data['front_obstacle']:
            self.obstacle_detected = True
            # 前方有障碍物，根据两侧距离决定转向
            if sensor_data['left_distance'] > sensor_data['right_distance']:
                avoidance_steering = 0.5  # 向左转 (正值表示左转)
            else:
                avoidance_steering = -0.5  # 向右转 (负值表示右转)

        elif sensor_data['left_obstacle']:
            avoidance_steering = -0.3  # 向右轻微转向

        elif sensor_data['right_obstacle']:
            avoidance_steering = 0.3  # 向左轻微转向

        # 合并转向控制 - 将目标导向和避障转向相结合
        total_steering = steering_error + avoidance_steering

        # 限制转向角度
        total_steering = np.clip(total_steering, -self.max_steering_angle, self.max_steering_angle)

        # 速度控制：根据障碍物距离调整速度 (距离越近，速度越慢)
        min_distance = min(sensor_data['front_distance'],      # 前方最近障碍物距离
                           sensor_data['left_distance'],      # 左方最近障碍物距离
                           sensor_data['right_distance'])     # 右方最近障碍物距离

        if min_distance < 1.0:
            speed_multiplier = 0.3
        elif min_distance < 2.0:
            speed_multiplier = 0.6
        else:
            speed_multiplier = 1.0

        target_speed_adjusted = self.target_speed * speed_multiplier

        # 简单的速度控制
        if self.current_speed < target_speed_adjusted:
            self.current_speed += 0.5 * dt
        elif self.current_speed > target_speed_adjusted:
            self.current_speed -= 0.5 * dt

        self.current_speed = np.clip(self.current_speed, 0, self.target_speed)

        # 记录路径 - 保存小车移动轨迹，用于后续分析或可视化
        self.path_history.append(car_pos.copy())  # 添加当前位置到路径历史
        if len(self.path_history) > 1000:  # 限制历史记录数量，避免内存占用过多
            self.path_history.pop(0)  # 移除最旧的位置记录

        # 生成控制信号
        control = np.zeros(self.model.nu)

        if hasattr(self.model, 'nu') and self.model.nu >= 6:
            # 完整模型的控制 (四轮独立驱动+前轮转向)
            control[0] = self.current_speed  # 前左轮驱动电机
            control[1] = self.current_speed  # 前右轮驱动电机
            control[2] = self.current_speed  # 后左轮驱动电机
            control[3] = self.current_speed  # 后右轮驱动电机
            control[4] = total_steering  # 前左轮转向电机
            control[5] = total_steering  # 前右轮转向电机
        else:
            # 简化模型的控制 (单电机驱动+单电机转向)
            control[0] = self.current_speed  # 驱动电机
            if len(control) > 1:
                control[1] = total_steering  # 转向电机

        return control

    def print_status(self):
        """打印状态信息
        
        在控制台打印当前模拟状态，包括位置、速度、方向等信息
        使用\r覆盖前一行以实现动态更新效果
        """
        car_pos = self.data.body('car').xpos
        target_pos = np.array([8, 0, 0.5])
        distance = np.linalg.norm(target_pos[:2] - car_pos[:2])

        print(f"\r时间: {self.simulation_time:.1f}s | "
              f"位置: ({car_pos[0]:.1f}, {car_pos[1]:.1f}) | "
              f"速度: {self.current_speed:.1f}m/s | "
              f"转向: {math.degrees(self.steering_angle):.0f}° | "
              f"距目标: {distance:.1f}m | "
              f"状态: {'避障' if self.obstacle_detected else '导航'} {'到达!' if self.target_reached else ''}",
              end="")

    def run_simulation(self):
        """运行模拟主循环
        
        启动MuJoCo可视化界面并运行无人小车模拟
        包括物理仿真、控制算法执行、状态显示和用户交互
        """
        print("无人小车模拟系统启动中...")
        print("=" * 80)
        print("控制说明:")
        print("  - 按ESC键退出模拟")
        print("  - 模拟会自动运行直到按下ESC或到达目标")
        print("  - 绿色球体是目标点")
        print("  - 红色物体是障碍物")
        print("  - 小车会自动导航并避开障碍物")
        print("=" * 80)

        # 设置模拟选项
        self.model.opt.gravity[2] = -9.81  # 确保重力设置正确

        # 重置模拟 - 将所有物理量重置为初始状态
        mujoco.mj_resetData(self.model, self.data)

        # 启动可视化查看器 - 创建3D可视化界面
        try:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as e:
            print(f"查看器启动失败: {e}")
            print("将以无界面模式运行模拟...")
            viewer = None

        last_time = time.time()
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                if viewer is not None and not viewer.is_running():
                    break

                # 计算时间步长 - 确保模拟时间正确更新
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                self.simulation_time += dt

                # 限制时间步长
                if dt > 0.1:
                    dt = 0.01

                # 应用自主驾驶控制 - 计算并应用控制信号
                control = self.autonomous_driving(dt)  # 获取控制信号
                self.data.ctrl[:] = control  # 将控制信号应用到模型

                # 执行物理模拟 - 更新物理状态
                mujoco.mj_step(self.model, self.data)

                # 更新查看器 - 同步可视化界面与物理模拟状态
                if viewer is not None:
                    viewer.sync()  # 刷新3D视图

                # 更新状态显示
                frame_count += 1
                if frame_count % 10 == 0:
                    self.print_status()

                # 检查是否到达目标
                if self.target_reached:
                    print(f"\n\n{'=' * 80}")
                    print("成功到达目标点！")
                    print(f"总时间: {self.simulation_time:.1f}秒")
                    print(f"平均速度: {np.mean([v for v in self.path_history if len(v) > 0]):.1f}m/s")
                    print(f"{'=' * 80}")
                    time.sleep(2)
                    break

                # 检查ESC键 - 修复后的代码
                if viewer is not None:
                    try:
                        # MuJoCo新版本使用不同的API来访问窗口句柄
                        if hasattr(viewer, 'context'):
                            # 检查查看器是否仍在运行
                            if not viewer.is_running():
                                break
                        else:
                            # 检查查看器是否仍在运行
                            if not viewer.is_running():
                                break
                    except:
                        # 如果任何API检查失败，检查查看器是否仍在运行
                        if viewer is not None and not viewer.is_running():
                            break

                # 控制帧率
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n用户中断模拟...")

        finally:
            if viewer is not None:
                viewer.close()

            # 显示模拟统计
            print(f"\n\n{'=' * 80}")
            print("模拟统计:")
            print(f"  总模拟时间: {self.simulation_time:.1f}秒")
            print(f"  总帧数: {frame_count}")
            print(f"  平均帧率: {frame_count / (time.time() - start_time):.1f} FPS")
            print(f"  路径点数: {len(self.path_history)}")
            print(f"{'=' * 80}")


def main():
    """主函数
    
    程序入口点，负责初始化系统、检查依赖、创建模拟实例并启动模拟
    包含错误处理和故障排除建议
    """
    print("正在初始化无人小车模拟系统...")
    print("=" * 80)

    try:
        # 检查必要的库 - 确保所有必需的库都已安装
        import importlib
        required_libs = ['mujoco', 'numpy', 'glfw']  # 必需的库列表
        missing_libs = []  # 缺失库列表

        for lib in required_libs:
            try:
                importlib.import_module(lib)  # 尝试导入库
            except ImportError:
                missing_libs.append(lib)  # 记录缺失的库

        if missing_libs:
            print(f"缺少必要的库: {missing_libs}")
            print("请使用以下命令安装:")
            print("pip install mujoco glfw numpy")
            return

        # 创建无人小车实例
        print("正在创建无人小车模型...")
        car_sim = AutonomousCar()

        print("模型创建成功！开始模拟...")
        time.sleep(1)

        # 运行模拟
        car_sim.run_simulation()

    except Exception as e:
        print(f"\n模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 提供故障排除建议
        print(f"\n{'=' * 80}")
        print("故障排除建议:")
        print("1. 确保已安装正确版本的MuJoCo:")
        print("   pip install mujoco")
        print("2. 如果使用简化模型，可能需要安装额外依赖:")
        print("   pip install glfw")
        print("3. 确保有足够的权限和磁盘空间")
        print("4. 尝试重启PyCharm或系统")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()