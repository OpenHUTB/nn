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
        """初始化无人小车模拟器"""
        # 如果没有提供模型文件，使用内置的XML模型
        if model_path is None:
            self.xml = """
            <mujoco>
                <option timestep="0.01" gravity="0 0 -9.81"/>

                <asset>
                    <texture type="skybox" builtin="flat" rgb1="0.3 0.5 0.7" rgb2="0.1 0.2 0.3"/>
                    <material name="grid" rgba="0.85 0.85 0.85 1"/>
                    <material name="body" rgba="0.2 0.6 0.8 1"/>
                    <material name="wheel" rgba="0.1 0.1 0.1 1"/>
                    <material name="obstacle" rgba="0.8 0.2 0.2 1"/>
                    <material name="target" rgba="0.2 0.8 0.2 1"/>
                    <material name="floor" rgba="0.9 0.9 0.9 1"/>
                </asset>

                <worldbody>
                    <!-- 地面 -->
                    <geom name="floor" type="plane" size="10 10 0.1" material="floor" pos="0 0 -0.1"/>

                    <!-- 无人小车 -->
                    <body name="car" pos="0 0 0.3">
                        <joint name="car_rot" type="free"/>
                        <geom name="car_body" type="box" size="0.3 0.5 0.2" material="body"/>
                        <geom name="car_front" type="box" size="0.3 0.1 0.15" pos="0 0.5 0" material="body"/>

                        <!-- 前轮 -->
                        <body name="front_left_wheel" pos="0.25 0.4 0">
                            <joint name="front_left_steer" type="hinge" axis="0 0 1" range="-30 30"/>
                            <joint name="front_left_roll" type="hinge" axis="0 1 0"/>
                            <geom name="wheel_fl" type="cylinder" size="0.08 0.05" material="wheel"/>
                        </body>

                        <body name="front_right_wheel" pos="-0.25 0.4 0">
                            <joint name="front_right_steer" type="hinge" axis="0 0 1" range="-30 30"/>
                            <joint name="front_right_roll" type="hinge" axis="0 1 0"/>
                            <geom name="wheel_fr" type="cylinder" size="0.08 0.05" material="wheel"/>
                        </body>

                        <!-- 后轮 -->
                        <body name="rear_left_wheel" pos="0.25 -0.4 0">
                            <joint name="rear_left_roll" type="hinge" axis="0 1 0"/>
                            <geom name="wheel_rl" type="cylinder" size="0.08 0.05" material="wheel"/>
                        </body>

                        <body name="rear_right_wheel" pos="-0.25 -0.4 0">
                            <joint name="rear_right_roll" type="hinge" axis="0 1 0"/>
                            <geom name="wheel_rr" type="cylinder" size="0.08 0.05" material="wheel"/>
                        </body>

                        <!-- 传感器位置 -->
                        <site name="front_sensor" pos="0 0.7 0.1" size="0.05"/>
                        <site name="left_sensor" pos="0.4 0 0.1" size="0.05"/>
                        <site name="right_sensor" pos="-0.4 0 0.1" size="0.05"/>
                    </body>

                    <!-- 目标点 -->
                    <body name="target" pos="8 0 0.5">
                        <geom name="target_geom" type="sphere" size="0.3" material="target"/>
                        <site name="target_site" pos="0 0 0" size="0.1"/>
                    </body>

                    <!-- 障碍物（带自由关节，可动态移动） -->
                    <body name="obstacle1_base" pos="3 2 0.5">
                        <joint name="obs1_joint" type="free"/>
                        <geom name="obs1" type="cylinder" size="0.4 0.8" material="obstacle"/>
                    </body>

                    <body name="obstacle2_base" pos="5 -1.5 0.5">
                        <joint name="obs2_joint" type="free"/>
                        <geom name="obs2" type="box" size="0.6 0.3 0.8" material="obstacle"/>
                    </body>

                    <body name="obstacle3_base" pos="2 -2 0.5">
                        <joint name="obs3_joint" type="free"/>
                        <geom name="obs3" type="sphere" size="0.5" material="obstacle"/>
                    </body>

                    <body name="obstacle4_base" pos="6 2 0.5">
                        <joint name="obs4_joint" type="free"/>
                        <geom name="obs4" type="cylinder" size="0.3 1.0" material="obstacle"/>
                    </body>

                    <!-- 路径轨迹标记 -->
                    <body name="trail_root" pos="0 0 -0.05">
                        <geom name="trail0" type="sphere" size="0.05" rgba="1 1 0 0.6"/>
                        <geom name="trail1" type="sphere" size="0.05" rgba="1 1 0 0.6"/>
                        <geom name="trail2" type="sphere" size="0.05" rgba="1 1 0 0.6"/>
                        <geom name="trail3" type="sphere" size="0.05" rgba="1 1 0 0.6"/>
                        <geom name="trail4" type="sphere" size="0.05" rgba="1 1 0 0.6"/>
                    </body>

                    <!-- 灯光 -->
                    <light name="top" pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>

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
        self.target_speed = 6.0
        self.max_steering_angle = 0.5
        self.avoidance_distance = 2.5
        self.avoidance_strength = 2.5

        # 状态变量
        self.current_speed = 0.0
        self.steering_angle = 0.0
        self.obstacle_detected = False
        self.simulation_time = 0.0
        self.target_reached = False
        self.path_history = []

        # PID控制器参数
        self.speed_Kp = 4.0
        self.speed_Ki = 0.1
        self.speed_Kd = 0.3
        self.speed_integral = 0.0
        self.speed_prev_error = 0.0
        self.speed_integral_max = 2.0

        self.steering_Kp = 6.0
        self.steering_Ki = 0.05
        self.steering_Kd = 0.2
        self.steering_integral = 0.0
        self.steering_prev_error = 0.0
        self.steering_integral_max = 1.0

        # 键盘状态
        self.keys = {}
        self.manual_mode = False
        self._last_mode_toggle = 0.0

        # 碰撞统计
        self.collision_count = 0
        self.collision_force_total = 0.0
        self.collision_force_max = 0.0
        self._prev_contact_count = 0
        self._collision_cooldown = 0.0

        # 障碍物运动参数
        self.obstacle_base_positions = {
            'obstacle1': np.array([3, 2, 0.5]),
            'obstacle2': np.array([5, -1.5, 0.5]),
            'obstacle3': np.array([2, -2, 0.5]),
            'obstacle4': np.array([6, 2, 0.5]),
        }
        self.obstacle_motions = {
            'obstacle1': {'type': 'circle', 'radius': 1.5, 'speed': 0.8, 'phase': 0.0},
            'obstacle2': {'type': 'linear_x', 'amplitude': 2.0, 'speed': 0.6, 'phase': 0.0},
            'obstacle3': {'type': 'circle', 'radius': 1.0, 'speed': 1.0, 'phase': math.pi / 2},
            'obstacle4': {'type': 'linear_y', 'amplitude': 1.8, 'speed': 0.7, 'phase': 0.0},
        }

        # 轨迹可视化
        self.trail_positions = []
        self.trail_max = 200
        self.trail_step = 0

    def create_simple_model(self):
        """创建简化模型（如果完整模型有问题）"""
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
        """清理临时文件"""
        if hasattr(self, 'temp_xml_path') and os.path.exists(self.temp_xml_path):
            try:
                os.remove(self.temp_xml_path)
            except:
                pass

    def get_sensor_readings(self):
        """获取传感器读数（动态读取障碍物位置）"""
        readings = {
            'front_distance': 10.0,
            'left_distance': 10.0,
            'right_distance': 10.0,
            'front_obstacle': False,
            'left_obstacle': False,
            'right_obstacle': False
        }

        car_pos = self.data.body('car').xpos
        car_orientation = self.data.body('car').xmat.reshape(3, 3)
        car_forward = car_orientation @ np.array([0, 1, 0])
        car_left = car_orientation @ np.array([1, 0, 0])

        obstacle_names = ['obstacle1_base', 'obstacle2_base', 'obstacle3_base', 'obstacle4_base']
        obstacle_sizes = [1.2, 1.4, 1.0, 1.3]

        for i, obs_name in enumerate(obstacle_names):
            try:
                obs_pos = self.data.body(obs_name).xpos.copy()
            except KeyError:
                # 回退到硬编码位置（简化模型）
                fallback = [
                    np.array([3, 2, 0.5]),
                    np.array([5, -1.5, 0.5]),
                    np.array([2, -2, 0.5]),
                    np.array([6, 2, 0.5])
                ]
                obs_pos = fallback[i]

            obs_vector = obs_pos - car_pos
            distance = np.linalg.norm(obs_vector[:2])

            if distance < self.avoidance_distance + obstacle_sizes[i]:
                obs_direction = obs_vector[:2] / distance if distance > 0 else np.array([0, 0])
                forward_2d = car_forward[:2]
                angle = math.atan2(
                    obs_direction[1] * forward_2d[0] - obs_direction[0] * forward_2d[1],
                    obs_direction[0] * forward_2d[0] + obs_direction[1] * forward_2d[1]
                )
                angle_deg = math.degrees(angle)

                if -45 < angle_deg < 45:
                    readings['front_distance'] = min(readings['front_distance'], distance)
                    if distance < 2.0:
                        readings['front_obstacle'] = True
                elif 45 <= angle_deg < 135:
                    readings['left_distance'] = min(readings['left_distance'], distance)
                    if distance < 1.5:
                        readings['left_obstacle'] = True
                elif -135 < angle_deg <= -45:
                    readings['right_distance'] = min(readings['right_distance'], distance)
                    if distance < 1.5:
                        readings['right_obstacle'] = True

        return readings

    def autonomous_driving(self, dt):
        """自主驾驶算法（集成 PID 控制、动态障碍物感知）"""

        def _pid_update(error, dt, key):
            integral = getattr(self, f'{key}_integral')
            prev_error = getattr(self, f'{key}_prev_error')
            Kp = getattr(self, f'{key}_Kp')
            Ki = getattr(self, f'{key}_Ki')
            Kd = getattr(self, f'{key}_Kd')
            integral_max = getattr(self, f'{key}_integral_max')

            integral += error * dt
            integral = max(-integral_max, min(integral_max, integral))

            derivative = 0.0
            if dt > 0.001:
                derivative = (error - prev_error) / dt

            setattr(self, f'{key}_integral', integral)
            setattr(self, f'{key}_prev_error', error)

            return Kp * error + Ki * integral + Kd * derivative

        if dt <= 0:
            dt = 0.01

        sensor_data = self.get_sensor_readings()

        try:
            target_pos = self.data.body('target').xpos.copy()
        except KeyError:
            target_pos = np.array([8, 0, 0.5])
        car_pos = self.data.body('car').xpos.copy()

        target_vector = target_pos - car_pos
        target_distance = np.linalg.norm(target_vector[:2])

        if target_distance < 0.5:
            self.target_reached = True
            return np.zeros(self.model.nu)

        if target_distance > 0:
            target_direction = target_vector[:2] / target_distance
        else:
            target_direction = np.array([0, 1])

        car_orientation = self.data.body('car').xmat.reshape(3, 3)
        car_direction = car_orientation @ np.array([0, 1, 0])
        car_direction_2d = car_direction[:2]
        if np.linalg.norm(car_direction_2d) > 0:
            car_direction_2d = car_direction_2d / np.linalg.norm(car_direction_2d)

        steering_error = math.atan2(
            target_direction[1] * car_direction_2d[0] - target_direction[0] * car_direction_2d[1],
            target_direction[0] * car_direction_2d[0] + target_direction[1] * car_direction_2d[1]
        )

        avoidance_steering = 0.0
        self.obstacle_detected = False

        if sensor_data['front_obstacle']:
            self.obstacle_detected = True
            if sensor_data['left_obstacle'] and sensor_data['right_obstacle']:
                avoidance_steering = -self.avoidance_strength * (
                    (2.0 - min(sensor_data['left_distance'], 2.0)) / 2.0)
            elif sensor_data['left_obstacle']:
                avoidance_steering = self.avoidance_strength * (
                    (1.5 - sensor_data['left_distance']) / 1.5)
            elif sensor_data['right_obstacle']:
                avoidance_steering = -self.avoidance_strength * (
                    (1.5 - sensor_data['right_distance']) / 1.5)
            else:
                if sensor_data['left_distance'] > sensor_data['right_distance']:
                    avoidance_steering = self.avoidance_strength * 0.5 * (
                        (2.0 - sensor_data['front_distance']) / 2.0)
                else:
                    avoidance_steering = -self.avoidance_strength * 0.5 * (
                        (2.0 - sensor_data['front_distance']) / 2.0)

        elif sensor_data['left_obstacle']:
            self.obstacle_detected = True
            avoidance_steering = -self.avoidance_strength * 0.6 * (
                (1.5 - sensor_data['left_distance']) / 1.5)

        elif sensor_data['right_obstacle']:
            self.obstacle_detected = True
            avoidance_steering = self.avoidance_strength * 0.6 * (
                (1.5 - sensor_data['right_distance']) / 1.5)

        avoidance_steering = max(-self.max_steering_angle * 2,
                                  min(self.max_steering_angle * 2, avoidance_steering))

        # PID 转向控制
        total_steering_error = steering_error * 2.0 + avoidance_steering
        total_steering = _pid_update(total_steering_error, dt, 'steering')
        total_steering = np.clip(total_steering, -self.max_steering_angle, self.max_steering_angle)

        min_distance = min(sensor_data['front_distance'],
                           sensor_data['left_distance'],
                           sensor_data['right_distance'])

        if min_distance < 1.0:
            speed_multiplier = 0.3
        elif min_distance < 2.0:
            speed_multiplier = 0.5
        elif min_distance < 3.0:
            speed_multiplier = 0.75
        else:
            speed_multiplier = 1.0

        target_speed_adjusted = self.target_speed * speed_multiplier
        car_vel = self.data.body('car').cvel[:2]
        current_speed = np.linalg.norm(car_vel)

        # PID 速度控制
        speed_error = target_speed_adjusted - current_speed
        self.current_speed = _pid_update(speed_error, dt, 'speed')
        self.current_speed = np.clip(self.current_speed, 0.0, 1.0)

        self.path_history.append(car_pos.copy())
        if len(self.path_history) > 1000:
            self.path_history.pop(0)

        control = np.zeros(self.model.nu)

        if hasattr(self.model, 'nu') and self.model.nu >= 6:
            control[0] = self.current_speed
            control[1] = self.current_speed
            control[2] = self.current_speed
            control[3] = self.current_speed
            control[4] = total_steering
            control[5] = total_steering
        else:
            control[0] = self.current_speed
            if len(control) > 1:
                control[1] = total_steering

        return control

    def print_status(self):
        """打印状态信息"""
        car_pos = self.data.body('car').xpos
        try:
            target_pos = self.data.body('target').xpos
        except KeyError:
            target_pos = np.array([8, 0, 0.5])
        distance = np.linalg.norm(target_pos[:2] - car_pos[:2])

        mode_str = '手动' if self.manual_mode else '自动'
        collision_str = f"碰撞:{self.collision_count}" if self.collision_count > 0 else ""

        print(f"\r时间: {self.simulation_time:.1f}s | "
              f"模式: {mode_str} | "
              f"位置: ({car_pos[0]:.1f}, {car_pos[1]:.1f}) | "
              f"速度: {self.current_speed:.2f} | "
              f"转向: {math.degrees(self.steering_angle):.0f}° | "
              f"距目标: {distance:.1f}m | "
              f"{collision_str} | "
              f"状态: {'避障' if self.obstacle_detected else '导航'}",
              end="")

    def _key_callback(self, window, key, scancode, action, mods):
        """键盘回调函数"""
        if action == glfw.PRESS or action == glfw.REPEAT:
            self.keys[key] = True
        elif action == glfw.RELEASE:
            self.keys[key] = False

    def _manual_driving(self, dt):
        """手动驾驶控制"""
        car_pos = self.data.body('car').xpos.copy()
        car_orientation = self.data.body('car').xmat.reshape(3, 3)
        car_direction = car_orientation @ np.array([0, 1, 0])
        car_direction_2d = car_direction[:2]
        if np.linalg.norm(car_direction_2d) > 0:
            car_direction_2d = car_direction_2d / np.linalg.norm(car_direction_2d)

        throttle = 0.0
        steer = 0.0

        if self.keys.get(glfw.KEY_W, False):
            throttle = 0.8
        elif self.keys.get(glfw.KEY_S, False):
            throttle = -0.3

        if self.keys.get(glfw.KEY_A, False):
            steer = self.max_steering_angle * 1.2
        elif self.keys.get(glfw.KEY_D, False):
            steer = -self.max_steering_angle * 1.2

        if self.keys.get(glfw.KEY_E, False):
            throttle = min(throttle + 0.3, 1.0)

        car_vel = self.data.body('car').cvel[:2]
        current_speed_fwd = np.dot(car_vel, car_direction_2d)

        if throttle >= 0:
            self.current_speed = min(throttle, self.current_speed + 2.0 * dt)
        else:
            self.current_speed = max(0.0, self.current_speed + 4.0 * throttle * dt)

        self.steering_angle = steer
        self.path_history.append(car_pos.copy())
        if len(self.path_history) > 1000:
            self.path_history.pop(0)

        control = np.zeros(self.model.nu)
        if hasattr(self.model, 'nu') and self.model.nu >= 6:
            control[0] = self.current_speed
            control[1] = self.current_speed
            control[2] = self.current_speed
            control[3] = self.current_speed
            control[4] = steer
            control[5] = steer
        else:
            control[0] = self.current_speed
            if len(control) > 1:
                control[1] = steer

        return control

    def _update_obstacle_motion(self, sim_time):
        """更新动态障碍物位置"""
        for obs_name, motion in self.obstacle_motions.items():
            base = self.obstacle_base_positions[obs_name]
            mtype = motion['type']
            speed = motion['speed']
            phase = motion['phase']
            angle = sim_time * speed + phase

            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'{obs_name}_base')
                if body_id < 0:
                    continue

                if mtype == 'circle':
                    radius = motion['radius']
                    new_x = base[0] + radius * math.cos(angle)
                    new_y = base[1] + radius * math.sin(angle)
                    self.data.qpos[body_id * 7:body_id * 7 + 3] = [new_x, new_y, base[2]]

                elif mtype == 'linear_x':
                    amp = motion['amplitude']
                    new_x = base[0] + amp * math.sin(angle)
                    self.data.qpos[body_id * 7:body_id * 7 + 3] = [new_x, base[1], base[2]]

                elif mtype == 'linear_y':
                    amp = motion['amplitude']
                    new_y = base[1] + amp * math.sin(angle)
                    self.data.qpos[body_id * 7:body_id * 7 + 3] = [base[0], new_y, base[2]]

            except Exception:
                pass

    def _update_trail_rendering(self, scene):
        """更新路径轨迹渲染（使用 MuJoCo 场景几何体）"""
        if len(self.path_history) < 2:
            return

        step = max(1, len(self.path_history) // self.trail_max)
        trail_points = self.path_history[::step][-self.trail_max:]

        yellow = [1.0, 1.0, 0.0, 0.7]
        red = [1.0, 0.3, 0.0, 0.9]

        for i, pt in enumerate(trail_points):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size[:] = [0.04, 0, 0]
            geom.pos[:] = [pt[0], pt[1], 0.03]
            t = i / max(len(trail_points) - 1, 1)
            geom.rgba[:] = [
                red[0] * t + yellow[0] * (1 - t),
                red[1] * t + yellow[1] * (1 - t),
                red[2] * t + yellow[2] * (1 - t),
                0.5 + 0.3 * t
            ]
            scene.ngeom += 1

    def _check_collisions(self, dt):
        """检测碰撞并更新统计信息"""
        collision_geom_pairs = [
            ('chassis', 'obs1'),
            ('chassis', 'obs2'),
            ('chassis', 'obs3'),
            ('chassis', 'obs4'),
        ]

        has_collision = False
        max_force = 0.0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            if geom1_name is None or geom2_name is None:
                continue

            for g1, g2 in collision_geom_pairs:
                if (geom1_name == g1 and geom2_name == g2) or (geom1_name == g2 and geom2_name == g1):
                    has_collision = True
                    force_magnitude = 0.0
                    for j in range(6):
                        force_magnitude += abs(contact.frame[j])
                    max_force = max(max_force, force_magnitude)

        if has_collision and self._collision_cooldown <= 0:
            self.collision_count += 1
            self.collision_force_total += max_force
            self.collision_force_max = max(self.collision_force_max, max_force)
            self._collision_cooldown = 0.3

        if self._collision_cooldown > 0:
            self._collision_cooldown -= dt

    def run_simulation(self):
        """运行模拟主循环"""
        print("无人小车模拟系统启动中...")
        print("=" * 80)
        print("控制说明:")
        print("  M        - 切换 自动/手动 模式")
        print("  W/S      - 手动模式: 前进/后退")
        print("  A/D      - 手动模式: 左转/右转")
        print("  E        - 手动模式: 加速")
        print("  ESC      - 退出模拟")
        print("=" * 80)
        print("  - 绿色球体是目标点")
        print("  - 红色物体是动态障碍物")
        print("  - 小车会自动导航或手动驾驶")
        print("  - 黄色轨迹显示小车行驶路径")
        print("=" * 80)

        self.model.opt.gravity[2] = -9.81
        mujoco.mj_resetData(self.model, self.data)

        try:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as e:
            print(f"查看器启动失败: {e}")
            print("将以无界面模式运行模拟...")
            viewer = None

        if viewer is not None and hasattr(viewer, 'window'):
            try:
                glfw.set_key_callback(viewer.window, self._key_callback)
            except Exception:
                pass

        scene = mujoco.MjvScene(self.model, maxgeom=500)

        last_time = time.time()
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                if viewer is not None and not viewer.is_running():
                    break

                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                self.simulation_time += dt

                if dt > 0.1:
                    dt = 0.01

                # M 键切换 自动/手动 模式
                if self.keys.get(glfw.KEY_M, False):
                    if self.simulation_time - self._last_mode_toggle > 0.5:
                        self.manual_mode = not self.manual_mode
                        self._last_mode_toggle = self.simulation_time
                        mode_name = '手动驾驶' if self.manual_mode else '自动驾驶'
                        print(f"\n>>> 切换到{mode_name}模式 <<<")
                        if self.manual_mode:
                            self.speed_integral = 0.0
                            self.speed_prev_error = 0.0
                            self.steering_integral = 0.0
                            self.steering_prev_error = 0.0

                # 更新动态障碍物
                self._update_obstacle_motion(self.simulation_time)

                # 选择控制模式
                if self.manual_mode:
                    control = self._manual_driving(dt)
                else:
                    control = self.autonomous_driving(dt)

                self.data.ctrl[:] = control

                mujoco.mj_step(self.model, self.data)

                # 碰撞检测
                self._check_collisions(dt)

                if viewer is not None:
                    viewer.sync()

                # 轨迹渲染
                if viewer is not None and hasattr(viewer, 'opt'):
                    try:
                        mujoco.mjv_updateScene(self.model, self.data,
                                                getattr(viewer, 'vopt', mujoco.MjvOption()),
                                                None, mujoco.mjtCatBit.mjCAT_ALL, scene)
                        self._update_trail_rendering(scene)
                        viewport = mujoco.MjrRect(0, 0,
                                                   getattr(viewer, 'viewport', mujoco.MjrRect(0, 0, 800, 600)).width,
                                                   getattr(viewer, 'viewport', mujoco.MjrRect(0, 0, 800, 600)).height)
                        ctx = getattr(viewer, 'con', None) or getattr(viewer, 'context', None)
                        if ctx is not None:
                            mujoco.mjr_render(viewport, scene, ctx)
                    except Exception:
                        pass

                frame_count += 1
                if frame_count % 10 == 0:
                    self.print_status()

                if self.target_reached:
                    print(f"\n\n{'=' * 80}")
                    print("成功到达目标点！")
                    print(f"总时间: {self.simulation_time:.1f}秒")
                    print(f"路径点数: {len(self.path_history)}")
                    print(f"{'=' * 80}")
                    time.sleep(2)
                    break

                if viewer is not None:
                    try:
                        if not viewer.is_running():
                            break
                    except Exception:
                        if viewer is not None and not viewer.is_running():
                            break

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n用户中断模拟...")

        finally:
            if viewer is not None:
                viewer.close()

            avg_force = self.collision_force_total / max(self.collision_count, 1)
            print(f"\n\n{'=' * 80}")
            print("模拟统计:")
            print(f"  总模拟时间: {self.simulation_time:.1f}秒")
            print(f"  总帧数: {frame_count}")
            print(f"  平均帧率: {frame_count / max(time.time() - start_time, 0.001):.1f} FPS")
            print(f"  路径点数: {len(self.path_history)}")
            print(f"  碰撞次数: {self.collision_count}")
            if self.collision_count > 0:
                print(f"  最大碰撞力: {self.collision_force_max:.1f}")
                print(f"  平均碰撞力: {avg_force:.1f}")
            print(f"{'=' * 80}")


def main():
    """主函数"""
    print("正在初始化无人小车模拟系统...")
    print("=" * 80)

    try:
        # 检查必要的库
        import importlib
        required_libs = ['mujoco', 'numpy', 'glfw']
        missing_libs = []

        for lib in required_libs:
            try:
                importlib.import_module(lib)
            except ImportError:
                missing_libs.append(lib)

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