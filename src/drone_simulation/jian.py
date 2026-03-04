"""
MuJoCo 四旋翼无人机仿真 - 完整监测系统版
✅ 多传感器融合感知
✅ 动态路径规划
✅ 智能避障决策
✅ 实时无人机状态监测
✅ 电池/高度/速度/姿态监测
✅ 故障预警系统
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os
from collections import deque
from datetime import datetime


class DroneMonitor:
    """无人机状态监测系统"""

    def __init__(self):
        # 电池状态
        self.battery_level = 100.0  # 百分比
        self.battery_voltage = 12.6  # 满电电压 (3S LiPo)
        self.battery_current = 0.0  # 当前电流
        self.battery_temperature = 25.0  # 电池温度
        self.battery_consumption_rate = 0.1  # 每秒耗电百分比

        # 飞行状态
        self.flight_time = 0.0  # 总飞行时间
        self.max_altitude = 0.0  # 最大高度
        self.max_speed = 0.0  # 最大速度
        self.distance_traveled = 0.0  # 总飞行距离
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.current_speed = 0.0  # 当前速度

        # 电机状态
        self.motor_rpm = [0.0, 0.0, 0.0, 0.0]  # 电机转速
        self.motor_temperature = [25.0, 25.0, 25.0, 25.0]  # 电机温度
        self.motor_current = [0.0, 0.0, 0.0, 0.0]  # 电机电流
        self.motor_throttle = [0.0, 0.0, 0.0, 0.0]  # 油门百分比

        # 姿态状态
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0

        # GPS状态
        self.gps_satellites = 8  # GPS卫星数量
        self.gps_accuracy = 0.5  # GPS精度(米)
        self.gps_fix = True  # GPS是否锁定

        # 传感器状态
        self.imu_status = True  # IMU状态
        self.barometer_status = True  # 气压计状态
        self.compass_status = True  # 指南针状态
        self.optical_flow_status = True  # 光流传感器状态

        # 警告和故障
        self.warnings = deque(maxlen=10)  # 警告信息队列
        self.faults = deque(maxlen=5)  # 故障信息队列
        self.low_battery_warning = False
        self.high_temperature_warning = False
        self.gps_loss_warning = False

        # 历史数据记录
        self.position_history = deque(maxlen=1000)
        self.altitude_history = deque(maxlen=1000)
        self.speed_history = deque(maxlen=1000)
        self.battery_history = deque(maxlen=1000)
        self.time_history = deque(maxlen=1000)

        # 日志记录
        self.log_file = None
        self.logging_enabled = True
        self.init_logging()

        print("📊 无人机监测系统初始化完成")

    def init_logging(self):
        """初始化日志文件"""
        if self.logging_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drone_log_{timestamp}.csv"
            self.log_file = open(filename, 'w')
            # 写入CSV头
            self.log_file.write("时间,高度,速度,电池,滚转,俯仰,偏航,温度,警告\n")

    def update(self, data, dt):
        """更新监测数据"""
        # 更新飞行时间
        self.flight_time += dt

        # 更新电池
        self.battery_level -= self.battery_consumption_rate * dt
        self.battery_level = max(0.0, self.battery_level)
        self.battery_voltage = 12.6 * (self.battery_level / 100.0)

        # 更新位置和距离
        current_pos = data.qpos[0:3].copy()
        if np.linalg.norm(self.last_position) > 0:
            self.distance_traveled += np.linalg.norm(current_pos - self.last_position)
        self.last_position = current_pos.copy()

        # 更新最大高度
        self.max_altitude = max(self.max_altitude, current_pos[2])

        # 更新速度
        current_vel = data.qvel[0:3].copy()
        self.current_speed = np.linalg.norm(current_vel)
        self.max_speed = max(self.max_speed, self.current_speed)

        # 更新姿态
        quat = data.qpos[3:7].copy()
        self.update_attitude(quat)

        # 更新角速度
        if data.qvel.shape[0] > 3:
            self.roll_rate = data.qvel[3]
            self.pitch_rate = data.qvel[4]
            self.yaw_rate = data.qvel[5]

        # 更新电机状态
        if hasattr(data, 'ctrl') and len(data.ctrl) >= 4:
            for i in range(4):
                self.motor_throttle[i] = data.ctrl[i] / 1000.0 * 100.0
                self.motor_rpm[i] = self.motor_throttle[i] * 100  # 简化计算
                # 电机温度随负载上升
                self.motor_temperature[i] = 25.0 + self.motor_throttle[i] * 0.5

        # 检查警告条件
        self.check_warnings()

        # 记录历史数据
        self.record_history(current_pos[2], self.current_speed, self.battery_level)

        # 写入日志
        if self.logging_enabled and self.log_file:
            self.log_file.write(f"{self.flight_time:.2f},{current_pos[2]:.2f}," +
                                f"{self.current_speed:.2f},{self.battery_level:.1f}," +
                                f"{self.roll:.1f},{self.pitch:.1f},{self.yaw:.1f}," +
                                f"{self.battery_temperature:.1f},{len(self.warnings)}\n")
            self.log_file.flush()

    def update_attitude(self, quat):
        """从四元数更新姿态"""
        # 四元数转欧拉角
        w, x, y, z = quat

        # 滚转 (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        self.roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        # 俯仰 (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.pitch = math.degrees(math.copysign(math.pi / 2, sinp))
        else:
            self.pitch = math.degrees(math.asin(sinp))

        # 偏航 (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def check_warnings(self):
        """检查警告条件"""
        # 低电量警告
        if self.battery_level < 20.0 and not self.low_battery_warning:
            self.add_warning(f"⚠️ 低电量: {self.battery_level:.1f}%")
            self.low_battery_warning = True
        elif self.battery_level < 10.0:
            self.add_fault(f"🔥 严重低电量: {self.battery_level:.1f}%")

        # 高温警告
        if self.battery_temperature > 50.0:
            self.add_warning(f"🌡️ 电池高温: {self.battery_temperature:.1f}°C")

        # 电机高温警告
        for i, temp in enumerate(self.motor_temperature):
            if temp > 60.0:
                self.add_warning(f"🌡️ 电机{i + 1}高温: {temp:.1f}°C")

        # GPS丢失警告
        if not self.gps_fix:
            if not self.gps_loss_warning:
                self.add_warning("📡 GPS信号丢失")
                self.gps_loss_warning = True
        else:
            self.gps_loss_warning = False

        # 姿态异常警告
        if abs(self.roll) > 45 or abs(self.pitch) > 45:
            self.add_warning(f"⚠️ 姿态异常: 滚转{self.roll:.1f}° 俯仰{self.pitch:.1f}°")

    def add_warning(self, warning):
        """添加警告信息"""
        timestamp = time.strftime("%H:%M:%S")
        self.warnings.append(f"[{timestamp}] {warning}")
        print(f"\n📢 {warning}")

    def add_fault(self, fault):
        """添加故障信息"""
        timestamp = time.strftime("%H:%M:%S")
        self.faults.append(f"[{timestamp}] {fault}")
        print(f"\n🚨 {fault}")

    def record_history(self, altitude, speed, battery):
        """记录历史数据"""
        self.position_history.append(self.last_position.copy())
        self.altitude_history.append(altitude)
        self.speed_history.append(speed)
        self.battery_history.append(battery)
        self.time_history.append(self.flight_time)

    def get_status_summary(self):
        """获取状态摘要"""
        return {
            'battery_level': self.battery_level,
            'battery': f"{self.battery_level:.1f}% ({self.battery_voltage:.1f}V)",
            'flight_time': f"{self.flight_time:.0f}s",
            'altitude': f"{self.last_position[2]:.1f}m",
            'max_altitude': f"{self.max_altitude:.1f}m",
            'speed': f"{self.current_speed:.1f}m/s",
            'max_speed': f"{self.max_speed:.1f}m/s",
            'distance': f"{self.distance_traveled:.1f}m",
            'attitude': f"R:{self.roll:.0f}° P:{self.pitch:.0f}° Y:{self.yaw:.0f}°",
            'gps': f"{self.gps_satellites} sat {'✅' if self.gps_fix else '❌'}",
            'temperature': f"{self.battery_temperature:.1f}°C",
            'warnings': len(self.warnings),
            'faults': len(self.faults)
        }

    def print_status(self):
        """打印状态信息"""
        status = self.get_status_summary()

        print("\n" + "=" * 60)
        print("📊 无人机状态监测系统")
        print("=" * 60)

        # 电池状态
        battery_color = "🟢" if status['battery_level'] > 50 else "🟡" if status['battery_level'] > 20 else "🔴"
        print(f"{battery_color} 电池: {status['battery']} | 温度: {status['temperature']}")

        # 飞行状态
        print(f"✈️ 飞行时间: {status['flight_time']} | 高度: {status['altitude']} (最大{status['max_altitude']})")
        print(f"💨 速度: {status['speed']} (最大{status['max_speed']}) | 距离: {status['distance']}")

        # 姿态
        print(f"🧭 姿态: {status['attitude']}")

        # GPS
        print(f"📡 GPS: {status['gps']}")

        # 警告和故障
        if status['warnings'] > 0 or status['faults'] > 0:
            print(f"\n⚠️ 警告: {status['warnings']} | 🚨 故障: {status['faults']}")

            if self.warnings:
                print("\n最新警告:")
                for warning in list(self.warnings)[-3:]:
                    print(f"  {warning}")

            if self.faults:
                print("\n最新故障:")
                for fault in list(self.faults)[-3:]:
                    print(f"  {fault}")

        print("=" * 60)

    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()
            print(f"📁 日志已保存")


class SensorSystem:
    """无人机感知系统"""

    def __init__(self, drone_pos, obstacle_positions, obstacle_sizes):
        self.drone_pos = drone_pos
        self.obstacle_positions = obstacle_positions
        self.obstacle_sizes = obstacle_sizes

        # 传感器参数
        self.lidar_range = 6.0  # 激光雷达探测距离
        self.lidar_resolution = 36  # 扫描分辨率（每10度一个点）
        self.ultrasonic_range = 3.0  # 超声波传感器距离
        self.camera_fov = 60  # 相机视场角（度）

        # 传感器数据
        self.lidar_points = []  # 激光雷达点云
        self.obstacle_directions = []  # 障碍物方向
        self.danger_zones = []  # 危险区域
        self.safe_directions = []  # 安全方向
        self.closest_obstacle = None  # 最近障碍物
        self.min_distance = float('inf')  # 最近距离

        # 历史数据
        self.history = deque(maxlen=10)

    def update(self, drone_pos):
        """更新传感器数据"""
        self.drone_pos = drone_pos
        self.lidar_points = []
        self.obstacle_directions = []
        self.danger_zones = []
        self.min_distance = float('inf')
        self.closest_obstacle = None

        # 激光雷达扫描
        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            direction = np.array([math.cos(rad), math.sin(rad), 0])

            hit_point, hit_dist, hit_obs = self.ray_cast(drone_pos, direction)
            if hit_point is not None:
                self.lidar_points.append({
                    'angle': angle,
                    'distance': hit_dist,
                    'point': hit_point,
                    'obstacle': hit_obs
                })

                if hit_dist < self.min_distance:
                    self.min_distance = hit_dist
                    self.closest_obstacle = hit_obs

                safety_dist = self.safety_distance()
                if hit_dist < safety_dist:
                    self.danger_zones.append({
                        'angle': angle,
                        'distance': hit_dist,
                        'obstacle': hit_obs
                    })

        self.calculate_safe_directions()

        self.history.append({
            'pos': drone_pos.copy(),
            'danger_zones': self.danger_zones.copy()
        })

    def ray_cast(self, start, direction, max_dist=6.0):
        """射线投射检测障碍物"""
        min_dist = max_dist
        hit_point = None
        hit_obs = None

        for obs_name, obs_pos in self.obstacle_positions.items():
            to_obs = obs_pos - start
            obs_size = self.obstacle_sizes.get(obs_name, 0.5)

            proj_dist = np.dot(to_obs, direction)
            if proj_dist < 0 or proj_dist > max_dist:
                continue

            perp_dist = np.linalg.norm(to_obs - proj_dist * direction)

            if perp_dist < obs_size:
                if proj_dist < min_dist:
                    min_dist = proj_dist
                    hit_point = start + direction * proj_dist
                    hit_obs = obs_name

        return hit_point, min_dist, hit_obs

    def safety_distance(self):
        """动态安全距离"""
        base_dist = 1.2
        if len(self.history) > 1:
            last_pos = self.history[-2]['pos']
            velocity = np.linalg.norm(self.drone_pos - last_pos) / 0.01
            return base_dist + velocity * 0.3
        return base_dist

    def calculate_safe_directions(self):
        """计算安全飞行方向"""
        self.safe_directions = []

        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            direction = np.array([math.cos(rad), math.sin(rad), 0])

            is_safe = True

            for danger in self.danger_zones:
                angle_diff = abs(angle - danger['angle'])
                if angle_diff < 30 and danger['distance'] < self.safety_distance() * 1.5:
                    is_safe = False
                    break

            if is_safe:
                self.safe_directions.append({
                    'angle': angle,
                    'direction': direction
                })

    def get_best_direction(self, target_direction):
        """获取最佳飞行方向"""
        if not self.safe_directions:
            return None, 0

        best_dir = None
        best_angle = 0
        best_score = -1

        for safe in self.safe_directions:
            direction = safe['direction']
            similarity = np.dot(direction, target_direction)
            score = (similarity + 1) / 2

            if score > best_score:
                best_score = score
                best_dir = direction
                best_angle = safe['angle']

        return best_dir, best_angle

    def get_avoidance_force(self, target_direction):
        """计算避障力"""
        if not self.danger_zones:
            return np.zeros(3), 1.0

        avoidance = np.zeros(3)

        for danger in self.danger_zones:
            angle = danger['angle']
            rad = math.radians(angle)
            danger_dir = np.array([math.cos(rad), math.sin(rad), 0])
            avoid_dir = -danger_dir

            dist = danger['distance']
            safety_dist = self.safety_distance()
            strength = 1.0 - min(dist / safety_dist, 1.0)
            strength = strength ** 2

            avoidance += avoid_dir * strength

        if np.linalg.norm(avoidance) > 0:
            avoidance = avoidance / np.linalg.norm(avoidance)

        safety_factor = max(0, 1.0 - len(self.danger_zones) * 0.1)
        safety_factor = max(0.2, min(1.0, safety_factor))

        return avoidance, safety_factor


class PathPlanner:
    """路径规划器"""

    def __init__(self):
        self.waypoints = []
        self.current_waypoint = 0
        self.path_history = []

        self.default_waypoints = [
            np.array([0.0, 0.0, 2.0]),
            np.array([4.0, 4.0, 2.0]),
            np.array([4.0, -4.0, 2.0]),
            np.array([-4.0, -4.0, 2.0]),
            np.array([-4.0, 4.0, 2.0]),
            np.array([0.0, 0.0, 2.0]),
            np.array([6.0, 0.0, 2.0]),
            np.array([0.0, 6.0, 2.0]),
            np.array([-6.0, 0.0, 2.0]),
            np.array([0.0, -6.0, 2.0]),
        ]

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint = 0

    def get_next_waypoint(self, current_pos):
        """获取下一个航点"""
        if not self.waypoints:
            self.waypoints = self.default_waypoints

        target = self.waypoints[self.current_waypoint]

        dist = np.linalg.norm(current_pos[:2] - target[:2])
        if dist < 2.0:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            target = self.waypoints[self.current_waypoint]

        return target


class QuadrotorSimulation:
    def __init__(self, xml_path="quadrotor_detailed_city.xml"):
        """初始化：从XML文件加载模型"""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到XML文件: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ 模型加载成功: {xml_path}")
        self.data = mujoco.MjData(self.model)
        self.n_actuators = self.model.nu

        # 基础推力
        self.base_thrust = 600
        if self.n_actuators > 0:
            self.data.ctrl[:] = [self.base_thrust] * self.n_actuators

        # ========== 飞行阶段 ==========
        self.flight_phase = "takeoff"
        self.phase_start_time = 0.0
        self.takeoff_height = 2.0
        self.cruise_height = 2.0
        self.landing_height = 0.2

        # ========== 运动参数 ==========
        self.max_speed = 2.0
        self.acceleration = 1.0
        self.current_velocity = np.zeros(3)
        self.target_pos = np.array([0, 0, self.takeoff_height])

        # ========== 感知系统 ==========
        self.obstacle_positions = {
            "building_office": np.array([5.0, 5.0, 1.0]),
            "building_tower": np.array([8.0, 4.0, 1.5]),
            "building_apartment": np.array([3.0, 8.0, 1.2]),
            "building_shop": np.array([-5.0, 5.0, 1.0]),
            "building_cafe": np.array([-8.0, 4.0, 1.0]),
            "building_house1": np.array([-5.0, -5.0, 0.8]),
            "building_house2": np.array([-8.0, -5.0, 0.8]),
            "building_school": np.array([5.0, -5.0, 1.2]),
            "building_library": np.array([8.0, -5.0, 1.0]),
            "tree_1": np.array([2.0, 2.0, 0.8]),
            "tree_2": np.array([-2.0, 2.0, 0.8]),
            "tree_3": np.array([2.0, -2.0, 0.8]),
            "tree_4": np.array([-2.0, -2.0, 0.8]),
            "light_1": np.array([3.0, 3.0, 0.6]),
            "light_2": np.array([-3.0, 3.0, 0.6]),
            "light_3": np.array([3.0, -3.0, 0.6]),
            "light_4": np.array([-3.0, -3.0, 0.6]),
            "car_1": np.array([2.0, 0.0, 0.3]),
            "car_2": np.array([-2.0, 0.0, 0.3])
        }
        self.obstacle_sizes = {
            "building_office": 1.5,
            "building_tower": 1.0,
            "building_apartment": 1.2,
            "building_shop": 1.2,
            "building_cafe": 1.0,
            "building_house1": 0.8,
            "building_house2": 0.8,
            "building_school": 1.2,
            "building_library": 1.0,
            "tree_1": 0.5,
            "tree_2": 0.5,
            "tree_3": 0.5,
            "tree_4": 0.5,
            "light_1": 0.3,
            "light_2": 0.3,
            "light_3": 0.3,
            "light_4": 0.3,
            "car_1": 0.5,
            "car_2": 0.5
        }

        # 初始化感知系统
        self.sensor_system = SensorSystem(
            self.data.qpos[0:3].copy(),
            self.obstacle_positions,
            self.obstacle_sizes
        )

        # 初始化路径规划器
        self.path_planner = PathPlanner()

        # 初始化无人机监测系统
        self.drone_monitor = DroneMonitor()

        # 状态记录
        self.avoidance_count = 0
        self.last_avoidance_time = 0

        # 飞行边界
        self.bounds = {'x': [-10, 10], 'y': [-10, 10], 'z': [0.2, 4.0]}

        # 状态显示间隔
        self.last_status_time = 0
        self.status_interval = 5.0  # 每5秒显示详细状态

    def update_flight_phase(self, current_time):
        """更新飞行阶段"""
        if self.flight_phase == "takeoff":
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * 0.5, 1.0)
            current_height = 0.2 + (self.takeoff_height - 0.2) * progress

            if progress >= 1.0:
                self.flight_phase = "cruise"
                self.phase_start_time = current_time
                print("\n🚁 起飞完成，开始巡航")

            return current_height

        elif self.flight_phase == "cruise":
            return self.cruise_height

        else:  # landing
            elapsed = current_time - self.phase_start_time
            progress = min(elapsed * 0.3, 1.0)
            current_height = self.cruise_height - (self.cruise_height - 0.2) * progress

            return current_height

    def calculate_safe_movement(self, current_pos, target_pos):
        """计算安全移动方向"""
        # 更新感知系统
        self.sensor_system.update(current_pos)

        to_target = target_pos - current_pos
        dist_to_target = np.linalg.norm(to_target)

        if dist_to_target < 0.1:
            return current_pos

        target_dir = to_target / dist_to_target

        avoidance_force, safety_factor = self.sensor_system.get_avoidance_force(target_dir)
        best_dir, best_angle = self.sensor_system.get_best_direction(target_dir)

        if safety_factor < 0.3:
            if np.linalg.norm(avoidance_force) > 0:
                move_dir = avoidance_force
                self.avoidance_count += 1
                if time.time() - self.last_avoidance_time > 2.0:
                    self.drone_monitor.add_warning("紧急避障")
                    self.last_avoidance_time = time.time()
            elif best_dir is not None:
                move_dir = best_dir
            else:
                move_dir = np.array([0, 0, 1])
        elif safety_factor < 0.7:
            if best_dir is not None:
                mix_weight = 0.4 + safety_factor * 0.3
                move_dir = target_dir * mix_weight + best_dir * (1 - mix_weight)
                move_dir = move_dir / np.linalg.norm(move_dir)
            else:
                move_dir = target_dir
        else:
            move_dir = target_dir

        target_velocity = move_dir * self.max_speed * (0.8 + safety_factor * 0.4)
        self.current_velocity += (target_velocity - self.current_velocity) * 0.15

        new_pos = current_pos + self.current_velocity * self.model.opt.timestep * 50

        new_pos[0] = np.clip(new_pos[0], self.bounds['x'][0], self.bounds['x'][1])
        new_pos[1] = np.clip(new_pos[1], self.bounds['y'][0], self.bounds['y'][1])
        new_pos[2] = np.clip(new_pos[2], self.bounds['z'][0], self.bounds['z'][1])

        return new_pos

    def get_next_target(self, current_pos):
        """获取下一个目标点"""
        if self.flight_phase == "takeoff":
            return np.array([0, 0, self.takeoff_height])
        elif self.flight_phase == "landing":
            return np.array([0, 0, 0.2])
        else:
            target = self.path_planner.get_next_waypoint(current_pos)
            target[2] = self.cruise_height
            return target

    def simulation_loop(self, viewer, duration):
        """主仿真循环"""
        start_time = time.time()
        last_print_time = time.time()
        last_monitor_update = time.time()
        self.phase_start_time = 0.0

        smooth_pos = np.array([0, 0, 0.2])

        while (viewer is None or (viewer and viewer.is_running())) and (time.time() - start_time) < duration:
            step_start = time.time()
            current_time = self.data.time
            wall_time = time.time()

            # 物理仿真步进
            mujoco.mj_step(self.model, self.data)

            # 获取当前位置
            current_pos = self.data.qpos[0:3].copy()

            # 更新飞行高度
            target_height = self.update_flight_phase(current_time)

            # 获取下一个目标点
            next_target = self.get_next_target(current_pos)
            next_target[2] = target_height

            # 计算安全移动
            new_pos = self.calculate_safe_movement(current_pos, next_target)

            # 更新监测系统（每秒10次）
            if wall_time - last_monitor_update > 0.1:
                self.drone_monitor.update(self.data, self.model.opt.timestep)
                last_monitor_update = wall_time

            # 平滑移动
            smooth_pos = smooth_pos + (new_pos - smooth_pos) * 0.3

            # 设置无人机位置
            self.data.qpos[0] = smooth_pos[0]
            self.data.qpos[1] = smooth_pos[1]
            self.data.qpos[2] = smooth_pos[2]

            # 设置无人机姿态
            if self.model.nq > 3:
                speed = np.linalg.norm(self.current_velocity)
                tilt = min(0.3, speed * 0.1)
                self.data.qpos[3] = math.cos(tilt / 2)
                self.data.qpos[4] = 0.0
                self.data.qpos[5] = math.sin(tilt / 2)
                self.data.qpos[6] = 0.0

            # 旋翼旋转视觉效果
            for i in range(4):
                if 7 + i < self.model.nq:
                    self.data.qpos[7 + i] += 30.0 * self.model.opt.timestep

            if viewer:
                viewer.sync()

            # 定期打印状态
            if wall_time - last_print_time > 1.0:
                # 获取感知数据
                danger_count = len(self.sensor_system.danger_zones)
                safe_dir_count = len(self.sensor_system.safe_directions)
                min_dist = self.sensor_system.min_distance
                closest = self.sensor_system.closest_obstacle

                _, safety_factor = self.sensor_system.get_avoidance_force(
                    next_target - current_pos
                )

                phase_names = {
                    "takeoff": "🔼 起飞",
                    "cruise": "✈️ 巡航",
                    "landing": "🔽 降落"
                }

                if safety_factor > 0.8:
                    safety_icon = "✅"
                elif safety_factor > 0.4:
                    safety_icon = "⚠️"
                else:
                    safety_icon = "🚨"

                # 获取监测数据
                monitor = self.drone_monitor.get_status_summary()

                print(f"\n{'=' * 80}")
                print(
                    f"时间: {current_time:.1f}s | 阶段: {phase_names[self.flight_phase]} | 航点: {self.path_planner.current_waypoint + 1}")
                print(f"位置: ({smooth_pos[0]:.2f}, {smooth_pos[1]:.2f}, {smooth_pos[2]:.2f})")
                print(f"速度: {np.linalg.norm(self.current_velocity):.2f} m/s")

                print(f"\n【感知系统】{safety_icon} 安全系数: {safety_factor:.2f}")
                print(f"  危险区域: {danger_count}处 | 安全方向: {safe_dir_count}个")
                print(f"  最近障碍: {closest} 距离 {min_dist:.2f}m | 避障次数: {self.avoidance_count}")

                print(f"\n【监测系统】")
                print(f"  🔋 电池: {monitor['battery']} | 🌡️ 温度: {monitor['temperature']}")
                print(
                    f"  ⏱️ 飞行时间: {monitor['flight_time']} | 📏 高度: {monitor['altitude']} (最大{monitor['max_altitude']})")
                print(f"  💨 速度: {monitor['speed']} (最大{monitor['max_speed']}) | 📍 距离: {monitor['distance']}")
                print(f"  🧭 姿态: {monitor['attitude']} | 📡 GPS: {monitor['gps']}")

                if monitor['warnings'] > 0:
                    print(f"  ⚠️ 警告: {monitor['warnings']}条")
                if monitor['faults'] > 0:
                    print(f"  🚨 故障: {monitor['faults']}条")

                print(f"{'=' * 80}")

                last_print_time = wall_time

            # 每30秒显示详细状态
            if wall_time - self.last_status_time > self.status_interval:
                self.drone_monitor.print_status()
                self.last_status_time = wall_time

            # 控制仿真速率
            elapsed = time.time() - step_start
            sleep_time = self.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_simulation(self, duration=120.0, use_viewer=True):
        """运行仿真"""
        print(f"\n{'🚁' * 10} 无人机完整监测系统 {'🚁' * 10}")
        print(f"▶ 激光雷达范围: {self.sensor_system.lidar_range}m")
        print(f"▶ 最大速度: {self.max_speed}m/s")
        print(f"▶ 障碍物数量: {len(self.obstacle_positions)}")
        print(f"▶ 飞行边界: X[-10,10] Y[-10,10] Z[0.2,4.0]")
        print(f"▶ 监测系统: 电池/姿态/GPS/温度")
        print(f"{'=' * 80}")

        try:
            if use_viewer:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = 30
                    viewer.cam.distance = 18.0
                    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

                    print("\n🔼 无人机开始起飞...")
                    print("感知系统已激活，正在扫描环境...")
                    print("监测系统已激活，实时监控无人机状态...")
                    self.simulation_loop(viewer, duration)
            else:
                self.simulation_loop(None, duration)
        except Exception as e:
            print(f"⚠ 仿真错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.drone_monitor.close()

        print(f"\n{'✅' * 10} 仿真结束 {'✅' * 10}")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 完整监测系统版")
    print("=" * 80)

    try:
        xml_path = "quadrotor_detailed_city.xml"
        sim = QuadrotorSimulation(xml_path)

        # ========== 可调参数 ==========
        sim.max_speed = 2.2
        sim.takeoff_height = 2.0
        sim.cruise_height = 2.0
        sim.sensor_system.lidar_range = 6.0

        print("✅ 初始化完成")
        print("▶ 所有系统已就绪")
        sim.run_simulation(duration=120.0, use_viewer=True)

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("请确保 quadrotor_detailed_city.xml 文件在同一目录下")
    except KeyboardInterrupt:
        print("\n\n⏹ 仿真被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()