import carla  # 导入Carla客户端库，用于与Carla模拟器交互
import numpy as np  # 导入numpy库，用于数值计算

# 从配置文件导入车辆控制相关的常量
from src.config import MAX_BRAKING_M_S_2, MAX_WHEEL_ANGLE_RAD, MAX_ACCELERATION_M_S_2


class CarlaSimulator:
    """Carla模拟器交互类，封装了与Carla模拟器的主要交互操作"""

    def __init__(self):
        """初始化Carla模拟器连接及相关参数"""
        # 连接到本地Carla服务器（默认地址localhost:2000）
        self.client = carla.Client('localhost', 2000)
        # 设置客户端超时时间（10秒）
        self.client.set_timeout(10.0)
        # 获取当前世界对象
        self.world = self.client.get_world()
        # 初始化主车辆（ego vehicle）为None
        self.ego_vehicle = None
        # 销毁场景中所有现有车辆（初始化环境）
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()

    def load_world(self, map_name):
        """
        加载指定名称的Carla地图

        参数:
            map_name: 地图名称（如'Town01'）
        """
        self.client.load_world(map_name)

    def spawn_ego_vehicle(self, vehicle_name, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        """
        在指定位置生成主车辆（ego vehicle）

        参数:
            vehicle_name: 车辆蓝图名称（用于筛选车辆模型）
            x, y, z: 生成位置的坐标（米）
            pitch, yaw, roll: 生成时的旋转角度（度），分别对应俯仰角、偏航角、横滚角
        """
        # 获取当前世界的蓝图库
        blueprint_library = self.world.get_blueprint_library()
        # 根据车辆名称筛选蓝图（取第一个匹配结果）
        vehicle_bp = blueprint_library.filter(vehicle_name)[0]
        # 设置生成位置
        spawn_location = carla.Location(x, y, z)
        # 设置生成旋转角度
        spawn_rotation = carla.Rotation(pitch, yaw, roll)
        # 组合位置和旋转为变换矩阵
        spawn_transform = carla.Transform(location=spawn_location, rotation=spawn_rotation)
        # 在指定位置生成车辆并赋值给主车辆
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)

    def set_spectator(self, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        """
        设置 spectator（ spectator 是Carla中的视角控制者）的位置和角度

        参数:
            x, y, z: spectator 位置坐标（米）
            pitch, yaw, roll: spectator 旋转角度（度）
        """
        # 获取当前spectator
        spectator = self.world.get_spectator()
        # 设置位置
        location = carla.Location(x=x, y=y, z=z)
        # 设置旋转角度
        rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        # 组合为变换矩阵
        spectator_transform = carla.Transform(location, rotation)
        # 应用变换到spectator
        spectator.set_transform(spectator_transform)

    def clean(self):
        """清理场景中所有车辆（销毁所有车辆actor）"""
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            vehicle.destroy()

    def draw_perception_planning(self, x_ref, y_ref, current_idx, look_ahead_points=20):
        """
        绘制感知规划信息：只显示车辆前方需要跟踪的轨迹点

        参数:
            x_ref: 参考轨迹x坐标
            y_ref: 参考轨迹y坐标
            current_idx: 当前轨迹点索引
            look_ahead_points: 向前看的点数
        """
        # 清除之前的绘制（通过绘制空列表）
        self.draw_trajectory([], [], life_time=0.1)

        # 计算要显示的轨迹点范围
        total_points = len(x_ref)
        if total_points == 0:
            return

        # 获取前方轨迹点
        end_idx = min(current_idx + look_ahead_points, total_points)

        # 提取要显示的轨迹点
        if current_idx < end_idx:
            display_x = x_ref[current_idx:end_idx]
            display_y = y_ref[current_idx:end_idx]
        else:
            # 处理循环轨迹
            display_x = list(x_ref[current_idx:]) + list(x_ref[:end_idx])
            display_y = list(y_ref[current_idx:]) + list(y_ref[:end_idx])

        # 绘制前方轨迹（绿色）
        if len(display_x) > 1:
            self.draw_trajectory(
                display_x,
                display_y,
                height=0.5,
                thickness=0.15,
                green=255,
                life_time=0.1
            )

        # 绘制当前目标点（红色）
        if len(display_x) > 0:
            target_location = carla.Location(
                x=display_x[0],
                y=display_y[0],
                z=0.5
            )
            self.world.debug.draw_point(
                target_location,
                size=0.3,
                color=carla.Color(255, 0, 0),
                life_time=0.1
            )

            # 绘制从车辆到目标点的连线
            vehicle_location = self.ego_vehicle.get_transform().location
            self.world.debug.draw_line(
                vehicle_location,
                target_location,
                thickness=0.1,
                color=carla.Color(0, 255, 255),
                life_time=0.1
            )

    def draw_frenet_frame(self, x_ref, y_ref, current_idx):
        """
        绘制Frenet坐标系参考线（用于显示横向偏差）
        """
        total_points = len(x_ref)
        if total_idx + 1 >= total_points:
            return

        # 获取当前参考线段
        x1, y1 = x_ref[current_idx], y_ref[current_idx]
        x2, y2 = x_ref[(current_idx + 1) % total_points], y_ref[(current_idx + 1) % total_points]

        # 绘制参考线段（蓝色）
        start_point = carla.Location(x=x1, y=y1, z=0.3)
        end_point = carla.Location(x=x2, y=y2, z=0.3)

        self.world.debug.draw_line(
            start_point,
            end_point,
            thickness=0.08,
            color=carla.Color(0, 0, 255),
            life_time=0.1
        )
    def draw_trajectory(self, x_traj, y_traj, height=0, thickness=0.1, red=0, green=0, blue=0, life_time=0.1):
        """
        在世界中绘制轨迹线

        参数:
            x_traj: 轨迹点的x坐标列表
            y_traj: 轨迹点的y坐标列表
            height: 轨迹线的z轴高度（米）
            thickness: 线的粗细
            red, green, blue: 线的颜色（0-255）
            life_time: 线在世界中的存在时间（秒）
        """
        # 遍历轨迹点，绘制连续线段
        for i in range(len(x_traj) - 1):
            # 线段起点
            start_point = carla.Location(x=x_traj[i], y=y_traj[i], z=height)
            # 线段终点
            end_point = carla.Location(x=x_traj[i + 1], y=y_traj[i + 1], z=height)
            # 在世界中绘制线段
            self.world.debug.draw_line(
                start_point,
                end_point,
                thickness=thickness,
                color=carla.Color(red, green, blue),
                life_time=life_time
            )

    def get_main_ego_vehicle_state(self):
        """
        获取主车辆的当前状态

        返回:
            x: 车辆位置x坐标（米）
            y: 车辆位置y坐标（米）
            theta: 车辆航向角（弧度，从yaw角度转换而来）
            v: 车辆瞬时速度（米/秒，x和y方向速度的合速度）
        """
        # 获取车辆变换信息（位置和旋转）
        transform = self.ego_vehicle.get_transform()
        x = transform.location.x
        y = transform.location.y
        # 将yaw角度（度）转换为弧度（航向角）
        theta = np.deg2rad(transform.rotation.yaw)
        # 计算x和y方向速度的合速度（标量）
        v = np.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2)
        return x, y, theta, v

    def apply_control(self, steer, throttle, brake):
        """
        向主车辆应用控制指令

        参数:
            steer: 转向指令（-1到1之间，对应左右转向）
            throttle: 油门指令（0到1之间）
            brake: 刹车指令（0到1之间）
        """
        # 调用Carla的VehicleControl接口应用控制
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    @staticmethod
    def process_control_inputs(wheel_angle_rad, acceleration_m_s_2):
        """
        处理原始控制输入（轮角和加速度），转换为Carla需要的标准化控制指令

        参数:
            wheel_angle_rad: 车轮转角（弧度）
            acceleration_m_s_2: 加速度指令（米/秒²，正为加速，负为刹车）

        返回:
            throttle: 标准化油门指令（0到1）
            brake: 标准化刹车指令（0到1）
            steer: 标准化转向指令（-1到1）
        """
        if acceleration_m_s_2 == 0:
            # 无加速度指令时，油门和刹车均为0
            throttle = 0
            brake = 0
        elif acceleration_m_s_2 < 0:
            # 负加速度（刹车），标准化到刹车指令（除以最大刹车加速度）
            throttle = 0
            brake = acceleration_m_s_2 / MAX_BRAKING_M_S_2
        else:
            # 正加速度（加速），标准化到油门指令（除以最大加速度）
            throttle = acceleration_m_s_2 / MAX_ACCELERATION_M_S_2
            brake = 0
        # 车轮转角标准化（除以最大轮角）
        steer = wheel_angle_rad / MAX_WHEEL_ANGLE_RAD
        return throttle, brake, steer

    def print_ego_vehicle_characteristics(self):
        """打印主车辆的物理特性参数（如车轮信息、扭矩曲线、质量等）"""
        if not self.ego_vehicle:
            print("Vehicle not spawned yet!")  # 车辆未生成时提示
            return None

        # 获取车辆物理控制参数
        physics_control = self.ego_vehicle.get_physics_control()

        print("Vehicle Physics Information.\n")

        # 打印车轮信息
        print("Wheel Information:")
        for i, wheel in enumerate(physics_control.wheels):
            print(f" Wheel {i + 1}:")
            print(f"   Tire Friction: {wheel.tire_friction}")  # 轮胎摩擦系数
            print(f"   Damping Rate: {wheel.damping_rate}")  # 阻尼率
            print(f"   Max Steer Angle: {wheel.max_steer_angle}")  # 最大转向角
            print(f"   Radius: {wheel.radius}")  # 车轮半径
            print(f"   Max Brake Torque: {wheel.max_brake_torque}")  # 最大刹车扭矩
            print(f"   Max Handbrake Torque: {wheel.max_handbrake_torque}")  # 最大手刹扭矩
            print(f"   Position (x, y, z): ({wheel.position.x}, {wheel.position.y}, {wheel.position.z})")  # 车轮位置

        # 打印扭矩曲线（RPM-扭矩关系）
        print(f" Torque Curve:")
        for point in physics_control.torque_curve:
            print(f"RPM: {point.x}, Torque: {point.y}")
        print(f" Max RPM: {physics_control.max_rpm}")  # 最大转速
        print(f" MOI (Moment of Inertia): {physics_control.moi}")  # 转动惯量
        print(f" Damping Rate Full Throttle: {physics_control.damping_rate_full_throttle}")  # 全油门时的阻尼率
        print(
            f" Damping Rate Zero Throttle Clutch Engaged: {physics_control.damping_rate_zero_throttle_clutch_engaged}")  # 零油门离合器结合时的阻尼率
        print(
            f" Damping Rate Zero Throttle Clutch Disengaged: {physics_control.damping_rate_zero_throttle_clutch_disengaged}")  # 零油门离合器分离时的阻尼率
        print(
            f" If True, the vehicle will have an automatic transmission: {physics_control.use_gear_autobox}")  # 是否自动变速箱
        print(f" Gear Switch Time: {physics_control.gear_switch_time}")  # 换挡时间
        print(f" Clutch Strength: {physics_control.clutch_strength}")  # 离合器强度
        print(f" Final Ratio: {physics_control.final_ratio}")  # 最终传动比
        print(f" Mass: {physics_control.mass}")  # 车辆质量
        print(f" Drag coefficient: {physics_control.drag_coefficient}")  # 空气阻力系数
        # 打印转向曲线（速度-转向关系）
        print(f" Steering Curve:")
        for point in physics_control.steering_curve:
            print(f"Speed: {point.x}, Steering: {point.y}")