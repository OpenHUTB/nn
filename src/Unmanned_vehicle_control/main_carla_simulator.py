import carla
import numpy as np
from src.config import MAX_BRAKING_M_S_2, MAX_WHEEL_ANGLE_RAD, MAX_ACCELERATION_M_S_2


class CarlaSimulator:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.ego_vehicle = None
        self.obstacles = []  # 存储障碍物对象
        self.destroy_all_actors()

        # 障碍物颜色
        self.obstacle_colors = {
            'vehicle.audi.tt': (255, 0, 0, 200),  # 红色半透明
            'vehicle.tesla.model3': (255, 165, 0, 200),  # 橙色半透明
            'static.prop.trafficcone': (255, 255, 0, 200),  # 黄色半透明
            'static.prop.streetbarrier': (128, 0, 128, 200),  # 紫色半透明
        }

    def destroy_all_actors(self):
        """销毁所有现有actor（车辆、障碍物等）"""
        for actor in self.world.get_actors():
            if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('static.prop.'):
                actor.destroy()
        self.obstacles = []

    def load_world(self, map_name):
        self.client.load_world(map_name)

    def spawn_ego_vehicle(self, vehicle_name, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_name)[0]
        spawn_location = carla.Location(x, y, z)
        spawn_rotation = carla.Rotation(pitch, yaw, roll)
        spawn_transform = carla.Transform(location=spawn_location, rotation=spawn_rotation)
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
        return self.ego_vehicle

    def spawn_obstacle(self, obstacle_type='vehicle.tesla.model3', x=0, y=0, z=0,
                       pitch=0, yaw=0, roll=0, static=True, color=None):
        """
        在指定位置生成障碍物

        参数:
            obstacle_type: 障碍物类型 ('vehicle.*', 'static.prop.*', 或 'walker.*')
            x, y, z: 位置坐标
            pitch, yaw, roll: 旋转角度
            static: 是否静态障碍物
            color: 自定义颜色 (R,G,B,A)
        """
        try:
            blueprint_library = self.world.get_blueprint_library()

            # 根据类型获取蓝图
            if obstacle_type.startswith('vehicle.'):
                obstacle_bp = blueprint_library.filter(obstacle_type)[0]
                # 设置车辆为静止
                if static:
                    obstacle_bp.set_attribute('role_name', 'static_obstacle')
            elif obstacle_type.startswith('static.prop.'):
                obstacle_bp = blueprint_library.filter(obstacle_type)[0]
            elif obstacle_type.startswith('walker.'):
                obstacle_bp = blueprint_library.filter(obstacle_type)[0]
            else:
                print(f"Unknown obstacle type: {obstacle_type}")
                return None

            # 设置障碍物颜色（如果提供）
            if color:
                if obstacle_bp.has_attribute('color'):
                    obstacle_bp.set_attribute('color', f'{color[0]},{color[1]},{color[2]}')

            spawn_location = carla.Location(x, y, z)
            spawn_rotation = carla.Rotation(pitch, yaw, roll)
            spawn_transform = carla.Transform(location=spawn_location, rotation=spawn_rotation)

            obstacle = self.world.spawn_actor(obstacle_bp, spawn_transform)
            self.obstacles.append(obstacle)

            print(f"Spawned obstacle at ({x:.1f}, {y:.1f}) of type {obstacle_type}")
            return obstacle

        except Exception as e:
            print(f"Failed to spawn obstacle: {e}")
            return None

    def spawn_spaced_obstacles(self):
        """
        生成间距合适的障碍物，避免过于密集
        在车辆前方不同距离和横向位置生成3个障碍物
        """
        if not self.ego_vehicle:
            print("No ego vehicle, cannot generate obstacles")
            return []

        vehicle_location = self.ego_vehicle.get_location()
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_yaw = vehicle_transform.rotation.yaw

        # 获取车辆前方的waypoint
        map = self.world.get_map()
        current_waypoint = map.get_waypoint(vehicle_location)

        if not current_waypoint:
            print("Cannot get current waypoint")
            return []

        # 定义障碍物配置：位置、类型、横向偏移
        obstacle_configs = [
            {'distance': 20.0, 'type': 'vehicle.audi.tt', 'lane_offset': 0.0, 'yaw_offset': 180,
             'color': (255, 0, 0, 200)},  # 正前方15米，掉头的车辆
            {'distance': 40.0, 'type': 'static.prop.trafficcone', 'lane_offset': 1.5, 'yaw_offset': 0,
             'color': (255, 255, 0, 200)},  # 前方30米，右侧交通锥
            {'distance': 50.0, 'type': 'vehicle.tesla.model3', 'lane_offset': -1.5, 'yaw_offset': 0,
             'color': (255, 165, 0, 200)},  # 前方45米，左侧车辆
        ]

        spawned_obstacles = []

        for i, config in enumerate(obstacle_configs):
            try:
                distance = config['distance']
                obstacle_type = config['type']
                lane_offset = config['lane_offset']  # 正值=右侧，负值=左侧
                yaw_offset = config['yaw_offset']
                color = config['color']

                # 获取前方waypoint
                next_waypoints = current_waypoint.next(distance)

                if not next_waypoints:
                    print(f"Warning: No waypoint found at distance {distance}m")
                    continue

                target_waypoint = next_waypoints[0]
                lane_width = target_waypoint.lane_width

                # 计算障碍物位置
                location = target_waypoint.transform.location

                # 如果有横向偏移，调整位置
                if lane_offset != 0:
                    # 计算横向偏移方向
                    perpendicular_yaw = vehicle_yaw + 90 if lane_offset > 0 else vehicle_yaw - 90
                    location.x += np.cos(np.radians(perpendicular_yaw)) * abs(lane_offset)
                    location.y += np.sin(np.radians(perpendicular_yaw)) * abs(lane_offset)

                # 设置朝向
                yaw = target_waypoint.transform.rotation.yaw + yaw_offset

                # 生成障碍物
                obstacle = self.spawn_obstacle(
                    obstacle_type=obstacle_type,
                    x=location.x,
                    y=location.y,
                    z=location.z + 0.1,
                    yaw=yaw,
                    static=True,
                    color=color
                )

                if obstacle:
                    spawned_obstacles.append(obstacle)

                    side = "center"
                    if lane_offset > 0:
                        side = "right"
                    elif lane_offset < 0:
                        side = "left"

                    print(f"Obstacle {i + 1}: {obstacle_type.split('.')[-1]} "
                          f"at {distance:.1f}m {side}, Position=({location.x:.1f}, {location.y:.1f})")

            except Exception as e:
                print(f"Error spawning obstacle {i + 1}: {e}")
                continue

        return spawned_obstacles

    def get_obstacle_positions(self):
        """获取所有障碍物的位置和边界框信息"""
        obstacle_info = []

        for obstacle in self.obstacles:
            try:
                transform = obstacle.get_transform()
                location = transform.location

                # 根据障碍物类型设置不同的安全距离
                obstacle_type = obstacle.type_id
                if 'vehicle' in obstacle_type:
                    safe_distance = 5.0  # 车辆类障碍物需要更大的安全距离
                    radius = 2.5
                elif 'streetbarrier' in obstacle_type:
                    safe_distance = 2.0
                    radius = 1.0
                elif 'trafficcone' in obstacle_type:
                    safe_distance = 1.5
                    radius = 0.5
                else:
                    safe_distance = 2.0
                    radius = 1.0

                obstacle_info.append({
                    'id': obstacle.id,
                    'type': obstacle_type,
                    'x': location.x,
                    'y': location.y,
                    'z': location.z,
                    'radius': radius,
                    'safe_distance': safe_distance
                })

            except Exception as e:
                print(f"Error getting obstacle info: {e}")
                continue

        return obstacle_info

    def draw_obstacles_with_info(self, obstacles_info, life_time=0.1):
        """绘制障碍物及其相关信息"""
        for obs_info in obstacles_info:
            try:
                location = carla.Location(x=obs_info['x'], y=obs_info['y'], z=obs_info['z'] + 0.5)
                radius = obs_info.get('radius', 2.0)
                safe_distance = obs_info.get('safe_distance', 3.0)

                # 根据障碍物类型设置颜色
                if 'vehicle' in obs_info['type']:
                    color = carla.Color(255, 0, 0)  # 红色
                elif 'streetbarrier' in obs_info['type']:
                    color = carla.Color(128, 0, 128)  # 紫色
                elif 'trafficcone' in obs_info['type']:
                    color = carla.Color(255, 255, 0)  # 黄色
                else:
                    color = carla.Color(255, 165, 0)  # 橙色

                # 绘制障碍物位置（大点）
                self.world.debug.draw_point(
                    location,
                    size=0.3,
                    color=color,
                    life_time=life_time
                )

                # 绘制障碍物半径
                self.draw_circle(location, radius, color=color, life_time=life_time)

                # 绘制安全距离圆（虚线）
                self.draw_dashed_circle(location, safe_distance,
                                        color=carla.Color(255, 255, 255, 100),  # 白色半透明
                                        dash_length=0.3, gap_length=0.3,
                                        life_time=life_time)

                # 绘制障碍物类型标签
                text_location = carla.Location(x=location.x, y=location.y, z=location.z + 1.5)
                self.world.debug.draw_string(
                    text_location,
                    obs_info['type'].split('.')[-1],
                    draw_shadow=True,
                    color=carla.Color(255, 255, 255),
                    life_time=life_time,
                    persistent_lines=False
                )

            except Exception as e:
                print(f"Error drawing obstacle info: {e}")
                continue

    def draw_circle(self, center, radius, color=carla.Color(255, 0, 0), life_time=0.1):
        """绘制圆形"""
        num_segments = 32
        for i in range(num_segments):
            angle1 = 2 * np.pi * i / num_segments
            angle2 = 2 * np.pi * (i + 1) / num_segments

            p1 = carla.Location(
                center.x + radius * np.cos(angle1),
                center.y + radius * np.sin(angle1),
                center.z
            )
            p2 = carla.Location(
                center.x + radius * np.cos(angle2),
                center.y + radius * np.sin(angle2),
                center.z
            )

            self.world.debug.draw_line(
                p1, p2,
                thickness=0.05,
                color=color,
                life_time=life_time
            )

    def draw_dashed_circle(self, center, radius, color=carla.Color(255, 255, 255),
                           dash_length=0.5, gap_length=0.5, life_time=0.1):
        """绘制虚线圆形"""
        num_segments = 64
        dash_pattern = True  # True表示绘制，False表示间隙

        for i in range(num_segments):
            angle1 = 2 * np.pi * i / num_segments
            angle2 = 2 * np.pi * (i + 1) / num_segments

            # 计算圆弧长度
            arc_length = radius * (angle2 - angle1)

            # 如果圆弧长度大于dash_length+gap_length，我们需要进一步细分
            if arc_length > dash_length + gap_length:
                sub_segments = int(arc_length / (dash_length + gap_length))
                for j in range(sub_segments):
                    sub_angle1 = angle1 + j * (angle2 - angle1) / sub_segments
                    sub_angle2 = angle1 + (j + 1) * (angle2 - angle1) / sub_segments

                    if dash_pattern:
                        p1 = carla.Location(
                            center.x + radius * np.cos(sub_angle1),
                            center.y + radius * np.sin(sub_angle1),
                            center.z
                        )
                        p2 = carla.Location(
                            center.x + radius * np.cos(sub_angle2),
                            center.y + radius * np.sin(sub_angle2),
                            center.z
                        )

                        self.world.debug.draw_line(
                            p1, p2,
                            thickness=0.02,
                            color=color,
                            life_time=life_time
                        )

                    dash_pattern = not dash_pattern
            else:
                if dash_pattern:
                    p1 = carla.Location(
                        center.x + radius * np.cos(angle1),
                        center.y + radius * np.sin(angle1),
                        center.z
                    )
                    p2 = carla.Location(
                        center.x + radius * np.cos(angle2),
                        center.y + radius * np.sin(angle2),
                        center.z
                    )

                    self.world.debug.draw_line(
                        p1, p2,
                        thickness=0.02,
                        color=color,
                        life_time=life_time
                    )

                # 切换绘制模式
                if arc_length > dash_length:
                    dash_pattern = not dash_pattern

    def set_spectator(self, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        spectator = self.world.get_spectator()
        location = carla.Location(x=x, y=y, z=z)
        rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        spectator_transform = carla.Transform(location, rotation)
        spectator.set_transform(spectator_transform)

    def clean(self):
        self.destroy_all_actors()

    def draw_perception_planning(self, x_ref, y_ref, current_idx, look_ahead_points=20, obstacles_info=None):
        """
        绘制感知规划信息，包括障碍物
        """
        # 清除之前的绘制
        self.draw_trajectory([], [], life_time=0.1)

        # 绘制障碍物信息
        if obstacles_info:
            self.draw_obstacles_with_info(obstacles_info, life_time=0.1)

        # 绘制前方轨迹
        total_points = len(x_ref)
        if total_points == 0:
            return

        end_idx = min(current_idx + look_ahead_points, total_points)

        if current_idx < end_idx:
            display_x = x_ref[current_idx:end_idx]
            display_y = y_ref[current_idx:end_idx]
        else:
            display_x = list(x_ref[current_idx:]) + list(x_ref[:end_idx])
            display_y = list(y_ref[current_idx:]) + list(y_ref[:end_idx])

        if len(display_x) > 1:
            self.draw_trajectory(
                display_x,
                display_y,
                height=0.5,
                thickness=0.15,
                green=255,
                life_time=0.1
            )

        if len(display_x) > 0:
            target_location = carla.Location(
                x=display_x[0],
                y=display_y[0],
                z=0.5
            )
            self.world.debug.draw_point(
                target_location,
                size=0.3,
                color=carla.Color(0, 255, 0),  # 绿色目标点
                life_time=0.1
            )

            if self.ego_vehicle:
                vehicle_location = self.ego_vehicle.get_transform().location
                self.world.debug.draw_line(
                    vehicle_location,
                    target_location,
                    thickness=0.1,
                    color=carla.Color(0, 255, 255),
                    life_time=0.1
                )

    def draw_frenet_frame(self, x_ref, y_ref, current_idx):
        total_points = len(x_ref)
        if current_idx + 1 >= total_points:
            return

        x1, y1 = x_ref[current_idx], y_ref[current_idx]
        x2, y2 = x_ref[(current_idx + 1) % total_points], y_ref[(current_idx + 1) % total_points]

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
        for i in range(len(x_traj) - 1):
            start_point = carla.Location(x=x_traj[i], y=y_traj[i], z=height)
            end_point = carla.Location(x=x_traj[i + 1], y=y_traj[i + 1], z=height)
            self.world.debug.draw_line(
                start_point,
                end_point,
                thickness=thickness,
                color=carla.Color(red, green, blue),
                life_time=life_time
            )

    def get_main_ego_vehicle_state(self):
        if not self.ego_vehicle:
            return 0, 0, 0, 0

        transform = self.ego_vehicle.get_transform()
        x = transform.location.x
        y = transform.location.y
        theta = np.deg2rad(transform.rotation.yaw)
        v = np.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2)
        return x, y, theta, v

    def apply_control(self, steer, throttle, brake):
        if self.ego_vehicle:
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    @staticmethod
    def process_control_inputs(wheel_angle_rad, acceleration_m_s_2):
        if acceleration_m_s_2 == 0:
            throttle = 0
            brake = 0
        elif acceleration_m_s_2 < 0:
            throttle = 0
            brake = max(min(acceleration_m_s_2 / MAX_BRAKING_M_S_2, 1.0), 0.0)
        else:
            throttle = max(min(acceleration_m_s_2 / MAX_ACCELERATION_M_S_2, 1.0), 0.0)
            brake = 0
        steer = max(min(wheel_angle_rad / MAX_WHEEL_ANGLE_RAD, 1.0), -1.0)
        return throttle, brake, steer

    def print_ego_vehicle_characteristics(self):
        if not self.ego_vehicle:
            print("Vehicle not spawned yet!")
            return None

        physics_control = self.ego_vehicle.get_physics_control()

        print("Vehicle Physics Information.\n")

        print("Wheel Information:")
        for i, wheel in enumerate(physics_control.wheels):
            print(f" Wheel {i + 1}:")
            print(f"   Tire Friction: {wheel.tire_friction}")
            print(f"   Damping Rate: {wheel.damping_rate}")
            print(f"   Max Steer Angle: {wheel.max_steer_angle}")
            print(f"   Radius: {wheel.radius}")
            print(f"   Max Brake Torque: {wheel.max_brake_torque}")
            print(f"   Max Handbrake Torque: {wheel.max_handbrake_torque}")
            print(f"   Position (x, y, z): ({wheel.position.x}, {wheel.position.y}, {wheel.position.z})")

        print(f" Torque Curve:")
        for point in physics_control.torque_curve:
            print(f"RPM: {point.x}, Torque: {point.y}")
        print(f" Max RPM: {physics_control.max_rpm}")
        print(f" MOI (Moment of Inertia): {physics_control.moi}")
        print(f" Damping Rate Full Throttle: {physics_control.damping_rate_full_throttle}")
        print(
            f" Damping Rate Zero Throttle Clutch Engaged: {physics_control.damping_rate_zero_throttle_clutch_engaged}")
        print(
            f" Damping Rate Zero Throttle Clutch Disengaged: {physics_control.damping_rate_zero_throttle_clutch_disengaged}")
        print(f" If True, the vehicle will have an automatic transmission: {physics_control.use_gear_autobox}")
        print(f" Gear Switch Time: {physics_control.gear_switch_time}")
        print(f" Clutch Strength: {physics_control.clutch_strength}")
        print(f" Final Ratio: {physics_control.final_ratio}")
        print(f" Mass: {physics_control.mass}")
        print(f" Drag coefficient: {physics_control.drag_coefficient}")
        print(f" Steering Curve:")
        for point in physics_control.steering_curve:
            print(f"Speed: {point.x}, Steering: {point.y}")