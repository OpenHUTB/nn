import gym
import carla
import numpy as np
import time
import sys
import random
import pygame
import os
from queue import Queue
from gym import spaces
from collections import deque
import math

# ========== 原有CarlaEnvironment类（重点修改视角平滑逻辑） ==========
class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        # 基础属性
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.settings = None
        self.vehicle = None
        self.npc_vehicles = []
        self.camera = None
        self.lidar = None
        self.imu = None
        self.spawn_points = []
        # TM核心配置（0.9.11专用 - 仅保留全局配置）
        self.traffic_manager = None
        self.tm_port = 8000
        self.tm_seed = 0  # 固定TM种子，保证行为一致
        # 数据队列
        self.image_queue = Queue(maxsize=1)
        self.lidar_queue = Queue(maxsize=1)
        self.imu_queue = Queue(maxsize=1)
        # 激光雷达3D点云专用队列
        self.raw_lidar_queue = Queue(maxsize=1)
        # 新增：当前视角模式（默认第三人称）
        self.current_view_mode = "third_person"  # first_person / third_person / bird_view
        
        # ========== 新增：视角平滑相关变量（解决抖动核心） ==========
        self.smooth_pos = None  # 平滑后的位置缓存
        self.smooth_rot = None  # 平滑后的旋转缓存
        self.pos_smooth_factor = 0.05 # 位置平滑系数（0.05~0.2，越小越平滑）
        self.rot_smooth_factor = 0.08  # 旋转平滑系数（比位置更平滑）
        self.view_switch_flag = False  # 视角切换标记（切换时跳过平滑）

        # 连接CARLA（强化严格同步模式）
        self._connect_carla()
        # 初始化TM（仅用0.9.11确实验证的全局API）
        self._init_traffic_manager()
        # 定义观测空间
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            'lidar_distances': gym.spaces.Box(low=0, high=50, shape=(360,), dtype=np.float32),
            'imu': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        })
        # 获取有效生成点（仅保留道路上的点，补充到60个以适配60辆NPC）
        self.spawn_points = self._get_valid_road_spawn_points()
        print(f"[场景初始化] 有效道路生成点数量: {len(self.spawn_points)}")
        sys.stdout.flush()

    def _connect_carla(self):
        """连接CARLA（极致同步配置）"""
        retry_count = 3
        for i in range(retry_count):
            try:
                print(f"[CARLA连接] 尝试第{i+1}次连接（localhost:2000）...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(30.0)
                self.world = self.client.get_world()
                self.blueprint_library = self.world.get_blueprint_library()
                
                # 终极同步配置（消除所有帧差）
                self.settings = self.world.get_settings()
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 1/60  # 60FPS仿真，物理更平滑
                self.settings.no_rendering_mode = False
                self.settings.substepping = True
                self.settings.max_substep_delta_time = 0.003  # 更小的子步（3ms），过滤微米级抖动
                self.settings.max_substeps = 30
                self.world.apply_settings(self.settings)
                time.sleep(1.5)  # 延长等待时间，确保配置完全生效
                current_settings = self.world.get_settings()
                print(f"[同步模式] 生效状态: {current_settings.synchronous_mode}, 固定帧间隔: {current_settings.fixed_delta_seconds}s")
                
                # 双重清理
                self._clear_all_non_ego_actors()
                self.world.tick()
                self._clear_all_non_ego_actors()
                
                # 版本检查
                server_version = self.client.get_server_version()
                print(f"[CARLA连接] 成功连接，服务器版本：{server_version}")
                if "0.9.11" not in server_version:
                    print("[警告] 检测到非0.9.11版本，可能存在兼容性问题！")
                return
            except Exception as e:
                print(f"[CARLA连接失败] {str(e)}")
                if i == retry_count - 1:
                    raise RuntimeError("无法连接CARLA，请检查模拟器是否启动（需0.9.11版本）")
                time.sleep(2)

    def _init_traffic_manager(self):
        """初始化交通管理器（强制同步）"""
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.tm_seed)
        self.traffic_manager.global_percentage_speed_difference(0.0)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(50.0)
        print("[TM配置] 交通管理器同步模式已启用")

    def _set_actor_tm_params(self, actor):
        """彻底移除所有Actor级TM配置"""
        if not self._is_actor_alive(actor):
            return
        pass

    def _get_valid_road_spawn_points(self):
        """过滤生成点：仅保留道路网络上的有效点"""
        map = self.world.get_map()
        valid_points = []
        for sp in map.get_spawn_points():
            waypoint = map.get_waypoint(sp.location)
            if waypoint and waypoint.road_id != -1:
                valid_points.append(sp)
        # 补充到60个
        if len(valid_points) < 60:
            for _ in range(60 - len(valid_points)):
                random_loc = self.world.get_random_location_from_navigation()
                if random_loc:
                    waypoint = map.get_waypoint(random_loc)
                    valid_points.append(carla.Transform(waypoint.transform.location, waypoint.transform.rotation))
        return valid_points

    def _is_actor_alive(self, actor):
        """安全检查Actor是否存活"""
        try:
            return actor is not None and actor.is_alive
        except Exception:
            return False

    def _safe_destroy_actor(self, actor):
        """安全销毁Actor"""
        try:
            if self._is_actor_alive(actor):
                actor.destroy()
        except Exception as e:
            if "has been destroyed" not in str(e) and "not found" not in str(e):
                print(f"[安全销毁警告] {str(e)}")

    def _clear_all_non_ego_actors(self):
        """清理非主车辆和NPC的Actor"""
        if not self.world:
            return
        actors = self.world.get_actors()
        cleared_count = {'vehicle': 0, 'bicycle': 0, 'static_vehicle': 0}
        keep_ids = set()
        if self._is_actor_alive(self.vehicle):
            keep_ids.add(self.vehicle.id)
        for npc in self.npc_vehicles:
            if self._is_actor_alive(npc):
                keep_ids.add(npc.id)
        
        for actor in actors:
            try:
                actor_type = actor.type_id
                if actor_type.startswith('vehicle.') and actor.id not in keep_ids:
                    self._safe_destroy_actor(actor)
                    cleared_count['vehicle'] += 1
                elif actor_type.startswith('walker.bicycle') and actor.id not in keep_ids:
                    self._safe_destroy_actor(actor)
                    cleared_count['bicycle'] += 1
                elif actor_type.startswith('static.vehicle') and actor.id not in keep_ids:
                    self._safe_destroy_actor(actor)
                    cleared_count['static_vehicle'] += 1
            except Exception as e:
                if "has been destroyed" not in str(e) and "not found" not in str(e):
                    print(f"[销毁Actor警告] {str(e)}")
        
        self.world.tick()
        print(f"[清理] 车辆{cleared_count['vehicle']} | 自行车{cleared_count['bicycle']} | 静态车辆{cleared_count['static_vehicle']}")

    def process_image(self, image):
        """处理摄像头数据"""
        try:
            if not hasattr(image, 'raw_data') or not hasattr(image, 'height') or not hasattr(image, 'width'):
                print("[图像处理错误] 无效的图像数据")
                return
                
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
            array = array[:, :, ::-1].copy()
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(array)
        except Exception as e:
            print(f"[图像处理错误] {str(e)}")

    def process_lidar(self, data):
        """处理激光雷达数据（保留原始点云用于3D可视化）"""
        try:
            if not hasattr(data, 'raw_data'):
                print("[激光雷达处理错误] 无效的激光雷达数据")
                return
                
            # 原始点云（x,y,z,intensity）- 用于3D可视化
            raw_points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)
            if self.raw_lidar_queue.full():
                self.raw_lidar_queue.get()
            self.raw_lidar_queue.put(raw_points)

            # 原有2D距离计算逻辑
            points = raw_points[:, :3]
            distances = np.linalg.norm(points, axis=1)
            angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
            angles = (angles + 360) % 360

            lidar_distances = np.full(360, 50.0, dtype=np.float32)
            for angle, dist in zip(angles, distances):
                angle_idx = int(round(angle)) % 360
                if dist < lidar_distances[angle_idx]:
                    lidar_distances[angle_idx] = dist

            if self.lidar_queue.full():
                self.lidar_queue.get()
            self.lidar_queue.put(lidar_distances)
        except Exception as e:
            print(f"[激光雷达处理错误] {str(e)}")

    def process_imu(self, data):
        """处理IMU数据"""
        try:
            if not hasattr(data, 'accelerometer') or not hasattr(data, 'gyroscope'):
                print("[IMU处理错误] 无效的IMU数据")
                return
                
            imu_data = np.array([
                data.accelerometer.x, data.accelerometer.y, data.accelerometer.z,
                data.gyroscope.x, data.gyroscope.y, data.gyroscope.z
            ], dtype=np.float32)
            if self.imu_queue.full():
                self.imu_queue.get()
            self.imu_queue.put(imu_data)
        except Exception as e:
            print(f"[IMU处理错误] {str(e)}")

    def reset(self):
        """重置环境"""
        self.close()
        time.sleep(1.0)
        self._clear_all_non_ego_actors()
        self._spawn_vehicle()
        
        if self.vehicle:
            self._spawn_sensors()
            self.vehicle.set_simulate_physics(True)
            # 优化车辆物理参数（彻底消除微小抖动）
            self._optimize_vehicle_physics()
            time.sleep(0.5)
            self.vehicle.set_autopilot(True, self.tm_port)
            self._spawn_npcs(60)
            self._clear_all_non_ego_actors()
            # 重置视角模式和平滑缓存
            self.current_view_mode = "third_person"
            self.smooth_pos = None
            self.smooth_rot = None
            self.view_switch_flag = False
        
        # 多次同步，物理稳定（延长时间）
        for _ in range(15):
            self.world.tick()
            time.sleep(0.05)
        
        print(f"[环境重置] 完成，主车辆1辆，NPC车辆{len(self.npc_vehicles)}辆")
        return self.get_observation()

    def _optimize_vehicle_physics(self):
        """终极物理优化：彻底消除车辆微小震动"""
        if not self._is_actor_alive(self.vehicle):
            return
        try:
            physics_control = self.vehicle.get_physics_control()
            # 进一步降低悬挂刚度（从1000→800），减少微米级震动
            for wheel in physics_control.wheels:
                wheel.suspension_stiffness = 800.0  # 更低的刚度，过滤路面微小颠簸
                wheel.suspension_damping = 250.0     # 更高的阻尼，快速衰减震动
                wheel.suspension_compression = 300.0  # 增加压缩阻尼
                wheel.max_suspension_travel = 0.08   # 减小悬挂行程，避免过度晃动
                wheel.friction_slip = 1.2            # 增加轮胎抓地力，减少打滑抖动
            self.vehicle.apply_physics_control(physics_control)
            print("[物理优化] 车辆悬挂+轮胎参数已终极优化，消除微小震动")
        except Exception as e:
            print(f"[物理优化警告] {str(e)}")

    def _spawn_vehicle(self):
        """生成主车辆"""
        self._safe_destroy_actor(self.vehicle)
        self.world.tick()

        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('color', '255,0,0')
        vehicle_bp.set_attribute('role_name', 'ego_vehicle')

        if not self.spawn_points:
            raise RuntimeError("无可用道路生成点")

        random.shuffle(self.spawn_points)
        for spawn_point in self.spawn_points[:10]:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle:
                self.vehicle.set_simulate_physics(True)
                print(f"[主车辆生成] 成功（ID: {self.vehicle.id}）")
                return

        raise RuntimeError("主车辆生成失败，请重启CARLA")

    def _spawn_npcs(self, count):
        """生成NPC车辆"""
        if not self.spawn_points:
            print("[NPC生成] 无可用生成点")
            return

        ego_transform = self.vehicle.get_transform()
        available_spawn_points = []
        for sp in self.spawn_points:
            distance = np.linalg.norm([
                sp.location.x - ego_transform.location.x,
                sp.location.y - ego_transform.location.y
            ])
            if distance > 15.0:
                available_spawn_points.append(sp)

        if len(available_spawn_points) < count:
            count = len(available_spawn_points)
            print(f"[NPC生成] 可用点不足，生成{count}辆")

        vehicle_bps = [bp for bp in self.blueprint_library.filter('vehicle.*') 
                       if bp.has_attribute('color') and bp.id != 'vehicle.tesla.model3']
        random.shuffle(vehicle_bps)
        if not vehicle_bps:
            vehicle_bps = self.blueprint_library.filter('vehicle.*')

        spawned_count = 0
        for i, spawn_point in enumerate(random.sample(available_spawn_points, count)):
            bp = vehicle_bps[i % len(vehicle_bps)]
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            bp.set_attribute('role_name', 'npc_vehicle')

            npc_vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if npc_vehicle:
                npc_vehicle.set_simulate_physics(True)
                time.sleep(0.05)
                npc_vehicle.set_autopilot(True, self.tm_port)
                self.npc_vehicles.append(npc_vehicle)
                spawned_count += 1

                if spawned_count % 5 == 0:
                    self.world.tick()

        self.world.tick()
        print(f"[NPC生成] 成功生成{spawned_count}辆（目标：{count}辆）")

    def _spawn_sensors(self):
        """生成传感器"""
        for sensor in [self.camera, self.lidar, self.imu]:
            self._safe_destroy_actor(sensor)

        # 摄像头（适配第一人称视角：调整安装位置）
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '100')
        # 第一人称视角摄像头（驾驶位）
        self.camera = self.world.spawn_actor(
            camera_bp, carla.Transform(carla.Location(x=1.0, y=0.0, z=1.2), carla.Rotation(pitch=0, yaw=0, roll=0)), 
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.camera.listen(self.process_image)

        # 激光雷达
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '200000')
        lidar_bp.set_attribute('rotation_frequency', '60')  # 与仿真帧率一致
        self.lidar = self.world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.0)), attach_to=self.vehicle
        )
        self.lidar.listen(self.process_lidar)

        # IMU
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.imu.listen(self.process_imu)
        print("[传感器] 初始化完成（帧率与仿真同步）")

    def get_observation(self):
        """获取观测数据"""
        while self.image_queue.empty() or self.lidar_queue.empty() or self.imu_queue.empty():
            time.sleep(0.001)
        return {
            'image': self.image_queue.get(),
            'lidar_distances': self.lidar_queue.get(),
            'imu': self.imu_queue.get()
        }

    def get_raw_lidar_points(self):
        """获取原始激光雷达点云（用于3D可视化）"""
        while self.raw_lidar_queue.empty():
            time.sleep(0.001)
        return self.raw_lidar_queue.get()

    def get_obstacle_directions(self, lidar_distances):
        """计算四向障碍物距离"""
        front_angles = np.concatenate([np.arange(345, 360), np.arange(0, 16)])
        rear_angles = np.arange(165, 196)
        left_angles = np.arange(75, 106)
        right_angles = np.arange(255, 286)
        return {
            'front': np.min(lidar_distances[front_angles]),
            'rear': np.min(lidar_distances[rear_angles]),
            'left': np.min(lidar_distances[left_angles]),
            'right': np.min(lidar_distances[right_angles])
        }

    def step(self, action=None):
        """环境交互步骤"""
        self.world.tick()
        
        # 清理无效NPC
        self.npc_vehicles = [npc for npc in self.npc_vehicles if self._is_actor_alive(npc)]
        
        # 打印NPC速度（调试用，降低频率）
        if random.random() < 0.02:
            for npc in self.npc_vehicles[:1]:
                try:
                    velocity = npc.get_velocity()
                    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z]) * 3.6
                    print(f"[NPC速度] ID:{npc.id} 速度:{speed:.1f}km/h")
                except Exception:
                    pass
        
        observation = self.get_observation()
        reward = 1.0
        done = False
        return observation, reward, done, {}

    def close(self):
        """清理资源"""
        if self.traffic_manager:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception as e:
                print(f"[TM清理警告] {str(e)}")
        for sensor in [self.camera, self.lidar, self.imu]:
            self._safe_destroy_actor(sensor)
        for npc in self.npc_vehicles:
            self._safe_destroy_actor(npc)
        self.npc_vehicles = []
        self._safe_destroy_actor(self.vehicle)
        self.vehicle = None
        self._clear_all_non_ego_actors()
        for q in [self.image_queue, self.lidar_queue, self.imu_queue, self.raw_lidar_queue]:
            while not q.empty():
                q.get()
        # 恢复世界设置
        if self.settings:
            try:
                self.settings.synchronous_mode = False
                self.settings.substepping = False
                self.world.apply_settings(self.settings)
            except Exception as e:
                print(f"[世界设置恢复警告] {str(e)}")
        print("[资源清理] 所有资源已销毁")

    # ========== 核心修改：视角切换+平滑跟随（解决抖动） ==========
    def switch_view_mode(self, mode):
        """切换视角模式（添加切换标记，跳过初始平滑）"""
        if mode in ["first_person", "third_person", "bird_view"]:
            self.current_view_mode = mode
            self.view_switch_flag = True  # 切换时强制重置平滑缓存
            print(f"[视角切换] 已切换至：{self.current_view_mode}")

    def _smooth_transform(self, target_loc, target_rot):
        """对位置和旋转进行指数移动平均平滑（核心防抖逻辑）"""
        # 首次初始化平滑缓存
        if self.smooth_pos is None or self.view_switch_flag:
            self.smooth_pos = target_loc
            self.smooth_rot = target_rot
            self.view_switch_flag = False  # 重置切换标记
            return self.smooth_pos, self.smooth_rot
        
        # 位置平滑（指数移动平均）
        self.smooth_pos.x = self.smooth_pos.x * (1 - self.pos_smooth_factor) + target_loc.x * self.pos_smooth_factor
        self.smooth_pos.y = self.smooth_pos.y * (1 - self.pos_smooth_factor) + target_loc.y * self.pos_smooth_factor
        self.smooth_pos.z = self.smooth_pos.z * (1 - self.pos_smooth_factor) + target_loc.z * self.pos_smooth_factor
        
        # 旋转平滑（指数移动平均，仅yaw角跟随，pitch/roll固定）
        self.smooth_rot.yaw = self.smooth_rot.yaw * (1 - self.rot_smooth_factor) + target_rot.yaw * self.rot_smooth_factor
        self.smooth_rot.pitch = target_rot.pitch  # 俯仰角固定，不平滑
        self.smooth_rot.roll = target_rot.roll    # 滚转角固定，不平滑
        
        return self.smooth_pos, self.smooth_rot

    def update_view(self, spectator):
        """根据当前视角模式更新 spectator 位置（添加平滑防抖）"""
        if not self._is_actor_alive(self.vehicle):
            return
        
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation

        # 计算目标视角位置和旋转
        if self.current_view_mode == "first_person":
            # 第一人称视角：驾驶位，向前看
            target_loc = vehicle_transform.transform(carla.Location(x=0.5, y=0.0, z=1.0))
            target_rot = vehicle_rotation
        elif self.current_view_mode == "third_person":
            # 第三人称视角：车辆后上方，平滑跟随
            target_loc = vehicle_transform.transform(carla.Location(x=-5.0, y=0.0, z=2.0))
            target_rot = carla.Rotation(pitch=-10.0, yaw=vehicle_rotation.yaw, roll=0.0)
        elif self.current_view_mode == "bird_view":
            # 鸟瞰视角：车辆正上方，全局俯视
            target_loc = carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 50.0)
            target_rot = carla.Rotation(pitch=-90.0, yaw=vehicle_rotation.yaw, roll=0.0)
        else:
            # 默认第三人称
            target_loc = vehicle_transform.transform(carla.Location(x=-5.0, y=0.0, z=2.0))
            target_rot = carla.Rotation(pitch=-10.0, yaw=vehicle_rotation.yaw, roll=0.0)
        
        # 对目标视角进行平滑处理（核心防抖）
        smooth_loc, smooth_rot = self._smooth_transform(target_loc, target_rot)
        
        # 更新 spectator（使用平滑后的位置和旋转）
        spectator.set_transform(carla.Transform(smooth_loc, smooth_rot))

# ========== 激光雷达3D点云可视化窗口（保留，单独运行） ==========
class Lidar3DVisualizer:
    def __init__(self, width=600, height=600):
        # 窗口参数
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        pygame.display.set_caption("激光雷达3D点云可视化")
        
        # 视角参数（默认从车辆后上方俯视）
        self.theta = math.radians(30)  # 水平旋转角（绕Y轴）
        self.phi = math.radians(-30)   # 垂直旋转角（绕X轴）
        self.scale = 10.0              # 缩放系数
        self.offset_x = width // 2
        self.offset_y = height // 2
        
        # 交互状态（即使暂未生效，保留逻辑）
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
        self.point_size = 2            # 点云渲染大小
        
        # 颜色映射（距离→RGB）
        self.color_map = {
            'red': (255, 0, 0),    # 0-10m（近距/高风险）
            'yellow': (255, 255, 0),# 10-30m（中距）
            'green': (0, 255, 0)    # 30-50m（远距/低风险）
        }

    def project_3d_to_2d(self, point):
        """透视投影：将3D点(x,y,z)转换为2D屏幕坐标"""
        x, y, z = point
        
        # 绕Y轴旋转（theta）
        x_rot = x * math.cos(self.theta) - z * math.sin(self.theta)
        z_rot = x * math.sin(self.theta) + z * math.cos(self.theta)
        
        # 绕X轴旋转（phi）
        y_rot = y * math.cos(self.phi) - z_rot * math.sin(self.phi)
        z_rot = y * math.sin(self.phi) + z_rot * math.cos(self.phi)
        
        # 透视投影 + 屏幕偏移
        if z_rot == 0:
            z_rot = 0.001  # 避免除零
        proj_x = self.offset_x + (x_rot * self.scale) / (z_rot / 5)
        proj_y = self.offset_y - (y_rot * self.scale) / (z_rot / 5)
        
        return int(proj_x), int(proj_y)

    def get_point_color(self, distance):
        """根据距离获取点云颜色（颜色编码）"""
        if distance < 10:
            return self.color_map['red']
        elif distance < 30:
            return self.color_map['yellow']
        else:
            return self.color_map['green']

    def filter_invalid_points(self, points):
        """过滤无效点：距离>50m/ <0.5m、z轴异常（地面/天空噪点）"""
        # 计算每个点的距离
        distances = np.linalg.norm(points[:, :3], axis=1)
        # 过滤条件
        mask = (distances > 0.5) & (distances < 50) & (points[:, 2] > -1) & (points[:, 2] < 5)
        return points[mask], distances[mask]

    def render(self, raw_points):
        """渲染3D点云"""
        # 1. 清空屏幕（深灰背景，高级感）
        self.screen.fill((20, 20, 20))
        
        # 2. 过滤无效点
        valid_points, distances = self.filter_invalid_points(raw_points)
        
        # 3. 逐点投影并渲染
        for point, dist in zip(valid_points, distances):
            x, y, z = point[:3]
            # 转换为车辆局部坐标系（点云以车辆为中心）
            local_point = (-x, -y, z)  # 反转x/y，让点云朝向正确
            # 3D→2D投影
            screen_x, screen_y = self.project_3d_to_2d(local_point)
            # 边界检查（避免渲染到窗口外）
            if 0 < screen_x < self.width and 0 < screen_y < self.height:
                color = self.get_point_color(dist)
                # 绘制点云（抗锯齿）
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), self.point_size)
        
        # 4. 绘制辅助信息（坐标系、说明文字）
        font = pygame.font.SysFont('Consolas', 12, bold=True)
        # 坐标系提示
        axis_text = font.render("X:前 Y:左 Z:上 | 激光雷达有效范围：0.5-50m", True, (200, 200, 200))
        self.screen.blit(axis_text, (10, 10))
        # 距离说明
        dist_text = font.render("红色:<10m 黄色:10-30m 绿色:>30m", True, (200, 200, 200))
        self.screen.blit(dist_text, (10, 30))
        
        # 5. 更新显示
        pygame.display.flip()

    def handle_events(self):
        """处理鼠标交互（保留逻辑，即使暂未生效）"""
        for event in pygame.event.get():
            # 鼠标按下
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self.mouse_down = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # 滚轮上滚（放大）
                    self.scale = min(self.scale + 1, 30)
                elif event.button == 5:  # 滚轮下滚（缩小）
                    self.scale = max(self.scale - 1, 5)
            # 鼠标松开
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
            # 鼠标拖拽（旋转视角）
            elif event.type == pygame.MOUSEMOTION and self.mouse_down:
                current_pos = pygame.mouse.get_pos()
                dx = current_pos[0] - self.last_mouse_pos[0]
                dy = current_pos[1] - self.last_mouse_pos[1]
                # 更新旋转角（灵敏度适配）
                self.theta += dx * 0.01
                self.phi += dy * 0.01
                self.last_mouse_pos = current_pos

# ========== 主可视化界面（3D视角增强+清晰交互） ==========
class MainVisualizer:
    def __init__(self, width=800, height=600):
        # 窗口基础配置
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("CARLA自动驾驶仿真 - 3D视角控制")
        
        # 字体配置（确保显示清晰，抗锯齿）
        self.font_large = pygame.font.SysFont('Microsoft YaHei', 16, bold=True)
        self.font_small = pygame.font.SysFont('Microsoft YaHei', 12, bold=True)
        
        # 颜色配置（工业风，高对比度，显示清晰）
        self.colors = {
            'bg_main': (18, 18, 28),       # 主背景（深蓝灰）
            'bg_panel': (30, 30, 45),      # 面板背景（深紫灰）
            'text_normal': (220, 220, 240),# 普通文字（浅灰蓝）
            'text_highlight': (0, 180, 255),# 高亮文字（天蓝色）
            'button_normal': (45, 45, 65), # 按钮常态
            'button_hover': (60, 60, 85),  # 按钮悬停
            'button_press': (0, 120, 180), # 按钮按下
            'border': (80, 80, 100)        # 边框颜色
        }
        
        # 交互按钮配置（位置+大小+文本）
        self.buttons = {
            'first_person': {
                'rect': pygame.Rect(20, 500, 120, 40),
                'text': '第一人称 (1)',
                'active': False
            },
            'third_person': {
                'rect': pygame.Rect(150, 500, 120, 40),
                'text': '第三人称 (2)',
                'active': True  # 默认激活
            },
            'bird_view': {
                'rect': pygame.Rect(280, 500, 120, 40),
                'text': '鸟瞰视角 (3)',
                'active': False
            },
            'pause': {
                'rect': pygame.Rect(410, 500, 120, 40),
                'text': '暂停 (空格)',
                'active': False
            }
        }
        
        # 状态变量
        self.paused = False
        self.fps = 0
        self.npc_count = 0
        self.obstacle_distances = {'front': 0, 'rear': 0, 'left': 0, 'right': 0}
        self.current_view = "第三人称"

    def draw_buttons(self):
        """绘制交互按钮（清晰易识别，有状态反馈）"""
        for btn_name, btn in self.buttons.items():
            # 确定按钮颜色（根据状态）
            if btn['active']:
                bg_color = self.colors['button_press']
            elif btn['rect'].collidepoint(pygame.mouse.get_pos()):
                bg_color = self.colors['button_hover']
            else:
                bg_color = self.colors['button_normal']
            
            # 绘制按钮（带边框，清晰）
            pygame.draw.rect(self.screen, bg_color, btn['rect'], border_radius=5)
            pygame.draw.rect(self.screen, self.colors['border'], btn['rect'], width=2, border_radius=5)
            
            # 绘制按钮文字（居中，抗锯齿）
            text_surf = self.font_small.render(btn['text'], True, self.colors['text_normal'])
            text_rect = text_surf.get_rect(center=btn['rect'].center)
            self.screen.blit(text_surf, text_rect)

    def draw_status_panel(self):
        """绘制状态面板（信息清晰，分区显示）"""
        # 面板背景
        panel_rect = pygame.Rect(20, 20, 300, 150)
        pygame.draw.rect(self.screen, self.colors['bg_panel'], panel_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.colors['border'], panel_rect, width=2, border_radius=5)
        
        # 面板标题
        title_surf = self.font_large.render("仿真状态信息", True, self.colors['text_highlight'])
        self.screen.blit(title_surf, (30, 30))
        
        # 状态文本（分行显示，对齐）
        status_texts = [
            f"当前视角：{self.current_view}",
            f"仿真帧率：{self.fps:.1f} FPS",
            f"NPC车辆数：{self.npc_count} 辆",
            f"前方障碍物：{self.obstacle_distances['front']:.1f} m",
            f"右侧障碍物：{self.obstacle_distances['right']:.1f} m"
        ]
        
        for i, text in enumerate(status_texts):
            y_pos = 60 + i * 20
            text_surf = self.font_small.render(text, True, self.colors['text_normal'])
            self.screen.blit(text_surf, (30, y_pos))

    def draw_help_text(self):
        """绘制操作提示（清晰易读）"""
        help_texts = [
            "快捷键：1-第一人称  2-第三人称  3-鸟瞰视角  空格-暂停/继续  ESC-退出",
            "视角说明：第一人称（驾驶位）| 第三人称（跟随）| 鸟瞰（全局俯视）"
        ]
        for i, text in enumerate(help_texts):
            y_pos = 450 + i * 20
            text_surf = self.font_small.render(text, True, self.colors['text_normal'])
            self.screen.blit(text_surf, (20, y_pos))

    def update_status(self, fps, npc_count, obstacle_distances, current_view):
        """更新状态信息（用于渲染）"""
        self.fps = fps
        self.npc_count = npc_count
        self.obstacle_distances = obstacle_distances
        self.current_view = current_view
        
        # 更新按钮激活状态
        for btn_name in self.buttons:
            self.buttons[btn_name]['active'] = False
        if current_view == "第一人称":
            self.buttons['first_person']['active'] = True
        elif current_view == "第三人称":
            self.buttons['third_person']['active'] = True
        elif current_view == "鸟瞰视角":
            self.buttons['bird_view']['active'] = True
        self.buttons['pause']['active'] = self.paused

    def handle_events(self, env):
        """处理交互事件（快捷键+按钮点击）"""
        for event in pygame.event.get():
            # 窗口关闭
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            # 键盘按键
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                elif event.key == pygame.K_1:
                    env.switch_view_mode("first_person")
                    self.current_view = "第一人称"
                elif event.key == pygame.K_2:
                    env.switch_view_mode("third_person")
                    self.current_view = "第三人称"
                elif event.key == pygame.K_3:
                    env.switch_view_mode("bird_view")
                    self.current_view = "鸟瞰视角"
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            # 鼠标点击按钮
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.buttons['first_person']['rect'].collidepoint(mouse_pos):
                        env.switch_view_mode("first_person")
                        self.current_view = "第一人称"
                    elif self.buttons['third_person']['rect'].collidepoint(mouse_pos):
                        env.switch_view_mode("third_person")
                        self.current_view = "第三人称"
                    elif self.buttons['bird_view']['rect'].collidepoint(mouse_pos):
                        env.switch_view_mode("bird_view")
                        self.current_view = "鸟瞰视角"
                    elif self.buttons['pause']['rect'].collidepoint(mouse_pos):
                        self.paused = not self.paused

    def render(self):
        """渲染主界面（所有元素）"""
        # 清空主背景
        self.screen.fill(self.colors['bg_main'])
        
        # 绘制状态面板
        self.draw_status_panel()
        
        # 绘制操作提示
        self.draw_help_text()
        
        # 绘制交互按钮
        self.draw_buttons()
        
        # 更新显示（双缓冲，无撕裂）
        pygame.display.flip()

# ========== 仿真运行函数 ==========
def run_simulation():
    pygame.init()
    # 初始化字体（确保中文显示清晰，抗锯齿）
    pygame.font.init()
    # 禁用pygame默认鼠标缩放，避免干扰
    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
    
    env = None
    lidar_visualizer = None
    main_visualizer = None
    try:
        print("\n[CARLA连接] 创建环境...")
        env = CarlaEnvironment()
        
        print("\n[环境重置] 生成车辆和传感器...")
        env.reset()
        
        if not env.vehicle or not env.vehicle.is_alive:
            raise RuntimeError("车辆生成失败，请检查CARLA是否正常运行")
        print(f"[车辆状态] 生成成功（ID: {env.vehicle.id}），已启用自动驾驶")

        # 初始化两个可视化窗口
        lidar_visualizer = Lidar3DVisualizer(width=600, height=600)
        main_visualizer = MainVisualizer(width=800, height=600)
        
        clock = pygame.time.Clock()
        spectator = env.world.get_spectator()
        
        # 初始化视角（强制一次平滑缓存）
        env.update_view(spectator)
        
        print("\n[仿真开始] 按ESC退出 | 快捷键1/2/3切换视角 | 空格暂停")
        print("[窗口说明] 主窗口（视角控制）| 独立窗口（激光雷达3D点云）")
        sys.stdout.flush()
        
        step = 0
        obstacle_distances = {'front': 0, 'rear': 0, 'left': 0, 'right': 0}
        while True:
            # 1. 处理所有交互事件
            main_visualizer.handle_events(env)
            lidar_visualizer.handle_events()
            
            # 2. 暂停逻辑
            if not main_visualizer.paused:
                # 严格同步的仿真帧推进
                env.world.tick()
                
                # 更新CARLA spectator视角（带平滑防抖）
                env.update_view(spectator)
                
                # 获取观测数据（低频率，减少消耗）
                if step % 3 == 0:
                    observation = env.get_observation()
                    obstacle_distances = env.get_obstacle_directions(observation['lidar_distances'])
                    # 渲染激光雷达3D点云
                    raw_lidar_points = env.get_raw_lidar_points()
                    lidar_visualizer.render(raw_lidar_points)
                
                # 打印日志（极低频率）
                if step % 120 == 0:
                    print(f"\n[步骤 {step}] 障碍物距离 - 前{obstacle_distances['front']:.1f}m | 后{obstacle_distances['rear']:.1f}m | "
                          f"左{obstacle_distances['left']:.1f}m | 右{obstacle_distances['right']:.1f}m")
                    sys.stdout.flush()
                
                step += 1
            
            # 3. 更新主界面状态并渲染
            current_fps = clock.get_fps()
            current_view_name = {
                "first_person": "第一人称",
                "third_person": "第三人称",
                "bird_view": "鸟瞰视角"
            }.get(env.current_view_mode, "第三人称")
            main_visualizer.update_status(
                fps=current_fps,
                npc_count=len(env.npc_vehicles),
                obstacle_distances=obstacle_distances,
                current_view=current_view_name
            )
            main_visualizer.render()
            
            # 4. 锁定渲染帧率（与仿真帧率一致）
            clock.tick_busy_loop(60)

    except KeyboardInterrupt:
        print("\n[用户终止] 收到退出信号")
    except Exception as e:
        print(f"\n[仿真错误] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        if env is not None:
            print("\n[资源清理] 销毁资源...")
            env.close()
        pygame.quit()
        print("\n[程序退出]")

if __name__ == "__main__":
    print("="*60)
    print(f"[启动时间] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Python解释器] {sys.executable}")
    print(f"[运行环境] Win11 + CARLA0.9.11 + Python3.7.5")
    print("="*60)
    sys.stdout.flush()
    run_simulation()