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
        """处理激光雷达数据"""
        try:
            if not hasattr(data, 'raw_data'):
                print("[激光雷达处理错误] 无效的激光雷达数据")
                return
                
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)[:, :3]
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

        # 摄像头
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '100')
        self.camera = self.world.spawn_actor(
            camera_bp, carla.Transform(carla.Location(x=2.0, z=1.5)), attach_to=self.vehicle
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
        for q in [self.image_queue, self.lidar_queue, self.imu_queue]:
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

    def init_spectator_smoother(self, window_size=15):
        """初始化镜头平滑器（增大滑动窗口到15帧）"""
        self.vehicle_pose_buffer = deque(maxlen=window_size)  # 15帧缓存，过滤更多高频抖动
        self.window_size = window_size
        # 初始化加权平均权重（近期帧权重更高，兼顾平滑和响应）
        self.weights = np.linspace(0.1, 1.0, window_size)  # 权重从0.1→1.0递增
        self.weights /= np.sum(self.weights)  # 归一化

    def update_spectator_ultra_smooth(self, spectator):
        """
        终极平滑镜头更新：加权滑动平均+微米级死区+完全锁定旋转
        """
        if not self._is_actor_alive(self.vehicle):
            return
        
        # 1. 获取同步帧快照的车辆位姿（仅用快照，杜绝异步）
        snapshot = self.world.get_snapshot()
        vehicle_snapshot = snapshot.find(self.vehicle.id)
        if not vehicle_snapshot:
            return
        current_pose = vehicle_snapshot.get_transform()
        
        # 2. 初始化关键变量
        avg_yaw = current_pose.rotation.yaw
        avg_pose = current_pose
        
        # 3. 加入滑动缓存并计算「加权平均」位姿（核心优化）
        self.vehicle_pose_buffer.append(current_pose)
        if len(self.vehicle_pose_buffer) >= self.window_size:
            # 提取缓存中的位姿
            poses = list(self.vehicle_pose_buffer)
            count = len(poses)
            
            # 加权平均位置（近期帧权重更高）
            avg_loc = carla.Location()
            for i in range(count):
                weight = self.weights[i]
                avg_loc.x += poses[i].location.x * weight
                avg_loc.y += poses[i].location.y * weight
                avg_loc.z += poses[i].location.z * weight  # Z轴也加权平均
            
            # 加权平均Yaw角（处理360度环绕）
            yaws = [pose.rotation.yaw for pose in poses]
            yaw_rads = np.radians(yaws)
            # 加权正弦和余弦
            weighted_sin = np.sum(np.sin(yaw_rads) * self.weights[:count])
            weighted_cos = np.sum(np.cos(yaw_rads) * self.weights[:count])
            avg_yaw = np.degrees(np.arctan2(weighted_sin, weighted_cos))
            
            # 构建平均位姿
            avg_pose = carla.Transform(avg_loc, carla.Rotation(pitch=0, yaw=avg_yaw, roll=0))
        
        # 4. 模拟父级绑定+Z轴强制锁定
        relative_loc = carla.Location(x=-5.0, y=0.0, z=2.0)
        target_loc = avg_pose.transform(relative_loc)
        target_loc.z = avg_pose.location.z + 2.0  # 强制锁定Z轴，不随任何波动
        
        # 5. 镜头旋转：完全锁定（仅Yaw跟随平均位姿）
        target_rot = carla.Rotation(
            pitch=-10.0,  # 完全固定，不参与任何平滑
            yaw=avg_yaw,
            roll=0.0      # 完全固定
        )
        
        # 6. EMA平滑+微米级死区过滤（终极去抖）
        current_transform = spectator.get_transform()
        alpha = 0.03  # 极致平滑系数（更小，更稳定）
        pos_deadzone = 0.005  # 微米级死区（5mm内波动不更新）
        
        # 位置平滑（带死区）
        loc_diff = np.array([
            target_loc.x - current_transform.location.x,
            target_loc.y - current_transform.location.y,
            target_loc.z - current_transform.location.z
        ])
        loc_diff_mag = np.linalg.norm(loc_diff)
        
        if loc_diff_mag > pos_deadzone:
            final_loc = carla.Location(
                x=alpha * target_loc.x + (1 - alpha) * current_transform.location.x,
                y=alpha * target_loc.y + (1 - alpha) * current_transform.location.y,
                z=target_loc.z  # Z轴直接锁定
            )
        else:
            # 死区内不更新，保持当前位置
            final_loc = current_transform.location
        
        # 旋转平滑（仅Yaw，带死区）
        yaw_diff = target_rot.yaw - current_transform.rotation.yaw
        yaw_diff = (yaw_diff + 180) % 360 - 180  # 归一化
        rot_deadzone = 0.02  # 0.02度死区
        
        if abs(yaw_diff) > rot_deadzone:
            final_yaw = current_transform.rotation.yaw + alpha * yaw_diff
            final_yaw = final_yaw % 360
        else:
            final_yaw = current_transform.rotation.yaw
        
        final_rot = carla.Rotation(
            pitch=-10.0,  # 完全固定
            yaw=final_yaw,
            roll=0.0      # 完全固定
        )
        
        # 7. 更新镜头（仅一次，同步帧内完成）
        spectator.set_transform(carla.Transform(final_loc, final_rot))


def run_simulation():
    pygame.init()
    env = None
    try:
        print("\n[CARLA连接] 创建环境...")
        env = CarlaEnvironment()
        
        # 初始化镜头平滑器（15帧加权滑动窗口）
        env.init_spectator_smoother(window_size=15)
        
        print("\n[环境重置] 生成车辆和传感器...")
        env.reset()
        
        if not env.vehicle or not env.vehicle.is_alive:
            raise RuntimeError("车辆生成失败，请检查CARLA是否正常运行")
        print(f"[车辆状态] 生成成功（ID: {env.vehicle.id}），已启用自动驾驶")

        clock = pygame.time.Clock()
        spectator = env.world.get_spectator()
        
        # 初始化镜头+填充缓存（延长初始化时间）
        env.world.tick()
        for _ in range(env.window_size * 2):  # 填充2倍窗口，确保加权平均生效
            env.update_spectator_ultra_smooth(spectator)
            env.world.tick()
        
        print("\n[仿真开始] 车辆将沿车道行驶，按Ctrl+C退出...")
        sys.stdout.flush()
        
        step = 0
        obstacle_distances = {'front': 0, 'rear': 0, 'left': 0, 'right': 0}
        while True:
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            # 严格同步的仿真帧推进
            env.world.tick()
            
            # 终极平滑的镜头更新
            env.update_spectator_ultra_smooth(spectator)
            
            # 极低频率获取观测（减少性能消耗）
            if step % 3 == 0:
                observation = env.get_observation()
                obstacle_distances = env.get_obstacle_directions(observation['lidar_distances'])
            
            # 极低频率打印（每2秒打印一次，减少IO抖动）
            if step % 120 == 0:
                print(f"\n[步骤 {step}] 障碍物距离 - 前{obstacle_distances['front']:.1f}m | 后{obstacle_distances['rear']:.1f}m | "
                      f"左{obstacle_distances['left']:.1f}m | 右{obstacle_distances['right']:.1f}m")
                sys.stdout.flush()
            
            # 锁定渲染帧率（与仿真帧率一致，避免波动）
            clock.tick_busy_loop(60)  # 更精准的帧率锁定
            step += 1

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
    print("="*60)
    sys.stdout.flush()
    run_simulation()