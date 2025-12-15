import gym
import carla
import numpy as np
import time
import sys
import random
from queue import Queue
from gym import spaces

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
        # TM核心配置（0.9.11专用）
        self.traffic_manager = None
        self.tm_port = 8000
        self.tm_seed = 0  # 固定TM种子，保证行为一致
        # 数据队列
        self.image_queue = Queue(maxsize=1)
        self.lidar_queue = Queue(maxsize=1)
        self.imu_queue = Queue(maxsize=1)
        # 车辆控制相关
        self.vehicle_control = carla.VehicleControl()
        self.last_steer = 0.0

        # 连接CARLA
        self._connect_carla()
        # 初始化TM（适配0.9.11 API）
        self._init_traffic_manager()
        # 定义观测空间
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            'lidar_distances': gym.spaces.Box(low=0, high=50, shape=(360,), dtype=np.float32),
            'imu': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        })
        # 获取有效生成点（仅保留道路上的点，修改：补充到60个以适配60辆NPC）
        self.spawn_points = self._get_valid_road_spawn_points()
        print(f"[场景初始化] 有效道路生成点数量: {len(self.spawn_points)}")
        sys.stdout.flush()

    def _connect_carla(self):
        """连接CARLA（增加版本检查）"""
        retry_count = 3
        for i in range(retry_count):
            try:
                print(f"[CARLA连接] 尝试第{i+1}次连接（localhost:2000）...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(20.0)
                self.world = self.client.get_world()
                self.blueprint_library = self.world.get_blueprint_library()
                
                # 同步模式配置（0.9.11最优参数）
                self.settings = self.world.get_settings()
                self.settings.synchronous_mode = True  # 启用同步模式
                self.settings.fixed_delta_seconds = 1/20  # 降低帧率，提升稳定性
                self.settings.no_rendering_mode = False  # 必须开启渲染，否则TM可能失效
                self.world.apply_settings(self.settings)
                
                # 双重清理
                self._clear_all_non_ego_actors()
                time.sleep(0.5)
                self.world.tick()  # 显式推进仿真帧
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
        """初始化交通管理器（严格适配0.9.11 API）"""
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        # 全局TM参数（0.9.11支持的全局方法）
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)  # 跟车距离（float）
        self.traffic_manager.set_synchronous_mode(True)  # 同步模式（bool）
        self.traffic_manager.set_random_device_seed(self.tm_seed)  # 随机种子（int）
        self.traffic_manager.global_percentage_speed_difference(0.0)  # 全速行驶（float）
        # 混合物理模式（0.9.11支持）
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(50.0)
        print("[TM配置] 交通管理器初始化完成（适配0.9.11 API）")

    def _set_actor_tm_params(self, actor):
        """为单个Actor设置TM参数（核心修改：提升主车辆速度，遵守交通规则）"""
        if not self._is_actor_alive(actor):
            return
        try:
            # 关键修改：设置为0%忽略交通规则，使车辆遵守红绿灯和标志
            self.traffic_manager.ignore_lights_percentage(actor, 0.0)  # 不忽略交通灯
            self.traffic_manager.ignore_signs_percentage(actor, 0.0)   # 不忽略交通标志
            self.traffic_manager.ignore_walkers_percentage(actor, 0.0) # 不忽略行人
            # 允许变道（float百分比）
            self.traffic_manager.allow_vehicle_lane_change(actor, 100.0)
            
            # 核心修改：提升速度参数（区分主车辆和NPC）
            if actor.attributes.get('role_name') == 'ego_vehicle':
                # 主车辆：速度限制因子提升到1.8（超速80%），最高速度设为100km/h
                self.traffic_manager.set_speed_limit_factor(actor, 1.8)
                self.traffic_manager.set_speed_limit(actor, 100.0)
            else:
                # NPC车辆：保持原有参数（可选：也可适当提升）
                self.traffic_manager.set_speed_limit_factor(actor, 1.2)
                self.traffic_manager.set_speed_limit(actor, 60.0)
                
        except Exception as e:
            print(f"[TM Actor配置警告] Actor ID {actor.id}: {str(e)}")

    def _update_vehicle_steering(self, vehicle):
        """更新车辆转向角，使轮胎转动跟随车辆运动"""
        if not self._is_actor_alive(vehicle):
            return
            
        try:
            # 获取车辆当前速度和变换
            velocity = vehicle.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
            
            # 获取车辆的物理控制
            physics_control = vehicle.get_physics_control()
            
            # 根据自动驾驶的控制指令获取转向角
            control = vehicle.get_control()
            
            # 平滑转向角变化，避免突变
            steer_factor = 0.3  # 转向灵敏度
            self.last_steer = self.last_steer * (1 - steer_factor) + control.steer * steer_factor
            
            # 应用转向角到所有车轮
            for wheel in physics_control.wheels:
                if wheel.type == carla.WheelType.Front:  # 只控制前轮转向
                    wheel.steer_angle = self.last_steer * 70  # 70度最大转向角
            
            # 应用物理控制
            vehicle.apply_physics_control(physics_control)
            
        except Exception as e:
            print(f"[车辆转向更新错误] {str(e)}")

    def _get_valid_road_spawn_points(self):
        """过滤生成点：仅保留道路网络上的有效点（修改：补充到60个以适配60辆NPC）"""
        map = self.world.get_map()
        valid_points = []
        for sp in map.get_spawn_points():
            # 获取生成点对应的道路点
            waypoint = map.get_waypoint(sp.location)
            if waypoint and waypoint.road_id != -1:  # 确保在道路上
                valid_points.append(sp)
        # 若有效点不足60，补充随机道路点（修改：从30改为60）
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
        
        self.world.tick()  # 显式推进仿真帧
        print(f"[清理] 车辆{cleared_count['vehicle']} | 自行车{cleared_count['bicycle']} | 静态车辆{cleared_count['static_vehicle']}")

    def process_image(self, image):
        """处理摄像头数据"""
        try:
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
        """重置环境（修改：生成主车辆+60辆NPC）"""
        self.close()
        time.sleep(1.0)
        self._clear_all_non_ego_actors()
        self._spawn_vehicle()
        
        if self.vehicle:
            self._spawn_sensors()
            # 启用物理模拟
            self.vehicle.set_simulate_physics(True)
            # 主车辆自动驾驶（延迟绑定+TM参数配置）
            time.sleep(0.5)
            self.vehicle.set_autopilot(True, self.tm_port)  # 启用自动控制指令输入
            self._set_actor_tm_params(self.vehicle)  # 为主车辆配置TM参数
            # 生成NPC车辆（核心修改：从20辆改为60辆）
            self._spawn_npcs(60)
            self._clear_all_non_ego_actors()
        
        # 多次同步，确保物理生效
        for _ in range(5):
            self.world.tick()  # 显式推进仿真帧
            time.sleep(0.2)
        
        print(f"[环境重置] 完成，主车辆1辆，NPC车辆{len(self.npc_vehicles)}辆")
        return self.get_observation()

    def _spawn_vehicle(self):
        """生成主车辆（特斯拉Model3）"""
        self._safe_destroy_actor(self.vehicle)
        self.world.tick()  # 显式推进仿真帧

        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('color', '255,0,0')
        vehicle_bp.set_attribute('role_name', 'ego_vehicle')

        if not self.spawn_points:
            raise RuntimeError("无可用道路生成点")

        random.shuffle(self.spawn_points)
        for spawn_point in self.spawn_points[:10]:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle:
                self.vehicle.set_simulate_physics(True)  # 启用物理模拟
                print(f"[主车辆生成] 成功（ID: {self.vehicle.id}）")
                return

        raise RuntimeError("主车辆生成失败，请重启CARLA")

    def _spawn_npcs(self, count):
        """生成NPC车辆（0.9.11关键：生成后延迟启用自动驾驶+单个Actor配置TM）"""
        if not self.spawn_points:
            print("[NPC生成] 无可用生成点")
            return

        # 过滤主车辆附近的点（修改：将距离从20米改为15米，释放更多生成点）
        ego_transform = self.vehicle.get_transform()
        available_spawn_points = []
        for sp in self.spawn_points:
            distance = np.linalg.norm([
                sp.location.x - ego_transform.location.x,
                sp.location.y - ego_transform.location.y
            ])
            if distance > 15.0:  # 从20米→15米，增加可用生成点数量
                available_spawn_points.append(sp)

        if len(available_spawn_points) < count:
            count = len(available_spawn_points)
            print(f"[NPC生成] 可用点不足，生成{count}辆")

        # 随机车辆蓝图
        vehicle_bps = [bp for bp in self.blueprint_library.filter('vehicle.*') 
                       if bp.has_attribute('color') and bp.id != 'vehicle.tesla.model3']
        random.shuffle(vehicle_bps)
        if not vehicle_bps:
            vehicle_bps = self.blueprint_library.filter('vehicle.*')

        # 生成NPC
        spawned_count = 0
        for i, spawn_point in enumerate(random.sample(available_spawn_points, count)):
            bp = vehicle_bps[i % len(vehicle_bps)]
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            bp.set_attribute('role_name', 'npc_vehicle')

            npc_vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if npc_vehicle:
                npc_vehicle.set_simulate_physics(True)  # 启用物理模拟
                # 0.9.11核心：生成后延迟0.1秒启用自动驾驶，让物理模拟生效
                time.sleep(0.1)
                npc_vehicle.set_autopilot(True, self.tm_port)  # 启用自动控制指令输入
                # 为单个NPC配置TM参数（关键：解决不动问题）
                self._set_actor_tm_params(npc_vehicle)
                self.npc_vehicles.append(npc_vehicle)
                spawned_count += 1

                if spawned_count % 5 == 0:
                    self.world.tick()  # 显式推进仿真帧

        self.world.tick()  # 显式推进仿真帧
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
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
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
        print("[传感器] 初始化完成")

    def get_observation(self):
        """获取观测数据"""
        while self.image_queue.empty() or self.lidar_queue.empty() or self.imu_queue.empty():
            time.sleep(0.01)
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
        """环境交互步骤（0.9.11同步关键）"""
        # 同步世界（TM会自动同步）
        self.world.tick()  # 显式推进仿真帧
        
        # 更新主车辆轮胎转向
        if self._is_actor_alive(self.vehicle):
            self._update_vehicle_steering(self.vehicle)
            
        # 更新NPC车辆轮胎转向
        for npc in self.npc_vehicles:
            if self._is_actor_alive(npc):
                self._update_vehicle_steering(npc)
        
        # 清理无效NPC
        self.npc_vehicles = [npc for npc in self.npc_vehicles if self._is_actor_alive(npc)]
        
        # 打印NPC速度（调试用）
        if random.random() < 0.1:  # 10%概率打印
            for npc in self.npc_vehicles[:1]:
                try:
                    velocity = npc.get_velocity()
                    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z]) * 3.6  # m/s → km/h
                    print(f"[NPC速度] ID:{npc.id} 速度:{speed:.1f}km/h")
                except Exception:
                    pass
        
        # 获取观测
        observation = self.get_observation()
        reward = 1.0
        done = False
        return observation, reward, done, {}

    def close(self):
        """清理资源"""
        # 停止TM
        if self.traffic_manager:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception as e:
                print(f"[TM清理警告] {str(e)}")
        # 销毁传感器
        for sensor in [self.camera, self.lidar, self.imu]:
            self._safe_destroy_actor(sensor)
        # 销毁NPC
        for npc in self.npc_vehicles:
            self._safe_destroy_actor(npc)
        self.npc_vehicles = []
        # 销毁主车辆
        self._safe_destroy_actor(self.vehicle)
        self.vehicle = None
        # 最后清理
        self._clear_all_non_ego_actors()
        # 清空队列
        for q in [self.image_queue, self.lidar_queue, self.imu_queue]:
            while not q.empty():
                q.get()
        # 恢复世界设置
        if self.settings:
            try:
                self.settings.synchronous_mode = False
                self.world.apply_settings(self.settings)
            except Exception as e:
                print(f"[世界设置恢复警告] {str(e)}")
        print("[资源清理] 所有资源已销毁")


if __name__ == "__main__":
    # 测试环境
    try:
        env = CarlaEnvironment()
        print("环境初始化完成，开始测试...")
        obs = env.reset()
        print(f"观测数据：图像{obs['image'].shape}，激光雷达{obs['lidar_distances'].shape}，IMU{obs['imu'].shape}")
        # 运行600步（约30秒）
        for i in range(600):
            obs, reward, done, _ = env.step()
            if i % 50 == 0:
                obstacle_info = env.get_obstacle_directions(obs['lidar_distances'])
                print(f"第{i}步 - 前向距离：{obstacle_info['front']:.2f}m，NPC数量：{len(env.npc_vehicles)}")
            time.sleep(0.05)
        env.close()
        print("测试完成")
    except Exception as e:
        print(f"测试出错：{str(e)}")
        if 'env' in locals():
            env.close()
        sys.exit(1)