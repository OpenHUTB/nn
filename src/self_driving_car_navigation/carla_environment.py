import gym
import carla
import numpy as np
import time
import sys
from queue import Queue
from gym import spaces

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.settings = None  # 用于保存世界设置
        self._connect_carla()

        # 观测空间定义
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            'lidar_distances': gym.spaces.Box(low=0, high=50, shape=(360,), dtype=np.float32),
            'imu': gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        })

        # 传感器和车辆实例
        self.vehicle = None
        self.camera = None
        self.lidar = None
        self.imu = None
        # 数据队列
        self.image_queue = Queue(maxsize=1)
        self.lidar_queue = Queue(maxsize=1)
        self.imu_queue = Queue(maxsize=1)
        # 生成点
        self.spawn_points = self.world.get_map().get_spawn_points()
        print(f"[CARLA场景] 检测到 {len(self.spawn_points)} 个车辆生成点")
        sys.stdout.flush()

    def _connect_carla(self):
        """连接CARLA服务器，支持重试并启用同步模式"""
        retry_count = 3
        for i in range(retry_count):
            try:
                print(f"[CARLA连接] 尝试第{i+1}次连接（localhost:2000）...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)
                self.world = self.client.get_world()
                self.blueprint_library = self.world.get_blueprint_library()
                
                # 关键修改：清除地图中所有默认静态车辆
                actors = self.world.get_actors()
                for actor in actors:
                    if actor.type_id.startswith('vehicle.'):  # 筛选所有车辆类型
                        actor.destroy()
                        print(f"[清除默认车辆] 销毁静态车辆（ID: {actor.id}）")
                
                # 启用同步模式（关键优化：解决数据不同步问题）
                self.settings = self.world.get_settings()
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 1/30  # 固定30帧
                self.world.apply_settings(self.settings)
                
                print("[CARLA连接] 成功连接到模拟器并启用同步模式")
                return
            except Exception as e:
                print(f"[CARLA连接失败] {str(e)}")
                if i == retry_count - 1:
                    raise RuntimeError("无法连接CARLA，请检查模拟器是否启动")
                time.sleep(2)

    def process_image(self, image):
        """处理摄像头数据，转换为RGB格式"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1].copy()  # BGR转RGB
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(array)
        except Exception as e:
            print(f"[图像处理错误] {str(e)}")

    def process_lidar(self, data):
        """处理激光雷达数据，生成360度距离数组"""
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
        """处理IMU数据，提取加速度和角速度"""
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
        """重置环境，生成车辆和传感器（启用CARLA原生自动驾驶）"""
        self.close()
        time.sleep(0.5)
        self._spawn_vehicle()
        if self.vehicle:
            self._spawn_sensors()
            # 启用CARLA原生自动驾驶（关键优化：使用成熟的车道保持逻辑）
            self.vehicle.set_autopilot(True)
        time.sleep(1.0)
        return self.get_observation()

    def _spawn_vehicle(self):
        """生成车辆（特斯拉Model3）- 选择车道内的生成点"""
        import random
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('color', '255,0,0')
        vehicle_bp.set_attribute('role_name', 'ego_vehicle')

        # 优先选择车道内的生成点（减少初始偏离）
        if self.spawn_points:
            random.shuffle(self.spawn_points)
            # 尝试前10个生成点，确保车辆在道路上
            for spawn_point in self.spawn_points[:10]:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle:
                    self.vehicle.set_autopilot(False)
                    self.vehicle.set_simulate_physics(True)
                    print(f"[车辆生成] 成功在道路生成点生成（ID: {self.vehicle.id}）")
                    return
        
        # 备用生成逻辑
        spawn_index = 10
        for i in range(3):
            spawn_point = self.spawn_points[(spawn_index + i) % len(self.spawn_points)]
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle:
                self.vehicle.set_autopilot(False)
                self.vehicle.set_simulate_physics(True)
                print(f"[车辆生成] 使用备用位置（ID: {self.vehicle.id}）")
                return
        
        raise RuntimeError("车辆生成失败，请重启CARLA或更换场景")

    def _spawn_sensors(self):
        """生成传感器（适配0.9.11版本参数，修正激光雷达垂直视野）"""
        # 前视摄像头（优化视角，更贴近驾驶视角）
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '100')  # 扩大视野
        camera_bp.set_attribute('sensor_tick', '0.033')  # 30Hz
        self.camera = self.world.spawn_actor(
            camera_bp, carla.Transform(carla.Location(x=2.0, z=1.5)), attach_to=self.vehicle
        )
        self.camera.listen(self.process_image)

        # 激光雷达（0.9.11兼容参数：用upper_fov和lower_fov替代vertical_fov）
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('horizontal_fov', '360')
        # 0.9.11专用：垂直视野通过上下视角差定义（15 - (-15) = 30度）
        lidar_bp.set_attribute('upper_fov', '15.0')   # 上视角
        lidar_bp.set_attribute('lower_fov', '-15.0')  # 下视角
        self.lidar = self.world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.0)), attach_to=self.vehicle
        )
        self.lidar.listen(self.process_lidar)

        # IMU传感器
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.033')
        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.imu.listen(self.process_imu)
        print("[传感器] 初始化成功")

    def get_observation(self):
        """获取传感器数据（确保数据就绪）"""
        while self.image_queue.empty() or self.lidar_queue.empty() or self.imu_queue.empty():
            time.sleep(0.01)
        return {
            'image': self.image_queue.get(),
            'lidar_distances': self.lidar_queue.get(),
            'imu': self.imu_queue.get()
        }

    def get_obstacle_directions(self, lidar_distances):
        """计算四个方向的最近障碍物距离"""
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
        """执行动作（使用自动驾驶时可忽略action）"""
        # 当启用autopilot时，不需要手动控制
        observation = self.get_observation()
        reward = 1.0
        done = False
        return observation, reward, done, {}

    def close(self):
        """清理资源并恢复世界设置"""
        # 销毁传感器
        for sensor in [self.camera, self.lidar, self.imu]:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        # 销毁车辆
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        # 清空队列
        for q in [self.image_queue, self.lidar_queue, self.imu_queue]:
            while not q.empty():
                q.get()
        # 恢复世界设置
        if self.settings:
            self.settings.synchronous_mode = False
            self.world.apply_settings(self.settings)
        print("[资源清理] 所有传感器和车辆已销毁")