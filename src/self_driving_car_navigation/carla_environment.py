import gym
import carla
import numpy as np
import sys
import time
from queue import Queue

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        self.client = None
        self.world = None
        self.blueprint_library = None
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
        """连接CARLA服务器，支持重试"""
        retry_count = 3
        for i in range(retry_count):
            try:
                print(f"[CARLA连接] 尝试第{i+1}次连接（localhost:2000）...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(15.0)  # 超时时间15秒
                self.world = self.client.get_world()
                self.blueprint_library = self.world.get_blueprint_library()
                print("[CARLA连接] 成功连接到模拟器")
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
            array = array[:, :, :3]  # 移除alpha通道
            array = array[:, :, ::-1]  # BGR转RGB
            array = array.copy()  # 消除负步长

            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(array)
        except Exception as e:
            print(f"[图像处理错误] {str(e)}")

    def process_lidar(self, data):
        """处理激光雷达数据，生成360度距离数组"""
        try:
            # 解析点云数据 (x,y,z,intensity)
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)[:, :3]
            distances = np.linalg.norm(points, axis=1)  # 计算每个点到车辆的距离
            angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi  # 计算角度（度）
            angles = (angles + 360) % 360  # 归一化到0-360度

            # 初始化360度距离数组（默认50米）
            lidar_distances = np.full(360, 50.0, dtype=np.float32)
            # 填充每个角度的最近距离
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
        """重置环境，生成车辆和传感器"""
        self.close()
        time.sleep(0.5)
        self._spawn_vehicle()
        if self.vehicle:

            self._spawn_camera()

        time.sleep(1.0)  # 等待传感器就绪
        return self.get_observation()

    def _spawn_vehicle(self):

        # 选择稳定车型（特斯拉Model3）
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色，便于观察
        vehicle_bp.set_attribute('role_name', 'drone')

        # 关键调整：使用第10个生成点（通常在主路中央，避免障碍物）
        spawn_index = 10  # 可根据场景调整（0~264）
        for i in range(3):
            # 优先用指定生成点，失败则重试
            spawn_point = self.spawn_points[(spawn_index + i) % len(self.spawn_points)]
            print(f"[车辆生成] 尝试在生成点 {spawn_index + i} 生成车辆（主路中央）...")
            sys.stdout.flush()
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle:
                self.vehicle.set_autopilot(False)
                self.vehicle.set_simulate_physics(True)  # 强制启用物理引擎
                print(f"[车辆生成] 成功（ID: {self.vehicle.id}）- 位置：主路中央")
                sys.stdout.flush()

                return
        raise RuntimeError("车辆生成失败，请重启CARLA或更换场景（如Town03）")

    def _spawn_sensors(self):
        """生成摄像头、激光雷达、IMU传感器（核心修正：兼容激光雷达参数）"""
        # 1. 前视摄像头
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')

        # 摄像头位置：车辆前方1.5米，高度2.4米（驾驶员视角）
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        self.camera = self.world.spawn_actor(
            camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=self.vehicle
        )
        self.camera.listen(self.process_image)

        # 2. 激光雷达（关键修正：用upper_fov和lower_fov替代vertical_fov）
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')  # 64线
        lidar_bp.set_attribute('range', '50')  # 最大50米
        lidar_bp.set_attribute('points_per_second', '200000')
        lidar_bp.set_attribute('rotation_frequency', '20')  # 20Hz
        lidar_bp.set_attribute('horizontal_fov', '360')  # 全向扫描
        # 垂直角度范围：-30°到30°（兼容所有版本）
        lidar_bp.set_attribute('upper_fov', '30.0')    # 上角度
        lidar_bp.set_attribute('lower_fov', '-30.0')   # 下角度
        self.lidar = self.world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.0)), attach_to=self.vehicle
        )
        self.lidar.listen(self.process_lidar)

        # 3. IMU传感器
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.imu.listen(self.process_imu)
        print("[传感器] 全向激光雷达+摄像头+IMU初始化成功")

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
        """计算前/后/左/右四个方向的最近障碍物距离"""
        # 角度范围定义（度）
        front_angles = np.concatenate([np.arange(345, 360), np.arange(0, 16)])  # 前方：-15~15°
        rear_angles = np.arange(165, 196)  # 后方：165~195°
        left_angles = np.arange(75, 106)  # 左方：75~105°
        right_angles = np.arange(255, 286)  # 右方：255~285°（-105~-75°）

        return {
            'front': np.min(lidar_distances[front_angles]),
            'rear': np.min(lidar_distances[rear_angles]),
            'left': np.min(lidar_distances[left_angles]),
            'right': np.min(lidar_distances[right_angles])
        }

    def step(self, action):
        """执行动作并返回环境反馈"""
        control = carla.VehicleControl(
            throttle=float(action[0]),
            steer=float(action[1]),
            brake=float(action[2])
        )
        self.vehicle.apply_control(control)
        observation = self.get_observation()
        reward = 1.0  # 基础存活奖励
        done = False
        return observation, reward, done, {}

    def close(self):
        """清理资源（传感器和车辆）"""
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
        print("[资源清理] 所有传感器和车辆已销毁")