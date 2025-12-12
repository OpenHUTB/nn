import carla
import numpy as np
import gym
import time

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        # 初始化CARLA客户端
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # ========== 同步模式核心配置（锁帧率，避免卡死） ==========
        self.sync_settings = self.world.get_settings()
        self.sync_settings.synchronous_mode = True  # 启用同步模式
        self.sync_settings.fixed_delta_seconds = 1.0 / 30  # 锁30fps（平衡流畅+稳定）
        self.sync_settings.no_rendering_mode = False  # 启用渲染（保证视角可见）
        self.world.apply_settings(self.sync_settings)

        # 动作/观测空间
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )

        # 核心对象
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image_data = None
        self.has_collision = False

        # 视角参数（同步模式最优配置）
        self.view_height = 3.8    # 超高视角（3.8米）
        self.view_pitch = -6.0    # 仅俯视6度（接近平视）
        self.view_distance = 4.5  # 正后方4.5米

    def reset(self):
        """重置环境（同步模式下安全初始化）"""
        # 清理旧资源
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        if self.camera is not None and self.camera.is_alive:
            self.camera.destroy()
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        self.image_data = None
        self.has_collision = False

        # 生成车辆（同步模式下容错处理）
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=100, y=100, z=0.5))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)

        # 初始化传感器（同步模式下低延迟配置）
        self._init_camera()
        self._init_collision_sensor()

        # 等待传感器就绪（同步模式下精确等待）
        timeout = 0
        while self.image_data is None and timeout < 30:  # 30帧超时（1秒）
            self.world.tick()  # 同步tick，保证传感器数据更新
            time.sleep(0.001)
            timeout += 1

        # 同步模式下强制绑定视角（无延迟）
        self.follow_vehicle()
        self.world.tick()
        return self.image_data.copy() if self.image_data is not None else np.zeros((128,128,3), dtype=np.uint8)

    def _init_camera(self):
        """同步模式下低延迟相机初始化"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.0')  # 同步模式下无传感器延迟
        # 相机挂载在车辆前方（不影响视角）
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda img: self._camera_callback(img))

    def _init_collision_sensor(self):
        """同步模式下碰撞传感器"""
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, collision_transform, attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._collision_callback(event))

    def _collision_callback(self, event):
        """碰撞回调（同步模式下即时响应）"""
        self.has_collision = True

    def _camera_callback(self, image):
        """相机回调（同步模式下无延迟处理）"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.image_data = array.reshape((image.height, image.width, 4))[:, :, :3]

    def follow_vehicle(self):
        """同步模式下丝滑视角更新（每帧必更，无延迟）"""
        spectator = self.world.get_spectator()
        if not spectator or not self.vehicle or not self.vehicle.is_alive:
            return

        # 同步模式下精准计算正后方位置（无误差）
        vehicle_tf = self.vehicle.get_transform()
        yaw_rad = np.radians(vehicle_tf.rotation.yaw)
        # 正后方绝对坐标（同步模式下无偏移）
        cam_x = vehicle_tf.location.x - (np.cos(yaw_rad) * self.view_distance)
        cam_y = vehicle_tf.location.y - (np.sin(yaw_rad) * self.view_distance)
        cam_z = self.view_height

        # 同步模式下强制更新视角（无延迟）
        spectator.set_transform(carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=self.view_pitch, yaw=vehicle_tf.rotation.yaw, roll=0.0)
        ))

    def get_observation(self):
        """同步模式下即时获取观测"""
        return self.image_data.copy() if self.image_data is not None else np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self, action):
        """同步模式下step（每帧同步，丝滑无延迟）"""
        if self.vehicle is None or not self.vehicle.is_alive:
            raise RuntimeError("车辆未初始化/已销毁，请先调用reset()")

        # 同步模式下极致平滑车辆控制（零抖动）
        throttle = 0.0
        steer = 0.0
        if action == 0:  # 前进
            throttle = 0.5  # 超平缓加速（匹配30fps）
        elif action == 1:  # 左转
            throttle = 0.4
            steer = -0.1    # 微转向（零物理抖动）
        elif action == 2:  # 右转
            throttle = 0.4
            steer = 0.1     # 微转向
        elif action == 3:  # 后退
            throttle = -0.2 # 超平缓后退

        # 同步模式下应用车辆控制（无物理波动）
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, 
            steer=steer,
            hand_brake=False,
            reverse=(throttle < 0),
            gear=1,
            manual_gear_shift=True
        ))
        
        # ========== 同步模式核心：先tick再更新视角（无延迟） ==========
        self.world.tick()  # 同步帧推进（30fps）
        self.follow_vehicle()  # 视角与帧同步更新（丝滑）

        # 同步模式下碰撞检测（即时响应）
        next_state = self.get_observation()
        reward = 0.1 if throttle > 0 else (-0.1 if throttle < 0 else 0.0)
        done = self.has_collision

        return next_state, reward, done, {}

    def close(self):
        """同步模式下安全关闭（必做：恢复异步模式）"""
        # 第一步：恢复CARLA异步模式（避免卡死）
        self.sync_settings.synchronous_mode = False
        self.world.apply_settings(self.sync_settings)

        # 第二步：销毁所有对象
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.destroy()
        if self.camera is not None and self.camera.is_alive:
            self.camera.destroy()
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()

        # 第三步：延迟释放（同步模式下必要）
        time.sleep(0.5)
        print("CARLA环境已关闭（同步模式已恢复为异步）")


