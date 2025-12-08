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

        # 定义动作空间和观测空间
        self.action_space = gym.spaces.Discrete(4)  # 前进、左转、右转、后退
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )  # 128x128 RGB图像

        # 核心对象初始化
        self.vehicle = None
        self.camera = None
        self.image_data = None  # 存储相机采集的图像数据

        # 镜头跟随参数（spectator视角）
        self.spectator_offset = carla.Location(x=0, y=0, z=2.5)  # 高度偏移
        self.spectator_distance = -5.0  # 车辆后方5米
        self.spectator_pitch = -10  # 向下俯视10度

    def reset(self):
        """重置环境，生成新车辆和相机，返回初始观测"""
        # 清理旧的车辆和相机资源
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()
            self.image_data = None

        # 生成车辆（特斯拉Model3）
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0, y=0, z=0))
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(False)

        # 初始化RGB相机
        self._init_camera()

        # 等待相机采集到第一帧数据
        while self.image_data is None:
            time.sleep(0.01)

        # 镜头跳转到车辆位置并跟随
        self.follow_vehicle()

        self.world.tick()
        return self.image_data.copy()

    def _init_camera(self):
        """初始化挂载在车辆上的RGB相机"""
        # 创建相机蓝图并设置参数
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '128')
        camera_bp.set_attribute('image_size_y', '128')
        camera_bp.set_attribute('fov', '90')  # 视野角度

        # 相机挂载位置：车辆前上方（x=1.5米，z=2.0米）
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        # 注册相机数据回调函数
        self.camera.listen(lambda img: self._camera_callback(img))

    def _camera_callback(self, image):
        """相机数据回调：将原始数据转换为RGB numpy数组"""
        # CARLA相机输出为RGBA格式（4通道），转换为RGB（3通道）
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.image_data = array[:, :, :3]  # 去掉Alpha通道

    def follow_vehicle(self):
        """调整spectator视角，让镜头跟随车辆"""
        spectator = self.world.get_spectator()
        if not spectator or not self.vehicle:
            return

        # 计算镜头位置（车辆后方+高度偏移）
        vehicle_transform = self.vehicle.get_transform()
        camera_location = vehicle_transform.location + carla.Location(x=self.spectator_distance) + self.spectator_offset
        # 计算镜头朝向（与车辆一致，向下俯视）
        camera_rotation = carla.Rotation(
            pitch=self.spectator_pitch,
            yaw=vehicle_transform.rotation.yaw,
            roll=0
        )
        # 设置镜头位置和朝向
        spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def get_observation(self):
        """获取当前相机图像（模型输入的观测值）"""
        if self.image_data is not None:
            return self.image_data.copy()
        # 兜底：相机未就绪时返回全零图像
        return np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self, action):
        """执行动作，返回新状态、奖励、终止标志等"""
        if self.vehicle is None:
            raise RuntimeError("车辆未初始化，请先调用reset()")

        # 将动作映射为车辆控制指令
        throttle = 0.0
        steer = 0.0
        if action == 0:  # 前进
            throttle = 1.0
        elif action == 1:  # 左转
            throttle = 0.5
            steer = -1.0
        elif action == 2:  # 右转
            throttle = 0.5
            steer = 1.0
        elif action == 3:  # 后退
            throttle = -1.0

        # 应用车辆控制
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        self.world.tick()

        # 镜头实时跟随车辆
        self.follow_vehicle()

        # 获取新状态、计算奖励、判断终止
        next_state = self.get_observation()
        reward = self._calculate_reward(throttle)  # 自定义奖励函数
        done = self._check_done()  # 自定义终止条件

        return next_state, reward, done, {}

    def _calculate_reward(self, throttle):
        """奖励函数：鼓励前进，惩罚后退"""
        if throttle > 0:
            return 0.1  # 前进奖励
        elif throttle < 0:
            return -0.1  # 后退惩罚
        return 0.0  # 无动作无奖励

    def _check_done(self):
        """终止条件：示例为永不终止（可根据需求修改）"""
        # 可扩展：碰撞检测、到达目标、超时等
        return False

    def close(self):
        """关闭环境，清理所有资源"""
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()
        print("CARLA环境已关闭，资源清理完成")


