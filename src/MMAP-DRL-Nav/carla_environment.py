import carla
import numpy as np
import gym

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)  # 增加超时时间，避免连接失败
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()  # 缓存蓝图库
        
        # 动作和观测空间保持不变
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        
        self.vehicle = None
        self.camera = None  # 后续可添加相机传感器
        
        # 初始化时直接生成车辆（或在首次reset时生成）
        self.reset()  # 确保车辆在环境创建时就初始化

    def reset(self):
        # 清理现有车辆
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        
        # 选择车辆蓝图（优先选可驾驶的车辆）
        vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        vehicle_bp.set_attribute('role_name', 'hero')  # 标记为主车辆
        
        # 选择有效的spawn点（优先用地图默认点）
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            # 如果没有默认点，使用自定义点（但尽量避免）
            spawn_point = carla.Transform(carla.Location(x=20, y=0, z=0.5))  # 调整到合理位置
        else:
            spawn_point = spawn_points[0]  # 用第一个默认点
        
        # 生成车辆并检查是否成功
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            # 尝试另一个spawn点（防止第一个点被占用）
            if len(spawn_points) > 1:
                spawn_point = spawn_points[1]
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is None:
                raise RuntimeError("无法生成车辆，请检查spawn点是否有效或CARLA是否正常运行")
        
        # 禁用自动驾驶，由代码控制
        self.vehicle.set_autopilot(False)
        self.world.tick()
        
        return self.get_observation()

    def get_observation(self):
        # 实际项目中需要添加相机传感器获取图像，这里先返回模拟数据
        # 示例：添加相机（简化版）
        if self.camera is None:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '128')
            camera_bp.set_attribute('image_size_y', '128')
            # 相机安装在车辆前方
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            # 这里需要设置相机回调函数获取图像，简化起见返回随机图像
        return np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("车辆未初始化，请先调用reset()")
        
        # 控制信号限制在有效范围（避免极端值）
        throttle = 0.0
        steer = 0.0
        if action == 0:  # 前进
            throttle = 0.5  # 适中油门，避免速度过快
        elif action == 1:  # 左转
            throttle = 0.3
            steer = -0.5
        elif action == 2:  # 右转
            throttle = 0.3
            steer = 0.5
        elif action == 3:  # 后退
            throttle = -0.3  # 后退油门绝对值较小
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        self.world.tick()
        
        next_state = self.get_observation()
        reward = 1.0  # 后续需根据任务设计奖励函数
        done = False
        return next_state, reward, done, {}

    def close(self):
        # 清理所有生成的actor（车辆、相机等）
        if self.camera is not None:
            self.camera.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        print("环境已清理")