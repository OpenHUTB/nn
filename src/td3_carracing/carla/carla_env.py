import carla
import gymnasium as gym
import numpy as np
import cv2
import time

class CarlaEnv(gym.Env):
    """Carla 模拟器环境包装器"""
    def __init__(self, town="Town03", render_mode="human", max_episode_steps=1000):
        super().__init__()

        self.town = town
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # 连接到 Carla 服务器
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)

        # 动作空间：[steer, throttle, brake]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 观察空间：RGB 图像 (84, 84, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.rgb_camera = None

        self.episode_steps = 0
        self.episode_reward = 0.0

        # 传感器数据
        self.image_data = None
        self.collision_data = None
        self.lane_invasion_data = None

        self.init_carla()

    def init_carla(self):
        """初始化 Carla 世界和车辆"""
        try:
            self.world = self.client.get_world()
            self.map = self.world.get_map()

            # 销毁所有现有车辆（可选）
            for actor in self.world.get_actors().filter('vehicle.*'):
                actor.destroy()

            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            vehicle_bp.set_attribute('role_name', 'hero')

            # 选择出生点
            spawn_points = self.map.get_spawn_points()
            if len(spawn_points) == 0:
                raise RuntimeError("找不到有效的出生点")

            self.spawn_point = np.random.choice(spawn_points)

            # 生成车辆
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
            self.vehicle.set_autopilot(False)

            # 安装传感器
            self.setup_sensors()

            print("Carla 环境初始化成功")

        except Exception as e:
            print(f"Carla 环境初始化失败: {e}")
            raise

    def setup_sensors(self):
        """安装车辆传感器"""
        blueprint_library = self.world.get_blueprint_library()

        # RGB 摄像头
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '320')
        camera_bp.set_attribute('image_size_y', '240')
        camera_bp.set_attribute('fov', '110')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
        self.rgb_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.rgb_camera.listen(lambda data: self.process_image(data))

        # 碰撞传感器
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda data: self.process_collision(data))

        # 车道入侵传感器
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda data: self.process_lane_invasion(data))

    def process_image(self, data):
        """处理摄像头图像数据"""
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        self.image_data = cv2.resize(array, (84, 84), interpolation=cv2.INTER_AREA)

    def process_collision(self, data):
        """处理碰撞数据"""
        self.collision_data = data

    def process_lane_invasion(self, data):
        """处理车道入侵数据"""
        self.lane_invasion_data = data

    def reset(self, **kwargs):
        """重置环境"""
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.image_data = None
        self.collision_data = None
        self.lane_invasion_data = None

        # 重置车辆位置
        try:
            self.vehicle.set_transform(self.spawn_point)
            self.vehicle.set_velocity(carla.Vector3D(0, 0, 0))
            self.vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
        except Exception as e:
            print(f"重置车辆位置失败: {e}")
            return self.get_observation(), {}

        # 等待传感器数据
        while self.image_data is None:
            time.sleep(0.01)

        return self.get_observation(), {}

    def get_observation(self):
        """获取观察数据"""
        return self.image_data.copy()

    def get_speed(self):
        """获取车辆速度（km/h）"""
        velocity = self.vehicle.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

    def step(self, action):
        """执行动作"""
        self.episode_steps += 1

        # 解析动作
        steer = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], 0.0, 1.0)
        brake = np.clip(action[2], 0.0, 1.0)

        # 设置车辆控制
        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer,
            hand_brake=False,
            reverse=False
        )
        self.vehicle.apply_control(control)

        # 等待传感器数据更新
        self.image_data = None
        while self.image_data is None:
            time.sleep(0.01)

        # 计算奖励
        reward = self.calculate_reward()

        # 检查是否结束
        terminated = False
        truncated = False

        if self.episode_steps >= self.max_episode_steps:
            truncated = True

        if self.collision_data is not None:
            terminated = True

        self.episode_reward += reward

        info = {
            'speed': self.get_speed(),
            'collision': self.collision_data is not None,
            'lane_invasion': self.lane_invasion_data is not None,
            'steps': self.episode_steps,
            'episode_reward': self.episode_reward
        }

        return self.get_observation(), reward, terminated, truncated, info

    def calculate_reward(self):
        """计算奖励"""
        reward = 0.0

        # 速度奖励
        speed = self.get_speed()
        reward += max(0.0, min(speed / 30.0, 1.0))

        # 碰撞惩罚
        if self.collision_data is not None:
            reward -= 5.0

        # 车道入侵惩罚
        if self.lane_invasion_data is not None:
            reward -= 2.0

        # 效率奖励（每步基本奖励）
        reward += 0.1

        return reward

    def render(self, mode='human'):
        """渲染环境"""
        pass

    def close(self):
        """关闭环境"""
        try:
            if self.rgb_camera:
                self.rgb_camera.stop()
                self.rgb_camera.destroy()

            if self.collision_sensor:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()

            if self.lane_invasion_sensor:
                self.lane_invasion_sensor.stop()
                self.lane_invasion_sensor.destroy()

            if self.vehicle:
                self.vehicle.destroy()

            print("Carla 环境已关闭")

        except Exception as e:
            print(f"关闭 Carla 环境时出错: {e}")

    def __del__(self):
        """删除对象时关闭环境"""
        self.close()
