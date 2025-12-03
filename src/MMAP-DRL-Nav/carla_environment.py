import carla
import numpy as np
import gym
import random  # 新增：用于随机选spawn点
import time    # 新增：用于销毁后延迟

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        
        self.vehicle = None
        self.camera = None
        
        # 新增：镜头跟随参数（仅用于初始化跳转）
        self.spectator_offset = carla.Location(x=0, y=0, z=2.5)
        self.spectator_distance = -5.0
        self.spectator_pitch = -10
        
        # 新增：初始化时先清理所有残留actor
        self._clean_all_actors()
        
        self.reset()

    # 新增：核心清理函数 - 销毁所有残留的车辆/传感器/行人等actor
    def _clean_all_actors(self):
        # 获取当前世界所有actor
        actor_list = self.world.get_actors()
        for actor in actor_list:
            # 筛选需要销毁的actor类型：车辆、传感器、行人（可根据需求调整）
            if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor') or actor.type_id.startswith('walker'):
                try:
                    actor.destroy()
                    time.sleep(0.05)  # 短暂延迟，确保销毁完成
                except Exception as e:
                    print(f"销毁actor失败: {e}")
        # 额外清理当前实例的车辆和相机
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

    def reset(self):
        # 第一步：先清理残留车辆（增强版）
        self._clean_all_actors()

        vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        vehicle_bp.set_attribute('role_name', 'hero')
        
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            spawn_point = carla.Transform(carla.Location(x=20, y=0, z=0.5))
        else:
            # 改动1：随机打乱spawn点，避免固定选前几个
            random.shuffle(spawn_points)
        
        # 改动2：循环尝试多个spawn点（最多尝试10个），提高生成成功率
        self.vehicle = None
        max_attempts = min(10, len(spawn_points))  # 最多试10个点（或所有点）
        for i in range(max_attempts):
            spawn_point = spawn_points[i] if spawn_points else carla.Transform(carla.Location(x=20, y=0, z=0.5))
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                break  # 生成成功，退出循环
            time.sleep(0.1)  # 失败后短暂延迟，再试下一个点
        
        # 最终检查：如果所有点都失败，抛出更友好的错误
        if self.vehicle is None:
            raise RuntimeError(f"尝试了{max_attempts}个spawn点仍无法生成车辆，请检查CARLA模拟器状态或手动清理地图")
        
        self.vehicle.set_autopilot(False)
        self.world.tick()

        # 仅保留：车辆生成时，镜头跳转到车辆旁（核心需求）
        self.follow_vehicle()
        
        return self.get_observation()

    # 镜头跳转核心方法（仅初始化时调用一次）
    def follow_vehicle(self):
        spectator = self.world.get_spectator()
        if not spectator or not self.vehicle:
            return
        vehicle_transform = self.vehicle.get_transform()
        camera_location = vehicle_transform.location + carla.Location(x=self.spectator_distance) + self.spectator_offset
        camera_rotation = carla.Rotation(
            pitch=self.spectator_pitch,
            yaw=vehicle_transform.rotation.yaw,
            roll=0
        )
        spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def get_observation(self):
        if self.camera is None:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '128')
            camera_bp.set_attribute('image_size_y', '128')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        return np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("车辆未初始化，请先调用reset()")
        
        throttle = 0.0
        steer = 0.0
        if action == 0:
            throttle = 0.5
        elif action == 1:
            throttle = 0.3
            steer = -0.5
        elif action == 2:
            throttle = 0.3
            steer = 0.5
        elif action == 3:
            throttle = -0.3
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        self.world.tick()
        
        next_state = self.get_observation()
        reward = 1.0
        done = False
        return next_state, reward, done, {}

    def close(self):
        # 改动3：关闭时调用全局清理，确保无残留
        self._clean_all_actors()
        print("环境已清理，所有actor已销毁")