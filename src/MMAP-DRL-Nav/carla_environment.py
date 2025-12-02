import carla
import numpy as np
import gym

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
        
        self.reset()

    def reset(self):

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        

        vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        vehicle_bp.set_attribute('role_name', 'hero')
        
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            spawn_point = carla.Transform(carla.Location(x=20, y=0, z=0.5))
        else:
            spawn_point = spawn_points[0]
        
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:

            if len(spawn_points) > 1:
                spawn_point = spawn_points[1]
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is None:

                raise RuntimeError("无法生成车辆")
        
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

        # 已删除：step里的镜头跟随逻辑 → 后续车辆移动，镜头不再更新
        # self.follow_vehicle()  # 这行已删掉
        
        next_state = self.get_observation()
        reward = 1.0

        done = False
        return next_state, reward, done, {}

    def close(self):
        
        if self.camera is not None:
            self.camera.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        print("环境已清理")