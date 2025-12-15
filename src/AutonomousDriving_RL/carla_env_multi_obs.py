# carla_env/carla_env_multi_obs.py
import carla
import numpy as np
import random
import time
from gymnasium import Env, spaces


class CarlaEnvMultiObs(Env):
    def __init__(self):
        super(CarlaEnvMultiObs, self).__init__()
        self.client = None
        self.world = None
        self.vehicle = None
        self.actor_list = []
        self.frame_count = 0
        self.max_frames = 1000

        # 观测空间：[x, y, speed_x, speed_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.client is None:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(60.0)
            print("🔄 连接 CARLA...")
            self.world = self.client.get_world()  # 不 reload_world！
        else:
            self._destroy_actors()

        self.spawn_vehicle()
        self.frame_count = 0
        return self.get_observation(), {}

    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, transform)
        self.actor_list.append(self.vehicle)
        print(f"🚗 车辆生成: {self.vehicle.type_id}")

    def _destroy_actors(self):
        for actor in self.actor_list:
            if actor and actor.is_alive:
                actor.destroy()
        self.actor_list.clear()
        for _ in range(3):
            self.world.tick()
            time.sleep(0.1)

    def get_observation(self):
        if not self.vehicle or not self.vehicle.is_alive:
            return np.zeros(4, dtype=np.float32)
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        return np.array([loc.x, loc.y, vel.x, vel.y], dtype=np.float32)

    def step(self, action):
        throttle, steer, brake = action
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)
        self.world.tick()
        self.frame_count += 1

        obs = self.get_observation()
        reward = float(np.linalg.norm([obs[2], obs[3]]))  # 速度奖励
        terminated = False
        truncated = self.frame_count >= self.max_frames
        return obs, reward, terminated, truncated, {}

    def close(self):
        self._destroy_actors()