import carla
import random
import math
from source.agent import ActorCar
from source.utility import get_env_settings, map2action

SETTING_FILE = "./config.yaml"


class CarlaEnv(object):
    def __init__(self):
        self.config = get_env_settings(SETTING_FILE)
        self.client = carla.Client(self.config['host'], self.config['port'])
        self.client.set_timeout(15)
        self.world = self.client.get_world()
        self.agent = None
        self.vehicle_control = None
        self.actor_list_env = []
        self.spectator = self.world.get_spectator()
        self.spectator_mode = 'follow'
        self.spectator_offset = carla.Location(x=-8.0, y=0, z=5.0)
        self.spectator_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)
        self.bp = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self._update_settings()
        self.world.apply_settings(self.world_settings)
        print("init actors num", len(self.world.get_actors().filter('vehicle')))

    def _update_settings(self):
        self.world_settings = self.world.get_settings()
        if self.config['syn'] is not None and self.config['substepping'] is not None:
            self.world_settings.synchronous_mode = True
            self.world_settings.fixed_delta_seconds = self.config['syn']['fixed_delta_seconds']
            self.world_settings.substepping = True
            self.world_settings.max_substep_delta_time = self.config['substepping']['max_substep_delta_time']
            self.world_settings.max_substeps = self.config['substepping']['max_substeps']

    def _set_env(self):
        cars = self.bp.filter("vehicle")
        print(f"set {self.config['car_num']} vehicles in the world")

        available_points = list(range(len(self.spawn_points)))
        random.shuffle(available_points)

        spawned = 0
        attempts = 0
        max_attempts = self.config['car_num'] * 3

        while spawned < self.config['car_num'] and attempts < max_attempts:
            attempts += 1
            if not available_points:
                available_points = list(range(len(self.spawn_points)))
                random.shuffle(available_points)

            idx = available_points.pop()
            try:
                car = self.world.spawn_actor(random.choice(cars), self.spawn_points[idx])
                car.set_autopilot(True)
                self.actor_list_env.append(car)
                spawned += 1
            except RuntimeError:
                continue

        print(f"Spawned {spawned} vehicles")

        self.agent = ActorCar(self.client, self.world, self.bp, self.spawn_points, self.config)
        self.vehicle_control = self.agent.actor_car.apply_control

    def step(self, action_index):
        action = map2action(action_index)
        assert isinstance(action, carla.VehicleControl), "action type is not vehicle control"
        self.vehicle_control(action)
        frame_index = self.world.tick()
        self.update_spectator()
        self.draw_speed_hud()
        observation, collision = self.agent.retrieve_data(frame_index)
        reward = self.get_reward(action_index, collision)
        self.draw_reward_hud(reward, collision)
        done = 1 if collision != 0 else 0
        return observation, reward, done

    def reset(self):
        print("initialize environment.")
        self.cleanup_world()
        self.client.set_timeout(15)
        self._update_settings()
        self._set_env()

        print("Waiting for sensors to initialize...")
        for i in range(10):
            self.world.tick()
            self.update_spectator()
            if i % 2 == 0:
                print(f"  Tick {i}...")

        frame_index = self.world.tick()
        self.update_spectator()
        print(f"after reset, current frame is: {frame_index}")

        print("Getting initial observation...")
        retry_count = 0
        obs, collision = self.agent.retrieve_data(frame_index)
        while obs is None and retry_count < 50:
            frame_index = self.world.tick()
            self.update_spectator()
            obs, collision = self.agent.retrieve_data(frame_index)
            retry_count += 1
            if retry_count % 10 == 0:
                print(f"  Waiting for camera... (retry {retry_count})")

        if obs is None:
            print("Warning: Failed to get initial observation, using None")

        print(f"Total vehicles: {len(self.world.get_actors().filter('*vehicle*'))}")
        return obs, collision

    def get_reward(self, action_index, intensity):
        if intensity != 0:
            return -200
        if action_index == 3:
            return -100
        elif action_index == 0:
            return 5
        else:
            return 1

    def cleanup_world(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list_env])
        if self.agent is not None:
            self.agent.cleanup()
        self.agent = None
        self.actor_list_env = []
        print("clean up the world, after cleanup world actors: ", len(self.world.get_actors().filter('vehicle')))
        assert len(self.world.get_actors().filter('vehicle')) == 0, "cleanup world wrong"

    def get_all_actors(self):
        return self.world.get_actors()

    def get_all_vehicles(self):
        return self.world.get_actors().filter('vehicle')

    def update_spectator(self):
        if self.agent is not None and self.agent.actor_car is not None:
            vehicle_transform = self.agent.actor_car.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation

            if self.spectator_mode == 'top':
                new_transform = carla.Transform(
                    carla.Location(x=vehicle_location.x, y=vehicle_location.y, z=50),
                    carla.Rotation(pitch=-90, yaw=vehicle_rotation.yaw, roll=0)
                )
            elif self.spectator_mode == 'first':
                cam_transform = carla.Transform(carla.Location(x=1.2, z=1.5))
                new_transform = vehicle_transform.transform(cam_transform)
                new_transform.rotation.roll = 0
            else:
                forward_vec = carla.Vector3D(x=vehicle_location.x + self.spectator_offset.x,
                                            y=vehicle_location.y + self.spectator_offset.y,
                                            z=vehicle_location.z + self.spectator_offset.z)
                new_transform = carla.Transform(
                    forward_vec,
                    carla.Rotation(pitch=self.spectator_rotation.pitch + vehicle_rotation.pitch,
                                  yaw=vehicle_rotation.yaw + self.spectator_rotation.yaw,
                                  roll=vehicle_rotation.roll + self.spectator_rotation.roll)
                )
            self.spectator.set_transform(new_transform)

    def set_spectator_mode(self, mode):
        if mode in ['follow', 'top', 'first']:
            self.spectator_mode = mode
            print(f"Spectator mode changed to: {mode}")
        else:
            print(f"Invalid mode: {mode}. Use 'follow', 'top', or 'first'")

    def get_speed(self):
        if self.agent and self.agent.actor_car:
            velocity = self.agent.actor_car.get_velocity()
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_ms * 3.6
            return speed_kmh, speed_ms
        return 0.0, 0.0

    def print_speed(self):
        speed_kmh, speed_ms = self.get_speed()
        print(f"Speed: {speed_kmh:.1f} km/h ({speed_ms:.1f} m/s)")
        return speed_kmh, speed_ms

    def draw_speed_hud(self):
        if self.agent and self.agent.actor_car:
            speed_kmh, speed_ms = self.get_speed()
            vehicle_transform = self.agent.actor_car.get_transform()
            hud_location = vehicle_transform.location + carla.Location(x=0, y=0, z=5)
            
            self.world.debug.draw_string(
                hud_location,
                f"Speed: {speed_kmh:.1f} km/h",
                color=carla.Color(255, 255, 0),
                life_time=0.1,
                draw_shadow=True
            )

    def draw_reward_hud(self, reward, collision):
        if self.agent and self.agent.actor_car:
            vehicle_transform = self.agent.actor_car.get_transform()
            vehicle_location = vehicle_transform.location
            
            # Draw reward info behind the vehicle
            behind_dir = carla.Vector3D(
                x=-math.cos(math.radians(vehicle_transform.rotation.yaw)) * 8,
                y=-math.sin(math.radians(vehicle_transform.rotation.yaw)) * 8,
                z=3
            )
            hud_location = carla.Location(
                x=vehicle_location.x + behind_dir.x,
                y=vehicle_location.y + behind_dir.y,
                z=vehicle_location.z + behind_dir.z
            )
            
            # Color based on reward
            if collision != 0:
                text = f"COLLISION! Reward: {reward:.1f}"
                color = carla.Color(255, 0, 0)
            elif reward > 0:
                text = f"Reward: +{reward:.1f}"
                color = carla.Color(0, 255, 0)
            else:
                text = f"Reward: {reward:.1f}"
                color = carla.Color(255, 255, 0)
            
            self.world.debug.draw_string(
                hud_location,
                text,
                color=color,
                life_time=0.5,
                draw_shadow=True
            )

    def exit_env(self):
        self.cleanup_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        print(f"before exited, there are {len(self.get_all_vehicles())} actors")
        print("exit world")

    def reward_sac(self, collision):
        if collision != 0:
            return -200
        else:
            return 1

    def step_sac(self, action):
        assert isinstance(action, carla.VehicleControl), "action is not the carla type."
        self.vehicle_control(action)
        frame_index = self.world.tick()
        self.update_spectator()
        self.draw_speed_hud()
        observation, collision = self.agent.retrieve_data(frame_index)
        reward = self.reward_sac(collision)
        self.draw_reward_hud(reward, collision)
        done = 1 if collision != 0 else 0
        return observation, reward, done