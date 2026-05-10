import carla
import random
from source.agent import ActorCar
from source.utility import get_env_settings, map2action

SETTING_FILE = "./config.yaml"


class CarlaEnv(object):
    def __init__(self):
        """initialize the environment.
        important members:
            self.config
            self.client
            self.world
            self.agent
            self.traffic_manager
        """
        self.config = get_env_settings(SETTING_FILE)
        self.client = carla.Client(self.config['host'], self.config['port'])
        self.client.set_timeout(15)
        self.world = self.client.get_world()
        self.agent = None
        self.vehicle_control = None
        self.actor_list_env = []
        self.spectator = self.world.get_spectator()
        self.spectator_offset = carla.Location(x=-8.0, y=0, z=5.0)
        self.spectator_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)
        # get blueprint and spawn_points
        self.bp = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        # update settings
        self._update_settings()
        self.world.apply_settings(self.world_settings)
        print("init actors num", len(self.world.get_actors().filter(
            'vehicle')))

    def _update_settings(self):
        self.world_settings = self.world.get_settings()
        if self.config['syn'] is not None and self.config['substepping'] is not None:
            self.world_settings.synchronous_mode = True
            self.world_settings.fixed_delta_seconds = self.config['syn']['fixed_delta_seconds']
            self.world_settings.substepping = True
            self.world_settings.max_substep_delta_time = self.config[
                'substepping']['max_substep_delta_time']
            self.world_settings.max_substeps = self.config['substepping']['max_substeps']

    def _set_env(self):
        """adding npc cars and create actor."""
        cars = self.bp.filter("vehicle")
        print(f"set {self.config['car_num']} vehicles in the world")
        for i in range(self.config['car_num']):
            car = self.world.spawn_actor(
                random.choice(cars), self.spawn_points[i])
            car.set_autopilot(True)
            self.actor_list_env.append(car)
        # print(f"setting {len(self.actor_list_env)} in _set_env")
        # adding agent(combination of car and camera)
        self.agent = ActorCar(self.client, self.world,
                              self.bp, self.spawn_points, self.config)
        self.vehicle_control = self.agent.actor_car.apply_control

    def step(self, action_index):
        """take an action.
        Args:
            action(carla.VehicleControl):throttle, steer, break, hand_break, reverse
        Returns:
            observation(np.array(640, 480, 3))
            reward(int)
            done(bool)
        """
        action = map2action(action_index)
        assert isinstance(
            action, carla.VehicleControl), "action type is not vehicle control"
        print("take: ", action)
        self.vehicle_control(action)
        frame_index = self.world.tick()
        self.update_spectator()
        print(f"after step, current frame is: {frame_index}")
        observation, collision = self.agent.retrieve_data(frame_index)
        reward = self.get_reward(action_index, collision)
        done = 1 if collision != 0 else 0
        return observation, reward, done

    def reset(self):
        """reset the environment while keeping the init settings."""
        # set false to keep the settings in sync
        print("initialize environment.")
        self.cleanup_world()
        self.client.set_timeout(15)
        # adding cars to env
        self._update_settings()
        self._set_env()
        # deploy env in sync mode
        frame_index = self.world.tick()
        self.update_spectator()
        print(f"after reset, current frame is: {frame_index}")
        assert len(self.world.get_actors().filter(
            '*vehicle*')) == (self.config['car_num'] + 1), "set env wrong"
        # return start image frame
        return self.agent.retrieve_data(frame_index)

    def get_reward(self, action_index, intensity):
        """reward policy.
        Args:
            action(carla.VehicleControl):only taking throttle[0, 1], steer[-1, 1], reverse[0, 1]
            intensity(float):the length of the collision_impluse
        Returns:
            reward:int
        """
        if intensity != 0:
            return -200
        if action_index == 3:
            return -100
        elif action_index == 0:
            return 5
        else:
            return 1

    def cleanup_world(self):
        # clean up the env
        # print("actorlist length: ", len(self.actor_list_env))
        self.client.apply_batch([carla.command.DestroyActor(x)
                                 for x in self.actor_list_env])
        # clean up the agent
        if self.agent is not None:
            # print("destroy agent")
            self.agent.cleanup()
        self.agent = None
        self.actor_list_env = []
        print("clean up the wrold, after cleanup world actors: ", len(self.world.get_actors().filter(
            'vehicle')))
        assert len(self.world.get_actors().filter(
            'vehicle')) == 0, "cleanup world wrong"

    def get_all_actors(self):
        """get all Actors in carla env.
        Returns:
            carla.ActorList
        """
        return self.world.get_actors()

    def get_all_vehicles(self):
        """get all vehicles in carla env including actor_car.
        Returns:
            carla.ActorList
        """
        return self.world.get_actors().filter('vehicle')

    def update_spectator(self):
        """update the spectator camera to follow the agent vehicle.
        This function should be called after each world tick to keep the camera following the car.
        """
        if self.agent is not None and self.agent.actor_car is not None:
            vehicle_transform = self.agent.actor_car.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            forward_vec = carla.Vector3D(x=vehicle_location.x + self.spectator_offset.x,
                                        y=vehicle_location.y + self.spectator_offset.y,
                                        z=vehicle_location.z + self.spectator_offset.z)
            
            new_transform = carla.Transform(
                forward_vec,
                carla.Rotation(pitch=self.spectator_rotation.pitch + vehicle_rotation.pitch,
                              yaw=vehicle_rotation.yaw + self.spectator_rotation.yaw,
                              roll=self.spectator_rotation.roll + vehicle_rotation.roll)
            )
            self.spectator.set_transform(new_transform)

    def exit_env(self):
        self.cleanup_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        print(
            f"before exited, there are { len(self.get_all_vehicles())} actors")
        print("exit world")

    # ==================================================================
    #  helper function for sac
    # ==================================================================

    def reward_sac(self, collision):
        if collision != 0:
            return -200
        else:
            return 1

    def step_sac(self, action):
        assert isinstance(
            action, carla.VehicleControl), "action is not the carla type."
        print("take: ", action)
        self.vehicle_control(action)
        frame_index = self.world.tick()
        self.update_spectator()
        print(f"after step, current frame is: {frame_index}")
        observation, collision = self.agent.retrieve_data(frame_index)
        # reward = self.get_reward(action_index, collision)
        reward = self.reward_sac(collision)
        done = 1 if collision != 0 else 0
        return observation, reward, done
