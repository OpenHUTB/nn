import carla
import random


class MatrixWorld(object):
    """Builds the world, cars, sensors etc."""

    # hardcoded 2d view properties
    _X = 3
    _Y = 0
    _Z = 10

    def __init__(self, client, im_width=480.0, im_height=480.0, render=True,
                 weather=None, fast=False, town='Town02'):
        """The MatrixWorld class manages the building properties of the Carla world. """
        self.im_width = im_width
        self.im_height = im_height
        self.client = client

        # prepare world
        self.world = client.load_world(town)
        self.weather = weather
        if self.weather is not None:
            self.world.set_weather(self.weather)

        if not render:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

        # Run sim faster
        if fast:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)

        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = []
        self.actor_list = []
        self.yaw = 0

    def change_map(self, map_name=None):
        if map_name is None:
            # map_id = random.randrange(1, 6)
            map_id = random.choice([7, 2])
            map_name = "Town0{}".format(map_id)
        self.world = self.client.load_world(map_name)
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = []
        return self.world

    def is_close_to_junction(self, location, max_d):
        """Checks the next waypoints to given locations are junction."""
        for d in range(5, max_d):
            wp = self.world.get_map().get_waypoint(location, carla.LaneType.Driving)
            target_wps = wp.next(d)
            for target_wp in target_wps:
                if target_wp.is_junction:
                    return True
        return False

    def generate_near_junction_points(self, max_d):
        """Generate spawn points to near junction."""
        spawn_points = self.world.get_map().get_spawn_points()
        near_junction_points = []
        for transform in spawn_points:
            if self.is_close_to_junction(transform.location, max_d):
                near_junction_points.append(transform)
        return near_junction_points

    def get_point_near_junction(self, max_d):
        """Returns a transform near junction waypoint."""
        if len(self.spawn_points) < 1:
            self.spawn_points = self.generate_near_junction_points(max_d)
        transform = random.choice(self.spawn_points)
        return transform

    def spawn_vehicle(self, transform=None, tr_spectator=True, near_junction=False):
        """Spawns a vehicle on transform point. If the transform
        not setted spawns on random spawn point. Optionally transfroms
        the spectator to spawn point."""
        # vehicle_bp = self.bp_lib.filter('vehicle.audi.tt')[0]


        vehicle_bp = self.bp_lib.filter('vehicle.tesla.model3')[0]
        #vehicle_bp = self.bp_lib.filter('vehicle.mini.cooperst')[0]
        if not transform and not near_junction:
            transform = random.choice(self.world.get_map().get_spawn_points())
        elif near_junction:
            transform = self.get_point_near_junction(max_d=50)

        vehicle = self.world.spawn_actor(vehicle_bp, transform)
        self.actor_list.append(vehicle)

        if tr_spectator:
            # Wait for world to get the vehicle actor
            self.world.tick()
            world_snapshot = self.world.wait_for_tick()
            actor_snapshot = world_snapshot.find(vehicle.id)
            # Set spectator at given transform (vehicle transform)
            spectator = self.world.get_spectator()
            actor_trans = actor_snapshot.get_transform()
            spectator.set_transform(actor_trans)
        return vehicle

    def spawn_sensor(self, sensor, vehicle, location, args=None):
        """Spawns image sensors (rgb or semantic) in 2d view."""
        sensor_bp = self.bp_lib.find(sensor)
        sensor_bp.set_attribute("image_size_x", str(self.im_width))
        sensor_bp.set_attribute("image_size_y", str(self.im_width))
        sensor_bp.set_attribute('sensor_tick', '0.1')

        actor_trans = vehicle.get_transform()

        # apply 2d view
        yaw = actor_trans.rotation.yaw
        roll = actor_trans.rotation.roll

        sensor_transform = carla.Transform(
            carla.Location(self._X, self._Y, self._Z),
            carla.Rotation(pitch=-90, roll=roll, yaw=yaw)
        )
        # store yaw to use it for rotate image
        # to make car always bottom center of image
        self.yaw = yaw

        sensor_actor = self.world.spawn_actor(sensor_bp, sensor_transform,
                                              attach_to=vehicle)
        self.actor_list.append(sensor_actor)
        return sensor_actor

    def spawn_collision_sensor(self, vehicle, location):
        col_sensor = self.bp_lib.find("sensor.other.collision")
        col_sensor = self.world.spawn_actor(col_sensor, carla.Transform(),
                                            attach_to=vehicle)
        self.actor_list.append(col_sensor)
        return col_sensor

    def spawn_lane_sensor(self, vehicle):
        lane_sensor_bp = self.bp_lib.find('sensor.other.lane_invasion')
        lane_sensor = self.world.spawn_actor(lane_sensor_bp, carla.Transform(),
                                             attach_to=vehicle)
        self.actor_list.append(lane_sensor)
        return lane_sensor

    def clean_world(self):
        for actor in self.actor_list:
            if actor is not None:
                actor.destroy()
        self.actor_list = []
