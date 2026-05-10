import cv2
import gym
import time
import math
import carla
import imutils
import numpy as np
from min_carla_env.matrix_world import MatrixWorld

CONFIG = {
    "width": 480,
    "height": 480,
    "max_step": 90000,
    "render": True
}


# only three actions for simplicty
# stable, left, right
ACTIONS = {
    0: [0.0, 0.0],    # Coast
    1: [0.0, -0.5],   # Turn Left
    2: [0.0, 0.5],    # Turn Right
}


class CarlaEnv(gym.Env):
    """Simple gym wrapper for Carla.
    Unfortunately it only uses the gym environment interface.
    Its not that much compatible with gym."""

    def __init__(self, client, config, world_config={}, debug=False, demo=False):
        self.debug = debug
        self.done = False
        self.rgb_data = None
        self.semantic_data = None
        self.steps = 0
        self.stuck_count = 0
        self.started = False
        self.collision_hist = []
        self.crossed_lane_hist = []
        self.config = config
        self.max_step = config["max_step"]
        self.demo = demo

        # build world
        self.mw = MatrixWorld(client, **world_config)
        self.world = self.mw.world
        self.measurements = {
            "kmh": 0.0,
            "prev_loc": None
        }
        self.spawn_actors()
        self.hist_wp = None

    def spawn_actors(self):
        """Spawns agent car and sensors."""
        self.vehicle = self.mw.spawn_vehicle()

        self.rgb_sensor = self.mw.spawn_sensor('sensor.camera.rgb',
                                               self.vehicle,
                                               carla.Location(x=2.5, z=0.7))
        self.rgb_sensor.listen(lambda image: self.process_img(image))
        self.semantic_sensor = self.mw.spawn_sensor(
            'sensor.camera.semantic_segmentation',
            self.vehicle, carla.Location(x=2.5, z=0.7))
        self.semantic_sensor.listen(lambda image: self.process_semantic(image))
        self.col_sensor = self.mw.spawn_collision_sensor(
                self.vehicle, carla.Location(x=2.5, z=0.7))
        self.col_sensor.listen(lambda event: self.collision_data(event))
        self.lane_sensor = self.mw.spawn_lane_sensor(self.vehicle)
        self.lane_sensor.listen(lambda event: self.lane_data(event))

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_data(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        # text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.crossed_lane_hist.extend(lane_types)

    def process_img(self, image):
        """Convert rgb image to array."""
        i = np.array(image.raw_data)
        i2 = i.reshape((self.config['width'], self.config['height'], 4))
        i3 = i2[:, :, :3]
        # rotate to make car bottom center
        i3 = imutils.rotate_bound(i3, self.mw.yaw)
        if self.debug:
            cv2.imwrite("i.png", i3)
        self.rgb_data = i3

    def semantic_mask(self, data):
        """Masks the current semantic data to desired labels
        classes I want to show:
        0: none
        1: road
        2: roadlines
        3: poles
        4: sidewalks
        5: vehicles
        """
        # replace sidewalks with others
        data[data == 1] = 4   # buildings
        data[data == 2] = 4   # fences
        data[data == 3] = 4   # other
        data[data == 4] = 4   # pedesterians
        data[data == 9] = 4   # vegetation
        data[data == 11] = 4  # walls
        data[data == 12] = 4  # TrafficSigns

        data[data == 5] = 3   # change poles
        data[data == 6] = 2   # change roadline
        data[data == 7] = 1   # change road
        data[data == 8] = 4   # change sidewalks
        data[data == 10] = 5  # change vehicles
        return data

    def process_semantic(self, image):
        """Convert a convert semantic image to array."""
        # if not isinstance(image, sensor.Image):
        #     raise ValueError("Argument must be a carla.sensor.Image")
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))

        # rotate to make car bottom center
        array = imutils.rotate_bound(array, self.mw.yaw)

        self.semantic_data = array[:, :, 2].copy()
        self.semantic_data = self.semantic_mask(self.semantic_data)
        # self.semantic_data = self.semantic_mask_simple(self.semantic_data)
        if self.debug:
            cc_semantic_data = self.labels_to_cityscapes_palette(
                self.semantic_data)
            cv2.imwrite("s.png", cc_semantic_data)

    def labels_to_cityscapes_palette(self, array):
        """
        Convert an image containing CARLA semantic segmentation labels to
        Cityscapes palette.
        """
        classes = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0]     # TrafficSigns
        }
        result = np.zeros((array.shape[0], array.shape[1], 3))
        for key, value in classes.items():
            result[np.where(array == key)] = value
        return result

    def get_measurements(self):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        measurements = {
            "kmh": kmh,
            "prev_loc": self.vehicle.get_transform().location
        }
        return measurements

    def reset(self):
        self.done = False
        self.steps = 0
        self.started = False
        self.rgb_data = None
        self.semantic_data = None
        self.collision_hist = []
        self.crossed_lane_hist = []
        self.hist_wp = None

        self.mw.clean_world()
        for _ in range(5):
            try:
                self.spawn_actors()
                break
            except Exception as e:
                print("Exception {}".format(e))
                self.mw.clean_world()

        self.measurements = {
                "kmh": 0.0,
                "prev_loc": None
        }
        time.sleep(1)
        # return self.rgb_data
        return self.semantic_data
        # return (self.rgb_data, self.semantic_data)

    def __euclid_dist(self, loc1, loc2):
        """Calc euclid distance of carla Locations."""
        dist = math.sqrt(
            (loc1.x - loc2.x)**2 +
            (loc1.y - loc2.y)**2 +
            (loc1.z - loc2.z)**2
            )
        return dist

    def simple_loc_reward(self, map: carla.Map, location: carla.Location):
        """Calculates simple reward for given location."""
        # calc closest drivable point distance
        reward = 0.0
        wp = map.get_waypoint(location, carla.LaneType.Driving)
        wp_location = wp.transform.location
        dist = self.__euclid_dist(wp_location, location)
        if dist < 0.5 and dist > -0.5:
            reward += 0.5
        else:
            reward -= np.exp(dist)

        return reward

    def is_stuck(self, location: carla.Location):
        prev_loc = self.measurements["prev_loc"]
        if prev_loc is not None and self.started:
            dist = self.__euclid_dist(prev_loc, location)
            if dist <= 0.05:
                return True
        return False

    def step(self, action):
        """Apply action, calculate reward, return observation."""
        # interpret actions
        action = ACTIONS[int(action)]
        steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False

        measurements = self.get_measurements()
        kmh = measurements["kmh"]
        if kmh >= 1 and not self.started:
            self.started = True
        else:
            self.collision_hist = []

        # make the car always in stable velocity
        brake = 0.0
        throttle = 0.3
        if kmh >= 20:
            brake = 0.2
            throttle = 0.0
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer,
            reverse=reverse, hand_brake=hand_brake))

        # calculate reward
        reward = 0.0
        vehicle_location = self.vehicle.get_transform().location
        reward += self.simple_loc_reward(self.mw.world.get_map(), vehicle_location)

        # count stuck to be able to stop running
        self.steps += 1
        if self.is_stuck(vehicle_location):
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        self.measurements = measurements
        if self.steps >= self.max_step:
            self.done = True

        current_w = self.mw.world.get_map().get_waypoint(vehicle_location)
        if reward <= -1000.0:   # limit the reward
            self.done = True
        if len(self.collision_hist) != 0:   # stop on collision
            self.done = True
            reward *= 2
        if len(self.crossed_lane_hist) != 0:    # stop on crossed lane
            for lane_marking in self.crossed_lane_hist:
                if lane_marking == carla.LaneMarkingType.Solid or\
                        lane_marking == carla.LaneMarkingType.NONE:
                    self.done = True
                    reward *= 2
                    break
            self.crossed_lane_hist = []
        if current_w.lane_type == carla.LaneType.Sidewalk:  # stop on out of road
            self.done = True
            reward *= 2
        if self.stuck_count > 20:   # stop on stuck
            self.done = True
            reward -= 100.0
        
        if self.demo and self.stuck_count < 20:
            self.done = False

        return self.semantic_data, reward, self.done, {}
        # return (self.rgb_data, self.semantic_data), reward, self.done, {}
