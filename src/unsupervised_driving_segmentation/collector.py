import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import string

from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy

from rails.bellman import BellmanUpdater
from rails.models import EgoModel
from autoagents.waypointer import Waypointer

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.tools.misc import is_within_distance_ahead

def get_entry_point():
    return 'QCollector'

FPS = 20.
STOP_THRESH = 0.1
MAX_STOP = 500

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.015, theta=0.015, dt=0.015, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class QCollector(AutonomousAgent):

    """
    action value agent but assumes a static world
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.track = Track.MAP
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)
        
        global width, height
        width = config['width']
        height = config['height']

        device = torch.device('cuda')
        ego_model = EgoModel(1./FPS*(self.num_repeat+1)).to(device)
        ego_model.load_state_dict(torch.load(self.ego_model_dir))
        ego_model.eval()
        BellmanUpdater.setup(config, ego_model, device=device)

        self.vizs = []
        self.rgbs = []
        self.segmentations = []
        self.lbls = []
        self.locs = [] # (x,y,z) absolute
        self.rots = [] # (pitch,yaw,roll) absolute
        self.spds = []
        self.cmds = []
        self.trafficlights = []
        self.cam_locations = [] # (x,y,z) absolute, differs from ego-vehicle location, offset between them always changes as expressed in world coordinates
        self.cam_rotations = [] # (pitch,yaw,roll) absolute, in our case same as ego-vehicle rotation as camera aligned

        self.waypointer = None

        if self.log_wandb:
            wandb.init(project='carla_data_phase1')

        self.noiser = OrnsteinUhlenbeckActionNoise(dt=1/FPS)
        self.prev_steer = 0

        self.stop_count = 0

        self._world = None
        self._vehicle = None
        self._map = None
        self.ego_cam = None

        self._proximity_tlight_threshold = 3  # meters

    def setup_environment(self):
        self._world = CarlaDataProvider._client.get_world()
        vehicles = self._world.get_actors().filter('vehicle.*')
        for v in vehicles:
            if v.attributes['role_name'] == 'hero':
                self._vehicle = v
                break

        self._map = self._world.get_map()

        cams = self._world.get_actors().filter('sensor.camera.rgb')
        for c in cams:
            print(c)
            if(c.attributes.get('fov') == '120'):
                self.ego_cam = c
                break;

    def destroy(self):
        if len(self.lbls) == 0:
            return

        self.flush_data()

    def flush_data(self):
        if len(self.vizs) == 0:
            return

        # Ensure all arrays in self.vizs have the same shape
        shapes = [viz.shape for viz in self.vizs]
        if len(set(shapes)) != 1:
            print("Mismatch in shapes of vizs. Skipping wandb.Video creation.")
            self.vizs.clear()
            self.rgbs.clear()
            self.segmentations.clear()
            self.lbls.clear()
            self.locs.clear()
            self.rots.clear()
            self.spds.clear()
            self.cmds.clear()
            self.trafficlights.clear()
            self.cam_locations.clear()
            self.cam_rotations.clear()
            return

        rs = _random_string()
        if self.log_wandb:
            try:
                # Separate RGB and segmentation parts from self.vizs
                segmentations = [viz[:, :viz.shape[1] // 2, :] for viz in self.vizs]
                rgbs = [viz[:, viz.shape[1] // 2:, :] for viz in self.vizs]

                # Ensure video_data has only RGB channels
                video_data = np.stack(rgbs).transpose((0, 3, 1, 2))
                num_frames, channels, height, width = video_data.shape

                if channels != 3:
                    raise ValueError("The channels dimension of the video data must be 3 for RGB.")

                # Log video with a unique name to avoid overwriting
                video_name = f'vid_{rs}.mp4'
                wandb.log({
                    video_name: wandb.Video(video_data, fps=20, format='mp4')
                })
                print(f"Logged video with name: {video_name}")

                # Log semantic segmentation video
                video_data_segmentation = np.stack(segmentations).transpose((0, 3, 1, 2))
                num_frames_segmentation, channels_segmentation, height_segmentation, width_segmentation = video_data_segmentation.shape

                if channels_segmentation != 3:
                    raise ValueError("The channels dimension of the video data must be 3 for semantic segmentation.")

                video_name_segmentation = f'vid_segmentation_{rs}.mp4'
                wandb.log({
                    video_name_segmentation: wandb.Video(video_data_segmentation, fps=20, format='mp4')
                })
                print(f"Logged segmentation video with name: {video_name_segmentation}")
            except Exception as e:
                print(f"Error creating wandb.Video: {e}")


        # Save data
        data_path = os.path.join(self.main_data_dir, rs)
        print ('Saving to {}'.format(data_path))

        lmdb_env = lmdb.open(data_path, map_size=int(1e10))

        length = len(self.lbls)
        with lmdb_env.begin(write=True) as txn:

            txn.put('len'.encode(), str(length).encode())

            for i in range(length):

                txn.put(
                    f'rgb_{i:04d}'.encode(),
                    np.ascontiguousarray(self.rgbs[i]).astype(np.uint8),
                )

                txn.put(
                    f'segmentation_{i:04d}'.encode(),
                    np.ascontiguousarray(self.segmentations[i]).astype(np.uint8),
                )

                txn.put(
                    f'birdview_{i:04d}'.encode(),
                    np.ascontiguousarray(self.lbls[i]).astype(np.uint8),
                )

                txn.put(
                    f'loc_{i:04d}'.encode(),
                    np.ascontiguousarray(self.locs[i]).astype(np.float32)
                )

                txn.put(
                    f'rot_{i:04d}'.encode(),
                    np.ascontiguousarray(self.rots[i]).astype(np.float32)
                )

                txn.put(
                    f'spd_{i:04d}'.encode(),
                    np.ascontiguousarray(self.spds[i]).astype(np.float32)
                )

                txn.put(
                    f'cmd_{i:04d}'.encode(),
                    np.ascontiguousarray(self.cmds[i]).astype(np.float32)
                )

                txn.put(
                    f'trafficlights_{i:04d}'.encode(),
                    np.ascontiguousarray(self.trafficlights[i]).astype(np.uint8),
                )

                txn.put(
                    f'cam_location_{i:04d}'.encode(),
                    np.ascontiguousarray(self.cam_locations[i]).astype(np.float32),
                )

                txn.put(
                    f'cam_rotation_{i:04d}'.encode(),
                    np.ascontiguousarray(self.cam_rotations[i]).astype(np.float32),
                )
        self.vizs.clear()
        self.rgbs.clear()
        self.segmentations.clear()
        self.lbls.clear()
        self.locs.clear()
        self.rots.clear()
        self.spds.clear()
        self.cmds.clear()
        self.trafficlights.clear()
        self.cam_locations.clear()
        self.cam_rotations.clear()

        lmdb_env.close()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.map', 'id': 'MAP'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
        ]

        # Add sensors
        sensors.append({
            'type': 'sensor.camera.rgb',
            'x': self.camera_x,
            'y': 0,
            'z': self.camera_z,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'width': width,
            'height': height,
            'fov': 120,
            'id': 'rgb'
        })
        sensors.append({
            'type': 'sensor.camera.semantic_segmentation',
            'x': self.camera_x,
            'y': 0,
            'z': self.camera_z,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'width': width,
            'height': height,
            'fov': 120,
            'id': 'segmentation'
        })
        return sensors

    def run_step(self, input_data, timestamp):
        rgbs = []
        segmentations = []

        _, rgb = input_data.get('rgb')
        _, segmentation = input_data.get('segmentation')

        rgb_image = rgb[...,:3]
        rgbs.append(rgb_image) # Keeping R, G and B channels
        segmentation = segmentation[..., :3] if segmentation.shape[-1] == 4 else segmentation[..., :3]
        segmentations.append(segmentation) 

        _, lbl = input_data.get('MAP')
        _, col = input_data.get('COLLISION')
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')

        if self._world is None:
            self.setup_environment()

        cam_location = self.ego_cam.get_transform().location
        cam_location = np.array([cam_location.x, cam_location.y, cam_location.z])

        cam_rotation = self.ego_cam.get_transform().rotation
        cam_rotation = np.array([cam_rotation.pitch, cam_rotation.yaw, cam_rotation.roll])

        # # Modify traffic light label of segmentation (discard it if no red light)
        # relevant_traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_actor, False)
        # if relevant_traffic_light is not None:
        #     is_relevant_red_traffic_light_red = relevant_traffic_light.get_state() != carla.TrafficLightState.Green # Yellow is considered Red
        # else:
        #     is_relevant_red_traffic_light_red = False
        # print(is_relevant_red_traffic_light_red)
        # print(1 in lbl[...,3])
        #
        #
        # tls = (1 in lbl[...,3]) and is_relevant_red_traffic_light_red # Relevant traffic light is red and visible in the birdview
        # print(tls)
        # print('-------------------------------')

        lights_list = self._world.get_actors().filter("*traffic_light*")
        tls, _ = self._is_light_red(lights_list)

        if not tls:
            segmentation[segmentation == 18] = 0

        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
            _, _, cmd = self.waypointer.tick(gps)
        else:
            _, _, cmd = self.waypointer.tick(gps)

        yaw = ego.get('rot')[-1]
        rot = ego.get('rot') # roll, pitch, yaw
        rot = [rot[1], rot[2], rot[0]] # pitch, yaw, roll
        spd = ego.get('spd')
        loc = ego.get('loc')

        delta_locs, delta_yaws, next_spds = BellmanUpdater.compute_table(yaw/180*math.pi)

        # Convert lbl to rew maps
        lbl_copy = lbl.copy()
        waypoint_rews, stop_rews, brak_rews, free = BellmanUpdater.get_reward(lbl_copy, [0,0], ref_yaw=yaw/180*math.pi, tls=tls)

        waypoint_rews = waypoint_rews[None].expand(self.num_plan, *waypoint_rews.shape)
        brak_rews = brak_rews[None].expand(self.num_plan, *brak_rews.shape)
        stop_rews = stop_rews[None].expand(self.num_plan, *stop_rews.shape)
        free = free[None].expand(self.num_plan, *free.shape)

        # If it is idle, make it LANE_FOLLOW
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        action_values, _ = BellmanUpdater.get_action(
            delta_locs, delta_yaws, next_spds,
            waypoint_rews[...,cmd_value], brak_rews, stop_rews, free,
            torch.zeros((self.num_plan,2)).float().to(BellmanUpdater._device),
            extract=(
                torch.tensor([[0.,0.]]),  # location
                torch.tensor([0.]),       # yaw
                torch.tensor([spd]),      # spd
            )
        )
        action_values = action_values.squeeze(0)

        action = int(Categorical(logits=action_values/self.temperature).sample())
        # action = int(action_values.argmax())

        steer, throt, brake = map(float, BellmanUpdater._actions[action])

        if self.noise_collect:
            steer += self.noiser()

        flush_length = len(self.vizs)
        if flush_length > self.num_per_flush:
            self.flush_data()

        if (flush_length % 100) == 0:
            print(f'{flush_length}')

        spd = ego.get('spd')
        visualization = visualize_obs(rgb, yaw/180*math.pi, (steer, throt, brake), spd, cmd=cmd.value, lbl=None, sem=segmentation, tls=tls)
        self.vizs.append(visualization)
        # Ensure visualization has the correct shape
        # if visualization.shape[-1] != 3:
        #     print(f"Visualization shape mismatch: {visualization.shape}")
        #     return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)
        # viz = visualize_obs(rgb, yaw/180*math.pi, (steer, throt, brake), spd, cmd=cmd.value, lbl=lbl_copy, sem=segmentation, tls=tls)
        # print(f'vizs shape: {self.vizs.shape}')
        # self.vizs.append(viz)
        
        if col:
            self.flush_data()
            raise Exception('Collector has collided!! Heading out :P')

        if spd < STOP_THRESH:
            self.stop_count += 1
        else:
            self.stop_count = 0

        if cmd_value in [4,5]:
            actual_steer = steer
        else:
            actual_steer = steer * 1.2

        self.prev_steer = actual_steer

        # Save data
        if self.num_frames % (self.num_repeat+1) == 0 and self.stop_count < MAX_STOP:
            # Note: basically, this is like fast-forwarding. should be okay tho as spd is 0.
            self.rgbs.append(rgbs)
            self.segmentations.append(segmentations)
            self.lbls.append(lbl)
            self.locs.append(loc)
            self.rots.append(rot)
            self.spds.append(spd)
            self.cmds.append(cmd_value)
            self.trafficlights.append(tls)
            self.cam_locations.append(cam_location)
            self.cam_rotations.append(cam_rotation)

        self.num_frames += 1

        return carla.VehicleControl(steer=actual_steer, throttle=throt, brake=brake)

    ################################################################################
    # agents.navigation.Agent utilities (as of Carla 0.9.10.1)
    ################################################################################

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform,
                                        self._vehicle.get_transform(),
                                        self._proximity_tlight_threshold):
                if traffic_light.state != carla.TrafficLightState.Green:
                    return (True, traffic_light)

        return (False, None)

    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)


def _random_string(length=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
