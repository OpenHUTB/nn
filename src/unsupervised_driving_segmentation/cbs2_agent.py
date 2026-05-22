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
from collections import deque
from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy

from autoagents.waypointer import Waypointer

from cbs2.bird_view.models import common
from cbs2.bird_view.models.controller import CustomController, PIDController
from cbs2.bird_view.models.controller import ls_circle
from cbs2.bird_view.models.image import PPM, ImagePolicyModelSS

import torchvision.transforms as transforms

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from guided_back_prop_cbs2 import gb_script

STEPS = 5
DT = 0.1

def get_entry_point():
    return 'CBS2Agent'

class CBS2Agent(AutonomousAgent):
    """
    CBS2 Image agent (Student)
    """

    def setup(self, path_to_conf_file):

        self.track = Track.SENSORS
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        self.model_name = 'Original'
        for key, value in config.items():
            setattr(self, key, value)

        if hasattr(self, 'ppm_bins'):
            self.ppm_bins = list(map(int, self.ppm_bins.split("-"))) # "1-2-3-6" --> [1, 2, 3, 6]
            self.model_name = 'PPM'
        else:
            self.ppm_bins = None

        if not hasattr(self, 'fpn'):
            self.fpn = False
        else:
            self.model_name = 'FPN'

        self.device = torch.device('cuda')
        self.vizs = []
        self.waypointer = None

        if self.log_wandb:
            wandb.init(project= path_to_conf_file.split('/')[-1].split('.')[0])

################################################################################
# CBS
        self.model = ImagePolicyModelSS(
            backbone='resnet34',
            # backbone='resnet50',
            all_branch=False,
            ppm_bins=self.ppm_bins,
            fpn=self.fpn
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.rgb_model_dir))
        self.model.eval()

        self.transform = transforms.ToTensor()
        self.one_hot = torch.FloatTensor(torch.eye(4))
        self.debug = dict()

        #self.fixed_offset = float(camera_args['fixed_offset'])
        self.fixed_offset = 4.0

        w = float(384)
        h = float(160)
        self.img_size = np.array([w,h])

        #self.gap = gap
        self.gap = 5

        self.steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

        pid = {
                "1" : {"Kp": 0.5, "Ki": 0.20, "Kd":0.0}, # Left
                "2" : {"Kp": 0.7, "Ki": 0.10, "Kd":0.0}, # Right
                "3" : {"Kp": 1.0, "Ki": 0.10, "Kd":0.0}, # Straight
                "4" : {"Kp": 1.0, "Ki": 0.50, "Kd":0.0}, # Follow
            }

        self.turn_control = CustomController(pid)

        self.speed_control = PIDController(K_P=.8, K_I=.08, K_D=0.)

        self.engine_brake_threshold_straight = 3.8
        self.brake_threshold_straight = 3.55

        #self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0

        self.last_brake = -1
################################################################################

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()
        self.prev_steer = 0

        del self.waypointer
        del self.model

    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        self.vizs.clear()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 160, 'fov': 120, 'id': f'RGB'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):

        _, rgb = input_data.get(f'RGB')
        rgb = np.array(rgb[...,:3])

        # Crop images
        #_rgb = rgb[self.crop_top:-self.crop_bottom,:,:3]
        _rgb = rgb[:,:,:3]

        _rgb = _rgb[...,::-1].copy()

        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')

        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)

        speed = ego.get('spd')

        _cmd = cmd.value
        command = self.one_hot[_cmd - 1]

        _rgb = torch.tensor(_rgb[None]).float().permute(0,3,1,2).to(self.device)

        _speed = torch.tensor([speed]).float().to(self.device)

        with torch.no_grad():
            _rgb = self.transform(rgb).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            model_pred = self.model(_rgb, _speed, _command)

        model_pred = model_pred.squeeze().detach().cpu().numpy()
        pixel_pred = model_pred
        # Project back to world coordinate
        model_pred = (model_pred+1)*self.img_size/2
        steer, throt, brake, target_speed = self.get_control(model_pred, _cmd, speed)

        # Plot RGB image with info
        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), speed, target_speed=target_speed, cmd=_cmd, pred=model_pred))

        # Plot RGB image with info + back prop viz
        # rgb_viz = visualize_obs(rgb, 0, (steer, throt, brake), speed, target_speed=target_speed, cmd=_cmd, pred=model_pred)
        # _speed = torch.FloatTensor([speed]).to(self.device)
        # _command = command.to(self.device).unsqueeze(0)
        # guided_back_prop_viz = gb_script.get_gb(rgb, self.model_name, _speed, _command)
        # canvas = np.hstack((rgb_viz, guided_back_prop_viz))
        # self.vizs.append(canvas)


        #Flush every 10k frames (instead of after episode finished)
        # if len(self.vizs) > 10000:
        #     self.flush_data()

        self.num_frames += 1

        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

    def get_control(self, model_pred, _cmd, speed):
        world_pred = self.unproject(model_pred)
        targets = [(0, 0)]

        for i in range(STEPS):
            pixel_dx, pixel_dy = world_pred[i]
            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy])

            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        targets = np.array(targets)

        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / (self.gap * DT)

        target_speed = np.clip(target_speed, 0.0, 5.0)

        c, r = ls_circle(targets)
        n = self.steer_points.get(str(_cmd), 1)
        closest = common.project_point_to_circle(targets[n], c, r)

        acceleration = target_speed - speed

        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)

        #steer = self.turn_control.run_step(alpha, _cmd) #original - outdated since new dataset
        steer = self.turn_control.run_step(alpha, _cmd)/3 #29dec
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        # Former braking threshold handling

        # if target_speed <= self.engine_brake_threshold:
        #     steer = 0.0
        #     throttle = 0.0
        #
        # if target_speed <= self.brake_threshold:
        #     brake = 1.0


        # New braking threshold handling
        # As we go faster when we go straight, we have different stopping threshold
        if np.abs(steer)<0.05:
            if target_speed <= self.engine_brake_threshold_straight:
                throttle = 0.0
            if target_speed <= self.brake_threshold_straight:
                brake = 1.0

        elif target_speed <= self.brake_threshold:
                throttle = 0.0
                brake = 1.0


        self.debug = {
                'target_speed': target_speed,
                'target': closest,
                'locations_world': targets,
                'locations_pixel': model_pred.astype(int),
                }

        steer, throt, brake = self.postprocess(steer, throttle, brake)

        return steer, throt, brake, target_speed

    def postprocess(self, steer, throttle, brake):
        control = carla.VehicleControl()
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return steer, throttle, brake

    def unproject(self, output, world_y=1.4, fov=90):

        cx, cy = self.img_size / 2

        w, h = self.img_size

        f = w /(2 * np.tan(fov * np.pi / 360))

        xt = (output[...,0:1] - cx) / f
        yt = (output[...,1:2] - cy) / f

        world_z = world_y / yt
        world_x = world_z * xt

        world_output = np.stack([world_x, world_z],axis=-1)

        if self.fixed_offset:
            world_output[...,1] -= self.fixed_offset

        world_output = world_output.squeeze()

        return world_output
