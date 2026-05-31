from os import path
import time
from typing import Any, Dict, Optional
import numpy as np
import cv2
import gymnasium as gym
import mujoco
import mujoco.viewer
from utils.mujoco_utils import MujocoModelNames
from utils.transform_utils import quat2mat
import random


class JointBinding:
    def __init__(self, data, joint_ids, model=None):
        self._data = data
        self._model = model
        # If joint_ids are joint ids (ints), convert to DOF addresses
        self._joint_ids = joint_ids

    @property
    def qfrc_applied(self):
        return self._data.qfrc_applied[self._joint_ids]

    @qfrc_applied.setter
    def qfrc_applied(self, value):
        self._data.qfrc_applied[self._joint_ids] = value

    @property
    def qpos(self):
        return self._data.qpos[self._joint_ids]

    @qpos.setter
    def qpos(self, value):
        self._data.qpos[self._joint_ids] = value

    @property
    def qvel(self):
        return self._data.qvel[self._joint_ids]

    @property
    def dofadr(self):
        return self._model.jnt_dofadr[self._joint_ids]

    @property
    def qfrc_bias(self):
        return self._data.qfrc_bias[self._joint_ids]

    @property
    def element_id(self):
        return self._joint_ids[0] if len(self._joint_ids) == 1 else self._joint_ids

    def __getattr__(self, name):
        return getattr(self._data, name)


class SiteBinding:
    def __init__(self, data, site_id):
        self._data = data
        self._site_id = site_id

    @property
    def xpos(self):
        return self._data.site_xpos[self._site_id]

    @property
    def xmat(self):
        return self._data.site_xmat[self._site_id].reshape(3, 3)

    @property
    def element_id(self):
        return self._site_id

    def __getattr__(self, name):
        return getattr(self._data, name)




class MujocoPhyEnv(gym.Env):
    """Superclass for MuJoCo environments using mujoco-py."""

    def __init__(
        self,
        model_path,
        frame_skip,
        render_mode: Optional[str] = None,
    ):
        self.TABLE_HEIGHT = 0.95
        self.IMAGE_WIDTH = 224
        self.IMAGE_HEIGHT = 224
        if model_path.startswith(".") or model_path.startswith("/"):
            self.fullpath = model_path
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        self.physics = self
        self.mjcf_model = self.model
        self.viewer = None
        self.renderer = mujoco.Renderer(self.model, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self._step_start = None
        self._timestep = self.model.opt.timestep
        self.arm_joints = [self.model.jnt_dofadr[i] for i in range(6)]
        self.eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'eef_site')
        self.gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_outer_knuckle_joint')
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.model_names = MujocoModelNames(self.model)
        self.cam_matrix = None
        self.cam_init = False
        self.camera_id = 0


    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)


    def step_mujoco_simulation(self, n_frames=1):
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
        self.render()

    def render(self):
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._step_start is None and self.render_mode == "human":
            self._step_start = time.time()

        if self.render_mode == "human":
            self.viewer.sync()
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self._step_start = time.time()

    def reset(self):
        self._reset_simulation()
        self.data.qpos[:6] = [
            1.5707,
            -1.5707,
            1.5707,
            -1.5707,
            -1.5707,
            0.0,
        ]
        return self.get_observation()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def get_body_com(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def get_ee_pos(self):
        return self.data.site_xpos[self.eef_site_id].copy()

    def get_observation(self):
        return {}

    def get_image_data(
        self,
        cam_name="top_down",
        depth=False,
        show=False
    ):
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb_data = self.renderer.render()
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)

        if show:
            cv2.imshow(cam_name, rgb_data)
            cv2.waitKey(1)

        if depth:
            depth_data = self.renderer.render_depth()
            return np.array(rgb_data), np.array(depth_data)

        return np.array(rgb_data)

    def pixel2world(self, cam_id, pixel_x, pixel_y, depth):
        if self.cam_matrix is None or not self.cam_init:
            self._init_camera(cam_id)
        depth = -depth
        camera_matrix = self.cam_matrix
        world_position = np.linalg.pinv(camera_matrix).dot(np.array([pixel_x, pixel_y, 1.0])) * depth
        wx, wy = world_position[:2]
        return [wx, wy, 1.16]

    def world2pixel(self, cam_id, x, y, z):
        if self.cam_matrix is None or not self.cam_init:
            self._init_camera(cam_id)
        camera_matrix = self.cam_matrix
        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))
        return [int(np.round(xs / s)), int(np.round(ys / s))]

    def _init_camera(self, cam_id):
        self.camera_id = cam_id
        cam = self.model.cam(cam_id)
        # Build full projection matrix (3x4) from camera intrinsics + extrinsics
        res = self.model.vis.global_.offwidth, self.model.vis.global_.offheight
        fovy = cam.fovy[0]  # fovy is an array
        aspect = res[0] / res[1]
        focal = 0.5 * res[1] / np.tan(fovy * np.pi / 360.0)
        cx, cy = res[0] / 2.0, res[1] / 2.0
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
        pos = cam.pos
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, cam.quat)
        rot_mat = rot_mat.reshape(3, 3)
        R = rot_mat.T
        t = -R @ pos
        RT = np.hstack([R, t.reshape(3, 1)])
        self.cam_matrix = K @ RT
        self.cam_init = True

    def set_body_pos(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        new_pos = [random.uniform(a=-0.224, b=0.224) for _ in range(2)]
        new_pos.append(1.1)
        self.data.xpos[body_id] = new_pos

    def find(self, obj_type, name):
        if obj_type == 'joint':
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return joint_id
        elif obj_type == 'site':
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            return site_id
        elif obj_type == 'body':
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            return body_id
        else:
            raise ValueError(f"Unknown obj_type: {obj_type}")

    def bind(self, obj, obj_type=None):
        if isinstance(obj, list):
            # Lists are always joints (list of joint ids)
            return JointBinding(self.data, obj, model=self.model)
        elif isinstance(obj, int):
            if obj_type == 'site':
                return SiteBinding(self.data, obj)
            elif obj_type == 'joint':
                return JointBinding(self.data, [obj], model=self.model)
            else:
                # Heuristic fallback (backward compat)
                if obj < self.model.njnt:
                    return JointBinding(self.data, [obj], model=self.model)
                else:
                    return SiteBinding(self.data, obj)
        elif isinstance(obj, tuple):
            # (id, type) tuple e.g. (site_id, 'site')
            return self.bind(obj[0], obj_type=obj[1])
        else:
            return JointBinding(self.data, [obj], model=self.model)