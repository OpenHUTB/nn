from os import path
from collections import defaultdict
import numpy as np
from controllers.operational_space_controller import OSC
from controllers.joint_effort_controller import GripperEffortCtrl
from gymnasium import spaces
from renderer.mujoco_env import MujocoPhyEnv
import random

_right_finger_name = "right_finger"
_left_finger_name = "left_finger"
_close_finger_dis = 0.06
_open_finger_dis = 0.152
_grasp_target_num = 6
_target_box = ["ball_3","ball_2","ball_1","box_2","box_1","box_3"]
eyehand_target = [-0.02,-0.13,1.45,0,0,1,1]

class GraspRobot(MujocoPhyEnv):
    def __init__(self, model_path="../worlds/grasp.xml", frame_skip=200, **kwargs):
        xml_file_path = path.join(path.dirname(path.realpath(__file__)), model_path)
        self.fullpath = xml_file_path
        super().__init__(xml_file_path, frame_skip, **kwargs)

        self.IMAGE_WIDTH = 64
        self.IMAGE_HEIGHT = 64
        self._set_observation_space()
        self.info = {}
        self._set_action_space()
        self.tolerance = 0.005
        self.drop_area = [0.6, 0.0, 1.15]
        self.arm_joints_names = list(self.model_names.joint_names[:6])

        # 修复：从 mjcf 模型中获取关节和末端执行器的 mjcf 元素（不是整数id）
        self.arm_joints = [self.mjcf_model.find('joint', name) for name in self.arm_joints_names]
        self.eef_name = self.model_names.site_names[1]
        self.eef_site = self.mjcf_model.find('site', self.eef_name)
        self.TABLE_HEIGHT = 1.0

        self.controller = OSC(
            physics=self.physics,
            joints=self.arm_joints,
            eef_site=self.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=80, ko=80, kv=50,
            vmax_xyz=1.0, vmax_abg=2.0
        )
        self.grp_ctrl = GripperEffortCtrl(physics=self.physics, gripper=self.gripper)
        self.target_objects = _target_box

    def before_grasp(self, show=False):
        self.reward = 0
        self.get_image_data("eyeinhand", depth=True, show=show)
        qpos = np.nan_to_num(self.physics.data.qpos.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        self.physics.data.qpos[:] = qpos
        for _ in range(self.frame_skip):
            self.controller.run(eyehand_target)
            self.grp_ctrl.run(signal=0)
            self.physics.data.qvel[:] = np.nan_to_num(self.physics.data.qvel, nan=0.0, posinf=0.0, neginf=0.0)
            self.physics.data.qacc[:] = np.nan_to_num(self.physics.data.qacc, nan=0.0, posinf=0.0, neginf=0.0)
            self.physics.data.ctrl[:] = np.nan_to_num(self.physics.data.ctrl, nan=0.0, posinf=1.0, neginf=-1.0)
            self.step_mujoco_simulation()
        rgb, depth = self.get_image_data("eyeinhand", depth=True, show=show)
        self.observation["rgb"] = rgb
        self.observation["depth"] = depth
        self.info['grasp'] = "Failed"
        self.info["move"] = "Failed"

    def after_grasp(self, show=False):
        self.get_image_data("eyeinhand", depth=True, show=show)
        for _ in range(self.frame_skip):
            self.controller.run(eyehand_target)
            self.grp_ctrl.run(signal=0)
            self.step_mujoco_simulation()
        rgb, depth = self.get_image_data("eyeinhand", depth=True, show=show)
        self.observation["rgb"] = rgb
        self.observation["depth"] = depth

    def move_eef(self, action):
        success = False
        target_pose = action.copy() + [0,0,1,1]
        for _ in range(self.frame_skip):
            self.controller.run(target_pose)
            self.physics.data.qpos[:] = np.nan_to_num(self.physics.data.qpos, nan=0.0, posinf=0.0, neginf=0.0)
            self.physics.data.ctrl[:] = np.nan_to_num(self.physics.data.ctrl, nan=0.0, posinf=1.0, neginf=-1.0)
            self.step_mujoco_simulation()
            ee_pos = self.get_ee_pos()
            if max(np.abs(ee_pos - action)) < self.tolerance:
                success = True
        if success:
            self.info["move"] = f"move to target {action}"
        return success

    def down_and_grasp(self, action):
        down_success = False
        target_pose = action.copy()
        target_pose[2] -= 0.05
        target_pose += [0,0,1,1]
        for _ in range(self.frame_skip):
            self.controller.run(target_pose)
            self.step_mujoco_simulation()
            if max(np.abs(self.get_ee_pos() - action)) < self.tolerance:
                down_success = True
        if down_success:
            for _ in range(self.frame_skip):
                self.controller.run(target_pose)
                self.grp_ctrl.run(signal=1)
                self.step_mujoco_simulation()
        return down_success

    def move_up_drop(self):
        success = False
        up_pose = list(self.get_ee_pos())
        up_pose[2] += 0.1
        up_pose += [0,0,1,1]
        drop_pose = self.drop_area + [0,0,1,1]
        dist = np.linalg.norm(self.get_ee_pos() - self.get_body_com(self.target_objects[0]))
        self.reward = -0.01 * dist
        for _ in range(self.frame_skip):
            self.controller.run(up_pose)
            self.step_mujoco_simulation()
        if self.check_grasp_success():
            self.info["grasp"] = "Success"
            self.grasped_num += 1
            self.reward = 1
            for _ in range(self.frame_skip):
                self.controller.run(drop_pose)
                self.step_mujoco_simulation()
                if max(np.abs(self.get_ee_pos() - self.drop_area)) < self.tolerance:
                    success = True
            if success:
                for _ in range(self.frame_skip):
                    self.controller.run(drop_pose)
                    self.grp_ctrl.run(signal=0)
                    self.step_mujoco_simulation()
                if self.physics.data.ctrl[0] == 0:
                    return success

    def check_terminated(self):
        for box in _target_box:
            if self.get_body_com(box)[2] >= self.TABLE_HEIGHT:
                return False
        return True

    def check_grasp_success(self):
        right = self.get_body_com(_right_finger_name)
        left = self.get_body_com(_left_finger_name)
        dist = max(np.abs(right - left))
        return (_close_finger_dis < dist < _open_finger_dis) and (self.physics.data.ctrl[0] == 255)

    def open_gripper(self):
        target_pose = list(self.get_ee_pos()) + [0,0,1,1]
        for _ in range(self.frame_skip):
            self.controller.run(target_pose)
            self.grp_ctrl.run(signal=0)
            self.step_mujoco_simulation()

    def move_and_grasp(self, action):
        action[2] = 1.15
        if self.move_eef(action):
            if self.down_and_grasp(action):
                self.move_up_drop()
            else:
                self.open_gripper()

    def _set_action_space(self):
        self.action_space1 = spaces.Box(low=-0.25, high=0.25, shape=[2])

    def _set_observation_space(self):
        self.observation = defaultdict()
        self.observation["rgb"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3))
        self.observation["depth"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

    def step(self, action):
        self.terminated = False
        self.info = {}
        self.before_grasp(show=False)
        self.move_and_grasp(action)
        self.after_grasp(show=False)
        ee_pos = self.get_ee_pos()
        obj_pos = self.get_body_com(self.target_objects[0])
        dist = np.linalg.norm(ee_pos - obj_pos)
        reward = -0.02 * dist
        if self.info.get("grasp") == "Success":
            reward += 10.0
        elif self.info.get("move") == "Failed":
            reward -= 1.0
        self.reward = reward
        if self.grasped_num == _grasp_target_num or self.grasp_step == 5:
            self.terminated = True
        if self.check_terminated():
            self.terminated = True
        self.grasp_step += 1
        return self.observation, self.reward, self.terminated, self.info

    def reset_object(self):
        for box_name in _target_box:
            self.set_body_pos(box_name)

    def reset(self):
        super().reset()
        self.reset_object()
        self.grasped_num = 0
        self.grasp_step = 0
        self.info["completion"] = "Failed"
        self.before_grasp(show=False)
        return self.observation

    def reset_without_random(self):
        super().reset()
        self.grasped_num = 0
        self.grasp_step = 0
        self.info["completion"] = "Failed"
        self.before_grasp(show=False)
        return self.observation

    def get_ee_pos(self):
        return self.physics.bind(self.eef_site).xpos.copy()

    def pixel2world(self, cam_id, px, py, depth):
        fx = fy = 500
        cx = self.IMAGE_WIDTH / 2
        cy = self.IMAGE_HEIGHT / 2
        x = (px - cx) * depth / fx
        y = (py - cy) * depth / fy
        return [x, y, depth]

    def world2pixel(self, cam_id, x, y, z):
        fx = fy = 500
        cx = self.IMAGE_WIDTH / 2
        cy = self.IMAGE_HEIGHT / 2
        px = int((x * fx / z) + cx)
        py = int((y * fy / z) + cy)
        return px, py

    def set_body_pos(self, body_name, pos=None):
        if pos is None:
            pos = [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), self.TABLE_HEIGHT + 0.05]
        body_id = self.physics.model.name2id(body_name, 'body')
        self.physics.model.body_pos[body_id] = pos