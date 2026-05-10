import os
import numpy as np
from collections import defaultdict
from gymnasium import spaces
from controllers.operational_space_controller import OSC
from controllers.joint_effort_controller import GripperEffortCtrl
from renderer.mujoco_env import MujocoPhyEnv

_target_box = ["ball_3","ball_2","ball_1","box_2","box_1","box_3"]
_right_finger_name = "right_finger"
_left_finger_name = "left_finger"
_grasp_target_num = 6

class GraspRobot(MujocoPhyEnv):
    def __init__(self, model_path="worlds/grasp.xml", frame_skip=40, render_mode=None):
        self.fullpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
        self.fullpath = model_path
        super().__init__(model_path, frame_skip=frame_skip)
        self.render_mode = render_mode
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = 64, 64
        self._set_observation_space()
        self._set_action_space()
        self.tolerance = 0.005
        self.drop_area = [0.6,0.0,1.15]
        self.TABLE_HEIGHT = 0.9
        self.GRASP_DEPTH = 0.08
        self.LIFT_HEIGHT = 0.18
        self.drop_area = [0.6, 0.0, 1.15]
    def __init__(self, model_path="worlds/grasp.xml", frame_skip=50, render_mode=None):
        self.fullpath = model_path
        super().__init__(model_path, frame_skip=frame_skip)
        self.render_mode = render_mode
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = 64,64
        self._set_observation_space()
        self._set_action_space()
        self.tolerance = 0.005
        self.drop_area = [0.6,0.0,1.15]
        self.TABLE_HEIGHT = 0.9

        self.arm_joints_names = list(self.model_names.joint_names[:6])
        self.arm_joints = [self.mjcf_model.find('joint', name) for name in self.arm_joints_names]
        self.eef_name = self.model_names.site_names[1]
        self.eef_site = self.mjcf_model.find('site', self.eef_name)

        self.controller = OSC(
            physics=self.physics,
            joints=self.arm_joints,
            eef_site=self.eef_site,
            min_effort=-200, max_effort=200,
            kp=100, ko=100, kv=60,
            vmax_xyz=0.8, vmax_abg=1.5
            min_effort=-150, max_effort=150,
            kp=80, ko=80, kv=50,
            vmax_xyz=1, vmax_abg=2
        )
        self.grp_ctrl = GripperEffortCtrl(physics=self.physics, gripper=self.gripper, effort=8.0)
        self.target_objects = _target_box
        self.grasped_num = 0
        self.grasp_step = 0
        self.object_positions_before_grasp = {}

    def _sanitize_physics_data(self):
        for attr in ['qpos','qvel','ctrl','qacc']:
            setattr(self.physics.data, attr, np.nan_to_num(getattr(self.physics.data, attr), nan=0.0, posinf=0.0, neginf=0.0))

    def get_ee_pos(self):
        return self.physics.bind(self.eef_site).xpos.copy()

    def get_body_com(self, body_name):
        body_id = self.physics.model.name2id(body_name, 'body')
        return self.physics.data.xpos[body_id].copy()

    def world2pixel(self, cam_id, x, y, z):
        fx = fy = 500
        cx = self.IMAGE_WIDTH / 2
        cy = self.IMAGE_HEIGHT / 2
        px = int((x * fx / z) + cx)
        py = int((y * fy / z) + cy)
        return px, py


    def _sanitize_physics_data(self):
        for attr in ['qpos', 'qvel', 'ctrl', 'qacc']:
            arr = getattr(self.physics.data, attr)
            setattr(self.physics.data, attr, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        for attr in ['qpos','qvel','ctrl','qacc']:
            setattr(self.physics.data, attr, np.nan_to_num(getattr(self.physics.data, attr), nan=0.0, posinf=0.0, neginf=0.0))

    def get_ee_pos(self):
        return self.physics.bind(self.eef_site).xpos.copy()

    def get_body_com(self, body_name):
        body_id = self.physics.model.name2id(body_name, 'body')
        return self.physics.data.xpos[body_id].copy()

    def set_body_pos(self, body_name, pos):
        body_id = self.physics.model.name2id(body_name, 'body')
        self.physics.model.body_pos[body_id] = pos

    def world2pixel(self, cam_id, x, y, z):
        fx = fy = 500
        cx = self.IMAGE_WIDTH / 2
        cy = self.IMAGE_HEIGHT / 2
        px = int((x * fx / z) + cx)
        py = int((y * fy / z) + cy)
        return px, py

    def pixel2world(self, cam_id, px, py, depth):
        x = (px / self.IMAGE_WIDTH - 0.5) * 0.48
        y = (py / self.IMAGE_HEIGHT - 0.5) * 0.48
        z = depth
        return np.array([x, y, z], dtype=np.float32)

    def _set_action_space(self):
        self.action_space = spaces.Box(low=-0.25, high=0.25, shape=[3], dtype=np.float32)

    def _set_observation_space(self):
        self.observation = defaultdict()
        self.observation["rgb"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3), dtype=np.float32)
        self.observation["depth"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), dtype=np.float32)
    def _set_action_space(self):
        self.action_space = spaces.Box(low=-0.25, high=0.25, shape=[3], dtype=np.float32)

    def _set_observation_space(self):
        self.observation = defaultdict()
        self.observation["rgb"] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT,3), dtype=np.float32)
        self.observation["depth"] = np.zeros((self.IMAGE_WIDTH,self.IMAGE_HEIGHT), dtype=np.float32)

    def move_eef(self, target):
        if hasattr(target, "tolist"):
            target = target.tolist()
        target_pose = target + [0, 0, 1, 1]
        target_pose = target + [0,0,1,1] 
        current_frame_skip = self.frame_skip if np.linalg.norm(np.array(self.get_ee_pos()) - np.array(target)) > 0.1 else 20
        for _ in range(current_frame_skip):
            self.controller.run(target_pose)
            self._sanitize_physics_data()
            self.step_mujoco_simulation()
            if np.allclose(self.get_ee_pos(), target, atol=self.tolerance):
                return True
        return False

    def down_and_grasp(self, target):
        down_pose = target.copy()
        down_pose[2] -= self.GRASP_DEPTH
        down_pose[2] = max(down_pose[2], self.TABLE_HEIGHT + 0.02)
        
        for obj_name in self.target_objects:
            pos = self.get_body_com(obj_name)
            self.object_positions_before_grasp[obj_name] = pos.copy()
        
        success = self.move_eef(down_pose)
        if success:
            for _ in range(self.frame_skip):

    def down_and_grasp(self, target):
        down_pose = target.copy()
        down_pose[2] -= 0.04
    def down_and_grasp(self, target):
        down_pose = target.copy()
        down_pose[2] -= 0.05
        success = self.move_eef(down_pose)
        if success:
            for _ in range(self.frame_skip // 2):
                self.grp_ctrl.run(signal=1)
                self.step_mujoco_simulation()
        return success

    def move_up_drop(self):
        up_pose = list(self.get_ee_pos())
        up_pose[2] += self.LIFT_HEIGHT
        self.move_eef(up_pose)
        
        grasp_success = self.check_grasp_success()
        
        if grasp_success:
            self.grasped_num += 1
            self.move_eef(self.drop_area)
            for _ in range(self.frame_skip // 2):
                self.grp_ctrl.run(signal=0)
                self.step_mujoco_simulation()
        
        self.object_positions_before_grasp.clear()
        up_pose[2] += 0.12
        up_pose[2] += 0.1
        drop_pose = self.drop_area + [0,0,1,1]
        self.move_eef(up_pose)
        grasp_success = self.check_grasp_success()
        if grasp_success:
            self.grasped_num += 1
            self.move_eef(drop_pose)
            for _ in range(self.frame_skip // 2):
                self.grp_ctrl.run(signal=0)
                self.step_mujoco_simulation()
        return grasp_success

    def check_grasp_success(self):
        right = self.get_body_com(_right_finger_name)
        left = self.get_body_com(_left_finger_name)
        finger_dist = np.linalg.norm(right - left)
        
        object_lifted = False
        for obj_name in self.target_objects:
            if obj_name in self.object_positions_before_grasp:
                prev_pos = self.object_positions_before_grasp[obj_name]
                curr_pos = self.get_body_com(obj_name)
                z_diff = curr_pos[2] - prev_pos[2]
                if z_diff > 0.03:
                    object_lifted = True
                    break
        
        return finger_dist < 0.16 and object_lifted
        dist = np.linalg.norm(right - left)
        return dist < 0.16
        dist = np.linalg.norm(self.get_body_com(_right_finger_name) - self.get_body_com(_left_finger_name))
        return dist < 0.12

    def open_gripper(self):
        for _ in range(self.frame_skip // 2):
            self.grp_ctrl.run(signal=0)
            self.step_mujoco_simulation()

    def step(self, action):
        self.info = {}
        self.open_gripper()
        
        moved = self.move_eef(action)
        grasped = self.down_and_grasp(action) if moved else False
        success = self.move_up_drop() if grasped else False

        ee_pos = self.get_ee_pos()
        closest_dist = float('inf')
        for obj_name in self.target_objects:
            obj_pos = self.get_body_com(obj_name)
            dist = np.linalg.norm(ee_pos - obj_pos)
            if dist < closest_dist:
                closest_dist = dist

        reward = 1.0 - closest_dist * 2.0
        if success:
            reward += 30.0
            self.info["grasp"] = "Success"
        else:
            self.info["grasp"] = "Failed"
            if moved and not grasped:
                reward -= 2.0
            elif grasped and not success:
                reward -= 5.0

        if not moved:
            reward -= 0.5

        self.grasp_step += 1
        done = self.grasped_num == _grasp_target_num or self.grasp_step >= 20
    # 修复 pixel2world，避免 cam_mat 报错
    def pixel2world(self, cam_id, px, py, depth):
        x = (px / self.IMAGE_WIDTH - 0.5) * 0.5
        y = (py / self.IMAGE_HEIGHT - 0.5) * 0.5
        z = depth
        return np.array([x, y, z], dtype=np.float32)

    def step(self, action):
        self.info = {}
        self.open_gripper()
        moved = self.move_eef(action)
        grasped = self.down_and_grasp(action) if moved else False
        success = self.move_up_drop() if grasped else False

        ee_pos = self.get_ee_pos()
        obj_pos = self.get_body_com(self.target_objects[0])
        dist = np.linalg.norm(ee_pos - obj_pos)
        reward = 0.5 - dist * 2.0

        if success:
            reward += 15.0
            self.info["grasp"] = "Success"
        else:
            self.info["grasp"] = "Failed"

        if not moved:
            reward -= 0.2

        self.grasp_step += 1
        done = self.grasped_num == _grasp_target_num or self.grasp_step >= 18

        dist = np.linalg.norm(self.get_ee_pos() - self.get_body_com(self.target_objects[0]))
        reward = 1.0 - np.tanh(3 * dist)
        if success:
            reward += 20
            self.info["grasp"] = "Success"
        else:
            self.info["grasp"] = "Failed"
        if not moved:
            reward -= 0.5

        self.grasp_step += 1
        done = self.grasped_num == _grasp_target_num or self.grasp_step >= 10
        return self.observation, reward, done, self.info

    def reset(self):
        super().reset()
        self.grasped_num = 0
        self.grasp_step = 0
        self.open_gripper()
        return self.observation

    def reset_without_random(self):
        super().reset()
        self.grasped_num = 0
        self.grasp_step = 0
        self.open_gripper()
        return self.observation


        return self.observation
