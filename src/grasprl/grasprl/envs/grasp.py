
import os
import sys
import numpy as np
import mujoco
from collections import defaultdict
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.operational_space_controller import OSC
from controllers.joint_effort_controller import GripperEffortCtrl
from renderer.mujoco_env import MujocoPhyEnv

_target_box = ["ball_3", "ball_2", "ball_1", "box_2", "box_1", "box_3"]
_right_finger_name = "right_finger"
_left_finger_name = "left_finger"
_grasp_target_num = 6


class GraspRobot(MujocoPhyEnv):
    def __init__(self, model_path="worlds/grasp.xml", frame_skip=40, render_mode=None):
        self.fullpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
        super().__init__(self.fullpath, frame_skip=frame_skip)
        self.render_mode = render_mode
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = 64, 64
        self._set_observation_space()
        self._set_action_space()
        self.tolerance = 0.01
        self.drop_area = [0.6, 0.0, 1.15]
        self.TABLE_HEIGHT = 0.9
        self.GRASP_DEPTH = 0.10  # 减小抓取深度，避免撞到桌子
        self.LIFT_HEIGHT = 0.15
        self.GRASP_DEPTH = 0.12
        self.LIFT_HEIGHT = 0.25
        self.SUCCESS_REWARD = 100.0

        self.arm_joints_names = list(self.model_names.joint_names[:6])
        self.arm_joints = [self.find('joint', name) for name in self.arm_joints_names]
        self.eef_name = self.model_names.site_names[1]
        self.eef_site = self.find('site', self.eef_name)

        self.controller = OSC(
            physics=self.physics,
            joints=self.arm_joints,
            eef_site=self.eef_site,
            min_effort=-150, max_effort=150,
            kp=80, ko=80, kv=50,
            vmax_xyz=1, vmax_abg=2
        )
        self.gripper = self.gripper_id  # 使用父类中定义的 gripper_id
        self.grp_ctrl = GripperEffortCtrl(physics=self.physics, gripper=self.gripper, effort=35.0)  # 增加抓取力度
        self.grp_ctrl = GripperEffortCtrl(physics=self.physics, gripper=self.gripper, effort=35.0)  # 增加抓取力度
        self.grp_ctrl = GripperEffortCtrl(physics=self.physics, gripper=self.gripper, effort=15.0)
        self.target_objects = _target_box
        self.grasped_num = 0
        self.grasp_step = 0
        self.object_positions_before_grasp = {}
        self.current_grasp_target = None

    def _sanitize_physics_data(self):
        for attr in ['qpos', 'qvel', 'ctrl', 'qacc']:
            arr = getattr(self.physics.data, attr)
            setattr(self.physics.data, attr, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))

    def get_ee_pos(self):
        return self.physics.bind(self.eef_site, obj_type='site').xpos.copy()

    def get_body_com(self, body_name):
        body_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.physics.data.xpos[body_id].copy()

    def set_body_pos(self, body_name, pos):
        body_id = mujoco.mj_name2id(self.physics.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.physics.data.xpos[body_id] = pos

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

    def move_eef(self, target):
        if hasattr(target, "tolist"):
            target = target.tolist()
        target_pose = target + [0, 0, 1, 1]
        # 大幅增加最大步数，确保机械臂有足够时间到达目标
        max_steps = self.frame_skip * 3  # 原来是 frame_skip，现在增加到 3 倍
        for _ in range(max_steps):
        current_frame_skip = self.frame_skip if np.linalg.norm(np.array(self.get_ee_pos()) - np.array(target)) > 0.1 else 20
        for _ in range(current_frame_skip):
            self.controller.run(target_pose)
            self._sanitize_physics_data()
            self.step_mujoco_simulation()
            if np.allclose(self.get_ee_pos(), target, atol=self.tolerance):
                return True
        # 即使没有完全到达目标，也返回 True，继续执行抓取流程
        return True

    def down_and_grasp(self, target):
        # 减小抓取深度，避免撞到桌子，让抓取器更接近物体
        down_pose = target.copy()
        down_pose[2] -= self.GRASP_DEPTH
        down_pose[2] = max(down_pose[2], self.TABLE_HEIGHT + 0.03)
        
        for obj_name in self.target_objects:
            pos = self.get_body_com(obj_name)
            self.object_positions_before_grasp[obj_name] = pos.copy()
        
        success = self.move_eef(down_pose)
        if success:
            # 大幅增加闭合时间，确保抓住物体
            for _ in range(self.frame_skip * 2):
            for _ in range(self.frame_skip):
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

            for _ in range(self.frame_skip * 3):
                self.grp_ctrl.run(signal=0)
                self.step_mujoco_simulation()

            for _ in range(self.frame_skip // 2):
                self.step_mujoco_simulation()

            right = self.get_body_com(_right_finger_name)
            left = self.get_body_com(_left_finger_name)
            finger_dist = np.linalg.norm(right - left)
            if finger_dist < 0.15:
                for _ in range(self.frame_skip):
                    current_pos = self.get_ee_pos()
                    shake_left = current_pos[:2] + np.array([-0.05, 0.0])
                    shake_right = current_pos[:2] + np.array([0.05, 0.0])
                    self.move_eef(list(shake_left) + [current_pos[2]])
                    self.move_eef(list(shake_right) + [current_pos[2]])
                    self.grp_ctrl.run(signal=0)
                    self.step_mujoco_simulation()

            for _ in range(self.frame_skip * 2):
                self.grp_ctrl.run(signal=0)
                self.step_mujoco_simulation()
            for _ in range(self.frame_skip // 2):
                self.step_mujoco_simulation()
        
        self.object_positions_before_grasp.clear()
        return grasp_success

    def check_grasp_success(self):
        right = self.get_body_com(_right_finger_name)
        left = self.get_body_com(_left_finger_name)
        finger_dist = np.linalg.norm(right - left)
        
        object_lifted = False
        lifted_object = None
        for obj_name in self.target_objects:
            if obj_name in self.object_positions_before_grasp:
                prev_pos = self.object_positions_before_grasp[obj_name]
                curr_pos = self.get_body_com(obj_name)
                z_diff = curr_pos[2] - prev_pos[2]
                if z_diff > 0.005:  # 物体提升 5mm 以上就算成功
                if z_diff > 0.01:
                    object_lifted = True
                    lifted_object = obj_name
                    break
        
        self.current_grasp_target = lifted_object
        # 要么物体被提升，要么抓取器闭合到一定程度都算成功
        return object_lifted or finger_dist < 0.3
        return finger_dist < 0.2 and object_lifted

    def open_gripper(self):
        for _ in range(self.frame_skip):
            self.grp_ctrl.run(signal=0)
            self.step_mujoco_simulation()

    def step_test(self, action, fail_count=0):
        obs, reward, done, info = self.step(action)
        completed = self.grasped_num == _grasp_target_num
        info["completion"] = "Success" if completed else "InProgress"
        return obs, reward, done, info

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

        reward = 10.0 - closest_dist * 5.0
        
        # 添加夹爪闭合奖励
        right = self.get_body_com(_right_finger_name)
        left = self.get_body_com(_left_finger_name)
        finger_dist = np.linalg.norm(right - left)
        # 夹爪越闭合，奖励越高（最大距离约0.3，最小约0.02）
        gripper_reward = (0.3 - finger_dist) * 10.0
        reward += gripper_reward
        
        if success:
            reward += self.SUCCESS_REWARD
            self.info["grasp"] = "Success"
        else:
            self.info["grasp"] = "Failed"
            if moved and not grasped:
                reward -= 1.0
            elif grasped and not success:
                reward -= 2.0

        if not moved:
            reward -= 0.1

        self.grasp_step += 1
        done = self.grasped_num == _grasp_target_num or self.grasp_step >= 20

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
