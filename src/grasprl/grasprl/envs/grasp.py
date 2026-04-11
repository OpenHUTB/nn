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


    def __init__(
        self,
        model_path="../worlds/grasp.xml",
        frame_skip=1000,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )
        self.fullpath = xml_file_path
        super().__init__(
            xml_file_path,
            frame_skip,
            **kwargs,
        )
        self.info = {}
        self._set_observation_space()
        self._set_action_space()
        self.tolerance = 0.005
        self.drop_area = [0.6, 0.0, 1.15]
        self.arm_joints_names = list(self.model_names.joint_names[:6])
        self.eef_name = self.model_names.site_names[1]
        self.controller = OSC(
            physics=self.physics,
            joints=self.arm_joints,
            eef_site=self.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0          
        )
        self.grp_ctrl = GripperEffortCtrl(
            physics=self.physics,
            gripper=self.gripper
        )
        self.target_objects = _target_box

    def before_grasp(self,show=False):
        self.reward = 0
        self.get_image_data("eyeinhand",depth=True,show=show)
        
        for i in range(self.frame_skip):
            self.controller.run(eyehand_target)
            self.grp_ctrl.run(signal=0)
            self.step_mujoco_simulation()
        
        rgb_data,depth_data = self.get_image_data("eyeinhand",depth=True,show=show)
        self.observation["rgb"] = rgb_data
        self.observation["depth"] = depth_data
        self.info['grasp'] = "Failed"
        self.info["move"] = "Failed"

    def after_grasp(self,show=False):
        self.get_image_data("eyeinhand",depth=True,show=show)
        for i in range(self.frame_skip):
            self.controller.run(eyehand_target)
            self.grp_ctrl.run(signal=0)
            self.step_mujoco_simulation()
        rgb_data,depth_data = self.get_image_data("eyeinhand",depth=True,show=show)
        self.observation["rgb"] = rgb_data
        self.observation["depth"] = depth_data

    def move_eef(self,action):
        success = False
        target_position = action.copy()
        target_pose = action.copy()
        target_pose.extend([0,0,1,1])
        for i in range(self.frame_skip):
            self.controller.run(
                target_pose
            )
            self.step_mujoco_simulation()
            ee_pos = self.get_ee_pos()
            detals = np.abs(ee_pos - target_position)
            if max(detals) < self.tolerance:
                success = True
        if success:
            self.info["move"] = "move to target {}".format(target_position)
        return success

    def down_and_grasp(self,action):
        down_success = False
        target_position = action.copy()
        target_pose = action.copy()
        target_pose[2] -= 0.05
        target_pose.extend([0,0,1,1])
        for i in range(self.frame_skip):
            self.controller.run(
                target_pose
            )
            self.step_mujoco_simulation()
            ee_pos = self.get_ee_pos()
            detals = np.abs(ee_pos - target_position)
            if max(detals) < self.tolerance:
                down_success = True
        if down_success:
            for i in range(self.frame_skip):
                self.controller.run(
                    target_pose
                )
                self.grp_ctrl.run(signal=1)
                self.step_mujoco_simulation()
                return down_success 


    def move_up_drop(self):
        success = False
        up_pose = list(self.get_ee_pos().copy())

        up_pose[2] += 0.1
        up_pose.extend([0,0,1,1])
        target_pose = self.drop_area.copy()
        target_pose.extend([0,0,1,1])
        for i in range(self.frame_skip):
            self.controller.run(
                up_pose
            )
            self.step_mujoco_simulation()
        grasp_success = self.check_grasp_success()
        if grasp_success:
            self.info["grasp"] = "Success"
            self.grasped_num +=1
            self.reward = 1
            for i in range(self.frame_skip):
                self.controller.run(
                    target_pose
                )
                self.step_mujoco_simulation()
                ee_pos = self.get_ee_pos()
                detals = np.abs(ee_pos - self.drop_area)
                if max(detals) < self.tolerance:
                    success = True
            if success:
                for i in range(self.frame_skip):
                    self.controller.run(
                        target_pose
                    )
                    self.grp_ctrl.run(signal=0)
                    self.step_mujoco_simulation()
                    #self.get_image_data(depth=True,show=True)
                if self.data.ctrl[0] == 0:
                        return success

    def check_terminated(self):
        terminated = False
        box_height = np.ones(len(_target_box))
        for i in range(len(_target_box)):
            box = _target_box[i]
            height = self.get_body_com(box)[2]
            box_height[i] = height
        if box_height.max() < self.TABLE_HEIGHT:
            terminated = True
        return terminated
            

    def check_grasp_success(self):
        grasp_success = False
        right_finger_xpos = self.get_body_com(_right_finger_name)
        left_finger_xpos = self.get_body_com(_left_finger_name)
        finger_distance = max(np.abs(right_finger_xpos-left_finger_xpos))
        if finger_distance>_close_finger_dis and finger_distance<_open_finger_dis and self.data.ctrl[0]==255:
            grasp_success = True
        return grasp_success


    def open_gripper(self):
        target_pose = list(self.get_ee_pos())
        target_pose.extend([0,0,1,1])
        for i in range(self.frame_skip):
            self.controller.run(
                target_pose
            )
            self.grp_ctrl.run(signal=0)
            self.step_mujoco_simulation()



    def move_and_grasp(self,action):
        action[2] = 1.15
        move_to_above = self.move_eef(action)

        if move_to_above:
            down_success = self.down_and_grasp(action)

            if down_success:
                self.move_up_drop()
            else:
                self.open_gripper()
                



    def _set_action_space(self):
        self.action_space1 = spaces.Box(low=-0.25,high=0.25,shape=[2])


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
        if self.grasped_num == _grasp_target_num or self.grasp_step==5:
            self.terminated = True
        if self.check_terminated():
            self.terminated = True
        self.grasp_step+=1
        return self.observation,self.reward,self.terminated,self.info

    def step_test(self, action,failed_num):
        self.terminated = False
 
        self.before_grasp(show=False)
        self.move_and_grasp(action)
        self.after_grasp(show=False)
        if self.grasped_num == _grasp_target_num: 
            self.terminated = True
            self.info["completion"] = "Success"
        if self.grasp_step==1:
            self.terminated = True
        if self.check_terminated():
            self.terminated = True
        self.grasp_step+=1
        return self.observation,self.reward,self.terminated,self.info


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