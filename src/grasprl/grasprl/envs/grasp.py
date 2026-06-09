
import os
import sys
import random
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
        # 放置区域必须在桌子范围内，确保物体不会掉到桌子外面
        # 桌子中心在 (0, 0.35)，所以放置在桌子右侧中间位置
        self.drop_area = [0.15, 0.15, 1.15]
        self.TABLE_HEIGHT = 0.95   # 桌面实际高度（geom center=0.9 + half_size=0.05）
        self.GRASP_DEPTH = 0.08    # 抓取时下压距离（确保夹爪碰到物体）
        self.LIFT_HEIGHT = 0.25    # 抓取成功后抬起高度
        self.SUCCESS_REWARD = 100.0
        # 密集奖励参数（辅助 DQN 更快收敛）
        self.APPROACH_REWARD_SCALE = 3.0   # 靠近目标奖励系数
        self.CLOSURE_REWARD_SCALE = 2.0    # 夹爪闭合奖励系数
        self.LIFT_REWARD_SCALE = 8.0       # 抬起高度奖励系数
        self.CONTACT_REWARD = 5.0          # 手指接触物体奖励
        self.BASE_FAIL_REWARD = -3.0       # 基础失败惩罚

        self.arm_joints_names = list(self.model_names.joint_names[:6])
        self.arm_joints = [self.find('joint', name) for name in self.arm_joints_names]
        self.eef_name = self.model_names.site_names[1]
        self.eef_site = self.find('site', self.eef_name)

        self.controller = OSC(
            physics=self.physics,
            joints=self.arm_joints,
            eef_site=self.eef_site,
            min_effort=-500, max_effort=500,
            kp=500, ko=300, kv=50,
            vmax_xyz=5, vmax_abg=5
        )
        # 使用新的 left/right_finger_act position actuators 控制夹爪
        self.left_finger_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_act'
        )
        self.right_finger_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_act'
        )
        # GripperEffortCtrl 作为备用（当 actuator 不存在时用 joint effort）
        self.gripper = self.gripper_id
        self.grp_ctrl = GripperEffortCtrl(physics=self.physics, gripper=self.gripper, effort=35.0)
        self.target_objects = _target_box
        self.grasped_num = 0
        self.grasp_step = 0
        self.object_positions_before_grasp = {}
        self.current_grasp_target = None

        # 给夹爪关节添加阻尼，稳定约束求解
        _gripper_joint_names = [
            'left_inner_knuckle_joint', 'left_outer_knuckle_joint', 'left_finger_joint',
            'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_finger_joint',
        ]
        for jnt_name in _gripper_joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                dof_id = self.model.jnt_dofadr[jnt_id]
                self.model.dof_damping[dof_id] = 0.5

    # ---------- 内部辅助方法 ----------
    def _sim_step(self, n=1):
        """纯物理仿真步，不渲染，用于内部密集循环提速。"""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def _try_render(self):
        """只在 human 模式下渲染一帧，用于阶段边界视觉更新。"""
        if self.render_mode == "human":
            self.render()

    def _sanitize_physics_data(self):
        """只清理 qvel/ctrl/qacc 中的 NaN/Inf， 不触碰 qpos 以免夹爪关节 NaN 导致夹爪重置到起始位置。"""
        for attr in ['qvel', 'ctrl', 'qacc']:
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
        # 增大动作范围，让机械臂能快速移动到目标位置
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=[3], dtype=np.float32)

    def _set_observation_space(self):
        self.observation = defaultdict()
        self.observation["rgb"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3), dtype=np.float32)
        self.observation["depth"] = np.zeros((self.IMAGE_WIDTH, self.IMAGE_HEIGHT), dtype=np.float32)

    def move_eef(self, target, max_steps=None):
        if hasattr(target, "tolist"):
            target = target.tolist()
        if max_steps is None:
            max_steps = self.frame_skip * 5
        ee_quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat_wxyz, self.data.site_xmat[self.eef_site].copy().flatten())
        ee_quat = ee_quat_wxyz[[1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
        target_pose = target + ee_quat.tolist()
        for _ in range(max_steps):
            current_frame_skip = self.frame_skip if np.linalg.norm(np.array(self.get_ee_pos()) - np.array(target)) > 0.1 else 20
            for _ in range(current_frame_skip):
                self.controller.run(target_pose)
                self._sanitize_physics_data()
                self._sim_step()
                self._clamp_arm_velocity()
                self._sync_arm_ctrl()
            if np.allclose(self.get_ee_pos(), np.array(target[:3]), atol=self.tolerance):
                return True
        return False

    def _move_eef_ik(self, target_pos, max_steps=800, pos_tol=0.008):
        """使用 OSC 迭代平滑移动到目标位置。

        不直接修改 qpos，完全依靠 OSC 控制器 + 物理仿真实现平滑运动。
        限制关节速度防止 NaN。
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        ee_quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat_wxyz, self.data.site_xmat[self.eef_site].copy().flatten())
        ee_quat = ee_quat_wxyz[[1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
        target_pose = target_pos.tolist() + ee_quat.tolist()

        for _ in range(max_steps):
            self.controller.run(target_pose)
            self._sanitize_physics_data()
            self._sim_step()
            self._clamp_arm_velocity()
            self._clamp_gripper_velocity()  # 添加夹爪速度钳制
            if np.linalg.norm(self.get_ee_pos() - target_pos) < pos_tol:
                return True

        final_err = np.linalg.norm(self.get_ee_pos() - target_pos)
        return final_err < pos_tol * 5

    def _clamp_arm_velocity(self, max_vel=5.0):
        """限制臂关节速度，防止仿真不稳定产生 NaN。"""
        for jn in self.arm_joints:
            vaddr = self.model.jnt_dofadr[jn]
            self.data.qvel[vaddr] = np.clip(self.data.qvel[vaddr], -max_vel, max_vel)

    def _clamp_gripper_velocity(self, max_vel=2.0):
        """限制夹爪关节速度，防止约束求解器产生 NaN。"""
        _gripper_joint_names = [
            'left_inner_knuckle_joint', 'left_outer_knuckle_joint', 'left_finger_joint',
            'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_finger_joint',
        ]
        for jnt_name in _gripper_joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                vaddr = self.model.jnt_dofadr[jnt_id]
                self.data.qvel[vaddr] = np.clip(self.data.qvel[vaddr], -max_vel, max_vel)

    def solve_ik_numerical(self, target_pos, max_iter=1000, tol=1e-4):
        """使用数值优化求解逆运动学。
        
        参数:
            target_pos: 目标EE位置 [x, y, z]
            max_iter: 最大迭代次数
            tol: 收敛容差
        
        返回:
            6维关节角度数组
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        
        # 保存当前状态
        current_qpos = self.data.qpos[:6].copy()
        
        # 目标函数：最小化 EE 位置误差
        def objective(qpos_6d):
            self.data.qpos[:6] = qpos_6d
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.get_ee_pos().copy()
            error = np.linalg.norm(ee_pos - target_pos)
            return error
        
        # 使用 Nelder-Mead 优化（不需要梯度）
        from scipy.optimize import minimize
        result = minimize(
            objective,
            current_qpos,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol}
        )
        
        # 恢复原始状态
        self.data.qpos[:6] = current_qpos
        mujoco.mj_forward(self.model, self.data)
        
        if result.fun > 0.01:  # 如果误差 > 1cm
            print(f"  ⚠️  IK 警告: 最终误差 {result.fun:.4f}m")
        
        return result.x
    
    def move_joints_smooth(self, target_qpos, steps=200):
        """从当前 qpos 平滑插值到目标 qpos。
        
        强制设置 qpos 并清零速度，确保手臂不会漂移。
        """
        target_qpos = np.asarray(target_qpos, dtype=np.float64)
        start_qpos = self.data.qpos[:6].copy()
        
        for i in range(steps + 1):  # +1 以确保到达最终位置
            alpha = i / steps
            # 五次多项式插值
            alpha_smooth = alpha**3 * (6*alpha**2 - 15*alpha + 10)
            interp_qpos = start_qpos + alpha_smooth * (target_qpos - start_qpos)
            
            # 强制设置关节位置
            self.data.qpos[:6] = interp_qpos
            
            # 清零臂关节速度，防止物理引擎干扰
            for jn in self.arm_joints:
                vaddr = self.model.jnt_dofadr[jn]
                self.data.qvel[vaddr] = 0.0
            
            # 运行物理步
            mujoco.mj_step(self.model, self.data)
            
            # 钳制速度防止 NaN
            self._clamp_arm_velocity(max_vel=1.0)
            self._clamp_gripper_velocity(max_vel=1.0)
        
        # 确保最终位置
        self.data.qpos[:6] = target_qpos
        for jn in self.arm_joints:
            vaddr = self.model.jnt_dofadr[jn]
            self.data.qvel[vaddr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def down_and_grasp(self, target):
        # 使用传入的目标位置，采用更稳定的两步抓取策略
        down_pose = np.array(target, dtype=np.float64).copy()
        
        # 步骤1：先定位到物体上方安全位置（避免IK直接下压导致不稳定）
        approach_pose = down_pose.copy()
        approach_pose[2] = max(down_pose[2], self.TABLE_HEIGHT + 0.2)  # 物体上方20cm
        self.move_eef(approach_pose)
        
        # 记录所有物体位置
        for obj_name in self.target_objects:
            pos = self.get_body_com(obj_name)
            self.object_positions_before_grasp[obj_name] = pos.copy()
        
        # 找到最近的物体并微调位置
        closest_obj = None
        closest_dist = float('inf')
        for obj_name in self.target_objects:
            pos = self.get_body_com(obj_name)
            dist = np.linalg.norm(pos[:2] - down_pose[:2])
            if dist < closest_dist:
                closest_dist = dist
                closest_obj = obj_name
        
        # 微调位置对准最近物体
        if closest_obj is not None:
            obj_pos = self.get_body_com(closest_obj)
            down_pose[0] = obj_pos[0]
            down_pose[1] = obj_pos[1]
            down_pose[2] = obj_pos[2]
        
        # 步骤2：限制下压深度，确保不会穿透桌面
        down_pose[2] = max(down_pose[2] - 0.03, self.TABLE_HEIGHT + 0.03)
        
        # 慢慢下降到抓取位置
        success = self.move_eef(down_pose)
        if success:
            # 闭合夹爪 (使用实际存在的 position actuator)
            left_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_act')
            right_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_act')
            for _ in range(self.frame_skip * 3):
                if left_act >= 0:
                    self.data.ctrl[left_act] = 0.95
                if right_act >= 0:
                    self.data.ctrl[right_act] = 0.95
                self._sim_step()
            # 再用力夹紧
            for _ in range(self.frame_skip * 2):
                if left_act >= 0:
                    self.data.ctrl[left_act] = 0.95
                if right_act >= 0:
                    self.data.ctrl[right_act] = 0.95
                self._sim_step()
        return success

    def move_up_drop(self):
        up_pose = list(self.get_ee_pos())
        up_pose[2] += self.LIFT_HEIGHT
        self.move_eef(up_pose)

        grasp_success = self.check_grasp_success()

        if grasp_success:
            self.grasped_num += 1
            # 确保放置位置的 z 坐标足够高，避免物体穿透桌子
            safe_drop_area = self.drop_area.copy()
            safe_drop_area[2] = max(safe_drop_area[2], self.TABLE_HEIGHT + 0.3)  # 桌面上方至少30cm
            self.move_eef(safe_drop_area)

            # 打开夹爪 (使用实际存在的 position actuator)
            left_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_act')
            right_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_act')
            for _ in range(self.frame_skip * 4):
                if left_act >= 0:
                    self.data.ctrl[left_act] = 0.0
                if right_act >= 0:
                    self.data.ctrl[right_act] = 0.0
                self._sim_step()

            # 增加更多物理仿真步让物体稳定下落到桌面上
            for _ in range(self.frame_skip * 3):
                self._sim_step()

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
                    if left_act >= 0:
                        self.data.ctrl[left_act] = 0.0
                    if right_act >= 0:
                        self.data.ctrl[right_act] = 0.0
                    self._sim_step()

            for _ in range(self.frame_skip * 2):
                self.grp_ctrl.run(signal=0)
                self.step_mujoco_simulation()
            for _ in range(self.frame_skip // 2):
                self.step_mujoco_simulation()
        
        self.object_positions_before_grasp.clear()
        return grasp_success

    def get_finger_contacts(self):
        """检测手指与物体之间的接触。
        
        Returns:
            contacts: {obj_name: total_force} 每个物体与手指的总接触力
            left_contacts: set of object names touching left finger
            right_contacts: set of object names touching right finger
        """
        finger_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, _left_finger_name),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, _right_finger_name),
        }
        obj_body_ids = {
            obj_name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            for obj_name in self.target_objects
        }

        contacts = {}
        left_contacts = set()
        right_contacts = set()
        left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, _left_finger_name)
        right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, _right_finger_name)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1, b2 = contact.geom1, contact.geom2
            body1 = self.model.geom_bodyid[b1]
            body2 = self.model.geom_bodyid[b2]

            # 检查手指-物体接触
            if body1 in finger_body_ids and body2 in obj_body_ids.values():
                finger_body = body1
                obj_body = body2
            elif body2 in finger_body_ids and body1 in obj_body_ids.values():
                finger_body = body2
                obj_body = body1
            else:
                continue

            # 计算接触力大小
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            total_force = np.linalg.norm(force[:3])

            # 找到对应的物体名
            for obj_name, oid in obj_body_ids.items():
                if oid == obj_body:
                    if obj_name not in contacts:
                        contacts[obj_name] = 0.0
                    contacts[obj_name] += total_force
                    if finger_body == left_finger_id:
                        left_contacts.add(obj_name)
                    if finger_body == right_finger_id:
                        right_contacts.add(obj_name)
                    break

        return contacts, left_contacts, right_contacts

    def check_grasp_success(self):
        """检查抓取是否成功。
        
        优化版：不仅检查物体 z 位移，还验证手指接触和夹爪闭合程度。
        - 物体必须被抬升 > 3mm
        - 手指间距必须小于张开值（夹爪确实闭合了）
        - 至少有一根手指与该物体有接触
        """
        right = self.get_body_com(_right_finger_name)
        left = self.get_body_com(_left_finger_name)
        finger_dist = np.linalg.norm(right - left)

        # 获取接触信息
        contacts, left_contacts, right_contacts = self.get_finger_contacts()

        object_lifted = False
        lifted_object = None
        for obj_name in self.target_objects:
            if obj_name in self.object_positions_before_grasp:
                prev_pos = self.object_positions_before_grasp[obj_name]
                curr_pos = self.get_body_com(obj_name)
                z_diff = curr_pos[2] - prev_pos[2]
                if z_diff > 0.003:  # 物体提升 3mm 以上
                    # 额外验证：手指间距合理 + 有接触
                    finger_closed = finger_dist < 0.12  # 夹爪已闭合
                    has_contact = (obj_name in left_contacts or obj_name in right_contacts)
                    if finger_closed and has_contact:
                        object_lifted = True
                        lifted_object = obj_name
                        break

        self.current_grasp_target = lifted_object
        self._last_finger_dist = finger_dist
        self._last_contacts = contacts
        self._last_lift_obj = lifted_object
        self._last_z_diffs = {}
        for obj_name in self.target_objects:
            if obj_name in self.object_positions_before_grasp:
                self._last_z_diffs[obj_name] = (
                    self.get_body_com(obj_name)[2] -
                    self.object_positions_before_grasp[obj_name][2]
                )

        return object_lifted

    def open_gripper(self, steps=40):
        """打开夹爪，同时用 OSC 保持臂位置不变，防止重力漂移。"""
        hold_ee = self.get_ee_pos().copy()
        ee_quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat_wxyz, self.data.site_xmat[self.eef_site].copy().flatten())
        ee_quat = ee_quat_wxyz[[1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
        hold_pose = hold_ee.tolist() + ee_quat.tolist()

        if self.left_finger_act >= 0:
            self.data.ctrl[self.left_finger_act] = 0.0
        if self.right_finger_act >= 0:
            self.data.ctrl[self.right_finger_act] = 0.0
        for _ in range(steps):
            self.controller.run(hold_pose)  # OSC 保持臂不动
            self._sanitize_physics_data()
            self._sim_step()
            self._clamp_arm_velocity()
            self._clamp_gripper_velocity()

    def close_gripper(self, target_val=0.95, steps=120):
        """闭合夹爪，渐进增加 ctrl 值到 target_val，同时用 OSC 保持臂不动。"""
        left_act_id = self.left_finger_act
        right_act_id = self.right_finger_act
        hold_ee = self.get_ee_pos().copy()
        ee_quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat_wxyz, self.data.site_xmat[self.eef_site].copy().flatten())
        ee_quat = ee_quat_wxyz[[1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
        hold_pose = hold_ee.tolist() + ee_quat.tolist()

        # 渐进闭合
        for step_i in range(steps):
            self.controller.run(hold_pose)
            progress = step_i / steps
            close_val = target_val * progress
            if left_act_id >= 0:
                self.data.ctrl[left_act_id] = close_val
            if right_act_id >= 0:
                self.data.ctrl[right_act_id] = close_val
            self._sanitize_physics_data()
            self._sim_step()
            self._clamp_arm_velocity()
            self._clamp_gripper_velocity()

        # 保压：维持闭合力
        for _ in range(steps // 2):
            self.controller.run(hold_pose)
            if left_act_id >= 0:
                self.data.ctrl[left_act_id] = target_val
            if right_act_id >= 0:
                self.data.ctrl[right_act_id] = target_val
            self._sanitize_physics_data()
            self._sim_step()
            self._clamp_arm_velocity()
            self._clamp_gripper_velocity()

    def step_test(self, action, fail_count=0):
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info

    def step(self, action):
        """
        执行一次完整抓取尝试。
        action = [x, y, z] 世界坐标系下的目标物体位置。
        流程: 打开夹爪→接近上方→下降→夹取（OSC 保持 arm）→抬起→检测→放置/松开
        
        奖励设计（密集+稀疏混合）：
        - 靠近奖励：EE 越接近目标 +reward
        - 闭合奖励：夹爪闭合程度 +reward
        - 接触奖励：手指碰到物体 +reward
        - 抬起奖励：物体抬起高度 +reward
        - 成功：+100，失败：小惩罚
        """
        self.info = {}
        reward = 0.0
        target = np.array(action, dtype=np.float64)

        # 确保 arm actuator ctrl 与 qpos 同步（reset 已做，但保险起见）
        self._sync_arm_ctrl()

        # 约束在桌面工作范围内
        target[0] = np.clip(target[0], -0.15, 0.15)
        target[1] = np.clip(target[1], 0.28, 0.42)
        target[2] = max(target[2], self.TABLE_HEIGHT + 0.02)

        # ========== Phase 1: 打开夹爪 ==========
        self.open_gripper()
        self._try_render()

        # ========== Phase 2: 粗定位 — 到物体上方 + 动态测量 offset ==========
        ee_before_approach = self.get_ee_pos().copy()
        approach_eef = np.array([target[0], target[1], target[2] + 0.20])
        approach_eef[2] = max(approach_eef[2], self.TABLE_HEIGHT + 0.15)
        self._move_eef_ik(approach_eef)
        self._try_render()

        # 靠近奖励：EE 向目标移动的距离
        approach_dist = np.linalg.norm(self.get_ee_pos() - target) - np.linalg.norm(ee_before_approach - target)
        approach_dist = max(-1.0, min(1.0, approach_dist))
        reward += self.APPROACH_REWARD_SCALE * approach_dist

        # ★ 测量真实 finger-to-EE 偏移量
        left_f = self.get_body_com(_left_finger_name)
        right_f = self.get_body_com(_right_finger_name)
        finger_center = (left_f + right_f) / 2
        real_offset = self.get_ee_pos() - finger_center   # ee = finger + offset

        self._try_render()

        # ========== Phase 3: 用实测 offset 精准下降 ==========
        grasp_eef = target + real_offset
        # 用 GRASP_DEPTH 控制下压深度，确保手指在物体两侧
        grasp_eef[2] = max(grasp_eef[2] + self.GRASP_DEPTH - 0.04, self.TABLE_HEIGHT + 0.01)

        ee_before_descent = self.get_ee_pos().copy()
        start_descent = ee_before_descent.copy()
        for seg_i in range(1, 4):
            alpha = seg_i / 3
            wp = start_descent + alpha * (grasp_eef - start_descent)
            self._move_eef_ik(wp)

        # 靠近奖励（精准下降阶段）
        final_dist = np.linalg.norm(self.get_ee_pos() - target)
        initial_dist = np.linalg.norm(ee_before_descent - target)
        descent_improvement = max(0.0, initial_dist - final_dist)
        reward += self.APPROACH_REWARD_SCALE * 0.5 * descent_improvement

        self._try_render()

        # ========== Phase 4: 保持 arm + 闭合夹爪 ==========
        self.object_positions_before_grasp = {}
        for obj_name in self.target_objects:
            self.object_positions_before_grasp[obj_name] = self.get_body_com(obj_name)

        self.close_gripper(target_val=0.95, steps=self.frame_skip * 3)

        # 闭合奖励：基于夹爪闭合程度
        finger_dist = self.get_finger_dist()
        max_open_dist = 0.15
        closure_ratio = max(0.0, 1.0 - finger_dist / max_open_dist)
        reward += self.CLOSURE_REWARD_SCALE * closure_ratio

        # 接触奖励：检查手指是否碰到物体
        contacts, _, _ = self.get_finger_contacts()
        if contacts:
            reward += self.CONTACT_REWARD

        self._try_render()

        # ========== Phase 5: 抬起 ==========
        ee = self.get_ee_pos()
        lift_pos = list(ee)
        lift_pos[2] += self.LIFT_HEIGHT
        self._move_eef_ik(lift_pos)
        self._try_render()

        # ========== Phase 6: 检测结果 ==========
        grasp_success = self.check_grasp_success()

        left_act_id = self.left_finger_act
        right_act_id = self.right_finger_act

        if grasp_success:
            self.grasped_num += 1
            # 抬起奖励：和实际物体被抬起的高度成正比
            if self._last_lift_obj and self._last_lift_obj in self._last_z_diffs:
                actual_lift = self._last_z_diffs[self._last_lift_obj]
                lift_ratio = min(1.0, actual_lift / self.LIFT_HEIGHT)
                reward += self.LIFT_REWARD_SCALE * lift_ratio
            reward += self.SUCCESS_REWARD
            self.info["grasp"] = "Success"
            self.info["lift_obj"] = self._last_lift_obj

            self._move_eef_ik(self.drop_area)
            self._try_render()
            # 打开夹爪
            for _ in range(self.frame_skip * 4):
                if left_act_id >= 0:
                    self.data.ctrl[left_act_id] = 0.0
                if right_act_id >= 0:
                    self.data.ctrl[right_act_id] = 0.0
                self._sim_step()
            self._try_render()
        else:
            reward += self.BASE_FAIL_REWARD
            self.info["grasp"] = "Failed"
            self.info["lift_obj"] = self._last_lift_obj
            for _ in range(self.frame_skip):
                if left_act_id >= 0:
                    self.data.ctrl[left_act_id] = 0.0
                if right_act_id >= 0:
                    self.data.ctrl[right_act_id] = 0.0
                self._sim_step()
            self._try_render()

        self.grasp_step += 1
        self.object_positions_before_grasp.clear()

        done = self.grasped_num >= _grasp_target_num or self.grasp_step >= 30

        self.info["grasped_num"] = self.grasped_num
        self.info["completion"] = "Success" if self.grasped_num >= _grasp_target_num else "InProgress"
        self.info["reward_breakdown"] = f"total={reward:.2f}"

        return self.observation, reward, done, self.info

    def get_finger_dist(self):
        right = self.get_body_com(_right_finger_name)
        left = self.get_body_com(_left_finger_name)
        return np.linalg.norm(right - left)

    def reset(self):
        self._reset_simulation()

        # 物体半高映射（用于放在桌面上）
        _obj_half = {
            'box_1': 0.025, 'box_2': 0.02, 'box_3': 0.03,
            'ball_1': 0.035, 'ball_2': 0.03, 'ball_3': 0.025,
        }

        # 随机化物体位置，直接放在桌面上（避免物理弹跳导致物体掉落）
        # 桌子范围: x=-0.3~0.3, y=-0.3~0.3, z=0.95(桌面)
        for obj_name in self.target_objects:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_name + '_x')
            if jnt_id >= 0:
                qpos_addr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_addr] = random.uniform(-0.2, 0.2)          # x 偏移（在桌子范围内）
                self.data.qpos[qpos_addr + 1] = random.uniform(-0.2, 0.2)      # y 偏移（在桌子范围内，避免超出）
                # 放在桌面上方 1cm: body_z + z_offset - half_size = TABLE_HEIGHT + 0.01
                body_z = self.model.body_pos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                ][2]
                half = _obj_half.get(obj_name, 0.02)
                self.data.qpos[qpos_addr + 2] = self.TABLE_HEIGHT + half - body_z + 0.01

        # UR5e 起始姿态 —— 朝桌子方向（+Y），降低起始高度以便更容易到达桌面
        # Base 在 (0, -0.45, 1.30)，桌子在 y=0~0.3。shoulder_pan=π/2 → 臂向 +Y 伸出。
        # 修改：增加 shoulder_lift 让臂更向下，使 EE 能够到达 Z≈1.0 的桌面高度
        self.data.qpos[:6] = [
            1.57,       # shoulder_pan_joint:  π/2 → EE 朝桌面中心 (+Y)
            -1.57,      # shoulder_lift_joint: 臂向下 90°（原来是 -1.3，降低以便到达桌面）
            1.57,       # elbow_joint:         前臂弯曲 90°（原来是 1.3，配合降低高度）
            -1.57,      # wrist_1_joint:       工具倾斜
            -1.57,      # wrist_2_joint:       手腕旋转
            0.0,        # wrist_3_joint
        ]
        # 打开夹爪
        if hasattr(self, 'grp_ctrl'):
            self.grp_ctrl.reset()

        mujoco.mj_forward(self.model, self.data)
        # 少量物理步让物体稳定贴合桌面（5 步足够，不会穿透）
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        # 清零速度，防止弹跳
        self.data.qvel[:] = 0

        # 同步 actuator ctrl 与当前 qpos，消除 position actuator 干扰力
        self._sync_arm_ctrl()

        self.grasped_num = 0
        self.grasp_step = 0
        return self.observation

    def _sync_arm_ctrl(self, include_gripper=False):
        """完全消除 arm position actuator 的影响，避免和 OSC 冲突。
        
        UR5e actuator: force = 2000*(ctrl - qpos) - 400*qvel
        设置 ctrl = qpos + 0.2*qvel → net force = 0（包括阻尼项）
        
        夹爪同步默认关闭，避免干扰 open/close 命令。
        仅在需要完全复位时传入 include_gripper=True。
        """
        # UR5e arm: gain=2000, damping=400 → velocity coefficient = 400/2000 = 0.2
        for jnt_name in self.arm_joints_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jnt_name)
            if act_id >= 0:
                jnt_id = self.find('joint', jnt_name)
                qaddr = self.model.jnt_qposadr[jnt_id]
                vaddr = self.model.jnt_dofadr[jnt_id]
                self.data.ctrl[act_id] = self.data.qpos[qaddr] + 0.2 * self.data.qvel[vaddr]

        if include_gripper:
            _gripper_joints = [
                'left_inner_knuckle_joint', 'left_outer_knuckle_joint', 'left_finger_joint',
                'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_finger_joint',
            ]
            for jnt_name in _gripper_joints:
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jnt_name)
                if act_id >= 0:
                    jnt_id = self.find('joint', jnt_name)
                    qaddr = self.model.jnt_qposadr[jnt_id]
                    vaddr = self.model.jnt_dofadr[jnt_id]
                    self.data.ctrl[act_id] = self.data.qpos[qaddr] + 0.01 * self.data.qvel[vaddr]

    def reset_without_random(self):
        return self.reset()
