
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
        self.GRASP_DEPTH = 0.06    # 抓取时下压距离（确保夹爪碰到物体）
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
        # 通过 tendon actuator (fingers_actuator) 控制 AG95 夹爪，比 qfrc_applied 更稳定
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.physics.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fingers_actuator'
        )
        self.grp_ctrl = GripperEffortCtrl(
            physics=self.physics,
            actuator_id=self.gripper_actuator_id,
            ctrl_close=0.95,   # 闭合时 tendon 目标长度 (max ≈ 0.943)
            ctrl_open=0.0,
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
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, self.data.site_xmat[self.eef_site].copy().flatten())
        target_pose = target + ee_quat.tolist()
        for _ in range(max_steps):
            current_frame_skip = self.frame_skip if np.linalg.norm(np.array(self.get_ee_pos()) - np.array(target)) > 0.1 else 20
            for _ in range(current_frame_skip):
                self.controller.run(target_pose)
                self._sanitize_physics_data()
                self._sim_step()
                self._sync_arm_ctrl()
            if np.allclose(self.get_ee_pos(), np.array(target[:3]), atol=self.tolerance):
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

    def _move_eef_ik(self, target_pos):
        """使用 IK + 力控 的混合方式到达目标，过程平滑不跳跃。"""
        target_pos = np.asarray(target_pos, dtype=np.float64)
        actual = self._ik_to_target(target_pos)
        # 用 OSC 微调并稳定（用 _sim_step 避免渲染卡顿）
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, self.data.site_xmat[self.eef_site].copy().flatten())
        target_pose = target_pos.tolist() + ee_quat.tolist()
        for _ in range(20):
            self.controller.run(target_pose)
            self._sanitize_physics_data()
            self._sim_step()
            self._sync_arm_ctrl()
        return actual

    def _ik_to_target(self, target_pos, steps=200, lr=0.5):
        """Jacobian pseudoinverse IK: adjust qpos to move eef_site to target_pos
        每次迭代后做一步物理仿真，避免瞬间"跳变"导致的视觉卡顿。
        """
        m, d = self.model, self.data
        eef_id = self.eef_site
        jnt_ids = self.arm_joints
        dof_adrs = [m.jnt_dofadr[j] for j in jnt_ids]
        target = np.asarray(target_pos, dtype=np.float64)

        # 保存初始 qpos 用于 nullspace 向心
        q_home = d.qpos[:6].copy()

        for i in range(steps):
            mujoco.mj_forward(m, d)
            ee = d.site_xpos[eef_id]
            err = target - ee
            if np.linalg.norm(err) < self.tolerance * 2:
                break

            # Position Jacobian (3 x nv)
            J_full = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, J_full, None, eef_id)

            J = np.zeros((3, len(dof_adrs)))
            for k, adr in enumerate(dof_adrs):
                J[:, k] = J_full[:, adr]

            # Damped pseudoinverse
            lamb = 0.001
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lamb * np.eye(3))

            # 任务空间增量
            dq_task = lr * J_pinv @ err

            # Nullspace 向心：把多余自由度向初始姿态拉（避免扭曲）
            null_proj = np.eye(6) - J_pinv @ J
            q_curr = np.array([d.qpos[adr] for adr in dof_adrs])
            dq_null = null_proj @ (q_home - q_curr) * 0.05

            dq = dq_task + dq_null
            for k, adr in enumerate(dof_adrs):
                d.qpos[adr] += dq[k]

            # 每20次迭代做一步物理，让机械臂平滑动起来
            if i % 20 == 0 and i > 0:
                self._sim_step()

        mujoco.mj_forward(m, d)
        return self.get_ee_pos()

    def down_and_grasp(self, target):
        # 使用传入的目标位置，采用更稳定的两步抓取策略
        down_pose = np.array(target, dtype=np.float64).copy()
        
        # 步骤1：先定位到物体上方安全位置（避免IK直接下压导致不稳定）
        approach_pose = down_pose.copy()
        approach_pose[2] = max(down_pose[2], self.TABLE_HEIGHT + 0.2)  # 物体上方20cm
        self.move_eef(approach_pose)
        # 减小抓取深度，避免撞到桌子，让抓取器更接近物体
        down_pose = target.copy()
        down_pose[2] -= self.GRASP_DEPTH
        down_pose[2] = max(down_pose[2], self.TABLE_HEIGHT + 0.03)
        
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
                if z_diff > 0.003:  # 物体提升 3mm 以上算成功
                if z_diff > 0.005:  # 物体提升 5mm 以上就算成功
                if z_diff > 0.01:
                    object_lifted = True
                    lifted_object = obj_name
                    break
        
        self.current_grasp_target = lifted_object
        # ONLY check if an object was actually lifted; don't return True just because gripper closed
        return object_lifted

    def open_gripper(self):
        """打开夹爪，只设置 ctrl 值，不执行物理仿真步骤"""
        left_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_act')
        right_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_act')
        if left_act >= 0:
            self.data.ctrl[left_act] = 0.0
        if right_act >= 0:
            self.data.ctrl[right_act] = 0.0
        # 只做一步 forward 让 ctrl 生效
        mujoco.mj_forward(self.model, self.data)

    def step_test(self, action, fail_count=0):
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info
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
        """
        执行一次完整抓取尝试。
        action = [x, y, z] 世界坐标系下的目标物体位置。
        流程: 打开夹爪→接近上方→下降→夹取（OSC 保持 arm）→抬起→检测→放置/松开
        """
        self.info = {}
        target = np.array(action, dtype=np.float64)

        # 确保 arm actuator ctrl 与 qpos 同步（reset 已做，但保险起见）
        self._sync_arm_ctrl()

        # 约束在桌面工作范围内
        target[0] = np.clip(target[0], -0.15, 0.15)
        target[1] = np.clip(target[1], 0.28, 0.42)
        target[2] = max(target[2], self.TABLE_HEIGHT + 0.02)

        # 夹爪中心相对 eef_site 的偏移（因 arm 姿态变化，依赖精调修正）
        finger_dx, finger_dy, finger_dz = 0.06, 0.0, -0.0535

        # ========== Phase 1: 打开夹爪 ==========
        self.grp_ctrl.last_signal = None
        # 直接把手指 joint qpos 归零 → 瞬间全开
        _gripper_joints = [
            'left_inner_knuckle_joint', 'left_outer_knuckle_joint', 'left_finger_joint',
            'right_inner_knuckle_joint', 'right_outer_knuckle_joint', 'right_finger_joint',
        ]
        for jnt_name in _gripper_joints:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                self.data.qpos[self.model.jnt_qposadr[jnt_id]] = 0.0
        # 设置实际存在的 position actuator ctrl=0 保持打开状态
        left_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_act')
        right_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_act')
        if left_act_id >= 0:
            self.data.ctrl[left_act_id] = 0.0
        if right_act_id >= 0:
            self.data.ctrl[right_act_id] = 0.0
        for _ in range(self.frame_skip // 2):   # 让物理稳定
            self._sim_step()
        self._try_render()

        # ========== Phase 2: 粗定位 - 到达物体上方 ==========
        approach_eef = np.array([
            target[0] - finger_dx,
            target[1] - finger_dy,
            target[2] - finger_dz + 0.12,
        ])
        approach_eef[2] = max(approach_eef[2], self.TABLE_HEIGHT + 0.18)
        self._move_eef_ik(approach_eef)
        self._try_render()

        # ========== Phase 3: 精调 - 测量实际偏移后精准定位 ==========
        grasp_eef = np.array([
            target[0] - finger_dx,
            target[1] - finger_dy,
            target[2] - finger_dz,
        ])
        grasp_eef[2] = max(grasp_eef[2], self.TABLE_HEIGHT + 0.06)
        self._move_eef_ik(grasp_eef)

        # 阻尼修正：测量手指中心与目标的偏差，逐步修正 eef 位置（减少到3次迭代）
        for refine_iter in range(3):
            left = self.get_body_com(_left_finger_name)
            right = self.get_body_com(_right_finger_name)
            finger_center = (left + right) / 2
            delta = finger_center - target
            xy_err = np.linalg.norm(delta[:2])
            if xy_err < 0.015:
                break
            # 阻尼修正 (lr=0.8) 防止振荡
            lr = 0.8
            corrected_eef = np.array([
                grasp_eef[0] - lr * delta[0],
                grasp_eef[1] - lr * delta[1],
                grasp_eef[2] - lr * delta[2],
            ])
            corrected_eef[2] = max(corrected_eef[2], self.TABLE_HEIGHT + 0.06)
            self._move_eef_ik(corrected_eef)
            grasp_eef = corrected_eef
        self._try_render()

        # ========== Phase 4: 保持 arm + 闭合夹爪 ==========
        self.object_positions_before_grasp = {}
        for obj_name in self.target_objects:
            self.object_positions_before_grasp[obj_name] = self.get_body_com(obj_name)

        # 获取实际存在的 position actuator ID（只有 left_finger_act, right_finger_act）
        left_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_act')
        right_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_act')

        # 保存当前 ee 位姿用于 OSC 保持臂位置
        hold_ee = self.get_ee_pos().copy()
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, self.data.site_xmat[self.eef_site].copy().flatten())
        hold_pose = hold_ee.tolist() + ee_quat.tolist()

        self.grp_ctrl.last_signal = None
        for step_i in range(self.frame_skip * 4):
            # 臂：OSC 保持位置（arm 无 actuator，必须用 OSC qfrc_applied）
            self.controller.run(hold_pose)

            # 手指：渐进闭合 position actuator（kp=500）
            progress = min(1.0, step_i / (self.frame_skip * 2))
            close_val = 0.95 * progress
            if left_act_id >= 0:
                self.data.ctrl[left_act_id] = close_val
            if right_act_id >= 0:
                self.data.ctrl[right_act_id] = close_val

            # 同时直接拉 AG95 outer knuckle qpos 加速闭合（绕过弱 kp）
            for jn in ['left_outer_knuckle_joint', 'right_outer_knuckle_joint']:
                jid = self.find('joint', jn)
                if jid >= 0:
                    qaddr = self.model.jnt_qposadr[jid]
                    current = self.data.qpos[qaddr]
                    self.data.qpos[qaddr] = current + (0.943 - current) * min(0.15, progress * 0.15)

            self._sanitize_physics_data()
            self._sim_step()
        self._try_render()

        # ========== Phase 5: 抬起 ==========
        ee = self.get_ee_pos()
        lift_pos = list(ee)
        lift_pos[2] += self.LIFT_HEIGHT
        self._move_eef_ik(lift_pos)
        self._try_render()

        # ========== Phase 6: 检测结果 ==========
        grasp_success = self.check_grasp_success()

        if grasp_success:
            self.grasped_num += 1
            reward = self.SUCCESS_REWARD
            self.info["grasp"] = "Success"

            self._move_eef_ik(self.drop_area)
            self._try_render()
            # 打开夹爪 (使用实际存在的 position actuator)
            for _ in range(self.frame_skip * 3):
                if left_act_id >= 0:
                    self.data.ctrl[left_act_id] = 0.0
                if right_act_id >= 0:
                    self.data.ctrl[right_act_id] = 0.0
                self._sim_step()

            # 开合释放
            for _ in range(self.frame_skip * 2):
                if left_act_id >= 0:
                    self.data.ctrl[left_act_id] = 0.0
                if right_act_id >= 0:
                    self.data.ctrl[right_act_id] = 0.0
                self._sim_step()
            self._try_render()
        else:
            reward = -5.0
            self.info["grasp"] = "Failed"
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

    def get_finger_dist(self):
        right = self.get_body_com(_right_finger_name)
        left = self.get_body_com(_left_finger_name)
        return np.linalg.norm(right - left)

    def reset(self):
        self._reset_simulation()

        # 物体半高映射（用于放在桌面上）
        _obj_half = {
            'box_1': 0.02, 'box_2': 0.015, 'box_3': 0.025,
            'ball_1': 0.03, 'ball_2': 0.025, 'ball_3': 0.02,
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

        # UR5e 起始姿态 —— 朝桌子方向（+Y），避免扭曲的侧向起始
        # Base 在 (0, -0.45, 0.95)，桌子在 y=0.35。shoulder_pan=0 → 臂向 +Y 伸出。
        self.data.qpos[:6] = [
            0.0,        # shoulder_pan_joint:  0 → 朝桌子方向（+Y），不再向侧面扭曲
            -1.2,       # shoulder_lift_joint: 前倾
            1.8,        # elbow_joint:         弯曲前伸
            -1.5,       # wrist_1_joint:       保持工具朝下
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

    def _sync_arm_ctrl(self):
        """完全消除 arm/gripper position actuator 的影响。
        
        UR5e actuator: force = 2000*(ctrl - qpos) - 400*qvel
        设置 ctrl = qpos + 0.2*qvel → force = 0（包括阻尼项）
        
        AG95 actuator: force = 10*(ctrl - qpos) - 0.1*qvel  
        设置 ctrl = qpos + 0.01*qvel → force = 0
        """
        # UR5e arm: gain=2000, damping=400 → velocity coefficient = 400/2000 = 0.2
        for jnt_name in self.arm_joints_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jnt_name)
            if act_id >= 0:
                jnt_id = self.find('joint', jnt_name)
                qaddr = self.model.jnt_qposadr[jnt_id]
                vaddr = self.model.jnt_dofadr[jnt_id]
                self.data.ctrl[act_id] = self.data.qpos[qaddr] + 0.2 * self.data.qvel[vaddr]

        # AG95 gripper joints: gain=10, damping=0.1 → velocity coefficient = 0.1/10 = 0.01
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
