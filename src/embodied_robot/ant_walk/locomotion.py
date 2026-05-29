# -*- coding: utf-8 -*-
import numpy as np
import mujoco


class AntLocomotion:
    """
    Ant 运动控制模块
    将高层指令（前进、转向）转化为具体的关节控制量。

    修复点：
    1. 不再把所有 ankle 目标都设为正值，而是根据 ant.xml 中每个关节的实际限位自动计算中位姿态。
    2. ankle_1/ankle_4 是正角度关节，ankle_2/ankle_3 是负角度关节，抬腿方向必须相反。
    3. target_vel 和 target_steer 真正参与步态幅值和左右差速控制。
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.nu = model.nu
        self.phase = 0.0

        # 执行器对应的关节顺序，必须与 ant.xml 的 <actuator> 顺序一致
        self.actuated_joint_names = [
            "hip_1", "ankle_1",
            "hip_2", "ankle_2",
            "hip_3", "ankle_3",
            "hip_4", "ankle_4",
        ]

        self.joint_ids = []
        self.qpos_indices = []
        self.qvel_indices = []
        self.joint_ranges = []

        for name in self.actuated_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"找不到关节: {name}")
            self.joint_ids.append(jid)
            self.qpos_indices.append(model.jnt_qposadr[jid])
            self.qvel_indices.append(model.jnt_dofadr[jid])
            self.joint_ranges.append(model.jnt_range[jid].copy())

        self.qpos_indices = np.asarray(self.qpos_indices, dtype=int)
        self.qvel_indices = np.asarray(self.qvel_indices, dtype=int)
        self.joint_ranges = np.asarray(self.joint_ranges, dtype=float)

        # 各关节的中位姿态。对当前 ant.xml：
        # hip ≈ 0 rad；ankle_1/4 ≈ +0.87 rad；ankle_2/3 ≈ -0.87 rad
        self.neutral_qpos = self.joint_ranges.mean(axis=1)

        # 髋关节前进方向符号。
        # 上一版 [1, 1, -1, -1] 对部分 ant.xml 会表现为“横向走”。
        # 当前模型的实际推进方向更接近 y 轴，因此目标和障碍物在 y 方向布置。
        # 这里保持原有步态符号，只由 Planner 负责朝向和转向控制。
        # 对应 hip_1, hip_2, hip_3, hip_4。
        self.hip_drive_sign = -np.array([1.0, -1.0, -1.0, 1.0], dtype=float)

        # 左右侧符号，用于转向差速。
        # 对应 hip_1, hip_2, hip_3, hip_4。
        self.turn_side_sign = np.array([1.0, 1.0, -1.0, -1.0], dtype=float)

        # 转向髋关节偏置符号。
        # 仅靠左右摆幅差时，当前开环步态容易继续直线前进；
        # 加入小幅度 hip bias 可以制造更明确的偏航力矩。
        self.turn_bias_sign = np.array([-1.0, 1.0, 1.0, -1.0], dtype=float)

        # 对角步态：1&3 同相，2&4 反相。
        # 对应 leg_1, leg_2, leg_3, leg_4
        self.leg_phase_offsets = np.array([0.0, np.pi, 0.0, np.pi], dtype=float)

    def _read_actuated_state(self, data):
        qpos = data.qpos[self.qpos_indices]
        qvel = data.qvel[self.qvel_indices]
        return qpos, qvel

    def apply_pd_control(self, data, target_qpos):
        """
        位置式 PD 控制。
        注意：data.ctrl 是 motor 控制量，会被 ant.xml 中的 gear 放大。
        因此这里输出后需要裁剪到 actuator_ctrlrange。
        """
        qpos_act, qvel_act = self._read_actuated_state(data)
        ctrl = self.config.balance_kp * (target_qpos - qpos_act) - self.config.balance_kd * qvel_act
        return ctrl

    def get_walking_signals(self, dt, forward_speed=1.0, turn_speed=0.0):
        """
        生成 8 个关节的目标角度。
        返回顺序：[hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4]
        """
        max_forward_speed = getattr(self.config, "max_forward_speed", 1.0)
        forward_speed = float(np.clip(forward_speed, -max_forward_speed, max_forward_speed))
        # turn_speed 由 Planner 输出，范围约为 [-max_steer_angle, max_steer_angle]。
        # 旧版本先裁剪到 [-1, 1]，再除以 max_steer_angle，会进一步削弱转向。
        turn_speed = float(np.clip(turn_speed, -self.config.max_steer_angle, self.config.max_steer_angle))

        # 当 planner 返回 0 速度和 0 转向时，进入站立保持状态。
        # 这样机器人接触远球后不会继续摆腿或缓慢移动。
        stop_eps = getattr(self.config, "stop_speed_epsilon", 0.03)
        if abs(forward_speed) < stop_eps and abs(turn_speed) < stop_eps:
            return self.neutral_qpos.copy()

        speed_ratio = min(abs(forward_speed) / max_forward_speed, 1.0)

        # 速度越大，相位推进越快；低速时仍保留最小频率，避免卡在静态姿态。
        phase_rate = self.config.gait_frequency * (0.35 + 0.65 * speed_ratio)
        self.phase = (self.phase + dt * phase_rate) % (2.0 * np.pi)

        target = self.neutral_qpos.copy()

        # 将 steer 转为 [-1, 1] 的转向强度。
        steer_gain = np.clip(turn_speed / self.config.max_steer_angle, -1.0, 1.0)
        hip_turn_bias_amp = float(getattr(self.config, "hip_turn_bias_amp", 0.0))

        for leg in range(4):
            hip_idx = leg * 2
            ankle_idx = hip_idx + 1

            p = self.phase + self.leg_phase_offsets[leg]

            # 髋关节：前后腿反向摆动，产生推进；转向时两侧幅值轻微不一致。
            turn_scale = 1.0 + self.config.turn_gait_gain * steer_gain * self.turn_side_sign[leg]
            turn_scale = float(np.clip(turn_scale, 0.55, 1.45))
            hip_swing = (
                self.config.hip_swing_amp
                * forward_speed
                * turn_scale
                * self.hip_drive_sign[leg]
                * np.sin(p)
            )
            hip_turn_bias = hip_turn_bias_amp * steer_gain * self.turn_bias_sign[leg]
            target[hip_idx] = self.neutral_qpos[hip_idx] + hip_swing + hip_turn_bias

            # 踝关节：只在摆动相增加弯曲，正限位关节向正方向弯，负限位关节向负方向弯。
            ankle_sign = 1.0 if self.neutral_qpos[ankle_idx] >= 0 else -1.0
            lift = max(0.0, np.sin(p))  # 摆动相抬腿，支撑相尽量贴地
            ankle_offset = ankle_sign * self.config.ankle_lift_amp * lift
            target[ankle_idx] = self.neutral_qpos[ankle_idx] + ankle_offset

        # 保证目标角度不越过 XML 关节限位，留一点安全边界。
        low = self.joint_ranges[:, 0] + self.config.joint_limit_margin
        high = self.joint_ranges[:, 1] - self.config.joint_limit_margin
        target = np.clip(target, low, high)
        return target

    def compute_control(self, data, dt, target_vel=1.0, target_steer=0.0):
        target_qpos = self.get_walking_signals(dt, target_vel, target_steer)
        ctrl = self.apply_pd_control(data, target_qpos)

        # 正确按 actuator 的上下界裁剪，而不是默认上下界对称。
        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        ctrl = np.clip(ctrl, ctrl_low, ctrl_high)
        return ctrl
