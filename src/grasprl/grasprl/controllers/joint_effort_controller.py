
# Import necessary modules and classes
import numpy as np


class JntEffortCtrl:
    def __init__(
        self,
        physics,
        joints,
        min_effort: np.ndarray,
        max_effort: np.ndarray,
    ) -> None:

        self._physics = physics
        self._joints = joints
        self._min_effort = min_effort
        self._max_effort = max_effort

    def run(self, target) -> None:
        # Clip the target efforts to ensure they are within the allowable effort range
        target_effort = np.clip(target, self._min_effort, self._max_effort)
        self._physics.bind(self._joints, obj_type='joint').qfrc_applied = target_effort
        self._physics.bind(self._joints).qfrc_applied = target_effort

    def reset(self) -> None:
        pass


class GripperEffortCtrl:
    def __init__(
        self,
        physics,
        gripper=None,
        actuator_id=None,
        ctrl_close=0.95,
        ctrl_open=0.0,
        effort=50.0,
        close_time=50,
    ) -> None:
        gripper,
        effort=50.0,  # 大幅增加抓取力度
        close_time=50
    ) -> None:
        effort=25.0,
        close_time=25
        ) -> None:
        self.physics = physics
        self.gripper = gripper
        self.actuator_id = actuator_id
        self.ctrl_close = ctrl_close
        self.ctrl_open = ctrl_open
        self.effort = effort
        self.close_time = close_time
        self.current_step = 0
        self.last_signal = None

    def run(self, signal):
        # 当信号改变时重置步数
        if self.last_signal is not None and self.last_signal != signal:
            self.current_step = 0
        self.last_signal = signal

        if signal == 1:
            self.close_gripper()
        else:
            self.open_gripper()

    def close_gripper(self):
        if self.actuator_id is not None:
            # 使用 actuator 控制
            self.physics.data.ctrl[self.actuator_id] = self.ctrl_close
        else:
            # 使用 joint effort 控制
            self.current_step += 1
            ramp_up_steps = 10
            if self.current_step <= ramp_up_steps:
                target_effort = self.effort * (self.current_step / ramp_up_steps)
            else:
                target_effort = self.effort
            self.physics.bind(self.gripper, obj_type='joint').qfrc_applied = target_effort

    def open_gripper(self):
        if self.actuator_id is not None:
            # 使用 actuator 控制
            self.physics.data.ctrl[self.actuator_id] = self.ctrl_open
        else:
            # 使用 joint effort 控制
            self.current_step += 1
            ramp_up_steps = 10
            max_open_effort = -self.effort * 4.0
            if self.current_step <= ramp_up_steps:
                target_effort = max_open_effort * (self.current_step / ramp_up_steps)
            else:
                target_effort = max_open_effort
            self.physics.bind(self.gripper, obj_type='joint').qfrc_applied = target_effort
        self.current_step += 1
        # 更快达到最大力度
        target_effort = min(self.effort * (self.current_step / 0.8), self.effort)
        self.physics.bind(self.gripper, obj_type='joint').qfrc_applied = target_effort

    def open_gripper(self):
        self.current_step = 0
        self.physics.bind(self.gripper, obj_type='joint').qfrc_applied = -self.effort * 2.0  # 更大的打开力度
        target_effort = min(self.effort * (self.current_step / 1.5), self.effort)
        self.physics.bind(self.gripper).qfrc_applied = target_effort

    def open_gripper(self):
        self.current_step = 0
        self.physics.bind(self.gripper).qfrc_applied = -self.effort * 1.5  # 更大的打开力度
        self.physics.bind(self.gripper).qfrc_applied = -self.effort * 1.0

    def reset(self):
        self.current_step = 0
