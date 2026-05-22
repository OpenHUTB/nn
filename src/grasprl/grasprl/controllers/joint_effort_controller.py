
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
        self._physics.bind(self._joints).qfrc_applied = target_effort

    def reset(self) -> None:
        pass


class GripperEffortCtrl:
    def __init__(
        self,
        physics,
        gripper,
        effort=25.0,
        close_time=25
        ) -> None:
        self.physics = physics
        self.gripper = gripper
        self.effort = effort
        self.close_time = close_time
        self.current_step = 0

    def run(self, signal):
        if signal == 1:
            self.close_gripper()
        else:
            self.open_gripper()

    def close_gripper(self):
        self.current_step += 1
        target_effort = min(self.effort * (self.current_step / 1.5), self.effort)
        self.physics.bind(self.gripper).qfrc_applied = target_effort

    def open_gripper(self):
        self.current_step = 0
        self.physics.bind(self.gripper).qfrc_applied = -self.effort * 1.0

    def reset(self):
        self.current_step = 0
