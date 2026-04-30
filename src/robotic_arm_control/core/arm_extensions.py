import numpy as np
import logging
import json
import os
import mujoco
import time

logger = logging.getLogger(__name__)


class PIDController:
    """PID关节控制器"""

    def __init__(self, kp=1500, ki=0.1, kd=10, joint_num=6):
        self.kp = np.array([kp] * joint_num, dtype=np.float64)
        self.ki = np.array([ki] * joint_num, dtype=np.float64)
        self.kd = np.array([kd] * joint_num, dtype=np.float64)

        self.error_sum = np.zeros(joint_num, dtype=np.float64)
        self.last_error = np.zeros(joint_num, dtype=np.float64)
        self.dt = 0.033

    def compute(self, current_joints, target_joints):
        """计算PID输出"""
        current = np.radians(np.array(current_joints))
        target = np.radians(np.array(target_joints))

        error = target - current
        self.error_sum += error * self.dt
        error_diff = (error - self.last_error) / self.dt

        # 积分饱和限制
        self.error_sum = np.clip(self.error_sum, -1.0, 1.0)

        # PID计算
        output = self.kp * error + self.ki * self.error_sum + self.kd * error_diff

        self.last_error = error
        return output.tolist()


class TrajectoryManager:
    """轨迹管理：保存/加载/规划"""

    def __init__(self, arm):
        self.arm = arm  # BaseRoboticArm实例
        self.trajectory = []
        self.trajectory_file = "trajectories/arm_trajectory.json"

        # 创建轨迹目录
        os.makedirs(os.path.dirname(self.trajectory_file), exist_ok=True)

    def generate_trajectory(self, start_joints, target_joints, num_points=100):
        """生成线性轨迹"""
        start = np.array(start_joints)
        target = np.array(target_joints)
        self.trajectory = np.linspace(start, target, num_points).tolist()

        # 裁剪每个轨迹点
        self.trajectory = [self.arm._clip_joint_angles(p) for p in self.trajectory]
        logger.info(f"生成轨迹：{num_points}个点")
        return self.trajectory

    def save_trajectory(self, filename=None):
        """保存轨迹到JSON文件"""
        if not self.trajectory:
            logger.warning("无轨迹可保存")
            return False

        filename = filename or self.trajectory_file
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joint_num": 6,
            "trajectory": self.trajectory
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        logger.info(f"轨迹已保存到：{filename}")
        return True

    def load_trajectory(self, filename=None):
        """从JSON文件加载轨迹"""
        filename = filename or self.trajectory_file
        if not os.path.exists(filename):
            logger.error(f"轨迹文件不存在：{filename}")
            return False

        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.trajectory = data["trajectory"]
        logger.info(f"加载轨迹：{len(self.trajectory)}个点，文件：{filename}")
        return True


class TargetVisualizer:
    """目标点可视化"""

    def __init__(self, sim):
        self.sim = sim  # BaseMuJoCoSim实例
        self.target_pos = [0.1, 0.0, 0.3]

        # 添加目标点几何（临时）
        self._add_target_geom()

    def _add_target_geom(self):
        """动态添加目标点几何"""
        self.target_geom = mujoco.MjvGeom()
        mujoco.mjv_initGeom(self.target_geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.02, 0.02, 0.02]), np.zeros(3), np.eye(3).flatten(), np.array([1.0, 0.0, 0.0, 1.0]))

    def update_target(self, pos):
        """更新目标点位置"""
        self.target_pos = pos

    def render(self, viewer):
        """在Viewer中渲染目标点（MuJoCo 3.x兼容）"""
        self.target_geom.pos[:] = np.array(self.target_pos)
        mujoco.mjv_addGeoms(viewer.model, viewer.data, viewer.user_scn, [self.target_geom])