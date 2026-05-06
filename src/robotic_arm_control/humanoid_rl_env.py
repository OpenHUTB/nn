"""
人形机器人强化学习环境 (优化版)
使用力矩控制，适合站立训练
"""
import os
import numpy as np
import mujoco
from gymnasium import Env, spaces
from gymnasium.utils import seeding


class HumanoidRLEnv(Env):
    """人形机器人强化学习环境"""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path=None, render_mode=None):
        super().__init__()

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model/humanoid_rl.xml")

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        # 关节名称 (11个关节)
        self.joint_names = [
            "neck",
            "l_shoulder", "l_elbow",
            "r_shoulder", "r_elbow",
            "l_hip", "l_knee", "l_ankle",
            "r_hip", "r_knee", "r_ankle"
        ]

        # 获取关节ID
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]

        # 状态维度
        self.n_joints = len(self.joint_names)
        # qpos: 根节点(7) + 关节(n_joints)
        # qvel: 根节点(6) + 关节(n_joints)
        obs_dim = 7 + self.n_joints + 6 + self.n_joints

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # 动作空间: 力矩控制
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float64
        )

        # 站立目标
        self.target_height = 1.2
        self.target_pos = np.array([0.0, 0.0, 1.2])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        for jid in self.joint_ids:
            self.data.joint(jid).qpos[0] = 0.0
            self.data.joint(jid).qvel[0] = 0.0

        self.data.qpos[0:3] = np.array([0, 0, 1.2])
        self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        self.data.qvel[0:6] = np.zeros(6)
        self.data.ctrl[:] = np.zeros(self.n_joints)

        mujoco.mj_forward(self.model, self.data)
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # ====================== 修复：力矩从 50 → 10 ======================
        torque = action * 10
        for i in range(self.n_joints):
            self.data.ctrl[i] = np.clip(torque[i], -50, 50)

        mujoco.mj_step(self.model, self.data)
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False

        return observation, reward, terminated, truncated, {}

    def _get_obs(self):
        root_qpos = self.data.qpos[:7].copy()
        root_qvel = self.data.qvel[:6].copy()
        joint_pos = np.array([self.data.joint(jid).qpos[0] for jid in self.joint_ids])
        joint_vel = np.array([self.data.joint(jid).qvel[0] for jid in self.joint_ids])
        return np.concatenate([root_qpos, joint_pos, root_qvel, joint_vel])

    # ====================== 修复：奖励函数全部缩小 ======================
    def _compute_reward(self):
        reward = 0.0

        # 1. 存活奖励（很小）
        reward += 0.1

        # 2. 高度奖励（超级安全，不会爆炸）
        torso_pos = self.data.body("torso").xpos[2]
        reward += 0.2 * torso_pos  # 只看高度，越高越好

        # 3. 直立奖励
        torso_quat = self.data.body("torso").xquat
        reward += 0.1 * torso_quat[0]

        return reward

    def _is_terminated(self):
        torso_pos = self.data.body("torso").xpos[2]
        if torso_pos < 0.5:
            return True

        torso_quat = self.data.body("torso").xquat
        if torso_quat[0] < 0.2:
            return True

        x_pos = self.data.qpos[0]
        y_pos = self.data.qpos[1]
        if x_pos**2 + y_pos**2 > 9:
            return True

        return False

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    import mujoco.viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except:
                    print("[文本模式]")
                    self.viewer = "text"
                    return
            if self.viewer != "text":
                self.viewer.sync()

    def close(self):
        if self.viewer and self.viewer != "text":
            self.viewer.close()
        self.viewer = None