"""用于无人机路径规划的 AirSim Gym 环境包装器"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
import time

from ..utils.airsim_utils import AirSimConnector, process_depth_image
from ..utils.reward_shaper import SimpleRewardShaper

logger = logging.getLogger(__name__)


class AirSimDroneEnv(gym.Env):
    """
    AirSim 多旋翼无人机的 OpenAI Gym 环境包装器。

    观测值：深度摄像机图像（84x84 灰度图）
    动作：6 个离散方向 + 悬停（0-6）
    奖励：距离惩罚 + 碰撞惩罚 + 成功奖励
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        ip_address: str = "127.0.0.1",
        image_shape: Tuple[int, int] = (84, 84),
        max_steps: int = 500,
        target_points: Optional[list] = None,
        action_duration: float = 0.5,
        velocity_step: float = 5.0,
        verbose: bool = False,
    ):
        """
        初始化 AirSim 无人机环境。

        参数：
            ip_address: AirSim 模拟器的 IP 地址
            image_shape: 目标图像大小（高, 宽）
            max_steps: 每个回合的最大步数
            target_points: 目标航点列表（3D 数组）
            action_duration: 每个运动的持续时间（秒）
            velocity_step: 运动速度大小（m/s）
            verbose: 启用日志记录
        """
        super().__init__()

        # Config
        self.image_shape = image_shape
        self.max_steps = max_steps
        self.action_duration = action_duration
        self.velocity_step = velocity_step
        self.verbose = verbose

        # AirSim 连接器
        self.airsim = AirSimConnector(ip_address=ip_address)

        # 奖励定形器
        default_targets = [
            np.array([0.0, 0.0, -10.0]),
            np.array([50.0, 0.0, -10.0]),
            np.array([50.0, 50.0, -10.0]),
        ]
        self.target_points = target_points or default_targets
        self.reward_shaper = SimpleRewardShaper(target_points=self.target_points)

        # 定义动作空间：7 个离散动作
        # 0: 向 X+ 运动，1: 向 Y+ 运动，2: 向 Z+ 运动
        # 3: 向 X- 运动，4: 向 Y- 运动，5: 向 Z- 运动
        # 6: 悬停（无运动）
        self.action_space = spaces.Discrete(7)

        # 定义观测空间：深度图像
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*image_shape, 1), dtype=np.uint8
        )

        # 状态跟踪
        self.state = {
            "position": np.zeros(3, dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
            "collision": False,
        }

        self.step_count = 0
        self.episode_reward = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        将环境重置为初始状态。

        参数：
            seed: 随机种子（用于可重复性）
            options: 附加选项（未使用）

        返回值：
            obs: 初始观测值
            info: 附加信息
        """
        super().reset(seed=seed)

        # 如果尚未连接到 AirSim，则连接
        if not self.airsim.is_connected():
            if self.verbose:
                logger.info("正在连接到 AirSim...")
            if not self.airsim.connect():
                raise RuntimeError("连接 AirSim 失败")

        # 重置模拟器
        if self.verbose:
            logger.info("正在重置 AirSim 环境...")

        try:
            self.airsim.reset()
            time.sleep(0.5)

            # 启用 API 控制并上锁
            self.airsim.enable_api_control(True)
            time.sleep(0.1)
            self.airsim.arm()
            time.sleep(0.1)

            # 起飞
            self.airsim.takeoff()
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"重置失败：{e}")
            raise RuntimeError(f"环境重置失败：{e}")

        # 重置内部状态
        self.step_count = 0
        self.episode_reward = 0.0
        self.reward_shaper.reset_targets()

        # 获取初始观测值
        obs = self._get_obs()
        info = self._get_info()

        if self.verbose:
            logger.info(f"环境已重置。初始状态：{self.state}")

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        在环境中执行一步。

        参数：
            action: 离散动作（0-6）

        返回值：
            obs: 动作后的观测值
            reward: 奖励信号
            terminated: 回合结束标志（达到目标）
            truncated: 截断标志（达到最大步数）
            info: 附加信息
        """
        self.step_count += 1

        # 执行动作
        vel_offset = self._interpret_action(action)
        success = self.airsim.move_by_velocity(
            vel_offset[0], vel_offset[1], vel_offset[2], duration=self.action_duration
        )

        if not success:
            logger.warning(f"第 {self.step_count} 步执行动作失败")

        time.sleep(0.05)  # 小延迟以确保状态更新

        # 获取新状态
        obs = self._get_obs()

        # 计算奖励
        current_pos = self.state["position"]
        current_target = self.reward_shaper.get_current_target()
        dist_to_target = np.linalg.norm(current_pos - current_target)
        reached_target = dist_to_target < 2.0

        reward, done = self.reward_shaper.compute_reward(
            position=current_pos,
            collision=self.state["collision"],
            reached_target=reached_target,
            info=None,
        )

        self.episode_reward += reward

        # 检查终止条件
        terminated = done  # 碰撞或达到目标
        truncated = self.step_count >= self.max_steps  # 达到最大步数

        info = self._get_info()
        info.update(
            {
                "step": self.step_count,
                "position": current_pos,
                "collision": self.state["collision"],
                "distance_to_target": dist_to_target,
            }
        )

        if self.verbose and (terminated or truncated or self.step_count % 50 == 0):
            logger.debug(
                f"第 {self.step_count} 步：奖励={reward:.2f}，"
                f"位置={current_pos}，碰撞={self.state['collision']}"
            )

        return obs, float(reward), terminated, truncated, info

    def _interpret_action(self, action: int) -> Tuple[float, float, float]:
        """
        将离散动作转换为速度偏移。

        参数：
            action: 动作索引（0-6）

        返回值：
            vel_offset: (vx, vy, vz) 速度分量
        """
        if action == 0:  # 向 X+ 运动（前进）
            return (self.velocity_step, 0.0, 0.0)
        elif action == 1:  # 向 Y+ 运动（右）
            return (0.0, self.velocity_step, 0.0)
        elif action == 2:  # 向 Z+ 运动（上升）
            return (0.0, 0.0, self.velocity_step)
        elif action == 3:  # 向 X- 运动（后退）
            return (-self.velocity_step, 0.0, 0.0)
        elif action == 4:  # 向 Y- 运动（左）
            return (0.0, -self.velocity_step, 0.0)
        elif action == 5:  # 向 Z- 运动（下降）
            return (0.0, 0.0, -self.velocity_step)
        else:  # 动作 6：悬停
            return (0.0, 0.0, 0.0)

    def _get_obs(self) -> np.ndarray:
        """
        获取当前观测值（深度图像）。

        返回值：
            观测值：(84, 84, 1) 灰度图
        """
        # 从 AirSim 更新无人机状态
        state = self.airsim.get_state()
        if state is not None:
            self.state = state

        # 获取深度图像
        raw_image = self.airsim.get_images(image_type="depth")

        if raw_image is None:
            # 失败时返回空白图像
            logger.warning("深度图像获取失败，返回空观测值")
            obs = np.zeros((*self.image_shape, 1), dtype=np.uint8)
        else:
            # 处理深度图像
            obs = process_depth_image(raw_image, target_size=self.image_shape)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """返回附加信息"""
        return {
            "episode_reward": self.episode_reward,
            "position": self.state["position"].copy(),
            "collision": self.state["collision"],
        }

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """渲染环境（可选）"""
        # 可以渲染 AirSim 场景或轨迹可视化
        # 目前只返回 None
        return None

    def close(self):
        """清理资源"""
        try:
            if self.airsim.is_connected():
                self.airsim.land()
                time.sleep(0.5)
                self.airsim.disconnect()
                logger.info("环境关闭成功")
        except Exception as e:
            logger.error(f"关闭环境时出错：{e}")

    def __del__(self):
        """析构函数以确保清理"""
        self.close()
