import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math
import time

# =========================================================
# 1. 坐标系精准换算 (UE -> AirSim)
# =========================================================
UE_START = np.array([1180.0, 610.0, 28.0])  # 出生点
UE_GOAL = np.array([790.0, 3360.0, -50.0])  # 正方体位置

# 计算相对向量 (单位: 米)
TARGET_POS_AIRSIM = (UE_GOAL - UE_START) / 100.0

print(f"========================================")
print(f"🚀 重启训练: 全新配置")
print(f"1. 目标相对坐标: {TARGET_POS_AIRSIM}")
print(f"2. 速度限制: 5.0 m/s (已加速)")
print(f"3. 判定半径: 5 米")
print(f"4. 防转圈机制: 已启用 Lidar 地面过滤")
print(f"========================================")


class AirSimMazeEnv(gym.Env):
    def __init__(self):
        super(AirSimMazeEnv, self).__init__()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 动作空间: [前进速度(0~1), 转向速度(-1~1)]
        self.action_space = spaces.Box(
            low=np.array([0, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # 观测空间 (Lidar 50m)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
            "lidar": spaces.Box(low=0, high=50, shape=(180,), dtype=np.float32)
        })

        self.last_dist = None
        self.prev_action = np.zeros(2)

    def step(self, action):
        # --- 1. 执行动作 (提速版) ---
        # 之前可能设低了，现在强制设为 5.0 m/s，保证飞得快
        fwd_vel = float(action[0]) * 5.0
        yaw_rate = float(action[1]) * 60

        # 锁定高度 Z=-1.5
        self.client.moveByVelocityZBodyFrameAsync(
            vx=fwd_vel,
            vy=0,
            z=-1.5,
            duration=0.1,
            yaw_mode=airsim.YawMode(True, yaw_rate)
        ).join()

        # --- 2. 获取观测 ---
        obs = self._get_obs()

        # --- 3. 计算奖励 ---
        reward, done = self._compute_reward_and_done(obs, action)

        truncated = False
        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 瞬移重置 (极速)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        pose = airsim.Pose(airsim.Vector3r(0, 0, -1.5), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose, True)

        self.prev_action = np.zeros(2)
        curr_pos = np.array([0.0, 0.0, -1.5])
        self.last_dist = np.linalg.norm(curr_pos - TARGET_POS_AIRSIM)

        return self._get_obs(), {}

    def _get_obs(self):
        # === 图像处理 ===
        img_obs = np.zeros((84, 84, 1), dtype=np.uint8)
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPlanar, True)
        ])
        if responses:
            response = responses[0]
            if response.width > 0:
                try:
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    img1d = np.clip(img1d, 0, 20)
                    img2d = img1d.reshape(response.height, response.width)
                    img_resize = cv2.resize(img2d, (84, 84))
                    img_uint8 = (img_resize / 20.0 * 255).astype(np.uint8)
                    img_obs = np.expand_dims(img_uint8, axis=-1)
                except:
                    pass

        # === Lidar 处理 (关键修复：地面过滤) ===
        lidar_scan = np.ones(180) * 20.0
        try:
            lidar_data = self.client.getLidarData("lidar_1")
            points = np.array(lidar_data.point_cloud, dtype=np.float32)

            if len(points) > 3:
                points = np.reshape(points, (-1, 3))

                # 【防止转圈的核心】
                # 你的 settings.json 是 -10 度，如果不加这个，它一定会把地板当墙
                # 我们只保留高度在 -1.0 到 0.5 之间的点 (水平视线附近的障碍物)
                z_mask = (points[:, 2] > -1.0) & (points[:, 2] < 0.5)
                points = points[z_mask]

                if len(points) > 0:
                    x = points[:, 0]
                    y = points[:, 1]
                    angles = np.arctan2(y, x) * 180 / np.pi
                    dists = np.linalg.norm(points[:, :2], axis=1)

                    valid_mask = (angles >= -90) & (angles < 90)
                    valid_angles = angles[valid_mask]
                    valid_dists = dists[valid_mask]

                    indices = ((valid_angles + 90).astype(int))
                    indices = np.clip(indices, 0, 179)
                    for i, d in zip(indices, valid_dists):
                        if d < lidar_scan[i]:
                            lidar_scan[i] = d
        except:
            pass

        return {"image": img_obs, "lidar": lidar_scan}

    def _compute_reward_and_done(self, obs, action):
        collision = self.client.simGetCollisionInfo().has_collided
        state = self.client.getMultirotorState().kinematics_estimated.position
        curr_pos = np.array([state.x_val, state.y_val, state.z_val])

        dist_to_goal = np.linalg.norm(curr_pos - TARGET_POS_AIRSIM)
        dist_from_start = np.linalg.norm(curr_pos)

        reward = 0
        done = False

        # 1. 撞墙
        if collision:
            reward = -50.0
            done = True
            print(f"❌ 撞墙!")
            return reward, done

        # 2. 成功 (5米内)
        if dist_to_goal < 5.0:
            reward = 100.0
            done = True
            print(f"✅ 任务完成! (距离: {dist_to_goal:.2f}m)")
            return reward, done

        # 3. 越界保护
        limit_dist = np.linalg.norm(TARGET_POS_AIRSIM) + 20.0
        if dist_from_start > limit_dist:
            reward = -20.0
            done = True
            print(f"⚠️ 飞出边界，重置")
            return reward, done

        # 4. 引导奖励
        if self.last_dist is not None:
            reward += (self.last_dist - dist_to_goal) * 10.0
        self.last_dist = dist_to_goal

        # 5. 避障惩罚 (防止死路)
        min_obs_dist = np.min(obs['lidar'])
        if min_obs_dist < 1.5:
            reward -= (1.5 - min_obs_dist) * 0.5

        # 6. 动作平滑 (防止抖动)
        reward -= np.linalg.norm(action - self.prev_action) * 0.1
        self.prev_action = action.copy()

        # 7. 步数惩罚
        reward -= 0.05

        return reward, done

    def close(self):
        self.client.enableApiControl(False)