import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math

# =========================================================
# 1. 坐标系精准换算
# =========================================================
# 请确保这三个坐标是你 UE 里显示的真实数值 (单位: cm)
UE_START = np.array([1180.0, 610.0, 28.0])  # 出生点
UE_GOAL = np.array([790.0, 3360.0, -50.0])  # 终点位置

# 计算 AirSim 中的相对目标向量 (单位: 米)
# 逻辑: (目标 - 起点) / 100
# 结果大约是: [-3.9, 27.5, -0.78]
TARGET_POS_AIRSIM = (UE_GOAL - UE_START) / 100.0

print(f"========================================")
print(f"环境配置信息:")
print(f"1. UE 目标坐标 (cm): {UE_GOAL}")
print(f"2. AirSim 相对目标 (m): {TARGET_POS_AIRSIM}")
print(f"3. 判定半径: 5 米 (球体)")
print(f"========================================")


class AirSimMazeEnv(gym.Env):
    def __init__(self):
        super(AirSimMazeEnv, self).__init__()

        # 连接 AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 动作空间
        # [0]: 前进速度 (0 ~ 4 m/s)
        # [1]: 转向速度 (-60 ~ 60 deg/s)
        self.action_space = spaces.Box(
            low=np.array([0, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # 观测空间 (为了兼容你之前的模型，Lidar 保持 high=50)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
            "lidar": spaces.Box(low=0, high=50, shape=(180,), dtype=np.float32)
        })

        self.last_dist = None

    def step(self, action):
        # --- 1. 执行动作 ---
        fwd_vel = float(action[0]) * 4.0
        yaw_rate = float(action[1]) * 60

        # 高度策略:
        # 目标 Z 约为 -0.78米。
        # 我们让无人机飞在 -1.5米 (离地1.5米)，这样它就在目标球体(半径5米)的内部
        # 不需要专门去对齐高度，只要飞过去就能触发
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
        reward, done = self._compute_reward_and_done(obs)

        truncated = False
        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 起飞并悬停
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-1.5, 1).join()

        # 初始化距离 (3D距离)
        state = self.client.getMultirotorState().kinematics_estimated.position
        curr_pos = np.array([state.x_val, state.y_val, state.z_val])
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
            if response.width > 0 and response.height > 0:
                try:
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    if img1d.size == response.width * response.height:
                        img1d = np.clip(img1d, 0, 20)
                        img2d = img1d.reshape(response.height, response.width)
                        img_resize = cv2.resize(img2d, (84, 84))
                        img_uint8 = (img_resize / 20.0 * 255).astype(np.uint8)
                        img_obs = np.expand_dims(img_uint8, axis=-1)
                except Exception:
                    pass

        # === Lidar 处理 ===
        lidar_scan = np.ones(180) * 20.0
        try:
            lidar_data = self.client.getLidarData("lidar_1")
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            if len(points) > 3:
                points = np.reshape(points, (-1, 3))
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
        except Exception:
            pass

        return {"image": img_obs, "lidar": lidar_scan}

    def _compute_reward_and_done(self, obs):
        # 1. 获取当前 3D 位置
        collision_info = self.client.simGetCollisionInfo()
        state = self.client.getMultirotorState().kinematics_estimated.position
        curr_pos = np.array([state.x_val, state.y_val, state.z_val])

        # 2. 计算到【正方体】的 3D 距离
        dist_to_goal = np.linalg.norm(curr_pos - TARGET_POS_AIRSIM)

        # 计算到【出生点】的距离 (用于判断是否飞太远)
        dist_from_start = np.linalg.norm(curr_pos)
        # 计算正方体到出生点的理论距离
        goal_distance_from_start = np.linalg.norm(TARGET_POS_AIRSIM)

        reward = 0
        done = False

        # --- A. 撞墙判定 ---
        if collision_info.has_collided:
            reward = -50.0
            done = True
            print(f"❌ 撞墙!")
            return reward, done

        # --- B. 成功判定: 进入正方体 5米 范围内 ---
        if dist_to_goal < 5.0:
            reward = 100.0  # 大奖
            done = True  # 结束回合 -> 触发 reset -> 回到出生点
            print(f"✅ 成功到达终点范围! (误差: {dist_to_goal:.2f}m)")
            return reward, done

        # --- C. 防止乱飞 (Geofence) ---
        # 逻辑: 如果无人机飞行的距离，比目标距离还要远 15 米，说明它已经飞过头并开始乱跑了
        # 强制结束，让它回出生点重来
        if dist_from_start > (goal_distance_from_start + 15.0):
            reward = -20.0
            done = True
            print(f"⚠️ 飞出边界 (距离起点 {dist_from_start:.1f}m)，强制重置")
            return reward, done

        # 补充: 如果高度异常 (Z > 5米, 钻地; Z < -10米, 飞太高)
        if curr_pos[2] > 5 or curr_pos[2] < -10:
            reward = -20.0
            done = True
            print("⚠️ 高度异常")
            return reward, done

        # --- D. 引导奖励 ---
        # 只要离正方体近了就给分
        if self.last_dist is not None:
            improvement = self.last_dist - dist_to_goal
            reward += improvement * 10.0

        self.last_dist = dist_to_goal

        # --- E. 避障惩罚 (防止死路) ---
        min_obstacle_dist = np.min(obs['lidar'])
        if min_obstacle_dist < 1.5:
            penalty = (1.5 - min_obstacle_dist) * 0.5
            reward -= penalty

        # --- F. 时间惩罚 ---
        reward -= 0.05

        return reward, done

    def close(self):
        self.client.enableApiControl(False)
