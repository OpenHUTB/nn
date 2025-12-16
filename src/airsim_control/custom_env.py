import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math

# 目标出口位置 (注意：这里要填换算成“米”后的单位)
# UE坐标: X=790cm, Y=3360cm -> AirSim: X=7.9m, Y=33.6m
EXIT_POS = np.array([7.9, 33.6])

# 允许的误差半径 (到达终点附近 2 米就算成功)
GOAL_THRESHOLD = 2.0


class AirSimMazeEnv(gym.Env):
    def __init__(self):
        super(AirSimMazeEnv, self).__init__()

        # 连接 AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 动作空间: [前进速度(0~1), 转向速度(-1~1)]
        # 我们不控制后退，只控制前进和左右转，这样更容易走出迷宫
        self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

        # 观测空间 (多模态):
        # 1. image: 84x84 的深度图
        # 2. lidar: 180 个维度的距离向量 (模拟前方180度扇区)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
            "lidar": spaces.Box(low=0, high=50, shape=(180,), dtype=np.float32)
        })

        self.state = None
        self.last_dist = None

    def step(self, action):
        # 1. 执行动作
        # 将动作映射到实际物理参数
        fwd_vel = float(action[0]) * 3.0  # 最大速度 3 m/s
        yaw_rate = float(action[1]) * 45  # 最大转向角速度 45度/s



        # 新的代码 (锁定高度在 -1.5 米):
        # 注意: NED坐标系中，负数是空中。-1.5 表示离地 1.5 米。
        self.client.moveByVelocityZBodyFrameAsync(
            vx=fwd_vel,
            vy=0,
            z=-1.5,  # <--- 强制固定在这个高度
            duration=0.1,
            yaw_mode=airsim.YawMode(True, yaw_rate)
        ).join()
        # ------------------- 修改结束 -------------------

        # 2. 获取观测
        obs = self._get_obs()

        # 3. 计算奖励和结束条件
        reward, done = self._compute_reward_and_done()

        truncated = False  # 对于 Gym 新版本
        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # 起飞到固定高度
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-1.5, 1).join()  # 保持在 1.5米高度

        # 初始化距离用于计算奖励
        pos = self.client.getMultirotorState().kinematics_estimated.position
        self.last_dist = np.linalg.norm(np.array([pos.x_val, pos.y_val]) - EXIT_POS)

        return self._get_obs(), {}

    def _get_obs(self):
        # --- 获取摄像头数据 (深度图) ---
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPlanar, True)
        ])

        if responses:
            img1d = np.array(responses[0].image_data_float, dtype=np.float32)
            img1d = np.clip(img1d, 0, 20)  # 限制最大深度为20米，归一化效果更好
            img2d = img1d.reshape(responses[0].height, responses[0].width)
            # 缩放到 84x84
            img_resize = cv2.resize(img2d, (84, 84))
            # 转换为 0-255 uint8 供 CNN 使用
            img_uint8 = (img_resize / 20.0 * 255).astype(np.uint8)
            img_obs = np.expand_dims(img_uint8, axis=-1)
        else:
            img_obs = np.zeros((84, 84, 1), dtype=np.uint8)

        # --- 获取 Lidar 数据 (并处理为 1D 向量) ---
        lidar_data = self.client.getLidarData("lidar_1")
        points = np.array(lidar_data.point_cloud, dtype=np.float32)

        # 初始化 180 个 bins (代表前方 -90度 到 +90度)
        lidar_scan = np.ones(180) * 20.0  # 默认最远距离 20米

        if len(points) > 3:
            points = np.reshape(points, (-1, 3))
            # 我们只关心局部坐标系下的 X(前) 和 Y(右)
            x = points[:, 0]
            y = points[:, 1]

            # 转换为极坐标: 角度和距离
            angles = np.arctan2(y, x) * 180 / np.pi  # -180 到 180
            dists = np.linalg.norm(points[:, :2], axis=1)

            # 筛选前方 -90 到 90 度的点
            valid_mask = (angles >= -90) & (angles < 90)
            valid_angles = angles[valid_mask]
            valid_dists = dists[valid_mask]

            # 将角度映射到 0-179 的索引
            indices = ((valid_angles + 90).astype(int))
            indices = np.clip(indices, 0, 179)

            # 更新每个角度扇区的最小距离 (避障需要知道最近的障碍物)
            for i, d in zip(indices, valid_dists):
                if d < lidar_scan[i]:
                    lidar_scan[i] = d

        return {"image": img_obs, "lidar": lidar_scan}

    def _compute_reward_and_done(self):
        # 获取状态
        collision = self.client.simGetCollisionInfo().has_collided
        pos = self.client.getMultirotorState().kinematics_estimated.position
        curr_dist = np.linalg.norm(np.array([pos.x_val, pos.y_val]) - EXIT_POS)

        reward = 0
        done = False

        # 1. 碰撞惩罚 (大惩罚并结束)
        if collision:
            reward = -50.0
            done = True
            return reward, done

        # 2. 到达终点奖励
        if curr_dist < GOAL_THRESHOLD:
            reward = 100.0
            done = True
            print(">>> REACHED EXIT! <<<")
            return reward, done

        # 3. 引导奖励 (Progress Reward)
        # 如果比上一步离终点近，给正奖励；远了给负奖励
        dist_improvement = self.last_dist - curr_dist
        reward += dist_improvement * 10.0
        self.last_dist = curr_dist

        # 4. 时间步惩罚 (迫使它快速移动)
        reward -= 0.05

        return reward, done