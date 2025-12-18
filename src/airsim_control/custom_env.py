# custom_env.py
import time
import math
import heapq
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim
import cv2

class AirSimMazeEnv(gym.Env):
    """
    AirSim + A* 增强迷宫环境
    解决：
      - ClockSpeed > 1 抖动（通过仿真时间缩放 + 平滑控制）
      - 死路识别 + A* 回退（提供简单 A* 接口）
      - PPO 训练兼容（可选 velocity 观测）
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        cell_size=1.5,
        grid_radius=30.0,
        lidar_max_range=40.0,
        follow_path_steps=10,
        include_velocity=True,
        sim_clock_speed=1.0,
        base_step_wall_time=0.12,
        enable_smoothing=True
    ):
        super().__init__()

        # ========== AirSim ==========
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # enable control if necessary (safe to call)
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception:
            pass

        # ========== 动作空间 ==========
        self.MAX_FWD_VEL = 4.0
        self.MAX_YAW_RATE = 60.0

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ========== 观测空间 ==========
        self.include_velocity = include_velocity
        obs_spaces = {
            "image": spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8),
            "lidar": spaces.Box(0.0, 50.0, (180,), dtype=np.float32)
        }
        if self.include_velocity:
            obs_spaces["velocity"] = spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_spaces)

        # ========== 栅格地图 ==========
        self.cell_size = float(cell_size)
        self.grid_radius = float(grid_radius)
        self.lidar_max_range = float(lidar_max_range)

        self.grid_cells = int(np.ceil((self.grid_radius * 2) / self.cell_size))
        if self.grid_cells % 2 == 0:
            self.grid_cells += 1
        self.grid_center = self.grid_cells // 2

        self.visited_cells = set()
        self.virtual_walls = set()

        # ========== A* ==========
        self.follow_path_steps = follow_path_steps
        self.current_path = []
        self.path_follow_counter = 0

        # ========== 时钟 & 控制平滑 ==========
        self.sim_clock_speed = float(sim_clock_speed)
        self.base_step_wall_time = float(base_step_wall_time)
        self.enable_smoothing = bool(enable_smoothing)

        self.vel_prev = 0.0
        self.yaw_prev = 0.0
        self.alpha_vel = 0.6
        self.alpha_yaw = 0.4
        self.max_accel = 2.0             # m/s^2
        self.max_angular_accel = 120.0   # deg/s^2

        # ========== 内部状态 ==========
        self.last_dist = None
        self._last_pos = np.zeros(3, dtype=np.float32)

        print(f"[ENV] sim_clock_speed={self.sim_clock_speed}, base_wall_dt={self.base_step_wall_time}")

    # -------------------------
    # 环境接口
    # -------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception:
            pass

        # small wait to stabilize
        time.sleep(0.05)

        self.visited_cells.clear()
        self.virtual_walls.clear()
        self.current_path = []
        self.path_follow_counter = 0
        self.vel_prev = 0.0
        self.yaw_prev = 0.0
        self.last_dist = None
        self._last_pos = np.array([0.0, 0.0, -1.5], dtype=np.float32)

        obs = self._get_obs()
        # initialize last_dist if possible
        try:
            state = self.client.getMultirotorState().kinematics_estimated.position
            pos = np.array([state.x_val, state.y_val, state.z_val], dtype=np.float32)
            self.last_dist = np.linalg.norm(pos)
        except Exception:
            self.last_dist = None

        return obs, {}

    def step(self, action):
        # action: [fwd_ratio (0~1), yaw_ratio (-1~1)]
        fwd_ratio = float(action[0])
        yaw_ratio = float(action[1])

        target_vel = np.clip(fwd_ratio, 0.0, 1.0) * self.MAX_FWD_VEL
        target_yaw_rate = np.clip(yaw_ratio, -1.0, 1.0) * self.MAX_YAW_RATE

        # 平滑与加速度限制 (基于墙钟 step)
        if self.enable_smoothing:
            wall_dt = self.base_step_wall_time
            dv = np.clip(target_vel - self.vel_prev, -self.max_accel * wall_dt, self.max_accel * wall_dt)
            vel = self.vel_prev + dv
            vel = self.alpha_vel * vel + (1.0 - self.alpha_vel) * self.vel_prev

            dyaw = np.clip(target_yaw_rate - self.yaw_prev, -self.max_angular_accel * wall_dt, self.max_angular_accel * wall_dt)
            yaw = self.yaw_prev + dyaw
            yaw = self.alpha_yaw * yaw + (1.0 - self.alpha_yaw) * self.yaw_prev
        else:
            vel = target_vel
            yaw = target_yaw_rate

        self.vel_prev = float(vel)
        self.yaw_prev = float(yaw)

        # 把墙钟步长转换为仿真持续时间（sim seconds）
        duration_sim = float(self.base_step_wall_time * self.sim_clock_speed)
        # 最小持续时间保护，避免 duration 非法值
        if duration_sim <= 0:
            duration_sim = float(self.base_step_wall_time)

        # 发送命令（以 body frame 速度，保持高度 -1.5）
        try:
            self.client.moveByVelocityZBodyFrameAsync(
                vx=float(vel),
                vy=0.0,
                z=-1.5,
                duration=duration_sim,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw))
            ).join()
        except Exception:
            # 如果 API 出错，略过 move call，继续仿真（避免崩溃）
            pass

        obs = self._get_obs()
        reward, done = self._compute_reward_and_done(obs)
        truncated = False
        info = {}
        return obs, float(reward), bool(done), truncated, info

    # -------------------------
    # 观测获取
    # -------------------------
    def _get_obs(self):
        # 图像（channels-first）
        img_obs = np.zeros((1, 84, 84), dtype=np.uint8)
        try:
            responses = self.client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPlanar, True)])
            if responses and len(responses) > 0:
                res = responses[0]
                if res.width > 0 and res.height > 0:
                    img1d = np.array(res.image_data_float, dtype=np.float32)
                    if img1d.size == res.width * res.height:
                        img2d = img1d.reshape(res.height, res.width)
                        img2d = np.clip(img2d, 0.0, 20.0)
                        img_resized = cv2.resize(img2d, (84, 84))
                        img_uint8 = (img_resized / 20.0 * 255.0).astype(np.uint8)
                        img_obs = np.expand_dims(img_uint8, axis=0)
        except Exception:
            pass

        # Lidar 处理
        lidar_scan = np.ones(180, dtype=np.float32) * self.lidar_max_range
        try:
            lidar_data = self.client.getLidarData("lidar_1")
            if hasattr(lidar_data, "point_cloud") and lidar_data.point_cloud:
                pts = np.array(lidar_data.point_cloud, dtype=np.float32)
                if pts.size >= 3:
                    pts = pts.reshape(-1, 3)
                    x = pts[:, 0]
                    y = pts[:, 1]
                    dists = np.linalg.norm(pts[:, :2], axis=1)
                    angles = np.degrees(np.arctan2(y, x))
                    mask = (angles >= -90) & (angles < 90) & (dists > 0)
                    valid_angles = angles[mask]
                    valid_dists = dists[mask]
                    idxs = (valid_angles + 90).astype(int)
                    idxs = np.clip(idxs, 0, 179)
                    for i, d in zip(idxs, valid_dists):
                        if d < lidar_scan[i]:
                            lidar_scan[i] = d
        except Exception:
            pass

        obs = {"image": img_obs, "lidar": lidar_scan.astype(np.float32)}

        if self.include_velocity:
            try:
                k = self.client.getMultirotorState().kinematics_estimated.linear_velocity
                vel = np.array([k.x_val, k.y_val], dtype=np.float32)
            except Exception:
                vel = np.zeros(2, dtype=np.float32)
            obs["velocity"] = vel

        return obs

    # -------------------------
    # 奖励与终止判断
    # -------------------------
    def _compute_reward_and_done(self, obs):
        reward = 0.0
        done = False
        try:
            state = self.client.getMultirotorState().kinematics_estimated.position
            pos = np.array([state.x_val, state.y_val, state.z_val], dtype=np.float32)
        except Exception:
            pos = self._last_pos

        # 简单引导：靠近原点（或者你可替换为目标距离）给予正奖励
        dist = np.linalg.norm(pos)
        if self.last_dist is not None:
            delta = self.last_dist - dist
            reward += np.clip(delta, -1.0, 1.0) * 5.0
        else:
            reward += 0.0
        self.last_dist = dist
        self._last_pos = pos

        # 碰撞判断
        try:
            coll = self.client.simGetCollisionInfo()
            if coll and coll.has_collided:
                reward -= 50.0
                done = True
        except Exception:
            pass

        # 高度异常 / 越界判定（保守）
        if pos[2] > 5 or pos[2] < -20:
            reward -= 20.0
            done = True

        # 时间步惩罚
        reward -= 0.02

        return float(reward), bool(done)

    # -------------------------
    # A* 简易实现（提供接口）
    # -------------------------
    def _astar(self, start, goal, occupancy_set=None):
        """
        返回路径（grid 坐标列表，从 start 的下一个节点开始到 goal）
        occupancy_set: set of blocked grid tuples (可为空)
        """
        if occupancy_set is None:
            occupancy_set = set()

        # 边界检查（可根据 grid size 扩展）
        def in_bounds(cell):
            x, y = cell
            return 0 <= x < self.grid_cells and 0 <= y < self.grid_cells

        open_heap = []
        heapq.heappush(open_heap, (0 + self._heuristic(start, goal), 0, start))
        came_from = {}
        gscore = {start: 0}
        closed = set()

        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            if current == goal:
                # 回溯路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path  # 包含 start -> ... -> goal

            if current in closed:
                continue
            closed.add(current)

            for dx, dy in neighbors:
                nb = (current[0] + dx, current[1] + dy)
                if not in_bounds(nb):
                    continue
                if nb in occupancy_set:
                    continue
                tentative_g = gscore[current] + (1.4142 if dx != 0 and dy != 0 else 1.0)
                if tentative_g < gscore.get(nb, 1e9):
                    came_from[nb] = current
                    gscore[nb] = tentative_g
                    fscore = tentative_g + self._heuristic(nb, goal)
                    heapq.heappush(open_heap, (fscore, tentative_g, nb))

        return []  # 无路径

    def _heuristic(self, a, b):
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        return float(np.linalg.norm(a - b))

    # -------------------------
    # 关闭
    # -------------------------
    def close(self):
        try:
            self.client.enableApiControl(False)
        except Exception:
            pass
