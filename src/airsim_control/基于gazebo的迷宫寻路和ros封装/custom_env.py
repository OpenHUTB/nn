import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from mavros_msgs.srv import CommandBool, SetMode
import math
import time
import threading

# ==========================================
# 🎯 迷宫目标配置
# ==========================================
TARGET_POS = np.array([10.0, 2.0, 2.0])
SUCCESS_DIST = 1.0


class AirSimMazeEnv(gym.Env):
    def __init__(self):
        super(AirSimMazeEnv, self).__init__()

        # --- 1. ROS 2 节点初始化 ---
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('gym_env_node')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

        # 发布者: 控制速度
        self.vel_pub = self.node.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # 订阅者: 雷达和位置
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.node.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)

        # === 关键新增：服务客户端 (用于解锁和切模式) ===
        self.arming_client = self.node.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.node.create_client(SetMode, '/mavros/cmd/set_mode')

        # 启动后台线程
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.latest_scan = None
        self.current_pose = None

        self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=30, shape=(180,), dtype=np.float32)

    # === 回调函数 ===
    def scan_callback(self, msg):
        raw = np.array(msg.ranges)
        raw[raw == float('inf')] = 30.0
        raw = np.nan_to_num(raw, nan=30.0)
        target_len = 180
        if len(raw) >= target_len:
            step = len(raw) // target_len
            self.latest_scan = raw[::step][:target_len]
        else:
            self.latest_scan = np.pad(raw, (0, target_len - len(raw)), constant_values=30.0)

    def pose_callback(self, msg):
        self.current_pose = msg.pose

    # === 辅助函数：自动起飞逻辑 ===
    def _arm_and_offboard(self):
        # 1. 发送心跳包
        for _ in range(10):
            self.vel_pub.publish(Twist())
            time.sleep(0.05)

        # 2. 切换 OFFBOARD 模式
        req_mode = SetMode.Request()
        req_mode.custom_mode = 'OFFBOARD'
        if self.set_mode_client.service_is_ready():
            self.set_mode_client.call_async(req_mode)

        # 3. 解锁 (Arm)
        req_arm = CommandBool.Request()
        req_arm.value = True
        if self.arming_client.service_is_ready():
            self.arming_client.call_async(req_arm)

    # === Step 函数 ===
    def step(self, action):
        fwd_speed = float(action[0]) * 2.0
        yaw_rate = float(action[1]) * 1.0

        vel_x, vel_y = 0.0, 0.0
        if self.current_pose:
            q = self.current_pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            current_yaw = math.atan2(siny_cosp, cosy_cosp)
            vel_x = fwd_speed * math.cos(current_yaw)
            vel_y = fwd_speed * math.sin(current_yaw)

        current_z = self.current_pose.position.z if self.current_pose else 0.0
        vel_z = (2.0 - current_z) * 1.0  # 保持2米高度

        cmd = Twist()
        cmd.linear.x = vel_x
        cmd.linear.y = vel_y
        cmd.linear.z = vel_z
        cmd.angular.z = yaw_rate
        self.vel_pub.publish(cmd)

        time.sleep(0.1)

        obs = self._get_obs()
        reward, done = self._compute_reward(obs)
        return obs, reward, done, False, {}

    def _get_obs(self):
        if self.latest_scan is None:
            return np.ones(180, dtype=np.float32) * 30.0
        return self.latest_scan.astype(np.float32)

    def _compute_reward(self, obs):
        if self.current_pose is None:
            return 0.0, False
        pos = np.array([self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z])
        dist = np.linalg.norm(pos - TARGET_POS)

        reward = -0.05
        done = False

        if np.min(obs) < 0.3:
            reward = -50.0
            done = True
            print("❌ 撞墙!")
        elif dist < SUCCESS_DIST:
            reward = 100.0
            done = True
            print("✅ 成功!")
        else:
            reward += (30.0 - dist) * 0.1
        return reward, done

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(">>> 新回合: 正在自动解锁起飞... <<<")
        self._arm_and_offboard()
        time.sleep(2.0)  # 等它飞起来
        return self._get_obs(), {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()