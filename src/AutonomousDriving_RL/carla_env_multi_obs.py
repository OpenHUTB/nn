# carla_env/carla_env_multi_obs.py
# 本文件定义了一个基于 CARLA 仿真器的自定义 Gymnasium 环境，
# 用于训练强化学习智能体（如 PPO）控制自动驾驶车辆。
# 观测空间为 4 维连续状态（位置 + 速度），动作空间为 3 维连续控制（油门、转向、刹车）。

import carla  # CARLA 仿真器 Python API
import numpy as np  # 数值计算库
import random  # 随机数生成
import time  # 时间控制（用于延迟）
import os  # 操作系统接口（文件操作）
import json  # JSON 文件读写（用于保存/加载车辆ID）
from gymnasium import Env, spaces  # Gymnasium 标准环境接口

# 定义临时文件路径，用于记录上一次运行生成的车辆ID（与脚本同目录）
VEHICLE_ID_FILE = ".last_vehicle_id.json"


class CarlaEnvMultiObs(Env):
    """
    自定义 CARLA 强化学习环境类，继承自 gymnasium.Env。
    支持自动清理历史车辆、多地图适配、第三人称视角跟随、安全spawn等特性。
    """

    def __init__(self, keep_alive_after_exit=True):
        """
        初始化环境。
        :param keep_alive_after_exit: 若为 True，close() 时不销毁车辆，便于人工观察或录屏。
        """
        super(CarlaEnvMultiObs, self).__init__()

        # CARLA 客户端与世界对象
        self.client = None
        self.world = None

        # 车辆相关
        self.vehicle = None  # 当前控制的车辆 Actor
        self._current_vehicle_id = None  # 本次生成的车辆 ID（用于下次清理）

        # 训练控制
        self.frame_count = 0  # 已执行的仿真步数
        self.max_frames = 1000  # 最大允许步数（用于 truncated 判定）
        self.prev_x = 0.0  # 上一帧的 x 坐标（用于计算位移奖励）

        # 视角控制
        self.spectator = None  # CARLA 观察者（摄像头）

        # 行为标志
        self.keep_alive = keep_alive_after_exit

        # 定义观测空间：[x, y, vx, vy] —— 位置 (m) + 速度 (m/s)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        # 定义动作空间：[throttle, steer, brake]
        # - throttle: [0.0, 1.0] 油门（0=松开，1=全踩）
        # - steer: [-1.0, 1.0] 转向（-1=左打满，1=右打满）
        # - brake: [0.0, 1.0] 刹车（0=松开，1=全刹）
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        :param seed: 随机种子（用于可复现性）
        :param options: 额外选项（本实现未使用）
        :return: 初始观测值 (obs), info 字典
        """
        super().reset(seed=seed)

        # 设置随机种子（确保行为可复现）
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        try:
            # 连接 CARLA 服务器（localhost:2000）
            if self.client is None:
                print("🔄 尝试连接 CARLA 服务器...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(20.0)  # 超时20秒
                self.world = self.client.get_world()
                if self.world is None:
                    raise RuntimeError("❌ 无法获取 CARLA 世界！")
                print(f"✅ 成功连接到 CARLA！地图: {self.world.get_map().name}")

            # 清理上一次运行残留的车辆（通过ID文件）
            self._destroy_last_run_vehicle()

            # 生成新车
            self.spawn_vehicle()

            # 同步几帧，确保车辆稳定
            for _ in range(5):
                self.world.tick()
                time.sleep(0.05)

            # 获取观察者并设置第三人称视角
            self.spectator = self.world.get_spectator()
            self._update_spectator_view()
            print("🎥 第三人称视角已激活（完整车身 + 前方道路可见）")

            # 重置计数器
            self.frame_count = 0
            obs = self.get_observation()
            self.prev_x = obs[0]  # 记录初始x位置
            return obs, {}

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

    def _destroy_last_run_vehicle(self):
        """
        安全销毁上一次运行留下的车辆。
        即使 .last_vehicle_id.json 文件损坏、为空或不存在，也能优雅处理，不抛出异常。
        """
        # 若无记录文件，直接跳过
        if not os.path.exists(VEHICLE_ID_FILE):
            print("ℹ️ 无历史车辆记录，跳过清理")
            return

        last_id = None
        try:
            # 安全读取 JSON 文件
            with open(VEHICLE_ID_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    print("⚠️ 车辆ID文件为空")
                    return
                data = json.loads(content)
                last_id = data.get("vehicle_id")
                if last_id is None:
                    print("⚠️ 车辆ID字段缺失")
                    return
        except (json.JSONDecodeError, OSError, ValueError) as e:
            # 文件损坏时，尝试删除它
            print(f"⚠️ 读取车辆ID文件失败（文件可能损坏）: {e}")
            try:
                os.remove(VEHICLE_ID_FILE)
            except OSError:
                pass
            return

        # 验证ID类型
        if not isinstance(last_id, int):
            print(f"⚠️ 车辆ID类型无效: {type(last_id)}")
            return

        # 发送销毁命令
        print(f"🧹 正在销毁上一次运行的车辆 (ID: {last_id})...")
        batch = [carla.command.DestroyActor(last_id)]
        responses = self.client.apply_batch_sync(batch, do_tick=True)

        if responses[0].error:
            print(f" - 销毁失败: {responses[0].error}")
        else:
            print("✅ 上次车辆已成功清理")

        # 清理后删除ID文件
        try:
            os.remove(VEHICLE_ID_FILE)
        except OSError as e:
            print(f"⚠️ 删除车辆ID文件失败: {e}")

    def spawn_vehicle(self):
        """
        在当前地图的安全位置生成一辆特斯拉 Model 3（若不可用则随机选车）。
        支持 Town01/03/05/10 的预设 spawn 点，其他地图自动选择最低 z 的点。
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

        # 根据地图名称选择 spawn 位置
        map_name = self.world.get_map().name.lower()
        if 'town01' in map_name:
            spawn_transform = carla.Transform(
                carla.Location(x=-60.0, y=20.0, z=0.3),
                carla.Rotation(yaw=90.0)
            )
        elif 'town03' in map_name:
            spawn_transform = carla.Transform(
                carla.Location(x=70.0, y=-10.0, z=0.3),
                carla.Rotation(yaw=180.0)
            )
        elif 'town05' in map_name:
            spawn_transform = carla.Transform(
                carla.Location(x=-75.0, y=16.0, z=0.3),
                carla.Rotation(yaw=90.0)
            )
        elif 'town10' in map_name:
            spawn_transform = carla.Transform(
                carla.Location(x=100.0, y=130.0, z=0.3),
                carla.Rotation(yaw=180.0)
            )
        else:
            # 通用 fallback：选择 z 最低的 spawn 点（更平坦安全）
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("❌ 地图中没有可用的 spawn points！")
            spawn_transform = min(spawn_points, key=lambda t: t.location.z)

        # 尝试生成车辆
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
            # 若失败，遍历所有 spawn 点，增加 z 安全余量
            spawn_points = self.world.get_map().get_spawn_points()
            for transform in spawn_points:
                safe_z = max(transform.location.z, 0.0) + 0.3
                safe_transform = carla.Transform(
                    carla.Location(x=transform.location.x, y=transform.location.y, z=safe_z),
                    transform.rotation
                )
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, safe_transform)
                if self.vehicle is not None:
                    break

        if self.vehicle is None:
            raise RuntimeError("❌ 无法生成车辆！")

        # 记录车辆信息
        self._current_vehicle_id = self.vehicle.id
        loc = self.vehicle.get_location()
        print(
            f"✅ 车辆生成成功: {self.vehicle.type_id} | ID={self._current_vehicle_id} | 位置: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        # 原子写入车辆ID文件（防止并发写入损坏）
        temp_file = VEHICLE_ID_FILE + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump({"vehicle_id": self._current_vehicle_id}, f)
            os.replace(temp_file, VEHICLE_ID_FILE)  # 原子操作
        except Exception as e:
            print(f"⚠️ 保存车辆ID失败（不影响运行）: {e}")

    def _update_spectator_view(self):
        """
        更新 CARLA 观察者视角，使其跟随车辆（第三人称）。
        相机位于车辆后上方，俯视前方道路。
        """
        if not (self.vehicle and self.spectator):
            return
        v_transform = self.vehicle.get_transform()
        # 相对偏移：后方8米，上方4米
        offset = carla.Location(x=-8.0, y=0.0, z=4.0)
        spectator_loc = v_transform.transform(offset)
        spectator_rot = carla.Rotation(
            pitch=-20.0,  # 俯视角
            yaw=v_transform.rotation.yaw,  # 跟随车辆朝向
            roll=0.0
        )
        self.spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

    def get_observation(self):
        """
        获取当前环境观测值。
        :return: np.array([x, y, vx, vy], dtype=np.float32)
        """
        if not self.vehicle or not self.vehicle.is_alive:
            # 车辆不存在时返回零向量（避免崩溃）
            return np.zeros(4, dtype=np.float32)
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        return np.array([loc.x, loc.y, vel.x, vel.y], dtype=np.float32)

    def step(self, action):
        """
        执行一步环境交互。
        :param action: [throttle, steer, brake]
        :return: obs, reward, terminated, truncated, info
        """
        throttle, steer, brake = action
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)
        self.world.tick()  # 推进仿真
        self.frame_count += 1
        self._update_spectator_view()  # 更新视角

        # 检查车辆是否被销毁
        if not self.vehicle or not self.vehicle.is_alive:
            return np.zeros(4, dtype=np.float32), -10.0, True, False, {}

        # 获取观测
        obs = self.get_observation()
        x, y, vx, vy = obs
        speed = np.linalg.norm([vx, vy])

        # ========================
        # ✅【强力推荐】使用车辆朝向速度作为主奖励
        # ========================
        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()  # 车头方向单位向量

        # 计算速度在车头方向的投影（鼓励向前行驶）
        forward_speed = vx * forward_vector.x + vy * forward_vector.y

        # 主奖励：只奖励正向前进（倒车不奖励）
        reward = 1.0 * max(forward_speed, 0.0)

        # 额外惩罚：如果几乎静止，施加较大惩罚（促进行动）
        if speed < 0.1:
            reward -= 0.5

        self.prev_x = x
        terminated = False  # 暂无终止条件（如碰撞）
        truncated = self.frame_count >= self.max_frames  # 超过最大步数则截断
        return obs, reward, terminated, truncated, {}

    def close(self):
        """
        关闭环境，释放资源。
        若 keep_alive=True，则保留车辆供人工观察；否则销毁。
        """
        if self.keep_alive:
            print("ℹ️ 车辆已保留（ID已记录，下次运行时将自动清理）")
            if self.vehicle:
                # 松开所有控制，让车自然停下
                self.vehicle.apply_control(carla.VehicleControl())
                for i in range(30):  # 同步30帧确保停止
                    self.world.tick()
                    self._update_spectator_view()
                    time.sleep(0.1)
                print("✅ 现在你可以自由操作 CARLA 视角（按 F1~F4）！")
        else:
            # 彻底清理
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
