# carla_env/carla_env_multi_obs.py
import carla
import numpy as np
import random
import time
import os
import json
from gymnasium import Env, spaces

# 用于记录上一次车辆ID的临时文件（与脚本同目录）
VEHICLE_ID_FILE = ".last_vehicle_id.json"


class CarlaEnvMultiObs(Env):
    def __init__(self, keep_alive_after_exit=True):  # 默认保留车辆
        super(CarlaEnvMultiObs, self).__init__()
        self.client = None
        self.world = None
        self.vehicle = None
        self._current_vehicle_id = None  # 记录本次生成的 vehicle ID
        self.frame_count = 0
        self.max_frames = 1000
        self.prev_x = 0.0
        self.spectator = None
        self.keep_alive = keep_alive_after_exit

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        try:
            if self.client is None:
                print("🔄 尝试连接 CARLA 服务器...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(20.0)
                self.world = self.client.get_world()
                if self.world is None:
                    raise RuntimeError("❌ 无法获取 CARLA 世界！")
                print(f"✅ 成功连接到 CARLA！地图: {self.world.get_map().name}")

            # 🔥 关键：只销毁上一次运行留下的车辆（通过ID文件）
            self._destroy_last_run_vehicle()

            self.spawn_vehicle()
            for _ in range(5):
                self.world.tick()
                time.sleep(0.05)

            self.spectator = self.world.get_spectator()
            self._update_spectator_view()
            print("🎥 第三人称视角已激活（完整车身 + 前方道路可见）")

            self.frame_count = 0
            obs = self.get_observation()
            self.prev_x = obs[0]
            return obs, {}

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

    def _destroy_last_run_vehicle(self):
        """仅销毁上一次运行留下的车辆（从 .last_vehicle_id.json 读取）"""
        if not os.path.exists(VEHICLE_ID_FILE):
            print("ℹ️ 无历史车辆记录，跳过清理")
            return

        try:
            with open(VEHICLE_ID_FILE, 'r') as f:
                data = json.load(f)
                last_id = data.get("vehicle_id")
                if last_id is None:
                    print("⚠️ 车辆ID文件内容无效")
                    return
        except Exception as e:
            print(f"⚠️ 读取车辆ID文件失败: {e}")
            return

        print(f"🧹 正在销毁上一次运行的车辆 (ID: {last_id})...")
        batch = [carla.command.DestroyActor(last_id)]
        responses = self.client.apply_batch_sync(batch, do_tick=True)

        if responses[0].error:
            print(f"   - 销毁失败: {responses[0].error}")
        else:
            print("✅ 上次车辆已成功清理")

        # 清理后删除文件（避免重复销毁）
        try:
            os.remove(VEHICLE_ID_FILE)
        except OSError:
            pass  # 忽略删除失败

    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

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
            # Town10HD_Opt 安全 spawn 点
            spawn_transform = carla.Transform(
                carla.Location(x=100.0, y=130.0, z=0.3),
                carla.Rotation(yaw=180.0)
            )
        else:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("❌ 地图中没有可用的 spawn points！")
            spawn_transform = spawn_points[0]

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
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

        self._current_vehicle_id = self.vehicle.id
        loc = self.vehicle.get_location()
        print(f"✅ 车辆生成成功: {self.vehicle.type_id} | ID={self._current_vehicle_id} | 位置: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        # ✅ 保存本次 vehicle ID 到文件（供下次运行清理）
        try:
            with open(VEHICLE_ID_FILE, 'w') as f:
                json.dump({"vehicle_id": self._current_vehicle_id}, f)
        except Exception as e:
            print(f"⚠️ 保存车辆ID失败: {e}")

    def _update_spectator_view(self):
        if not (self.vehicle and self.spectator):
            return

        v_transform = self.vehicle.get_transform()
        offset = carla.Location(x=-8.0, y=0.0, z=4.0)
        spectator_loc = v_transform.transform(offset)

        spectator_rot = carla.Rotation(
            pitch=-20.0,
            yaw=v_transform.rotation.yaw,
            roll=0.0
        )
        self.spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

    def get_observation(self):
        if not self.vehicle or not self.vehicle.is_alive:
            return np.zeros(4, dtype=np.float32)
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        return np.array([loc.x, loc.y, vel.x, vel.y], dtype=np.float32)

    def step(self, action):
        throttle, steer, brake = action
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)
        self.world.tick()
        self.frame_count += 1
        self._update_spectator_view()

        if not self.vehicle or not self.vehicle.is_alive:
            return np.zeros(4, dtype=np.float32), -10.0, True, False, {}

        obs = self.get_observation()
        x, y, vx, vy = obs
        speed = np.linalg.norm([vx, vy])
        reward = 0.1 * (x - self.prev_x) + 0.5 * speed
        self.prev_x = x

        terminated = False
        truncated = self.frame_count >= self.max_frames
        return obs, reward, terminated, truncated, {}

    def close(self):
        # ✅ 关键：不销毁车辆！保留用于观察或录屏
        if self.keep_alive:
            print("ℹ️ 车辆已保留（ID已记录，下次运行时将自动清理）")
            if self.vehicle:
                # 松开油门刹车，让车静止
                self.vehicle.apply_control(carla.VehicleControl())
            # 同步几帧确保控制生效
            for i in range(30):
                self.world.tick()
                self._update_spectator_view()
                time.sleep(0.1)
            print("✅ 现在你可以自由操作 CARLA 视角（按 F1~F4）！")
        else:
            # 如果需要立即清理（如训练模式），可启用
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
