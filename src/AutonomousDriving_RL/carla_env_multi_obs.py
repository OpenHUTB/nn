# carla_env/carla_env_multi_obs.py
import carla
import numpy as np
import random
import time
import os
import json
from gymnasium import Env, spaces

VEHICLE_ID_FILE = ".last_vehicle_id.json"
TRAJECTORY_LOG_FILE = "trajectory.csv"


class CarlaEnvMultiObs(Env):
    def __init__(self, keep_alive_after_exit=True, log_trajectory=True):
        super(CarlaEnvMultiObs, self).__init__()
        self.client = None
        self.world = None
        self.vehicle = None
        self._current_vehicle_id = None
        self.frame_count = 0
        self.max_frames = 1000
        self.spectator = None
        self.keep_alive = keep_alive_after_exit
        self.log_trajectory = log_trajectory
        self.trajectory_data = []

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

    def _connect_carla(self, max_retries=3):
        """自动重试连接 CARLA"""
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试连接 CARLA 服务器 (第 {attempt + 1} 次)...")
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.get_world()
                if self.world is not None:
                    print(f"✅ 成功连接到 CARLA！地图: {self.world.get_map().name}")
                    return True
            except Exception as e:
                print(f"⚠️ 连接失败: {e}")
                time.sleep(2)
        raise RuntimeError("❌ 无法连接到 CARLA 服务器，请确保 CARLA 已启动！")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._connect_carla()
        self._destroy_last_run_vehicle()
        self.spawn_vehicle()

        for _ in range(5):
            self.world.tick()
            time.sleep(0.05)

        self.spectator = self.world.get_spectator()
        self._update_spectator_view()

        self.trajectory_data = []
        self.frame_count = 0
        obs = self.get_observation()
        return obs, {}

    def _destroy_last_run_vehicle(self):
        if not os.path.exists(VEHICLE_ID_FILE):
            return
        try:
            with open(VEHICLE_ID_FILE, 'r') as f:
                data = json.load(f)
                last_id = data.get("vehicle_id")
            if isinstance(last_id, int):
                self.client.apply_batch_sync([carla.command.DestroyActor(last_id)], do_tick=True)
        except Exception:
            pass
        try:
            os.remove(VEHICLE_ID_FILE)
        except OSError:
            pass

    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        # 优先使用特斯拉，否则随机选一个
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp or not vehicle_bp.has_attribute('number_of_wheels'):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

        # 设置颜色（可选）
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)

        map_name = self.world.get_map().name.lower()
        spawn_transform = None

        # 针对 Town10HD_Opt 使用已知安全点
        if 'town10' in map_name:
            spawn_transform = carla.Transform(
                carla.Location(x=100.0, y=130.0, z=0.3),
                carla.Rotation(yaw=180.0)
            )
        else:
            # 通用 fallback：使用第一个 spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if spawn_points:
                spawn_transform = spawn_points[0]
            else:
                # 极端 fallback：原点上方
                spawn_transform = carla.Transform(carla.Location(x=0, y=0, z=1.0), carla.Rotation())

        # 尝试主位置
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)

        # 如果失败，遍历所有 spawn points
        if self.vehicle is None:
            print("⚠️ 主 spawn 点失败，尝试遍历所有可用点...")
            all_spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(all_spawn_points)  # 随机顺序避免总用同一个
            for sp in all_spawn_points:
                # 抬高一点防止穿地
                safe_z = max(sp.location.z, 0.0) + 0.3
                safe_sp = carla.Transform(
                    carla.Location(x=sp.location.x, y=sp.location.y, z=safe_z),
                    sp.rotation
                )
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, safe_sp)
                if self.vehicle is not None:
                    print(f"✅ 在备用点成功生成车辆: ({safe_sp.location.x:.1f}, {safe_sp.location.y:.1f})")
                    break

        if self.vehicle is None:
            raise RuntimeError("❌ 所有 spawn 点均无法生成车辆！请检查地图或 CARLA 状态。")

        self._current_vehicle_id = self.vehicle.id
        loc = self.vehicle.get_location()
        print(
            f"✅ 车辆生成成功: {self.vehicle.type_id} | ID={self._current_vehicle_id} | 位置: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        try:
            with open(VEHICLE_ID_FILE, 'w') as f:
                json.dump({"vehicle_id": self._current_vehicle_id}, f)
        except Exception as e:
            print(f"⚠️ 保存车辆ID失败: {e}")

    def _update_spectator_view(self):
        """修复视角：温和第三人称，确保看到整车"""
        if not (self.vehicle and self.spectator):
            return
        try:
            v_transform = self.vehicle.get_transform()
            # 相机：后方5米，右侧1米，上方2.2米（更低更稳）
            offset = carla.Location(x=-5.0, y=1.0, z=2.2)
            camera_loc = v_transform.transform(offset)
            # 俯角 -10°（不要太陡），yaw 跟随车辆
            spectator_rot = carla.Rotation(
                pitch=-10.0,
                yaw=v_transform.rotation.yaw,
                roll=0.0
            )
            self.spectator.set_transform(carla.Transform(camera_loc, spectator_rot))
        except Exception:
            pass  # 容错

    def _log_trajectory(self, x, y, speed):
        if self.log_trajectory:
            self.trajectory_data.append((x, y, speed))

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

        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        forward_speed = vx * forward_vector.x + vy * forward_vector.y
        reward = 1.0 * max(forward_speed, 0.0)
        if speed < 0.1:
            reward -= 0.5

        self._log_trajectory(x, y, speed)

        terminated = False
        truncated = self.frame_count >= self.max_frames
        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.log_trajectory and self.trajectory_data:
            try:
                with open(TRAJECTORY_LOG_FILE, 'w') as f:
                    f.write("x,y,speed\n")
                    for x, y, speed in self.trajectory_data:
                        f.write(f"{x:.3f},{y:.3f},{speed:.3f}\n")
                print(f"📊 轨迹已保存至: {TRAJECTORY_LOG_FILE}")
            except Exception as e:
                print(f"⚠️ 轨迹保存失败: {e}")

        if self.keep_alive:
            print("ℹ️ 车辆已保留（ID已记录，下次运行时将自动清理）")
            if self.vehicle:
                self.vehicle.apply_control(carla.VehicleControl())
                for i in range(30):
                    self.world.tick()
                    self._update_spectator_view()
                    time.sleep(0.1)
                print("✅ 现在你可以自由操作 CARLA 视角（按 F1~F4）！")
        else:
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
