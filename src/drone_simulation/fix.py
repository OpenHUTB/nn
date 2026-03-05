"""
MuJoCo 四旋翼无人机仿真 - 智能自修复系统版（优化版）
✅ 多传感器融合感知
✅ 动态路径规划
✅ 智能避障决策
✅ 实时无人机状态监测
✅ 故障自诊断与自动修复
✅ 紧急情况处理机制
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import os
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


# ========== 数据类定义 ==========

@dataclass
class BatteryStatus:
    """电池状态数据类"""
    level: float = 100.0
    voltage: float = 12.6
    current: float = 0.0
    temperature: float = 25.0
    consumption_rate: float = 0.1


@dataclass
class MotorStatus:
    """电机状态数据类"""
    rpm: List[float] = None
    temperature: List[float] = None
    current: List[float] = None
    throttle: List[float] = None

    def __post_init__(self):
        if self.rpm is None:
            self.rpm = [0.0, 0.0, 0.0, 0.0]
        if self.temperature is None:
            self.temperature = [25.0, 25.0, 25.0, 25.0]
        if self.current is None:
            self.current = [0.0, 0.0, 0.0, 0.0]
        if self.throttle is None:
            self.throttle = [0.0, 0.0, 0.0, 0.0]


@dataclass
class AttitudeStatus:
    """姿态状态数据类"""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class GPSStatus:
    """GPS状态数据类"""
    satellites: int = 8
    accuracy: float = 0.5
    fix: bool = True


@dataclass
class SensorStatus:
    """传感器状态数据类"""
    imu: bool = True
    barometer: bool = True
    compass: bool = True
    optical_flow: bool = True


@dataclass
class FlightStatus:
    """飞行状态数据类"""
    time: float = 0.0
    max_altitude: float = 0.0
    max_speed: float = 0.0
    distance: float = 0.0
    last_position: np.ndarray = None
    current_speed: float = 0.0

    def __post_init__(self):
        if self.last_position is None:
            self.last_position = np.array([0.0, 0.0, 0.0])


# ========== 修复记录类 ==========

class RepairLog:
    """修复日志记录类"""
    def __init__(self, maxlen=50):
        self.logs = deque(maxlen=maxlen)

    def add(self, issue: str, success: bool):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            'time': timestamp,
            'issue': issue,
            'success': success
        })

    def get_recent(self, n=5):
        return list(self.logs)[-n:]


# ========== 警告/故障管理类 ==========

class MessageQueue:
    """消息队列管理类"""
    def __init__(self, maxlen=10):
        self.queue = deque(maxlen=maxlen)

    def add(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.queue.append(f"[{timestamp}] {msg}")

    def get_recent(self, n=3):
        return list(self.queue)[-n:]

    def __len__(self):
        return len(self.queue)


# ========== 自动修复系统 ==========

class AutoRepairSystem:
    """无人机自动修复系统"""

    def __init__(self, monitor):
        self.monitor = monitor
        self.repair_attempts = 0
        self.successful_repairs = 0
        self.repair_log = RepairLog()

        # 修复策略映射
        self.repair_strategies = {
            'battery': self._repair_battery,
            'motor': self._repair_motor,
            'gps': self._repair_gps,
            'attitude': self._repair_attitude,
            'sensor': self._repair_sensor,
        }

        # 自动修复开关
        self.auto_repair_enabled = True

        print("🔧 自动修复系统初始化完成")

    def check_and_repair(self, data) -> bool:
        """检查并修复问题"""
        if not self.auto_repair_enabled:
            return False

        issues = self._diagnose_issues()
        repaired = False

        for issue in issues:
            if issue in self.repair_strategies:
                success = self.repair_strategies[issue](data)
                if success:
                    self.repair_attempts += 1
                    self.successful_repairs += 1
                    self.repair_log.add(issue, success)
                    repaired = True

        return repaired

    def _diagnose_issues(self) -> List[str]:
        """诊断系统问题"""
        issues = []
        m = self.monitor

        # 电池问题诊断
        if m.battery.level < 15.0:
            issues.append('battery')
        elif m.battery.level < 25.0:
            m.add_warning("电池电量不足，建议返航")

        # 电机问题诊断
        for i, temp in enumerate(m.motor.temperature):
            if temp > 70.0:
                issues.append('motor')
                m.add_fault(f"电机{i+1}过热")

        # GPS问题诊断
        if not m.gps.fix or m.gps.satellites < 4:
            issues.append('gps')

        # 姿态问题诊断
        if abs(m.attitude.roll) > 60 or abs(m.attitude.pitch) > 60:
            issues.append('attitude')

        # 传感器问题诊断
        if not m.sensor.imu or not m.sensor.barometer:
            issues.append('sensor')

        return issues

    def _repair_battery(self, data) -> bool:
        """修复电池问题"""
        print("🔋 执行电池修复：降低功耗模式")

        # 降低飞行速度
        if hasattr(data, 'max_speed'):
            data.max_speed = min(data.max_speed, 1.5)

        # 切换到节能模式
        self.monitor.battery.consumption_rate *= 0.7

        self.monitor.add_warning("已切换到节能模式")
        return True

    def _repair_motor(self, data) -> bool:
        """修复电机问题"""
        print("⚙️ 执行电机修复：降低负载")

        # 降低油门限制
        if hasattr(data, 'ctrl') and len(data.ctrl) >= 4:
            for i in range(4):
                data.ctrl[i] = min(data.ctrl[i], 700)

        # 降低飞行速度
        if hasattr(data, 'max_speed'):
            data.max_speed = min(data.max_speed, 1.2)

        self.monitor.add_warning("电机负载已降低")
        return True

    def _repair_gps(self, data) -> bool:
        """修复GPS问题"""
        print("📡 执行GPS修复：切换到视觉定位")

        self.monitor.sensor.optical_flow = True
        self.monitor.gps.fix = True
        self.monitor.gps.satellites = max(self.monitor.gps.satellites, 4)

        self.monitor.add_warning("已切换到视觉定位")
        return True

    def _repair_attitude(self, data) -> bool:
        """修复姿态问题"""
        print("🧭 执行姿态修复：自动调平")

        if hasattr(data, 'qpos') and data.qpos.shape[0] > 6:
            current_pos = data.qpos[0:3].copy()
            data.qpos[0:3] = current_pos
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        self.monitor.add_warning("已执行自动调平")
        return True

    def _repair_sensor(self, data) -> bool:
        """修复传感器问题"""
        print("📊 执行传感器修复：重启传感器")

        self.monitor.sensor.imu = True
        self.monitor.sensor.barometer = True
        self.monitor.sensor.compass = True

        self.monitor.add_warning("传感器已重启")
        return True

    def get_stats(self) -> Dict:
        """获取修复统计"""
        return {
            'attempts': self.repair_attempts,
            'successful': self.successful_repairs,
            'success_rate': (self.successful_repairs / max(self.repair_attempts, 1)) * 100,
            'recent_repairs': self.repair_log.get_recent()
        }


# ========== 无人机监测系统 ==========

class DroneMonitor:
    """无人机状态监测系统"""

    def __init__(self):
        # 使用数据类组织状态
        self.battery = BatteryStatus()
        self.flight = FlightStatus()
        self.motor = MotorStatus()
        self.attitude = AttitudeStatus()
        self.gps = GPSStatus()
        self.sensor = SensorStatus()

        # 警告和故障队列
        self.warnings = MessageQueue(maxlen=10)
        self.faults = MessageQueue(maxlen=5)

        # 警告标志
        self._warning_flags = {
            'low_battery': False,
            'high_temp': False,
            'gps_loss': False
        }

        # 历史数据记录
        self.history = {
            'position': deque(maxlen=1000),
            'altitude': deque(maxlen=1000),
            'speed': deque(maxlen=1000),
            'battery': deque(maxlen=1000),
            'time': deque(maxlen=1000)
        }

        # 自动修复系统
        self.repair_system = AutoRepairSystem(self)

        # 紧急状态
        self.emergency_mode = False
        self.return_to_home = False
        self.auto_landing = False

        # 日志记录
        self.log_file = None
        self._init_logging()

        print("📊 无人机监测系统初始化完成")

    def _init_logging(self):
        """初始化日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drone_log_{timestamp}.csv"
        self.log_file = open(filename, 'w')
        self.log_file.write("时间,高度,速度,电池,滚转,俯仰,偏航,温度,警告,修复\n")

    def update(self, data, dt):
        """更新监测数据"""
        self.flight.time += dt

        # 更新电池
        self._update_battery(dt)

        # 更新位置和距离
        self._update_position(data)

        # 更新速度和高度
        self._update_velocity(data)

        # 更新姿态
        self._update_attitude(data)

        # 更新电机状态
        self._update_motors(data)

        # 检查警告条件
        self._check_warnings()

        # 自动修复
        repaired = self.repair_system.check_and_repair(data)

        # 记录历史数据
        self._record_history()

        # 写入日志
        self._write_log(repaired)

    def _update_battery(self, dt):
        """更新电池状态"""
        self.battery.level -= self.battery.consumption_rate * dt
        self.battery.level = max(0.0, self.battery.level)
        self.battery.voltage = 12.6 * (self.battery.level / 100.0)

    def _update_position(self, data):
        """更新位置和距离"""
        current_pos = data.qpos[0:3].copy()
        if np.linalg.norm(self.flight.last_position) > 0:
            self.flight.distance += np.linalg.norm(current_pos - self.flight.last_position)
        self.flight.last_position = current_pos.copy()
        self.flight.max_altitude = max(self.flight.max_altitude, current_pos[2])

    def _update_velocity(self, data):
        """更新速度"""
        current_vel = data.qvel[0:3].copy()
        self.flight.current_speed = np.linalg.norm(current_vel)
        self.flight.max_speed = max(self.flight.max_speed, self.flight.current_speed)

    def _update_attitude(self, data):
        """更新姿态"""
        quat = data.qpos[3:7].copy()
        w, x, y, z = quat

        # 滚转
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        self.attitude.roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        # 俯仰
        sinp = 2.0 * (w * y - z * x)
        self.attitude.pitch = math.degrees(math.asin(max(-1, min(1, sinp))))

        # 偏航
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.attitude.yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        # 角速度
        if data.qvel.shape[0] > 3:
            self.attitude.roll_rate = data.qvel[3]
            self.attitude.pitch_rate = data.qvel[4]
            self.attitude.yaw_rate = data.qvel[5]

    def _update_motors(self, data):
        """更新电机状态"""
        if hasattr(data, 'ctrl') and len(data.ctrl) >= 4:
            for i in range(4):
                self.motor.throttle[i] = data.ctrl[i] / 1000.0 * 100.0
                self.motor.rpm[i] = self.motor.throttle[i] * 100
                self.motor.temperature[i] = 25.0 + self.motor.throttle[i] * 0.5

    def _check_warnings(self):
        """检查警告条件"""
        # 低电量警告
        if self.battery.level < 20.0 and not self._warning_flags['low_battery']:
            self.add_warning(f"⚠️ 低电量: {self.battery.level:.1f}%")
            self._warning_flags['low_battery'] = True

            if self.battery.level < 15.0:
                self.emergency_mode = True
                self.add_fault("🔥 严重低电量，准备自动返航")
                self.return_to_home = True

        if self.battery.level < 10.0:
            self.add_fault(f"🔥 严重低电量: {self.battery.level:.1f}%，执行紧急降落")
            self.auto_landing = True

        # 高温警告
        if self.battery.temperature > 50.0:
            self.add_warning(f"🌡️ 电池高温: {self.battery.temperature:.1f}°C")
        if self.battery.temperature > 60.0:
            self.add_fault("🔥 电池过热，执行紧急降落")
            self.auto_landing = True

        # 电机高温警告
        for i, temp in enumerate(self.motor.temperature):
            if temp > 60.0:
                self.add_warning(f"🌡️ 电机{i+1}高温: {temp:.1f}°C")
            if temp > 75.0:
                self.add_fault(f"🔥 电机{i+1}过热，降低负载")

        # GPS丢失警告
        if not self.gps.fix and not self._warning_flags['gps_loss']:
            self.add_warning("📡 GPS信号丢失")
            self._warning_flags['gps_loss'] = True
        elif self.gps.fix:
            self._warning_flags['gps_loss'] = False

        # 姿态异常警告
        if abs(self.attitude.roll) > 45 or abs(self.attitude.pitch) > 45:
            self.add_warning(f"⚠️ 姿态异常: 滚转{self.attitude.roll:.1f}° 俯仰{self.attitude.pitch:.1f}°")
        if abs(self.attitude.roll) > 60 or abs(self.attitude.pitch) > 60:
            self.add_fault("🚨 严重姿态异常，执行紧急调平")

    def _record_history(self):
        """记录历史数据"""
        self.history['position'].append(self.flight.last_position.copy())
        self.history['altitude'].append(self.flight.last_position[2])
        self.history['speed'].append(self.flight.current_speed)
        self.history['battery'].append(self.battery.level)
        self.history['time'].append(self.flight.time)

    def _write_log(self, repaired):
        """写入日志"""
        if self.log_file:
            self.log_file.write(
                f"{self.flight.time:.2f},{self.flight.last_position[2]:.2f},"
                f"{self.flight.current_speed:.2f},{self.battery.level:.1f},"
                f"{self.attitude.roll:.1f},{self.attitude.pitch:.1f},{self.attitude.yaw:.1f},"
                f"{self.battery.temperature:.1f},{len(self.warnings)},{1 if repaired else 0}\n"
            )
            self.log_file.flush()

    def add_warning(self, warning: str):
        """添加警告信息"""
        self.warnings.add(warning)
        print(f"\n📢 {warning}")

    def add_fault(self, fault: str):
        """添加故障信息"""
        self.faults.add(fault)
        print(f"\n🚨 {fault}")

    def get_status_summary(self) -> Dict:
        """获取状态摘要"""
        repair_stats = self.repair_system.get_stats()

        return {
            'battery_level': self.battery.level,
            'battery': f"{self.battery.level:.1f}% ({self.battery.voltage:.1f}V)",
            'flight_time': f"{self.flight.time:.0f}s",
            'altitude': f"{self.flight.last_position[2]:.1f}m",
            'max_altitude': f"{self.flight.max_altitude:.1f}m",
            'speed': f"{self.flight.current_speed:.1f}m/s",
            'max_speed': f"{self.flight.max_speed:.1f}m/s",
            'distance': f"{self.flight.distance:.1f}m",
            'attitude': f"R:{self.attitude.roll:.0f}° P:{self.attitude.pitch:.0f}° Y:{self.attitude.yaw:.0f}°",
            'gps': f"{self.gps.satellites} sat {'✅' if self.gps.fix else '❌'}",
            'temperature': f"{self.battery.temperature:.1f}°C",
            'warnings': len(self.warnings),
            'faults': len(self.faults),
            'repair_attempts': repair_stats['attempts'],
            'repair_success': repair_stats['successful'],
            'repair_rate': f"{repair_stats['success_rate']:.1f}%",
            'emergency': self.emergency_mode,
            'return_home': self.return_to_home,
            'auto_landing': self.auto_landing
        }

    def print_status(self):
        """打印状态信息"""
        s = self.get_status_summary()

        print("\n" + "="*70)
        print("📊 无人机状态监测系统")
        print("="*70)

        # 电池状态
        color = "🟢" if s['battery_level'] > 50 else "🟡" if s['battery_level'] > 20 else "🔴"
        print(f"{color} 电池: {s['battery']} | 温度: {s['temperature']}")

        # 飞行状态
        print(f"✈️ 飞行时间: {s['flight_time']} | 高度: {s['altitude']} (最大{s['max_altitude']})")
        print(f"💨 速度: {s['speed']} (最大{s['max_speed']}) | 距离: {s['distance']}")

        # 姿态和GPS
        print(f"🧭 姿态: {s['attitude']}")
        print(f"📡 GPS: {s['gps']}")

        # 修复统计
        print(f"🔧 修复: {s['repair_attempts']}次 (成功率{s['repair_rate']})")

        # 紧急状态
        emergency = []
        if s['emergency']: emergency.append("🚨紧急")
        if s['return_home']: emergency.append("🏠返航")
        if s['auto_landing']: emergency.append("🛬降落")
        if emergency:
            print(f"⚡ 状态: {' '.join(emergency)}")

        # 警告和故障
        if s['warnings'] > 0 or s['faults'] > 0:
            print(f"\n⚠️ 警告: {s['warnings']} | 🚨 故障: {s['faults']}")

            if self.warnings.queue:
                print("最新警告:")
                for w in self.warnings.get_recent():
                    print(f"  {w}")

            if self.faults.queue:
                print("最新故障:")
                for f in self.faults.get_recent():
                    print(f"  {f}")

        print("="*70)

    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()
            print(f"📁 日志已保存")


# ========== 感知系统 ==========

class SensorSystem:
    """无人机感知系统"""

    def __init__(self, obstacle_positions: Dict, obstacle_sizes: Dict):
        self.obstacle_positions = obstacle_positions
        self.obstacle_sizes = obstacle_sizes

        # 传感器参数
        self.lidar_range = 6.0
        self.lidar_resolution = 36

        # 传感器数据
        self.danger_zones = []
        self.safe_directions = []
        self.closest_obstacle = None
        self.min_distance = float('inf')

        # 历史数据
        self.history = deque(maxlen=10)

    def update(self, drone_pos):
        """更新传感器数据"""
        self.danger_zones = []
        self.min_distance = float('inf')
        self.closest_obstacle = None

        # 激光雷达扫描
        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            direction = np.array([math.cos(rad), math.sin(rad), 0])

            hit_dist, hit_obs = self._ray_cast(drone_pos, direction)
            if hit_dist < self.lidar_range:
                if hit_dist < self.min_distance:
                    self.min_distance = hit_dist
                    self.closest_obstacle = hit_obs

                if hit_dist < self._safety_distance():
                    self.danger_zones.append({
                        'angle': angle,
                        'distance': hit_dist,
                        'obstacle': hit_obs
                    })

        self._calculate_safe_directions()
        self.history.append({'pos': drone_pos.copy(), 'danger_zones': self.danger_zones.copy()})

    def _ray_cast(self, start, direction) -> Tuple[float, Optional[str]]:
        """射线投射检测障碍物"""
        min_dist = self.lidar_range
        hit_obs = None

        for obs_name, obs_pos in self.obstacle_positions.items():
            to_obs = obs_pos - start
            obs_size = self.obstacle_sizes.get(obs_name, 0.5)

            proj_dist = np.dot(to_obs, direction)
            if proj_dist < 0 or proj_dist > self.lidar_range:
                continue

            perp_dist = np.linalg.norm(to_obs - proj_dist * direction)
            if perp_dist < obs_size and proj_dist < min_dist:
                min_dist = proj_dist
                hit_obs = obs_name

        return min_dist, hit_obs

    def _safety_distance(self) -> float:
        """动态安全距离"""
        base_dist = 1.2
        if len(self.history) > 1:
            last_pos = self.history[-2]['pos']
            velocity = np.linalg.norm(self.history[-1]['pos'] - last_pos) / 0.01
            return base_dist + velocity * 0.3
        return base_dist

    def _calculate_safe_directions(self):
        """计算安全飞行方向"""
        self.safe_directions = []
        safety_dist = self._safety_distance() * 1.5

        for angle in range(0, 360, 10):
            is_safe = all(abs(angle - d['angle']) >= 30 or d['distance'] >= safety_dist
                         for d in self.danger_zones)
            if is_safe:
                rad = math.radians(angle)
                self.safe_directions.append({
                    'angle': angle,
                    'direction': np.array([math.cos(rad), math.sin(rad), 0])
                })

    def get_best_direction(self, target_dir) -> Tuple[Optional[np.ndarray], float]:
        """获取最佳飞行方向"""
        if not self.safe_directions:
            return None, 0.0

        best = max(self.safe_directions,
                  key=lambda d: (np.dot(d['direction'], target_dir) + 1) / 2)
        return best['direction'], best['angle']

    def get_avoidance_force(self, target_dir) -> Tuple[np.ndarray, float]:
        """计算避障力"""
        if not self.danger_zones:
            return np.zeros(3), 1.0

        force = np.zeros(3)
        for danger in self.danger_zones:
            angle = math.radians(danger['angle'])
            danger_dir = np.array([math.cos(angle), math.sin(angle), 0])
            strength = (1.0 - danger['distance'] / self._safety_distance()) ** 2
            force -= danger_dir * strength

        if np.linalg.norm(force) > 0:
            force /= np.linalg.norm(force)

        safety = max(0.2, min(1.0, 1.0 - len(self.danger_zones) * 0.1))
        return force, safety


# ========== 路径规划器 ==========

class PathPlanner:
    """路径规划器"""

    def __init__(self):
        self.waypoints = []
        self.current = 0
        self.home = np.array([0.0, 0.0, 0.2])

        self.default_waypoints = [
            np.array([0.0, 0.0, 2.0]),
            np.array([4.0, 4.0, 2.0]), np.array([4.0, -4.0, 2.0]),
            np.array([-4.0, -4.0, 2.0]), np.array([-4.0, 4.0, 2.0]),
            np.array([0.0, 0.0, 2.0]), np.array([6.0, 0.0, 2.0]),
            np.array([0.0, 6.0, 2.0]), np.array([-6.0, 0.0, 2.0]),
            np.array([0.0, -6.0, 2.0]),
        ]

    def get_next(self, current_pos, monitor=None) -> np.ndarray:
        """获取下一个航点"""
        if monitor:
            if monitor.return_to_home:
                return self.home
            if monitor.auto_landing:
                return np.array([self.home[0], self.home[1], 0.2])

        if not self.waypoints:
            self.waypoints = self.default_waypoints

        target = self.waypoints[self.current]
        if np.linalg.norm(current_pos[:2] - target[:2]) < 2.0:
            self.current = (self.current + 1) % len(self.waypoints)
            target = self.waypoints[self.current]

        return target


# ========== 无人机主仿真类 ==========

class QuadrotorSimulation:
    """四旋翼无人机主仿真类"""

    # 障碍物配置
    OBSTACLES = {
        'positions': {
            "building_office": [5.0, 5.0, 1.0], "building_tower": [8.0, 4.0, 1.5],
            "building_apartment": [3.0, 8.0, 1.2], "building_shop": [-5.0, 5.0, 1.0],
            "building_cafe": [-8.0, 4.0, 1.0], "building_house1": [-5.0, -5.0, 0.8],
            "building_house2": [-8.0, -5.0, 0.8], "building_school": [5.0, -5.0, 1.2],
            "building_library": [8.0, -5.0, 1.0], "tree_1": [2.0, 2.0, 0.8],
            "tree_2": [-2.0, 2.0, 0.8], "tree_3": [2.0, -2.0, 0.8],
            "tree_4": [-2.0, -2.0, 0.8], "light_1": [3.0, 3.0, 0.6],
            "light_2": [-3.0, 3.0, 0.6], "light_3": [3.0, -3.0, 0.6],
            "light_4": [-3.0, -3.0, 0.6], "car_1": [2.0, 0.0, 0.3],
            "car_2": [-2.0, 0.0, 0.3]
        },
        'sizes': {
            "building_office": 1.5, "building_tower": 1.0, "building_apartment": 1.2,
            "building_shop": 1.2, "building_cafe": 1.0, "building_house1": 0.8,
            "building_house2": 0.8, "building_school": 1.2, "building_library": 1.0,
            "tree_1": 0.5, "tree_2": 0.5, "tree_3": 0.5, "tree_4": 0.5,
            "light_1": 0.3, "light_2": 0.3, "light_3": 0.3, "light_4": 0.3,
            "car_1": 0.5, "car_2": 0.5
        }
    }

    def __init__(self, xml_path="quadrotor_detailed_city.xml"):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到XML文件: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        print(f"✓ 模型加载成功: {xml_path}")

        # 初始化子系统
        self._init_subsystems()
        self._init_flight_params()

    def _init_subsystems(self):
        """初始化子系统"""
        # 转换为numpy数组
        obs_pos = {k: np.array(v) for k, v in self.OBSTACLES['positions'].items()}

        self.sensor = SensorSystem(obs_pos, self.OBSTACLES['sizes'])
        self.planner = PathPlanner()
        self.monitor = DroneMonitor()

    def _init_flight_params(self):
        """初始化飞行参数"""
        self.phase = "takeoff"
        self.phase_start = 0.0
        self.takeoff_height = 2.0
        self.cruise_height = 2.0

        self.max_speed = 2.2
        self.current_vel = np.zeros(3)

        self.bounds = {'x': [-10, 10], 'y': [-10, 10], 'z': [0.2, 4.0]}
        self.avoidance_count = 0
        self.last_avoid_time = 0
        self.last_status_time = 0
        self.status_interval = 5.0

        # 基础推力
        if self.model.nu > 0:
            self.data.ctrl[:] = [600] * self.model.nu

    def _update_height(self, current_time) -> float:
        """更新飞行高度"""
        if self.phase == "takeoff":
            progress = min((current_time - self.phase_start) * 0.5, 1.0)
            height = 0.2 + (self.takeoff_height - 0.2) * progress
            if progress >= 1.0:
                self.phase = "cruise"
                self.phase_start = current_time
                print("\n🚁 起飞完成")
            return height
        elif self.phase == "cruise":
            return self.cruise_height
        else:  # landing
            progress = min((current_time - self.phase_start) * 0.3, 1.0)
            return self.cruise_height - (self.cruise_height - 0.2) * progress

    def _calc_movement(self, current_pos, target_pos) -> np.ndarray:
        """计算移动方向"""
        self.sensor.update(current_pos)

        to_target = target_pos - current_pos
        dist = np.linalg.norm(to_target)
        if dist < 0.1:
            return current_pos

        target_dir = to_target / dist
        force, safety = self.sensor.get_avoidance_force(target_dir)
        best_dir, _ = self.sensor.get_best_direction(target_dir)

        # 选择移动方向
        if self.monitor.emergency_mode:
            move_dir = force if np.linalg.norm(force) > 0 else best_dir or np.array([0, 0, 1])
        elif safety < 0.3:
            move_dir = force if np.linalg.norm(force) > 0 else best_dir or np.array([0, 0, 1])
            if time.time() - self.last_avoid_time > 2.0:
                self.monitor.add_warning("紧急避障")
                self.last_avoid_time = time.time()
                self.avoidance_count += 1
        elif safety < 0.7 and best_dir is not None:
            w = 0.4 + safety * 0.3
            move_dir = target_dir * w + best_dir * (1 - w)
            move_dir /= np.linalg.norm(move_dir)
        else:
            move_dir = target_dir

        # 速度因子
        speed_factor = 1.0
        if self.monitor.emergency_mode:
            speed_factor = 0.6
        elif self.monitor.return_to_home:
            speed_factor = 0.8
        elif self.monitor.auto_landing:
            speed_factor = 0.5

        target_vel = move_dir * self.max_speed * (0.8 + safety * 0.4) * speed_factor
        self.current_vel += (target_vel - self.current_vel) * 0.15

        new_pos = current_pos + self.current_vel * self.model.opt.timestep * 50
        for axis in 'xyz':
            new_pos[0 if axis=='x' else 1 if axis=='y' else 2] = np.clip(
                new_pos[0 if axis=='x' else 1 if axis=='y' else 2],
                self.bounds[axis][0], self.bounds[axis][1]
            )
        return new_pos

    def _get_target(self, current_pos) -> np.ndarray:
        """获取目标点"""
        if self.phase == "takeoff":
            return np.array([0, 0, self.takeoff_height])
        elif self.phase == "landing":
            return np.array([0, 0, 0.2])
        else:
            target = self.planner.get_next(current_pos, self.monitor)
            target[2] = self.cruise_height
            return target

    def _set_attitude(self, pos):
        """设置姿态"""
        if self.model.nq > 3:
            speed = np.linalg.norm(self.current_vel)
            tilt = min(0.3, speed * 0.1)
            self.data.qpos[3] = math.cos(tilt/2)
            self.data.qpos[4] = 0.0
            self.data.qpos[5] = math.sin(tilt/2)
            self.data.qpos[6] = 0.0

        # 旋翼旋转
        for i in range(min(4, self.model.nq - 7)):
            self.data.qpos[7 + i] += 30.0 * self.model.opt.timestep

    def _print_status(self, current_time, pos, target):
        """打印状态"""
        danger = len(self.sensor.danger_zones)
        safe = len(self.sensor.safe_directions)
        _, safety = self.sensor.get_avoidance_force(target - pos)

        phase_icons = {"takeoff": "🔼", "cruise": "✈️", "landing": "🔽"}
        safety_icon = "✅" if safety > 0.8 else "⚠️" if safety > 0.4 else "🚨"

        m = self.monitor.get_status_summary()

        print(f"\n{'='*90}")
        print(f"时间: {current_time:.1f}s | {phase_icons[self.phase]} 航点{self.planner.current + 1}")
        print(f"位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | 速度: {np.linalg.norm(self.current_vel):.2f}")
        print(f"\n【感知】{safety_icon} 安全:{safety:.2f} | 危险:{danger} 安全方向:{safe} | 最近:{self.sensor.closest_obstacle} {self.sensor.min_distance:.2f}m")
        print(f"【电池】{m['battery']} | 温度:{m['temperature']} | 飞行:{m['flight_time']}")
        print(f"【姿态】{m['attitude']} | GPS:{m['gps']} | 修复:{m['repair_attempts']}({m['repair_rate']})")
        print(f"{'='*90}")

    def run(self, duration=120.0):
        """运行仿真"""
        print(f"\n{'🚁'*10} 无人机智能自修复系统 {'🚁'*10}")
        print(f"▶ 激光雷达: {self.sensor.lidar_range}m | 速度: {self.max_speed}m/s")
        print(f"▶ 障碍物: {len(self.OBSTACLES['positions'])} | 边界: X±10 Y±10 Z0.2-4.0")
        print(f"{'='*90}")

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.azimuth = -45
                viewer.cam.elevation = 30
                viewer.cam.distance = 18.0
                viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

                print("\n🔼 起飞... 系统已激活")

                start_time = time.time()
                last_print = 0
                last_monitor = 0
                pos_smooth = np.array([0, 0, 0.2])

                while viewer.is_running() and time.time() - start_time < duration:
                    step_start = time.time()
                    current_time = self.data.time
                    wall_time = time.time()

                    mujoco.mj_step(self.model, self.data)
                    current_pos = self.data.qpos[0:3].copy()

                    # 更新系统
                    if wall_time - last_monitor > 0.1:
                        self.monitor.update(self.data, self.model.opt.timestep)
                        last_monitor = wall_time

                    # 飞行控制
                    height = self._update_height(current_time)
                    target = self._get_target(current_pos)
                    target[2] = height
                    new_pos = self._calc_movement(current_pos, target)

                    # 平滑移动
                    pos_smooth = pos_smooth + (new_pos - pos_smooth) * 0.3
                    self.data.qpos[:3] = pos_smooth

                    self._set_attitude(pos_smooth)
                    viewer.sync()

                    # 状态显示
                    if wall_time - last_print > 1.0:
                        self._print_status(current_time, pos_smooth, target)
                        last_print = wall_time

                    if wall_time - self.last_status_time > self.status_interval:
                        self.monitor.print_status()
                        self.last_status_time = wall_time

                    # 速率控制
                    elapsed = time.time() - step_start
                    if self.model.opt.timestep - elapsed > 0:
                        time.sleep(self.model.opt.timestep - elapsed)

        except Exception as e:
            print(f"⚠ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.monitor.close()

        print(f"\n{'✅'*10} 仿真结束 {'✅'*10}")


def main():
    print("🚁 MuJoCo 四旋翼无人机 - 智能自修复系统版")
    print("=" * 90)

    try:
        sim = QuadrotorSimulation("quadrotor_detailed_city.xml")
        print("✅ 初始化完成")
        sim.run(duration=120.0)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except KeyboardInterrupt:
        print("\n\n⏹ 仿真中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()