# -*- coding: utf-8 -*-
import time
import numpy as np
from typing import Optional, List, Dict, Tuple

from core import BaseDroneController, ConfigManager, Logger


class WaypointMarker:
    """航点标记"""
    def __init__(self, position: np.ndarray, mode: str, label: str = ""):
        self.timestamp = time.time()
        self.position = position.copy()
        self.mode = mode
        self.label = label  # 航点标签，如 "起飞点"、"转弯点"、"降落点"
        self.index = 0  # 航点序号


class SimulationDroneController(BaseDroneController):
    def __init__(self, config: Optional[ConfigManager] = None, simulation_mode: bool = True):
        super().__init__(config)
        self.simulation_mode: bool = simulation_mode
        self.master = None

        # 航点管理
        self.waypoints: List[WaypointMarker] = []
        self.current_waypoint_index = 0
        self.waypoint_labels = ["起飞", "左转", "右转", "上升", "下降", "悬停", "降落"]

        # 轨迹录制功能
        self.is_recording: bool = False
        self.recorded_trajectory: List[Dict] = []
        self.record_start_time: float = 0.0

        # 轨迹回放功能
        self.is_replaying: bool = False
        self.replay_trajectory: List[Dict] = []
        self.replay_index: int = 0
        self.replay_speed: float = 1.0
        self.replay_start_time: float = 0.0

    def disconnect(self) -> bool:
        """断开连接"""
        if self.master:
            try:
                self.master.close()
            except:
                pass
        self.connected = False
        self.logger.info("仿真无人机: 已断开连接")
        return True

    def takeoff(self, altitude: float = None) -> bool:
        """起飞"""
        if altitude is None:
            altitude = self.config.get("drone.takeoff_altitude", 2.0) if self.config else 2.0
        return self._takeoff_simulation(altitude)

    def _connect_to_real_drone(self):
        try:
            from pymavlink import mavutil
            self.master = mavutil.mavlink_connection('udp:127.0.0.1:14540')
            self.master.wait_heartbeat()
            self.logger.info("成功连接到无人机仿真器！")
            self.connected = True
            self.simulation_mode = False
        except ImportError as e:
            self.logger.warning(f"未安装pymavlink库，自动切换到仿真模式: {e}")
            self.simulation_mode = True
        except Exception as e:
            self.logger.warning(f"连接无人机失败: {e}")
            self.simulation_mode = True

    def connect(self) -> bool:
        """连接无人机"""
        if not self.simulation_mode:
            self._connect_to_real_drone()
        else:
            self.connected = True
            self.logger.info("仿真无人机: 已连接")
        return self.connected

    def _simulate_command(self, command, intensity):
        """仿真模式命令处理"""
        intensity = max(0.1, min(1.0, intensity))

        if command == "takeoff":
            self._takeoff_simulation(intensity)
        elif command == "land":
            self._land_simulation(intensity)
        elif command == "forward":
            self._move_simulation('forward', intensity)
        elif command == "backward":
            self._move_simulation('backward', intensity)
        elif command == "up":
            self._move_simulation('up', intensity)
        elif command == "down":
            self._move_simulation('down', intensity)
        elif command == "left":
            self._move_simulation('left', intensity)
        elif command == "right":
            self._move_simulation('right', intensity)
        elif command == "turn_left":
            self._rotate_simulation('yaw_left', intensity)
        elif command == "turn_right":
            self._rotate_simulation('yaw_right', intensity)
        elif command == "hover":
            self._hover_simulation()
        elif command == "stop":
            self._stop_simulation()
        else:
            self._simulate_takeoff(altitude)

        return True

    def _simulate_takeoff(self, altitude: float):
        self.logger.info(f"仿真: 无人机起飞到 {altitude} 米高度")
        self.state['armed'] = True
        self.state['mode'] = 'TAKEOFF'
        self.state['velocity'][1] = 1.0

    def land(self) -> bool:
        if not self.connected:
            self.logger.error("未连接")
            return False

        if not self.simulation_mode and self.master:
            self._send_mavlink_land()
        else:
            self._simulate_land()

        return True

    def _simulate_land(self):
        self.logger.info("仿真: 无人机降落")
        if self.state['armed']:
            self.state['mode'] = 'LAND'
            self.state['velocity'][1] = -1.0

    def hover(self):
        if not self.connected:
            return

        if not self.simulation_mode and self.master:
            self._send_mavlink_hover()
        else:
            self._simulate_hover()

    def _simulate_hover(self):
        if self.state['armed']:
            self.state['velocity'] = np.array([0.0, 0.0, 0.0])
            self.state['mode'] = 'HOVER'
            self.logger.info("仿真: 无人机悬停")

    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float = 0.5):
        if not self.connected:
            return

        if not self.simulation_mode and self.master:
            self._send_mavlink_velocity(vx, vy, vz)
        else:
            self._simulate_move(vx, vy, vz)

    def _simulate_move(self, vx: float, vy: float, vz: float):
        if not self.state['armed']:
            self.logger.warning("警告: 无人机未解锁，无法移动")
            return

        self.state['velocity'] = np.array([vx, vy, vz])

        if vx > 0:
            self.state['mode'] = 'FORWARD'
        elif vx < 0:
            self.state['mode'] = 'BACKWARD'
        elif vy > 0:
            self.state['mode'] = 'RIGHT'
        elif vy < 0:
            self.state['mode'] = 'LEFT'
        elif vz < 0:
            self.state['mode'] = 'UP'
        elif vz > 0:
            self.state['mode'] = 'DOWN'

        self.logger.info(f"仿真: 无人机移动，速度: ({vx:.2f}, {vy:.2f}, {vz:.2f})")

    def _rotate_simulation(self, direction, intensity):
        """仿真旋转"""
        if not self.state['armed']:
            print("[ERROR] 警告：无人机未解锁，无法旋转")
            print("   请先做出'张开手掌'手势进行起飞解锁")
            return

        rotation_speed = 30.0 * intensity  # 度/秒

        if direction == 'yaw_left':
            self.state['orientation'][2] += rotation_speed  # yaw左转
            self.state['mode'] = 'YAW_LEFT'
        elif direction == 'yaw_right':
            self.state['orientation'][2] -= rotation_speed  # yaw右转
            self.state['mode'] = 'YAW_RIGHT'

        # 保持位置不变
        self.state['velocity'] = np.array([0.0, 0.0, 0.0])

        print(f"[OK] 仿真：无人机{direction}，速度{rotation_speed:.1f}度/秒")

    def _hover_simulation(self):
        """仿真悬停"""
        self.state['velocity'] = np.array([0.0, 0.0, 0.0])
        self.state['mode'] = 'HOVER'
        self.logger.info("仿真: 无人机悬停")

    def _emergency_land(self):
        self.logger.warning("警告: 电池耗尽，紧急降落！")
        self._simulate_land()

    def _send_mavlink_takeoff(self):
        try:
            from pymavlink import mavutil
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
            )
            self._set_mavlink_mode("TAKEOFF")
            self.logger.info("真实无人机: 已解锁并起飞")
        except Exception as e:
            self.logger.error(f"起飞失败: {e}")

    def _send_mavlink_land(self):
        try:
            self._set_mavlink_mode("LAND")
            self.logger.info("真实无人机: 开始降落")
        except Exception as e:
            self.logger.error(f"降落失败: {e}")

    def _send_mavlink_hover(self):
        try:
            self._set_mavlink_mode("LOITER")
        except Exception as e:
            self.logger.error(f"设置悬停模式失败: {e}")

    def _send_mavlink_velocity(self, vx: float, vy: float, vz: float):
        pass

    def update_physics(self, dt: float):
        """更新物理仿真"""
        if not self.state['armed']:
            return

        # 更新位置
        self.state['position'] += self.state['velocity'] * dt

        # 处理起飞高度
        if self.state['mode'] == 'TAKEOFF':
            target_height = self.config.get("drone.takeoff_altitude", 2.0) if self.config else 2.0
            if self.state['position'][1] >= target_height:
                self.state['velocity'][1] = 0.0
                self.state['mode'] = 'HOVER'
                self.logger.info("仿真: 无人机已达到目标高度，开始悬停")

        # 处理降落
        elif self.state['mode'] == 'LAND' and self.state['position'][1] <= 0.1:
            self.state['position'][1] = 0.0
            self.state['velocity'][1] = 0.0
            self.state['armed'] = False
            self.state['mode'] = 'LANDED'
            self.logger.info("仿真: 无人机已降落")

        # 边界检查
        if self.state['position'][1] < 0:
            self.state['position'][1] = 0
            self.state['velocity'][1] = max(self.state['velocity'][1], 0)

        max_altitude = self.config.get("drone.max_altitude", 10.0) if self.config else 10.0
        if self.state['position'][1] > max_altitude:
            self.state['position'][1] = max_altitude
            self.state['velocity'][1] = min(self.state['velocity'][1], 0)

        self._record_trajectory()

        # 如果在录制模式，也录制详细数据
        if self.is_recording:
            self._record_point()

        # 电池消耗
        battery_drain = 0.001 * dt * 60
        if np.linalg.norm(self.state['velocity']) > 0.1:
            battery_drain *= 1.5
        self.state['battery'] -= battery_drain

        if self.state['battery'] < 0:
            self.state['battery'] = 0
            self._emergency_land()

    def add_waypoint(self, label: str = "") -> 'WaypointMarker':
        """添加航点标记"""
        waypoint = WaypointMarker(
            position=self.state['position'],
            mode=self.state['mode'],
            label=label
        )
        waypoint.index = len(self.waypoints)
        self.waypoints.append(waypoint)
        self.logger.info(f"[航点] 已标记 #{waypoint.index}: {label} @ 位置({waypoint.position[0]:.1f}, {waypoint.position[1]:.1f}, {waypoint.position[2]:.1f})")
        return waypoint

    def add_waypoint_by_index(self, label_index: int) -> 'WaypointMarker':
        """通过标签索引添加航点"""
        if 0 <= label_index < len(self.waypoint_labels):
            return self.add_waypoint(self.waypoint_labels[label_index])
        return self.add_waypoint(f"航点{len(self.waypoints)}")

    def clear_waypoints(self):
        """清除所有航点"""
        count = len(self.waypoints)
        self.waypoints.clear()
        self.current_waypoint_index = 0
        self.logger.info(f"[航点] 已清除 {count} 个航点")

    def get_waypoints(self) -> List[Dict]:
        """获取航点列表（用于保存）"""
        return [
            {
                'index': wp.index,
                'timestamp': wp.timestamp,
                'position': wp.position.tolist() if isinstance(wp.position, np.ndarray) else wp.position,
                'mode': wp.mode,
                'label': wp.label
            }
            for wp in self.waypoints
        ]

    def get_waypoints_for_display(self) -> List['WaypointMarker']:
        """获取航点列表（用于3D显示）"""
        return self.waypoints

    def load_waypoints_from_dict(self, waypoints_data: List[Dict]):
        """从数据加载航点"""
        self.waypoints.clear()
        for wp_data in waypoints_data:
            wp = WaypointMarker(
                position=np.array(wp_data['position']),
                mode=wp_data['mode'],
                label=wp_data.get('label', '')
            )
            wp.index = wp_data['index']
            wp.timestamp = wp_data.get('timestamp', 0)
            self.waypoints.append(wp)
        self.current_waypoint_index = 0
        self.logger.info(f"[航点] 已加载 {len(self.waypoints)} 个航点")

    # ========== 轨迹录制和回放功能 ==========

    def start_recording(self):
        """开始录制轨迹"""
        if self.is_recording:
            self.logger.warning("[录制] 已经在录制中")
            return

        self.is_recording = True
        self.recorded_trajectory.clear()
        self.record_start_time = time.time()
        self.logger.info("[录制] 开始录制轨迹")

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            self.logger.warning("[录制] 未在录制中")
            return

        self.is_recording = False
        duration = time.time() - self.record_start_time
        self.logger.info(f"[录制] 停止录制，共记录 {len(self.recorded_trajectory)} 个点，耗时 {duration:.1f}秒")

    def _record_point(self):
        """录制当前点"""
        if not self.is_recording:
            return

        point = {
            'timestamp': time.time() - self.record_start_time,
            'position': self.state['position'].copy(),
            'velocity': self.state['velocity'].copy(),
            'orientation': self.state['orientation'].copy(),
            'mode': self.state['mode'],
            'battery': self.state['battery']
        }
        self.recorded_trajectory.append(point)

    def save_trajectory_to_file(self, filename: str = None):
        """保存轨迹到文件"""
        if not self.recorded_trajectory:
            self.logger.warning("[保存] 没有可保存的轨迹数据")
            return False

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"

        try:
            import json
            # 准备可序列化的数据
            data_to_save = {
                'trajectory': [
                    {
                        'timestamp': p['timestamp'],
                        'position': p['position'].tolist() if hasattr(p['position'], 'tolist') else list(p['position']),
                        'velocity': p['velocity'].tolist() if hasattr(p['velocity'], 'tolist') else list(p['velocity']),
                        'orientation': p['orientation'].tolist() if hasattr(p['orientation'], 'tolist') else list(p['orientation']),
                        'mode': p['mode'],
                        'battery': p['battery']
                    }
                    for p in self.recorded_trajectory
                ],
                'waypoints': self.get_waypoints(),
                'record_duration': time.time() - self.record_start_time if self.is_recording else 0
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

            self.logger.info(f"[保存] 轨迹已保存到: {filename}")
            return True

        except Exception as e:
            self.logger.error(f"[保存] 保存轨迹失败: {e}")
            return False

    def load_trajectory_from_file(self, filename: str) -> bool:
        """从文件加载轨迹"""
        try:
            import json

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.replay_trajectory = data.get('trajectory', [])

            # 加载航点（如果有）
            waypoints_data = data.get('waypoints', [])
            if waypoints_data:
                self.load_waypoints_from_dict(waypoints_data)

            self.logger.info(f"[加载] 已加载轨迹: {len(self.replay_trajectory)} 个点")
            return True

        except Exception as e:
            self.logger.error(f"[加载] 加载轨迹失败: {e}")
            return False

    def list_saved_trajectories(self) -> List[str]:
        """列出已保存的轨迹文件"""
        import glob
        import os

        pattern = "trajectory_*.json"
        files = glob.glob(pattern)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files

    def start_replay(self, speed: float = 1.0):
        """开始回放轨迹"""
        if not self.replay_trajectory:
            self.logger.warning("[回放] 没有可回放的轨迹")
            return

        if self.is_replaying:
            self.logger.warning("[回放] 已经在回放中")
            return

        self.is_replaying = True
        self.replay_index = 0
        self.replay_speed = speed
        self.replay_start_time = time.time()
        self.logger.info(f"[回放] 开始回放，速度: {speed}x")

    def update_replay(self) -> Optional[Tuple[np.ndarray, str]]:
        """更新回放状态"""
        if not self.is_replaying or not self.replay_trajectory:
            return None

        # 计算当前应该显示的点的索引
        elapsed = (time.time() - self.replay_start_time) * self.replay_speed
        target_time = elapsed

        # 找到对应时间的点
        while self.replay_index < len(self.replay_trajectory):
            point = self.replay_trajectory[self.replay_index]
            if point['timestamp'] <= target_time:
                self.replay_index += 1
            else:
                break

        # 检查是否回放结束
        if self.replay_index >= len(self.replay_trajectory):
            self.is_replaying = False
            self.logger.info("[回放] 回放结束")
            return None

        # 返回当前位置和模式
        point = self.replay_trajectory[self.replay_index]
        position = np.array(point['position'])
        mode = point['mode']

        return (position, mode)

    def _set_mavlink_mode(self, mode: str):
        try:
            from pymavlink import mavutil
            mode_id = self.master.mode_mapping()[mode]
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id
            )
        except Exception as e:
            self.logger.error(f"设置模式失败: {e}")


# 向后兼容的别名
DroneController = SimulationDroneController
