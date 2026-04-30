# -*- coding: utf-8 -*-
import time
import numpy as np
from typing import Optional

from core import BaseDroneController, ConfigManager, Logger


class SimulationDroneController(BaseDroneController):
    def __init__(self, config: Optional[ConfigManager] = None, simulation_mode: bool = True):
        super().__init__(config)
        self.simulation_mode: bool = simulation_mode
        self.master = None

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
        if not self.simulation_mode:
            self._connect_to_real_drone()
        else:
            self._simulate_command(command, intensity)

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
        if self.state['armed']:
            self.state['position'] += self.state['velocity'] * dt

            if self.state['mode'] == 'TAKEOFF':
                target_height = self.config.get("drone.takeoff_altitude", 2.0)
                if self.state['position'][1] >= target_height:
                    self.state['velocity'][1] = 0.0
                    self.state['mode'] = 'HOVER'
                    self.logger.info("仿真: 无人机已达到目标高度，开始悬停")

            elif self.state['mode'] == 'LAND' and self.state['position'][1] <= 0.1:
                self.state['position'][1] = 0.0
                self.state['velocity'][1] = 0.0
                self.state['armed'] = False
                self.state['mode'] = 'LANDED'
                self.logger.info("仿真: 无人机已降落")

            if self.state['position'][1] < 0:
                self.state['position'][1] = 0
                self.state['velocity'][1] = max(self.state['velocity'][1], 0)

            max_altitude = self.config.get("drone.max_altitude", 10.0)
            if self.state['position'][1] > max_altitude:
                self.state['position'][1] = max_altitude
                self.state['velocity'][1] = min(self.state['velocity'][1], 0)

            self._record_trajectory()

            drain_rate = self.config.get("drone.battery_drain_rate", 0.01)
            if self.state['battery'] > 0:
                battery_drain = drain_rate * dt * 60
                if np.linalg.norm(self.state['velocity']) > 0.1:
                    battery_drain *= 1.5
                self.state['battery'] -= battery_drain

                if self.state['battery'] < 0:
                    self.state['battery'] = 0
                    self._emergency_land()

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
