import time
import math
import numpy as np


class DroneController:
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.connected = False
        self.master = None

        # 无人机状态
        self.state = {
            'position': np.array([0.0, 0.0, 0.0]),  # [x, y, z]
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),  # [roll, pitch, yaw]
            'battery': 100.0,
            'armed': False,
            'mode': 'DISARMED'
        }

        # 控制参数
        self.control_params = {
            'max_speed': 2.0,  # 最大速度 m/s
            'max_altitude': 10.0,  # 最大高度 m
            'takeoff_altitude': 2.0,  # 起飞高度 m
            'hover_threshold': 0.1,  # 悬停阈值
            'battery_drain_rate': 0.01,  # 电池消耗率（每分钟百分比）
        }

        # 轨迹记录
        self.trajectory = []
        self.max_trajectory_points = 500

        # 如果连接真实无人机（保留原有功能）
        if not simulation_mode:
            self._connect_to_real_drone()

    def _connect_to_real_drone(self):
        """连接真实无人机（延迟导入pymavlink）"""
        try:
            # 延迟导入，避免仿真模式下也需要pymavlink
            from pymavlink import mavutil
            self.master = mavutil.mavlink_connection('udp:127.0.0.1:14540')
            self.master.wait_heartbeat()
            print("成功连接到无人机仿真器！")
            self.connected = True
        except ImportError as e:
            print(f"未安装pymavlink库，自动切换到仿真模式: {e}")
            print("请运行: pip install pymavlink")
            self.simulation_mode = True
        except Exception as e:
            print(f"连接无人机失败: {e}")
            self.simulation_mode = True

    def send_command(self, command, intensity=1.0):
        """发送控制命令"""
        print(f"[DEBUG] 收到命令: {command}, 强度: {intensity}")
        print(f"[DEBUG] 当前状态: armed={self.state['armed']}, mode={self.state['mode']}")

        if not self.simulation_mode and self.connected:
            self._send_mavlink_command(command)
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
        elif command == "left":  # 新增
            self._move_simulation('left', intensity)
        elif command == "right":  # 新增
            self._move_simulation('right', intensity)
        elif command == "hover":
            self._hover_simulation()
        elif command == "stop":
            self._stop_simulation()
        else:
            print(f"未知命令: {command}")

    def _takeoff_simulation(self, intensity):
        """仿真起飞"""
        print(f"[DEBUG] 执行起飞命令, 强度: {intensity}")
        print(f"[DEBUG] 当前armed状态: {self.state['armed']}")

        if not self.state['armed']:
            self.state['armed'] = True
            self.state['mode'] = 'TAKEOFF'
            # 设置目标高度和速度
            target_height = self.control_params['takeoff_altitude'] * intensity
            # 设置向上的速度
            self.state['velocity'][1] = 1.0 * intensity  # Y轴向上
            print(f"✅ 仿真：无人机已解锁并起飞到 {target_height:.1f} 米高度")
        else:
            print("⚠️ 无人机已经解锁，无需再次起飞")

    def _land_simulation(self, intensity):
        """仿真降落"""
        if self.state['armed']:
            self.state['mode'] = 'LAND'
            self.state['velocity'][1] = -1.0 * intensity  # 向下速度
            print("仿真：无人机降落")

    def _move_simulation(self, direction, intensity):
        """仿真移动"""
        print(f"[DEBUG] 尝试移动: {direction}, 强度: {intensity}")
        print(f"[DEBUG] 当前armed状态: {self.state['armed']}")

        if not self.state['armed']:
            print("❌ 警告：无人机未解锁，无法移动")
            print("   请先做出'张开手掌'手势进行起飞解锁")
            return

        speed = self.control_params['max_speed'] * intensity

        if direction == 'forward':
            self.state['velocity'][2] = speed  # 向前（Z轴正方向）
            self.state['mode'] = 'FORWARD'
        elif direction == 'backward':
            self.state['velocity'][2] = -speed  # 向后（Z轴负方向）
            self.state['mode'] = 'BACKWARD'
        elif direction == 'up':
            self.state['velocity'][1] = speed  # 向上（Y轴正方向）
            self.state['mode'] = 'UP'
        elif direction == 'down':
            self.state['velocity'][1] = -speed  # 向下（Y轴负方向）
            self.state['mode'] = 'DOWN'
        elif direction == 'left':
            self.state['velocity'][0] = -speed  # 向左（X轴负方向）
            self.state['mode'] = 'LEFT'
        elif direction == 'right':
            self.state['velocity'][0] = speed  # 向右（X轴正方向）
            self.state['mode'] = 'RIGHT'

        print(f"✅ 仿真：无人机{direction}移动，速度{speed:.1f}m/s")

    def _hover_simulation(self):
        """仿真悬停"""
        if self.state['armed']:
            self.state['velocity'] = np.array([0.0, 0.0, 0.0])
            self.state['mode'] = 'HOVER'
            print("仿真：无人机悬停")

    def _stop_simulation(self):
        """仿真停止"""
        self.state['armed'] = False
        self.state['mode'] = 'DISARMED'
        self.state['velocity'] = np.array([0.0, 0.0, 0.0])
        print("仿真：无人机停止")

    def update_physics(self, dt):
        """更新物理状态"""
        if self.state['armed']:
            # 更新位置
            self.state['position'] += self.state['velocity'] * dt

            # 自动起飞完成检测
            if self.state['mode'] == 'TAKEOFF':
                target_height = self.control_params['takeoff_altitude']
                if self.state['position'][1] >= target_height:
                    self.state['velocity'][1] = 0.0  # 停止上升
                    self.state['mode'] = 'HOVER'
                    print("仿真：无人机已达到目标高度，开始悬停")

            # 自动降落检测
            elif self.state['mode'] == 'LAND' and self.state['position'][1] <= 0.1:
                self.state['position'][1] = 0.0
                self.state['velocity'][1] = 0.0
                self.state['armed'] = False
                self.state['mode'] = 'LANDED'
                print("仿真：无人机已降落")

            # 限制高度
            if self.state['position'][1] < 0:
                self.state['position'][1] = 0
                self.state['velocity'][1] = max(self.state['velocity'][1], 0)  # 不允许继续下降

            if self.state['position'][1] > self.control_params['max_altitude']:
                self.state['position'][1] = self.control_params['max_altitude']
                self.state['velocity'][1] = min(self.state['velocity'][1], 0)  # 不允许继续上升

            # 记录轨迹
            self._record_trajectory()

            # 消耗电池
            if self.state['battery'] > 0:
                battery_drain = self.control_params['battery_drain_rate'] * dt * 60
                # 移动时消耗更多电池
                if np.linalg.norm(self.state['velocity']) > 0.1:
                    battery_drain *= 1.5
                self.state['battery'] -= battery_drain

                if self.state['battery'] < 0:
                    self.state['battery'] = 0
                    self._emergency_land()

    def _record_trajectory(self):
        """记录飞行轨迹"""
        self.trajectory.append(tuple(self.state['position']))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)

    def _emergency_land(self):
        """紧急降落"""
        print("警告：电池耗尽，紧急降落！")
        self._land_simulation(1.0)

    def get_state(self):
        """获取当前状态"""
        return {
            'position': self.state['position'].copy(),
            'velocity': self.state['velocity'].copy(),
            'orientation': self.state['orientation'].copy(),
            'battery': self.state['battery'],
            'armed': self.state['armed'],
            'mode': self.state['mode']
        }

    def get_trajectory(self):
        """获取飞行轨迹"""
        return self.trajectory.copy()

    def get_status_string(self):
        """获取状态字符串"""
        pos = self.state['position']
        return (f"模式: {self.state['mode']} | "
                f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                f"电池: {self.state['battery']:.1f}% | "
                f"解锁: {'是' if self.state['armed'] else '否'}")

    def reset(self, position=None, orientation=None):
        """重置无人机状态"""
        if position is not None:
            self.state['position'] = np.array(position)
        else:
            self.state['position'] = np.array([0.0, 0.0, 0.0])

        if orientation is not None:
            self.state['orientation'] = np.array(orientation)
        else:
            self.state['orientation'] = np.array([0.0, 0.0, 0.0])

        self.state['velocity'] = np.array([0.0, 0.0, 0.0])
        self.state['battery'] = 100.0
        self.state['armed'] = False
        self.state['mode'] = 'DISARMED'
        self.trajectory.clear()
        print("仿真：无人机状态已重置")

    # MAVLink相关方法（只有在真实模式下才需要）
    def _send_mavlink_command(self, command):
        """发送MAVLink命令"""
        try:
            from pymavlink import mavutil

            if command == "takeoff":
                self._arm_and_takeoff()
            elif command == "land":
                self._land()
            elif command == "hover":
                self._set_mode("LOITER")
            elif command == "stop":
                self._set_mode("HOLD")
            else:
                print(f"MAVLink模式下不支持的命令: {command}")

        except Exception as e:
            print(f"MAVLink命令执行错误: {e}")

    def _arm_and_takeoff(self):
        """MAVLink解锁并起飞"""
        try:
            from pymavlink import mavutil
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            self._set_mode("TAKEOFF")
            print("真实无人机：已解锁并起飞")
        except Exception as e:
            print(f"起飞失败: {e}")

    def _land(self):
        """MAVLink降落"""
        try:
            self._set_mode("LAND")
            print("真实无人机：开始降落")
        except Exception as e:
            print(f"降落失败: {e}")

    def _set_mode(self, mode):
        """MAVLink设置模式"""
        try:
            from pymavlink import mavutil
            mode_id = self.master.mode_mapping()[mode]
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
        except Exception as e:
            print(f"设置模式失败: {e}")