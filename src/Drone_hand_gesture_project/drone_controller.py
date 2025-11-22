import time
from pymavlink import mavutil


class DroneController:
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.connected = False
        self.master = None

        if not simulation_mode:
            self._connect_to_drone()

    def _connect_to_drone(self):
        try:
            self.master = mavutil.mavlink_connection('udp:127.0.0.1:14540')
            self.master.wait_heartbeat()
            print("成功连接到无人机仿真器！")
            self.connected = True
        except Exception as e:
            print(f"连接无人机失败: {e}")
            self.simulation_mode = True

    def send_command(self, command):
        print(f"执行命令: {command}")

        if not self.simulation_mode and self.connected:
            self._send_mavlink_command(command)
        else:
            self._simulate_command(command)

    def _send_mavlink_command(self, command):
        try:
            if command == "takeoff":
                self._arm_and_takeoff()
            elif command == "land":
                self._land()
            elif command == "hover":
                self._set_mode("LOITER")
            elif command == "stop":
                self._set_mode("HOLD")
        except Exception as e:
            print(f"命令执行错误: {e}")

    def _simulate_command(self, command):
        commands = {
            "takeoff": "无人机起飞",
            "land": "无人机降落",
            "up": "无人机上升",
            "down": "无人机下降",
            "forward": "无人机前进",
            "backward": "无人机后退",
            "hover": "无人机悬停",
            "stop": "无人机停止",
            "none": "等待指令..."
        }
        message = commands.get(command, f"未知命令: {command}")
        print(message)

    def _arm_and_takeoff(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        self._set_mode("TAKEOFF")
        print("无人机已解锁并起飞")

    def _land(self):
        self._set_mode("LAND")
        print("无人机开始降落")

    def _set_mode(self, mode):
        mode_id = self.master.mode_mapping()[mode]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)