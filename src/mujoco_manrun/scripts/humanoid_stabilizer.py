import sys
import os
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import mujoco
import numpy as np
import time
from cpg_oscillator import CPGOscillator
from sensor_simulator import SensorSimulator
from utils import quat_to_euler_xyz, clip_value

# ===================== 键盘输入 =====================
class KeyboardInputHandler(threading.Thread):
    def __init__(self, stabilizer):
        super().__init__(daemon=True)
        self.stabilizer = stabilizer
        self.running = True

    def run(self):
        print("\n===== 控制指令说明 =====")
        print("w: 开始行走 | s: 停止行走 | e: 紧急停止 | r: 恢复站立")
        print("a: 左转 | d: 右转 | 空格: 原地转向 | z: 减速 | x: 加速")
        print("1: 慢走 | 2: 正常走 | 3: 小跑 | 4: 原地踏步")
        print("========================\n")
        while self.running:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        self._handle_key(key)
                else:
                    import select
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1).lower()
                        self._handle_key(key)
                time.sleep(0.01)
            except:
                continue

    def _handle_key(self, key):
        if key == 'w':
            current_gait = self.stabilizer.gait_mode
            self.stabilizer.set_state("WALK")
            self.stabilizer.set_gait_mode(current_gait)
            print(f"[指令] 切换为行走状态 | 当前步态: {current_gait}")
        elif key == 's':
            self.stabilizer.set_state("STOP")
            print("[指令] 切换为停止状态")
        elif key == 'e':
            self.stabilizer.set_state("EMERGENCY")
            print("[指令] 触发紧急停止")
        elif key == 'r':
            self.stabilizer.set_state("STAND")
            print("[指令] 恢复站立姿态")
        elif key == 'a':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle + 0.05)
            print(f"[指令] 左转 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")
        elif key == 'd':
            self.stabilizer.set_turn_angle(self.stabilizer.turn_angle - 0.05)
            print(f"[指令] 右转 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")
        elif key == ' ':
            self.stabilizer.set_turn_angle(0.2 if self.stabilizer.turn_angle <= 0 else -0.2)
            print(f"[指令] 原地转向 | 当前转向角度: {self.stabilizer.turn_angle:.2f}rad")
        elif key == 'z':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed - 0.1)
            print(f"[指令] 减速 | 当前速度: {self.stabilizer.walk_speed:.2f}")
        elif key == 'x':
            self.stabilizer.set_walk_speed(self.stabilizer.walk_speed + 0.1)
            print(f"[指令] 加速 | 当前速度: {self.stabilizer.walk_speed:.2f}")
        elif key == 'm':
            self.stabilizer.enable_sensor_simulation = not self.stabilizer.enable_sensor_simulation
            print(f"[指令] 传感器模拟{'开启' if self.stabilizer.enable_sensor_simulation else '关闭'}")
        elif key == 'p':
            self.stabilizer.print_sensor_data()
        elif key == '1':
            self.stabilizer.set_gait_mode("SLOW")
            print(f"[指令] 切换为慢走模式")
        elif key == '2':
            self.stabilizer.set_gait_mode("NORMAL")
            print(f"[指令] 切换为正常走模式")
        elif key == '3':
            self.stabilizer.set_gait_mode("TROT")
            print(f"[指令] 切换为小跑模式")
        elif key == '4':
            self.stabilizer.set_gait_mode("STEP_IN_PLACE")
            print(f"[指令] 切换为原地踏步模式")

# ===================== 以下完全不动 =====================

class HumanoidStabilizer:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.cpg = CPGOscillator()
        self.sensor = SensorSimulator(self.model, self.data)

        self.velocity = 0.0
        self.turn_rate = 0.0
        self.gait_mode = "NORMAL"
        self.turn_angle = 0.0
        self.walk_speed = 0.5
        self.state = "STAND"
        self.enable_sensor_simulation = True

        self.has_ros = False
        self.ros_handler = None

    def _update_control(self):
        dt = 0.005
        cpg_output = self.cpg.update(dt, speed_factor=self.walk_speed, turn_factor=self.turn_rate)

        joint_targets = {
            "left_hip": cpg_output,
            "right_hip": -cpg_output,
            "left_knee": -cpg_output * 0.8,
            "right_knee": cpg_output * 0.8
        }

        for joint_name, target_pos in joint_targets.items():
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.data.ctrl[joint_id] = target_pos
            except:
                continue

        self.data.ctrl = clip_value(self.data.ctrl, -1.5, 1.5)

    def set_gait_mode(self, mode):
        valid_modes = ["NORMAL", "SLOW", "FAST", "TROT", "STEP_IN_PLACE"]
        self.gait_mode = mode if mode in valid_modes else "NORMAL"
        gait_params = {
            "SLOW": {"freq": 0.3, "amp": 0.3},
            "NORMAL": {"freq": 0.5, "amp": 0.4},
            "FAST": {"freq": 0.7, "amp": 0.5},
            "TROT": {"freq": 0.8, "amp": 0.6},
            "STEP_IN_PLACE": {"freq": 0.4, "amp": 0.3}
        }
        self.cpg.base_freq = gait_params[self.gait_mode]["freq"]
        self.cpg.base_amp = gait_params[self.gait_mode]["amp"]

    def set_velocity(self, v):
        self.velocity = clip_value(v, -1.0, 1.0, "速度")

    def set_turn_rate(self, tr):
        self.turn_rate = clip_value(tr, -1.0, 1.0, "转向速率")

    def set_turn_angle(self, ta):
        self.turn_angle = clip_value(ta, -np.pi / 4, np.pi / 4, "转向角度")

    def set_walk_speed(self, ws):
        self.walk_speed = clip_value(ws, 0.1, 1.0, "行走速度")

    def set_state(self, state):
        valid_states = ["STAND", "WALK", "STOP", "EMERGENCY"]
        self.state = state if state in valid_states else "STAND"
        if self.state == "EMERGENCY":
            self.data.ctrl[:] = 0

    def print_sensor_data(self):
        self.sensor.print_sensor_data()

    def simulate(self):
        keyboard_handler = KeyboardInputHandler(self)
        keyboard_handler.start()

        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            use_new_viewer = True
        except:
            import mujoco.glfw as glfw
            glfw.init()
            window = glfw.create_window(1280, 720, "Humanoid Simulation", None, None)
            glfw.make_context_current(window)
            viewer = mujoco.MjViewer(window)
            viewer.set_model(self.model)
            use_new_viewer = False

        print("仿真启动！按H查看控制帮助")
        while True:
            if use_new_viewer:
                if not viewer.is_running():
                    break
                viewer.sync()
            else:
                if glfw.window_should_close(window):
                    break

            if self.state == "STOP":
                self.velocity = 0
                self.turn_rate = 0

            self._update_control()
            self.sensor.get_sensor_data(self.gait_mode)
            mujoco.mj_step(self.model, self.data)

            if not use_new_viewer:
                viewer.render()
                glfw.swap_buffers(window)
                glfw.poll_events()

            time.sleep(0.005)

        keyboard_handler.running = False
        keyboard_handler.join()
        print("仿真结束！")

if __name__ == "__main__":
    parent_dir = os.path.dirname(SCRIPT_DIR)
    model_path = os.path.join(parent_dir, "models", "humanoid.xml")
    if not os.path.exists(model_path):
        model_path = os.path.join(SCRIPT_DIR, "models", "humanoid.xml")
    stabilizer = HumanoidStabilizer(model_path)
    stabilizer.simulate()