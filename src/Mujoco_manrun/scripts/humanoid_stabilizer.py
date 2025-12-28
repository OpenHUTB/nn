import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import mujoco
import numpy as np
import time
from cpg_oscillator import CPGOscillator
from sensor_simulator import SensorSimulator
from utils import quat_to_euler_xyz, clip_value
from keyboard_handler import KeyboardInputHandler  # 新增导入

class HumanoidStabilizer:
    def __init__(self, model_path):
        # 加载Mujoco模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化CPG振荡器（步态生成）
        self.cpg = CPGOscillator()
        
        # 初始化传感器模拟
        self.sensor = SensorSimulator(self.model, self.data)
        
        # 控制参数
        self.velocity = 0.0  # 前进速度
        self.turn_rate = 0.0  # 转向速率
        self.gait_mode = "NORMAL"  # 默认步态
        self.turn_angle = 0.0  # 新增转向角度参数
        self.walk_speed = 0.5  # 新增行走速度参数
        self.state = "STAND"  # 新增状态参数
        self.enable_sensor_simulation = True  # 传感器模拟开关
        
        # ROS相关初始化（兼容处理）
        self.has_ros = False
        self.ros_handler = None

    def _update_control(self):
        """更新机器人控制指令（融合键盘/ROS输入）"""
        # 1. 从CPG生成步态关节指令（修复原代码中cpg.generate_gait不存在的问题）
        # 这里使用基础的CPG输出作为关节目标（实际需根据模型关节结构调整）
        dt = 0.005  # 与仿真步长一致
        cpg_output = self.cpg.update(dt, speed_factor=self.walk_speed, turn_factor=self.turn_rate)
        
        # 示例关节映射（需根据实际模型关节名称调整）
        joint_targets = {
            "left_hip": cpg_output,
            "right_hip": -cpg_output,
            "left_knee": -cpg_output * 0.8,
            "right_knee": cpg_output * 0.8
        }
        
        # 2. 设置关节控制指令
        for joint_name, target_pos in joint_targets.items():
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.data.ctrl[joint_id] = target_pos
            except:
                continue
        
        # 3. 限制控制指令范围
        self.data.ctrl = clip_value(self.data.ctrl, -1.5, 1.5)

    def set_gait_mode(self, mode):
        """设置步态模式（扩展支持更多模式）"""
        valid_modes = ["NORMAL", "SLOW", "FAST", "TROT", "STEP_IN_PLACE"]
        self.gait_mode = mode if mode in valid_modes else "NORMAL"
        # 根据步态模式调整CPG参数
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
        """设置前进速度（-1.0 ~ 1.0）"""
        self.velocity = clip_value(v, -1.0, 1.0, "速度")

    def set_turn_rate(self, tr):
        """设置转向速率（-1.0 ~ 1.0）"""
        self.turn_rate = clip_value(tr, -1.0, 1.0, "转向速率")

    def set_turn_angle(self, ta):
        """设置转向角度"""
        self.turn_angle = clip_value(ta, -np.pi/4, np.pi/4, "转向角度")

    def set_walk_speed(self, ws):
        """设置行走速度"""
        self.walk_speed = clip_value(ws, 0.1, 1.0, "行走速度")

    def set_state(self, state):
        """设置机器人状态"""
        valid_states = ["STAND", "WALK", "STOP", "EMERGENCY"]
        self.state = state if state in valid_states else "STAND"
        if self.state == "EMERGENCY":
            self.data.ctrl[:] = 0  # 紧急停止时归零控制

    def print_sensor_data(self):
        """打印传感器数据"""
        self.sensor.print_sensor_data()

    def simulate(self):
        """主仿真循环：控制+可视化"""
        # 启动键盘监听线程
        keyboard_handler = KeyboardInputHandler(self)
        keyboard_handler.start()

        # ========== 兼容新旧Mujoco版本的Viewer启动 ==========
        try:
            # 新版Mujoco（2.3.0+）
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            use_new_viewer = True
            viewer_running = True  # 新增状态标记
        except (ImportError, AttributeError):
            # 旧版Mujoco（2.1.0-）或mujoco-py
            import mujoco.glfw as glfw
            glfw.init()
            window = glfw.create_window(1280, 720, "Humanoid Simulation", None, None)
            glfw.make_context_current(window)
            glfw.swap_interval(1)
            viewer = mujoco.MjViewer(window)
            viewer.set_model(self.model)
            use_new_viewer = False

        # 主仿真循环
        print("仿真启动！按H查看控制帮助")
        while True:
            # 退出条件（修复新版Viewer判断）
            if use_new_viewer:
                try:
                    viewer.sync()  # 尝试同步，失败则表示窗口已关闭
                except:
                    viewer_running = False
                if not viewer_running:
                    break
            else:
                if glfw.window_should_close(window):
                    break

            # 状态机控制
            if self.state == "STOP":
                self.velocity = 0
                self.turn_rate = 0

            # 2. 更新控制指令（融合ROS/键盘）
            self._update_control()

            # 3. 更新传感器数据
            self.sensor.get_sensor_data(self.gait_mode)

            # 4. 执行一步仿真
            mujoco.mj_step(self.model, self.data)

            # 5. 更新可视化
            if not use_new_viewer:
                viewer.render()
                glfw.swap_buffers(window)
                glfw.poll_events()

            # 6. 控制仿真频率（200Hz）
            time.sleep(0.005)

        # 清理资源
        keyboard_handler.running = False  # 停止键盘线程
        keyboard_handler.join()
        if not use_new_viewer:
            glfw.destroy_window(window)
            glfw.terminate()
        print("仿真结束！")

# 测试代码（单独运行时）
if __name__ == "__main__":
    model_path = os.path.join(SCRIPT_DIR, "../models/humanoid.xml")
    # 检查备用路径
    if not os.path.exists(model_path):
        model_path = os.path.join(SCRIPT_DIR, "models/humanoid.xml")
    stabilizer = HumanoidStabilizer(model_path)
    stabilizer.simulate()
