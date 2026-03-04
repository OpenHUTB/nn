import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
from pynput import keyboard
from typing import Dict
import os


class KeyboardControlledArm:
    def __init__(self, model_path: str):
        """初始化键盘控制机械臂（修复版）"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 关节配置（和你的XML保持一致）
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.actuator_ids = {}
        for name in self.joint_names:
            act_name = f"motor_{name}"  # 执行器命名规则：motor_joint1...
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self.actuator_ids[name] = act_id
            print(f"✅ {act_name} ID: {act_id}")  # 调试：确认执行器ID有效（应为0-5，不是-1）

        # 关节角度限制
        self.joint_limits = {
            "joint1": (-np.pi, np.pi),
            "joint2": (-2.0, 2.0),
            "joint3": (-2.5, 0.5),
            "joint4": (-2.0, 2.0),
            "joint5": (-1.8, 1.8),
            "joint6": (-1.8, 1.8)
        }

        # 控制参数
        self.joint_speed = 1.2  # 关节运动速度（调大更易感知）
        self.control_state = {name: 0.0 for name in self.joint_names}  # 0=停止, ±1=方向
        self.running = True

        # 启动键盘监听
        self.key_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.key_listener.start()
        self.print_controls()

    def print_controls(self):
        """打印控制说明"""
        print("\n" + "="*50)
        print("🎮 机械臂键盘控制说明（请先点击MuJoCo窗口获取焦点！）")
        print("← → 方向键：joint1 左右旋转")
        print("↑ ↓ 方向键：joint2 上下摆动")
        print("PageUp/PageDown：joint3 上下摆动")
        print("A/D：joint4 左右旋转")
        print("W/S：joint5 上下摆动")
        print("Q/E：joint6 上下摆动")
        print("空格键：紧急停止所有关节")
        print("ESC：退出程序")
        print("="*50 + "\n")

    def on_key_press(self, key):
        """按键按下处理（带调试打印）"""
        print(f"🔍 捕获按键: {key}")  # 调试：确认按键被程序收到

        try:
            # 方向键控制
            if key == keyboard.Key.left:
                self.control_state["joint1"] = -self.joint_speed
            elif key == keyboard.Key.right:
                self.control_state["joint1"] = self.joint_speed
            elif key == keyboard.Key.up:
                self.control_state["joint2"] = self.joint_speed
            elif key == keyboard.Key.down:
                self.control_state["joint2"] = -self.joint_speed
            elif key == keyboard.Key.page_up:
                self.control_state["joint3"] = self.joint_speed
            elif key == keyboard.Key.page_down:
                self.control_state["joint3"] = -self.joint_speed

            # 字母键控制（确保英文输入法）
            elif key == keyboard.KeyCode.from_char('a'):
                self.control_state["joint4"] = -self.joint_speed
            elif key == keyboard.KeyCode.from_char('d'):
                self.control_state["joint4"] = self.joint_speed
            elif key == keyboard.KeyCode.from_char('w'):
                self.control_state["joint5"] = self.joint_speed
            elif key == keyboard.KeyCode.from_char('s'):
                self.control_state["joint5"] = -self.joint_speed
            elif key == keyboard.KeyCode.from_char('q'):
                self.control_state["joint6"] = self.joint_speed
            elif key == keyboard.KeyCode.from_char('e'):
                self.control_state["joint6"] = -self.joint_speed

            # 功能键
            elif key == keyboard.Key.space:
                self.control_state = {name: 0.0 for name in self.joint_names}
                print("⏹️  所有关节已停止！")
            elif key == keyboard.Key.esc:
                self.running = False
                print("🛑 收到退出指令...")

        except AttributeError:
            pass  # 忽略未知按键

    def on_key_release(self, key):
        """按键松开：停止对应关节运动"""
        try:
            if key in [keyboard.Key.left, keyboard.Key.right]:
                self.control_state["joint1"] = 0.0
            elif key in [keyboard.Key.up, keyboard.Key.down]:
                self.control_state["joint2"] = 0.0
            elif key in [keyboard.Key.page_up, keyboard.Key.page_down]:
                self.control_state["joint3"] = 0.0
            elif key in [keyboard.KeyCode.from_char('a'), keyboard.KeyCode.from_char('d')]:
                self.control_state["joint4"] = 0.0
            elif key in [keyboard.KeyCode.from_char('w'), keyboard.KeyCode.from_char('s')]:
                self.control_state["joint5"] = 0.0
            elif key in [keyboard.KeyCode.from_char('q'), keyboard.KeyCode.from_char('e')]:
                self.control_state["joint6"] = 0.0
        except AttributeError:
            pass

    def update_control(self):
        """更新关节控制信号（简化版，更稳定）"""
        for joint_name in self.joint_names:
            act_id = self.actuator_ids[joint_name]
            if act_id == -1:
                print(f"❌ 警告：{joint_name} 执行器ID无效，请检查XML！")
                continue

            # 直接输出速度控制信号（更直观，避免角度计算问题）
            ctrl_signal = self.control_state[joint_name] * 10  # 放大信号
            ctrl_signal = np.clip(ctrl_signal, -10, 10)  # 限幅
            self.data.ctrl[act_id] = ctrl_signal

    def run(self):
        """运行仿真主循环"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 调整相机视角
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -25
            viewer.cam.lookat = [0.1, 0.0, 0.2]

            print("✅ 仿真启动！请点击MuJoCo窗口后开始控制...")

            while self.running and viewer.is_running():
                step_start = time.time()

                # 更新控制
                self.update_control()

                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 控制仿真帧率
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        # 清理资源
        self.key_listener.stop()
        print("👋 程序已退出")


def main():
    """主函数"""
    # 先安装依赖（如果没装）：pip install pynput numpy mujoco
    model_path = "arm6dof_final.xml"  # 你的模型文件名

    try:
        arm = KeyboardControlledArm(model_path)
        arm.run()
    except FileNotFoundError as e:
        print(f"❌ 错误：{e}，请确认XML文件在当前目录！")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()