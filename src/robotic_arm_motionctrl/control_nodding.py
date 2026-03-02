import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
from pynput import keyboard
import os


class KeyboardControlledArm:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.actuator_ids = {}

        for name in self.joint_names:
            act_name = f"motor_{name}"
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self.actuator_ids[name] = act_id
            if act_id == -1:
                print(f"⚠️  执行器 {act_name} 未找到")
            else:
                print(f"✅ {act_name}  ID: {act_id}")

        self.joint_speed = 1.0
        self.control = {j: 0.0 for j in self.joint_names}
        self.auto_nod_enabled = False
        self.running = True

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.daemon = True
        self.listener.start()

        self._print_help()

    def _print_help(self):
        print("\n" + "="*50)
        print("🎮 机械臂键盘控制")
        print("← →         joint1 旋转")
        print("↑ ↓         joint2 摆动")
        print("PgUp/PgDn   joint3 摆动")
        print("A / D       joint4 旋转")
        print("W / S       joint5 摆动")
        print("Q / E       joint6 旋转")
        print("空格        急停")
        print("N           轻柔点头开关")
        print("ESC         退出")
        print("="*50 + "\n")

    def _on_press(self, key):
        try:
            if key == keyboard.Key.left:
                self.control["joint1"] = -self.joint_speed
            elif key == keyboard.Key.right:
                self.control["joint1"] = self.joint_speed

            elif key == keyboard.Key.up:
                self.control["joint2"] = self.joint_speed
            elif key == keyboard.Key.down:
                self.control["joint2"] = -self.joint_speed

            elif key == keyboard.Key.page_up:
                self.control["joint3"] = self.joint_speed
            elif key == keyboard.Key.page_down:
                self.control["joint3"] = -self.joint_speed

            elif key == keyboard.KeyCode.from_char('a'):
                self.control["joint4"] = -self.joint_speed
            elif key == keyboard.KeyCode.from_char('d'):
                self.control["joint4"] = self.joint_speed

            elif key == keyboard.KeyCode.from_char('w'):
                self.control["joint5"] = self.joint_speed
            elif key == keyboard.KeyCode.from_char('s'):
                self.control["joint5"] = -self.joint_speed

            elif key == keyboard.KeyCode.from_char('q'):
                self.control["joint6"] = self.joint_speed
            elif key == keyboard.KeyCode.from_char('e'):
                self.control["joint6"] = -self.joint_speed

            elif key == keyboard.KeyCode.from_char('n'):
                self.auto_nod_enabled = not self.auto_nod_enabled
                print("🔄 轻柔点头：", "开启" if self.auto_nod_enabled else "关闭")

            elif key == keyboard.Key.space:
                self.control = {j: 0.0 for j in self.joint_names}
                self.auto_nod_enabled = False
                print("⏹️  所有关节已停止")

            elif key == keyboard.Key.esc:
                self.running = False

        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            if key in [keyboard.Key.left, keyboard.Key.right]:
                self.control["joint1"] = 0.0
            elif key in [keyboard.Key.up, keyboard.Key.down]:
                self.control["joint2"] = 0.0
            elif key in [keyboard.Key.page_up, keyboard.Key.page_down]:
                self.control["joint3"] = 0.0
            elif key == keyboard.KeyCode.from_char('a') or key == keyboard.KeyCode.from_char('d'):
                self.control["joint4"] = 0.0
            elif key == keyboard.KeyCode.from_char('w') or key == keyboard.KeyCode.from_char('s'):
                self.control["joint5"] = 0.0
            elif key == keyboard.KeyCode.from_char('q') or key == keyboard.KeyCode.from_char('e'):
                self.control["joint6"] = 0.0
        except AttributeError:
            pass

    def auto_nod(self):
        """✅ 超轻柔轻微点头（已调小幅度）"""
        while self.running:
            if self.auto_nod_enabled:
                # 轻轻向下
                self.control["joint2"] = 0.21
                time.sleep(1.0)
                self.control["joint2"] = 0
                time.sleep(0.5)

                # 轻轻向上
                self.control["joint2"] = -0.21
                time.sleep(1.0)
                self.control["joint2"] = 0
                time.sleep(1.33)
            else:
                time.sleep(0.05)

    def _update_ctrl(self):
        for jname in self.joint_names:
            aid = self.actuator_ids[jname]
            if aid == -1:
                continue
            val = self.control[jname] * 10
            val = np.clip(val, -10, 10)
            self.data.ctrl[aid] = val

    def run(self):
        threading.Thread(target=self.auto_nod, daemon=True).start()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 1.9
            viewer.cam.azimuth = 50
            viewer.cam.elevation = -28
            viewer.cam.lookat = [0.0, 0.0, 0.2]

            print("✅ 仿真已启动")

            while self.running and viewer.is_running():
                step_start = time.time()
                self._update_ctrl()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                elapsed = time.time() - step_start
                if elapsed < self.model.opt.timestep:
                    time.sleep(self.model.opt.timestep - elapsed)

        print("👋 程序已退出")


def main():
    arm = KeyboardControlledArm("arm6dof_final.xml")
    arm.run()


if __name__ == "__main__":
    main()