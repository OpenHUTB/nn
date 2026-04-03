import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import ctypes

# ===================== 键盘控制（Windows） =====================
class Keyboard:
    def __init__(self):
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.running = True
        self.target = 0.0
        self.step = 0.05

    def down(self, key):
        return self.user32.GetAsyncKeyState(key) & 0x8000 != 0

    def loop(self):
        while self.running:
            if self.down(0x51): self.running = False  # Q
            if self.down(0x41): self.target -= self.step  # A
            if self.down(0x44): self.target += self.step  # D
            if self.down(0x53): self.target = 0.0         # S
            self.target = np.clip(self.target, -1.8*np.pi, 1.8*np.pi)
            time.sleep(0.02)

# ===================== PID 控制器（带抗饱和） =====================
class PID:
    def __init__(self, kp, ki, kd, max_out=10, max_i=5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.max_i = max_i
        self.integral = 0.0
        self.last_err = 0.0

    def step(self, e, dt):
        self.integral = np.clip(self.integral + e*dt, -self.max_i, self.max_i)
        der = (e - self.last_err)/dt
        out = self.kp*e + self.ki*self.integral + self.kd*der
        out = np.clip(out, -self.max_out, self.max_out)
        self.last_err = e
        return out

# ===================== 主程序 =====================
def main():
    XML_PATH = "arm6dof_final.xml"
    JOINT = 1
    DT = 0.01

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    nu = model.nu

    kb = Keyboard()
    pid = PID(kp=35, ki=2.0, kd=3.0)

    th = threading.Thread(target=kb.loop, daemon=True)
    th.start()

    print("✅ 单关节高级位置伺服（工程版）")
    print("A: 左   D: 右   S: 归零   Q: 退出")
    print("👉 关节2 大幅度闭环运动")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and kb.running:
            q_current = data.qpos[JOINT]
            q_target = kb.target

            error = q_target - q_current
            tau = pid.step(error, DT)

            ctrl = np.zeros(nu)
            ctrl[JOINT] = tau
            data.ctrl[:] = ctrl

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(DT)

    print("✅ 已退出")

if __name__ == "__main__":
    main()