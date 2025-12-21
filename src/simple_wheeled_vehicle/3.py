"""
自动巡航小车
- 恒定速度 0.0005 m/s
- 转向回正（直走）
- R 键仍可复位
"""
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

# ---------------- 键盘 ------------------
KEYS = {keyboard.KeyCode.from_char('r'): False}

def on_press(k):
    if k in KEYS: KEYS[k] = True
def on_release(k):
    if k in KEYS: KEYS[k] = False
keyboard.Listener(on_press=on_press, on_release=on_release).start()

# ---------------- 加载模型 --------------
model = mujoco.MjModel.from_xml_path("wheeled_car.xml")
data = mujoco.MjData(model)

# ---------------- 参数 ------------------
CRUISE_SPEED = 0.0005      # ← 龟速恒定前进
steer_target = 0.0         # 直走

# ---------------- 复位 ------------------
def reset_car():
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.03
    print("\r>>> 已复位 <<<", end='', flush=True)

# ---------------- 主循环 ----------------
mujoco.mj_resetData(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -25
    while viewer.is_running():
        if KEYS[keyboard.KeyCode.from_char('r')]:
            reset_car()
            KEYS[keyboard.KeyCode.from_char('r')] = False

        # ---- 自动巡航：恒定速度 + 直走 ----
        data.ctrl[0] = steer_target   # 前左转向 = 0
        data.ctrl[1] = steer_target   # 前右转向 = 0
        data.ctrl[2] = CRUISE_SPEED   # 前左驱动
        data.ctrl[3] = CRUISE_SPEED   # 前右驱动
        data.ctrl[4] = CRUISE_SPEED   # 后左驱动
        data.ctrl[5] = CRUISE_SPEED   # 后右驱动

        # ---- 仿真 + HUD ----
        mujoco.mj_step(model, data)
        vel = np.linalg.norm(data.qvel[:3])
        print(f"\rspeed: {vel:7.5f} m/s", end='', flush=True)
        viewer.sync()

    print()