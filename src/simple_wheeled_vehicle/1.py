"""
超低速 + 顺滑转向小车
- 速度 0.4 m/s 封顶
- 转向：方向键只给“目标角度”，伺服自带阻尼慢回正
- R 复位
"""
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

# ---------------- 键盘 ------------------
KEYS = {keyboard.Key.up: False,
        keyboard.Key.down: False,
        keyboard.Key.left: False,
        keyboard.Key.right: False,
        keyboard.KeyCode.from_char('r'): False}

def on_press(k):
    if k in KEYS: KEYS[k] = True
def on_release(k):
    if k in KEYS: KEYS[k] = False

keyboard.Listener(on_press=on_press, on_release=on_release).start()

# ---------------- 加载模型 --------------
model = mujoco.MjModel.from_xml_path("wheeled_car.xml")
data = mujoco.MjData(model)

# ---------------- 参数 ------------------
MAX_SPEED   = 0.4          # ← 封顶速度（m/s）
ACCEL_RAMP  = 0.02         # ← 爬升系数
steer_target = 0.0
speed_target = 0.0
speed = 0.0                # 当前速度

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

        # ---- 超低速软启动 ----
        if KEYS[keyboard.Key.up]:
            speed_target = min(speed_target + ACCEL_RAMP, MAX_SPEED)
        elif KEYS[keyboard.Key.down]:
            speed_target = max(speed_target - ACCEL_RAMP, -MAX_SPEED * 0.7)
        else:
            speed_target = 0.0

        # ---- 顺滑转向：只给目标角度，伺服自动慢回正 ----
        if KEYS[keyboard.Key.left]:
            steer_target = 0.4          # 最大 0.4 rad
        elif KEYS[keyboard.Key.right]:
            steer_target = -0.4
        else:
            steer_target = 0.0          # 松键即回正（伺服自带阻尼）
        # ---- 速度斜坡 ----
        speed = speed_target   # 电机直接吃斜坡值即可（已很柔）

        # ---- 输出 ----
        data.ctrl[0] = steer_target   # 前左转向
        data.ctrl[1] = steer_target   # 前右转向
        data.ctrl[2] = speed - steer_target * 0.3
        data.ctrl[3] = speed + steer_target * 0.3
        data.ctrl[4] = speed - steer_target * 0.15
        data.ctrl[5] = speed + steer_target * 0.15

        # ---- 仿真 + HUD ----
        mujoco.mj_step(model, data)
        vel = np.linalg.norm(data.qvel[:3])
        print(f"\rspeed: {vel:5.2f} m/s", end='', flush=True)
        viewer.sync()

    print()