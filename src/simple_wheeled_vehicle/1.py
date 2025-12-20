"""
抗翻小车遥控器
- 方向键驾驶
- 电机软启动（扭矩斜坡）
- 限速 / 终端 HUD
- R 键复位
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

# ---------------- 控制参数 --------------
MAX_SPEED   = 2.0           # 最大目标速度（m/s）
ACCEL_RAMP  = 0.08          # 电机扭矩斜坡系数（越小越柔）
steer_target = 0.0
speed_target = 0.0
steer = 0.0
speed = 0.0
alpha_steer = 0.15
auto_center = 0.92

# ---------------- 复位函数 --------------
def reset_car():
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.07
    print("\r>>> 已复位 <<<", end='', flush=True)

# ---------------- 主循环 ----------------
mujoco.mj_resetData(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -25
    while viewer.is_running():
        # ---- 复位键 ----
        if KEYS[keyboard.KeyCode.from_char('r')]:
            reset_car()
            KEYS[keyboard.KeyCode.from_char('r')] = False

        # ---- 驾驶指令 ----
        if KEYS[keyboard.Key.up]:
            speed_target = min(speed_target + ACCEL_RAMP, MAX_SPEED)
        elif KEYS[keyboard.Key.down]:
            speed_target = max(speed_target - ACCEL_RAMP, -MAX_SPEED * 0.7)
        else:
            speed_target = 0.0

        if KEYS[keyboard.Key.left]:
            steer_target = 0.5
        elif KEYS[keyboard.Key.right]:
            steer_target = -0.5
        else:
            steer_target *= auto_center

        # ---- 平滑转向 ----
        steer = alpha_steer * steer_target + (1 - alpha_steer) * steer

        # ---- 输出到电机 ----
        data.ctrl[0] = steer
        data.ctrl[1] = steer
        data.ctrl[2] = speed_target - steer * 0.5
        data.ctrl[3] = speed_target + steer * 0.5
        data.ctrl[4] = speed_target - steer * 0.25
        data.ctrl[5] = speed_target + steer * 0.25

        # ---- 仿真步进 + HUD ----
        mujoco.mj_step(model, data)
        vel = np.linalg.norm(data.qvel[:3])
        print(f"\rspeed: {vel:5.2f} m/s", end='', flush=True)
        viewer.sync()

    print()