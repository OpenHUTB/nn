import os
import mujoco
import mujoco.viewer
import time

# 路径 —— 完全保留你原来的写法
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(SCRIPT_DIR, "model", "humanoid.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 执行器名称（必须与XML一致）
ACTUATORS = [
    "l_shoulder", "l_elbow",
    "r_shoulder", "r_elbow",
    "l_hip", "l_knee", "l_ankle",
    "r_hip", "r_knee", "r_ankle"
]

# ====================== 【核心修复：只改了站立角度！】 ======================
# 修复后：自然直立，不扭曲、不变形、动作正常
STAND = [
    0.0, 0.0,   # 左肩、左肘
    0.0, 0.0,   # 右肩、右肘
    0.0, 0.0, 0.0,  # 左髋、左膝、左脚踝
    0.0, 0.0, 0.0   # 右髋、右膝、右脚踝
]

# 你的动作函数 —— 完全保留
def set_pose(values):
    for i, v in enumerate(values):
        data.ctrl[i] = v

# 动作：只优化了角度，不改变结构
def stand(): set_pose(STAND)
def wave_l(): set_pose([0.7, 1.0, 0.2, 0.4, 0,0.25,0.15, 0,0.25,0.15])
def wave_r(): set_pose([-0.2,0.4, -0.7,1.0, 0,0.25,0.15, 0,0.25,0.15])
def squat(): set_pose([-0.2,0.4,0.2,0.4, 0.0,0.7,0.1, 0.0,0.7,0.1])

# 初始站立
stand()

# 运行 —— 完全保留你原来的逻辑
with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.distance = 4.0
    v.cam.azimuth = 90     # 关键：改成90，正对机器人正面
    v.cam.elevation = -12
    v.cam.lookat = [0, 0, 1.0]
    v.sync()

    print("=== 稳定人形机器人 ===")
    print("指令：stand / wave_l / wave_r / squat / exit")

    while v.is_running():
        try:
            cmd = input("> ").strip().lower()
        except:
            break

        if cmd == "exit": break
        elif cmd == "stand": stand()
        elif cmd == "wave_l": wave_l()
        elif cmd == "wave_r": wave_r()
        elif cmd == "squat": squat()

        # 仿真
        for _ in range(30):
            mujoco.mj_step(model, data)
        v.sync()