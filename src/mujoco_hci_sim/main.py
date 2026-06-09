import time
import mujoco
from mujoco import viewer
import numpy as np
import math

def main():
    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_path("humanoid.xml")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    data = mujoco.MjData(model)

    # 加载站立姿势
    mujoco.mj_resetDataKeyframe(model, data, 0)
    qpos0 = data.qpos.copy()

    print("✅ 稳定站立 + 大幅快速挥手启动！")

    with viewer.launch_passive(model, data) as v:
        while True:
            # ========== 核心：保持站立 ==========
            kp = 100.0
            kd = 10.0
            data.ctrl[:] = kp * (qpos0[7:] - data.qpos[7:]) - kd * data.qvel[6:]

            # ========== 大幅+快速挥手 ==========
            # 1. 幅度 1.4（挥手抬得更高）
            # 2. 速度从 10（挥得更快）
            wave = math.sin(data.time * 10) * 1.4

            # 只控制肩膀关节，这样挥手更明显
            data.ctrl[16] = wave  # 右肩（大臂）
            # 小臂不动，只挥大臂，动作更像真实挥手
            # data.ctrl[17] = wave

            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()