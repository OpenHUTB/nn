import time
import mujoco
from mujoco import viewer
import numpy as np

def main():
    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_path("humanoid.xml")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    data = mujoco.MjData(model)

    # 1. 加载站立关键帧
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 2. 保存初始关节位置（PD控制目标）
    qpos0 = data.qpos.copy()

    print("✅ 启动PD控制，人形将永久站立")

    # 运行仿真
    with viewer.launch_passive(model, data) as v:
        while True:
            # PD控制：让关节保持初始位置
            kp = 100.0  # 比例增益
            kd = 10.0   # 微分增益
            data.ctrl[:] = kp * (qpos0[7:] - data.qpos[7:]) - kd * data.qvel[6:]

            mujoco.mj_step(model, data)
            
            print(f"时间: {data.time:.2f}s | 身高: {data.qpos[2]:.2f}m")
            v.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()