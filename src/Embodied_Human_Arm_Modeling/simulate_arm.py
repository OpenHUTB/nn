import os
import sys
os.environ['MUJOCO_GL'] = 'egl'
# 关键设置：获取当前脚本文件的绝对目录，并设置为工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
# simulate_arm.py
import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path('arm_model.xml')
    data = mujoco.MjData(model)

    # 2. 设置初始关节位置（例如，让肘部稍微弯曲）
    data.qpos[model.joint('elbow').id] = -np.radians(45)  # -45度

    # 3. 使用 viewer 进行交互式仿真
    print("正在启动仿真查看器...")
    print("提示：")
    print("  - 在查看器窗口按 [Space] 暂停/继续物理仿真。")
    print("  - 在左侧‘关节’面板，可以拖拽滑块实时驱动电机，观察手臂运动。")

    with mujoco.viewer.launch(model, data) as viewer:
        # 设置一个简单的摆动目标
        target_positions = np.array([0.3, 0.1, -0.5, 0.0])  # 对应四个电机的目标位置
        time_per_cycle = 5.0  # 摆动周期（秒）
        start_time = time.time()

        # 保持查看器窗口开启
        while viewer.is_running():
            step_start = time.time()

            # 计算基于时间的周期性目标
            t = time.time() - start_time
            phase = 2 * np.pi * t / time_per_cycle
            current_target = target_positions * np.sin(phase)

            # 将目标位置设置为电机的控制信号（简单的位置控制）
            data.ctrl[:] = current_target

            # 执行一步物理仿真
            mujoco.mj_step(model, data)

            # 同步查看器
            viewer.sync()

            # 粗略的时间同步，避免占用过多CPU
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(min(time_until_next_step, model.opt.timestep))

    print("仿真结束。")

if __name__ == "__main__":
    main()