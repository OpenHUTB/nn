import mujoco
from mujoco import viewer
import time
import numpy as np


def control_robot(model_path):
    """
    控制优化后的拟人化机器人模型行走。
    适配新的关节结构和初始姿态，实现更自然的步态。
    """
    # 加载优化后的机器人模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 启动可视化器
    with viewer.launch_passive(model, data) as viewer_instance:
        print("仿真开始。按ESC或关闭窗口停止...")
        start_time = time.time()

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                # 步态周期：2.5秒一步，保持稳定性
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % 2.5

                # -------------------------- 核心步态逻辑 --------------------------
                # 使用正弦波生成平滑的控制信号，围绕初始姿态(ref)进行摆动
                # 0~1.25秒：左腿摆动，右腿支撑；左臂后摆，右臂前摆
                if cycle < 1.25:
                    # swing_phase 从 0 变化到 1，表示摆动腿的一个完整摆动过程
                    swing_phase = (cycle / 1.25)

                    # --- 腿部控制 ---
                    # 左腿（摆动腿）：髋关节前摆，膝关节弯曲
                    data.ctrl[0] = 0.05 + 0.25 * np.sin(swing_phase * np.pi)  # left_hip
                    data.ctrl[1] = -0.15 - 0.3 * np.sin(swing_phase * np.pi)  # left_knee
                    data.ctrl[2] = 0.0 + 0.1 * np.cos(swing_phase * np.pi)  # left_ankle

                    # 右腿（支撑腿）：保持稳定，轻微调整以平衡身体
                    data.ctrl[3] = -0.05 - 0.05 * np.sin(swing_phase * np.pi)  # right_hip
                    data.ctrl[4] = -0.15 + 0.05 * np.sin(swing_phase * np.pi)  # right_knee
                    data.ctrl[5] = 0.0 - 0.05 * np.cos(swing_phase * np.pi)  # right_ankle

                    # --- 手臂协同控制 ---
                    # 左臂（与摆动腿反向）：肩关节后摆，肘关节随动弯曲
                    data.ctrl[6] = 0.0 - 0.2 * np.sin(swing_phase * np.pi)  # left_shoulder
                    data.ctrl[7] = -0.8 + 0.15 * np.sin(swing_phase * np.pi)  # left_elbow

                    # 右臂（与摆动腿同向）：肩关节前摆，肘关节随动弯曲
                    data.ctrl[9] = 0.0 + 0.2 * np.sin(swing_phase * np.pi)  # right_shoulder
                    data.ctrl[10] = -0.8 - 0.15 * np.sin(swing_phase * np.pi)  # right_elbow

                # 1.25~2.5秒：右腿摆动，左腿支撑；右臂后摆，左臂前摆
                else:
                    # swing_phase 从 0 变化到 1
                    swing_phase = ((cycle - 1.25) / 1.25)

                    # --- 腿部控制 ---
                    # 右腿（摆动腿）
                    data.ctrl[3] = -0.05 - 0.25 * np.sin(swing_phase * np.pi)  # right_hip
                    data.ctrl[4] = -0.15 - 0.3 * np.sin(swing_phase * np.pi)  # right_knee
                    data.ctrl[5] = 0.0 + 0.1 * np.cos(swing_phase * np.pi)  # right_ankle

                    # 左腿（支撑腿）
                    data.ctrl[0] = 0.05 + 0.05 * np.sin(swing_phase * np.pi)  # left_hip
                    data.ctrl[1] = -0.15 + 0.05 * np.sin(swing_phase * np.pi)  # left_knee
                    data.ctrl[2] = 0.0 - 0.05 * np.cos(swing_phase * np.pi)  # left_ankle

                    # --- 手臂协同控制 ---
                    # 右臂（与摆动腿反向）
                    data.ctrl[9] = 0.0 - 0.2 * np.sin(swing_phase * np.pi)  # right_shoulder
                    data.ctrl[10] = -0.8 + 0.15 * np.sin(swing_phase * np.pi)  # right_elbow

                    # 左臂（与摆动腿同向）
                    data.ctrl[6] = 0.0 + 0.2 * np.sin(swing_phase * np.pi)  # left_shoulder
                    data.ctrl[7] = -0.8 - 0.15 * np.sin(swing_phase * np.pi)  # left_elbow

                # 固定关节：腕关节保持下垂，颈部保持中立
                data.ctrl[8] = -0.2  # left_wrist (保持微微下垂)
                data.ctrl[11] = -0.2  # right_wrist (保持微微下垂)
                data.ctrl[12] = 0.5  # neck (中间位置)

                # -------------------------- 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()
                # 控制仿真速度，使其更易于观察
                time.sleep(model.opt.timestep * 1.5)

        except KeyboardInterrupt:
            print("\n仿真被用户中断")


if __name__ == "__main__":
    # 确保模型文件名与实际保存的一致
    model_file = "Robot_move_straight.xml"
    control_robot(model_file)