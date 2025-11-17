import mujoco
from mujoco import viewer
import time
import numpy as np


def control_robot(model_path):
    # 加载修复后的直立模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 启动可视化器
    with viewer.launch_passive(model, data) as viewer_instance:
        print("仿真开始。按ESC或关闭窗口停止...")
        start_time = time.time()
        # 初始平衡控制：启动时施加微小扭矩保持直立
        for i in range(6):  # 腿部关节优先激活
            data.ctrl[i] = 0.3

        try:
            while True:
                if not viewer_instance.is_running():
                    break

                # 步态周期：延长至2.5秒，降低步频增强稳定性
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % 2.5
                phase = cycle / 2.5  # 0~1标准化周期

                # -------------------------- 核心控制逻辑 --------------------------
                # 0~1.25秒：左腿向前迈步，右腿支撑；左臂后摆，右臂前摆
                if cycle < 1.25:
                    # 腿部控制：基于正弦曲线的平滑过渡，避免动作突变
                    swing_phase = phase * 2  # 0~1
                    # 左腿（摆动腿）：髋关节前摆+膝关节弯曲，动作更柔和
                    data.ctrl[0] = 0.3 + 0.2 * np.sin(swing_phase * np.pi)  # left_hip
                    data.ctrl[1] = 0.6 - 0.3 * np.sin(swing_phase * np.pi)  # left_knee（弯曲幅度减小）
                    data.ctrl[2] = 0.2 + 0.1 * np.sin(swing_phase * np.pi)  # left_ankle

                    # 右腿（支撑腿）：保持稳定，轻微调整平衡
                    data.ctrl[3] = 0.25 - 0.05 * np.sin(swing_phase * np.pi)  # right_hip
                    data.ctrl[4] = 0.15  # right_knee（保持微屈增强缓冲）
                    data.ctrl[5] = 0.15  # right_ankle

                    # 手臂协同：与腿部反向摆动，幅度减小避免失衡
                    data.ctrl[6] = 0.25 - 0.15 * np.sin(swing_phase * np.pi)  # left_shoulder（后摆）
                    data.ctrl[7] = 0.4  # left_elbow（保持弯曲）
                    data.ctrl[9] = 0.6 + 0.15 * np.sin(swing_phase * np.pi)  # right_shoulder（前摆）
                    data.ctrl[10] = 0.4  # right_elbow（保持弯曲）

                # 1.25~2.5秒：右腿向前迈步，左腿支撑；右臂后摆，左臂前摆
                else:
                    swing_phase = (phase - 0.5) * 2  # 0~1
                    # 右腿（摆动腿）：对称动作
                    data.ctrl[3] = 0.3 + 0.2 * np.sin(swing_phase * np.pi)  # right_hip
                    data.ctrl[4] = 0.6 - 0.3 * np.sin(swing_phase * np.pi)  # right_knee
                    data.ctrl[5] = 0.2 + 0.1 * np.sin(swing_phase * np.pi)  # right_ankle

                    # 左腿（支撑腿）：保持稳定
                    data.ctrl[0] = 0.25 - 0.05 * np.sin(swing_phase * np.pi)  # left_hip
                    data.ctrl[1] = 0.15  # left_knee（保持微屈）
                    data.ctrl[2] = 0.15  # left_ankle

                    # 手臂协同：对称摆动
                    data.ctrl[6] = 0.6 + 0.15 * np.sin(swing_phase * np.pi)  # left_shoulder（前摆）
                    data.ctrl[7] = 0.4  # left_elbow
                    data.ctrl[9] = 0.25 - 0.15 * np.sin(swing_phase * np.pi)  # right_shoulder（后摆）
                    data.ctrl[10] = 0.4  # right_elbow

                # 固定关节：腕关节和颈部保持稳定
                data.ctrl[8] = 0.0  # left_wrist
                data.ctrl[11] = 0.0  # right_wrist
                data.ctrl[12] = 0.5  # neck（中间位置）

                # -------------------------- 仿真推进 --------------------------
                mujoco.mj_step(model, data)
                viewer_instance.sync()
                # 降低仿真速度至原速的80%，便于观察平衡状态
                time.sleep(model.opt.timestep * 1.25)

        except KeyboardInterrupt:
            print("\n仿真被用户中断")


if __name__ == "__main__":
    model_file = "Robot_move_straight.xml"  # 与修复后模型文件名一致
    control_robot(model_file)