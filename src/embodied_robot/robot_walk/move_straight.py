import mujoco
from mujoco import viewer
import time

def control_robot(model_path):
    # 加载优化后的拟人化机器人模型
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

                # 步态周期：2秒1步（双腿+双臂协同）
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % 2

                # -------------------------- 核心控制逻辑 --------------------------
                # 0~1秒：左腿向前迈步，右腿支撑；左臂向后摆，右臂向前摆
                if cycle < 1:
                    # ------------------- 腿部控制 -------------------
                    # 左腿：髋关节向前转，膝关节弯曲（迈步）
                    data.ctrl[0] = 0.4  # left_hip_motor（扭矩增大适配腿部质量）
                    data.ctrl[1] = 0.7  # left_knee_motor
                    data.ctrl[2] = 0.2  # left_ankle_motor（轻微调整保持平衡）
                    # 右腿：髋关节向后转，膝关节伸直（支撑）
                    data.ctrl[3] = 0.2  # right_hip_motor
                    data.ctrl[4] = 0.1  # right_knee_motor
                    data.ctrl[5] = 0.1  # right_ankle_motor

                    # ------------------- 手臂控制（协同摆动） -------------------
                    # 左臂：肩关节向后摆，肘关节微屈
                    data.ctrl[6] = 0.2  # left_shoulder_motor
                    data.ctrl[7] = 0.5  # left_elbow_motor
                    data.ctrl[8] = 0.0  # left_wrist_motor（固定）
                    # 右臂：肩关节向前摆，肘关节微屈
                    data.ctrl[9] = 0.7  # right_shoulder_motor
                    data.ctrl[10] = 0.5  # right_elbow_motor
                    data.ctrl[11] = 0.0  # right_wrist_motor（固定）

                # 1~2秒：右腿向前迈步，左腿支撑；右臂向后摆，左臂向前摆
                else:
                    # ------------------- 腿部控制 -------------------
                    # 左腿：髋关节向后转，膝关节伸直（支撑）
                    data.ctrl[0] = 0.2  # left_hip_motor
                    data.ctrl[1] = 0.1  # left_knee_motor
                    data.ctrl[2] = 0.1  # left_ankle_motor
                    # 右腿：髋关节向前转，膝关节弯曲（迈步）
                    data.ctrl[3] = 0.4  # right_hip_motor
                    data.ctrl[4] = 0.7  # right_knee_motor
                    data.ctrl[5] = 0.2  # right_ankle_motor

                    # ------------------- 手臂控制（协同摆动） -------------------
                    # 左臂：肩关节向前摆，肘关节微屈
                    data.ctrl[6] = 0.7  # left_shoulder_motor
                    data.ctrl[7] = 0.5  # left_elbow_motor
                    data.ctrl[8] = 0.0  # left_wrist_motor
                    # 右臂：肩关节向后摆，肘关节微屈
                    data.ctrl[9] = 0.2  # right_shoulder_motor
                    data.ctrl[10] = 0.5  # right_elbow_motor
                    data.ctrl[11] = 0.0  # right_wrist_motor

                # 颈部固定（避免头部晃动）
                data.ctrl[12] = 0.5  # neck_motor（中间位置固定）

                # -------------------------- 仿真推进 --------------------------
                mujoco.mj_step(model, data)  # 运行一步物理仿真
                viewer_instance.sync()       # 同步渲染（解决模糊问题）
                time.sleep(model.opt.timestep)  # 控制仿真速度与物理步长匹配

        except KeyboardInterrupt:
            print("\n仿真被用户中断")

if __name__ == "__main__":
    # 确保模型文件名与实际保存的一致（这里用你要求的"Robot_move_straight.xml"）
    model_file = "Robot_move_straight.xml"
    control_robot(model_file)