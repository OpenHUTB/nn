import mujoco
from mujoco import viewer
import time

def control_robot(model_path):
    # 加载模型和数据（修复后模型可正常加载）
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

                # 步态周期：2秒1步（交替迈腿）
                elapsed_time = time.time() - start_time
                cycle = elapsed_time % 2

                # -------------------------- 双腿交替控制 --------------------------
                # 0~1秒：左腿向前迈步，右腿支撑
                if cycle < 1:
                    # 左腿：髋关节向前转（控制信号0.3，对应关节range 0~1.0）
                    data.ctrl[0] = 0.3  # left_hip_motor
                    # 左腿：膝关节弯曲（控制信号0.6）
                    data.ctrl[1] = 0.6  # left_knee_motor
                    # 右腿：髋关节向后转（控制信号0.2，支撑身体）
                    data.ctrl[3] = 0.2  # right_hip_motor
                    # 右腿：膝关节伸直（控制信号0.1）
                    data.ctrl[4] = 0.1  # right_knee_motor

                # 1~2秒：右腿向前迈步，左腿支撑
                else:
                    # 左腿：髋关节向后转（控制信号0.2，支撑身体）
                    data.ctrl[0] = 0.2  # left_hip_motor
                    # 左腿：膝关节伸直（控制信号0.1）
                    data.ctrl[1] = 0.1  # left_knee_motor
                    # 右腿：髋关节向前转（控制信号0.3）
                    data.ctrl[3] = 0.3  # right_hip_motor
                    # 右腿：膝关节弯曲（控制信号0.6）
                    data.ctrl[4] = 0.6  # right_knee_motor

                # 踝关节固定（暂时不转动，保证站立稳定）
                data.ctrl[2] = 0.0  # left_ankle_motor
                data.ctrl[5] = 0.0  # right_ankle_motor

                # 运行仿真步
                mujoco.mj_step(model, data)
                # 同步可视化（解决模糊问题，与仿真步同步）
                viewer_instance.sync()
                # 控制仿真速度（与模型timestep一致）
                time.sleep(model.opt.timestep)

        except KeyboardInterrupt:
            print("\n仿真被用户中断")

if __name__ == "__main__":
    # 模型文件名用你提供的 "Robot_with_Legs.xml"（若要改名为Robot_move_straight.xml，这里同步改）
    model_file = "Robot_move_straight.xml"
    control_robot(model_file)