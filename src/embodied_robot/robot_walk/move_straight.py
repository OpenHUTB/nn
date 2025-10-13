import mujoco
from mujoco import viewer  # 显式导入viewer模块
import time


def control_robot(model_path):
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 启动可视化器（MuJoCo 3.x版本的正确用法）
    with viewer.launch_passive(model, data) as viewer_instance:
        print("仿真开始。按ESC或关闭窗口停止...")
        start_time = time.time()

        try:
            while True:
                # 检查窗口是否已关闭
                if not viewer_instance.is_running():
                    break

                # 计算控制信号
                elapsed_time = time.time() - start_time
                data.ctrl[0] = 0.5 if (elapsed_time % 20 < 10) else -0.5

                # 运行仿真步
                mujoco.mj_step(model, data)

                # 同步可视化器
                viewer_instance.sync()

                # 控制仿真速度
                time.sleep(model.opt.timestep)


        except KeyboardInterrupt:
            print("\n仿真被用户中断")


if __name__ == "__main__":
    model_file = "Robot_move_straight.xml"  # 确保模型文件路径正确
    control_robot(model_file)
