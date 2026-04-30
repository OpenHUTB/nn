import time
import mujoco
from mujoco import viewer

def main():
    model_path = "src/mujoco01/humanoid.xml" 
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    print("启动模拟器...")
    with viewer.launch_passive(model, data) as v:
        last_print_time = 0
        
        while v.is_running():
            # 右手平稳抬手
            data.ctrl[19] = 0.4

            # 仿真步进
            mujoco.mj_step(model, data)

            # 打印传感器数据
            if data.time - last_print_time > 0.3:
                print("======================================")
                print(f"加速度: {data.sensordata[0]:.2f}, {data.sensordata[1]:.2f}, {data.sensordata[2]:.2f}")
                print(f"速度: {data.sensordata[3]:.2f}, {data.sensordata[4]:.2f}, {data.sensordata[5]:.2f}")
                print(f"足部受力: {data.sensordata[6]:.2f}, {data.sensordata[7]:.2f}, {data.sensordata[8]:.2f}")
                last_print_time = data.time

            v.sync()

if __name__ == "__main__":
    main()