import time
import csv
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

    # 最大力矩限制（防止抖动）
    max_torque = 0.4

    # CSV 记录
    csv_file = open("simulation_data.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    header = ["time"] + [f"qpos_{i}" for i in range(model.nq)] + [f"qvel_{i}" for i in range(model.nv)]
    writer.writerow(header)

    print("启动模拟器...")
    with viewer.launch_passive(model, data) as v:
        last_print_time = 0

        while v.is_running():
            # 目标控制量
            target_ctrl = 0.4

            # 限制最大力矩，防止抖动（新功能）
            if target_ctrl > max_torque:
                target_ctrl = max_torque
            if target_ctrl < -max_torque:
                target_ctrl = -max_torque

            # 施加控制
            data.ctrl[19] = target_ctrl

            # 仿真步进
            mujoco.mj_step(model, data)

            # 写入CSV
            row = [data.time]
            row += data.qpos.tolist()
            row += data.qvel.tolist()
            writer.writerow(row)

            # 打印信息
            if data.time - last_print_time > 0.3:
                print("======================================")
                print(f"加速度: {data.sensordata[0]:.2f}, {data.sensordata[1]:.2f}, {data.sensordata[2]:.2f}")
                print(f"速度: {data.sensordata[3]:.2f}, {data.sensordata[4]:.2f}, {data.sensordata[5]:.2f}")
                print(f"足部受力: {data.sensordata[6]:.2f}, {data.sensordata[7]:.2f}, {data.sensordata[8]:.2f}")
                print(f"当前关节控制输出值: {target_ctrl:.2f}") 
                last_print_time = data.time

            v.sync()

    csv_file.close()
    print("✅ 数据已保存到 simulation_data.csv")

if __name__ == "__main__":
    main()