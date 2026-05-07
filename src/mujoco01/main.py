import time
import csv  # 只新增这个库
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
    # 修复：必须传 keyframe 索引 0
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # ======================
    # 新增：CSV 数据记录（不影响任何原有功能）
    # ======================
    csv_file = open("simulation_data.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    
    # 写入表头
    header = ["time"]
    header += [f"qpos_{i}" for i in range(model.nq)]
    header += [f"qvel_{i}" for i in range(model.nv)]
    writer.writerow(header)

    print("启动模拟器...")
    with viewer.launch_passive(model, data) as v:
        last_print_time = 0
        
        while v.is_running():
            # 右手平稳抬手（原功能完全保留）
            data.ctrl[19] = 0.4

            # 仿真步进（原功能）
            mujoco.mj_step(model, data)

            # ======================
            # 新增：每帧写入数据（不影响原来任何代码）
            # ======================
            row = [data.time]
            row += data.qpos.tolist()
            row += data.qvel.tolist()
            writer.writerow(row)

            # 打印传感器数据（原功能完全保留）
            if data.time - last_print_time > 0.3:
                print("======================================")
                print(f"加速度: {data.sensordata[0]:.2f}, {data.sensordata[1]:.2f}, {data.sensordata[2]:.2f}")
                print(f"速度: {data.sensordata[3]:.2f}, {data.sensordata[4]:.2f}, {data.sensordata[5]:.2f}")
                print(f"足部受力: {data.sensordata[6]:.2f}, {data.sensordata[7]:.2f}, {data.sensordata[8]:.2f}")
                last_print_time = data.time

            v.sync()

    # 关闭文件
    csv_file.close()
    print("✅ 数据已保存到 simulation_data.csv")

if __name__ == "__main__":
    main()