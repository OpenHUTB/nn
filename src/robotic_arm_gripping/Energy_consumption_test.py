import mujoco
import mujoco.viewer
import time
import os

MODEL_PATH = "arm_model.xml"
STEP_DELAY = 0.005

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        return None
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        return model, data
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return None

def energy_consumption_test():
    model_data = load_model(MODEL_PATH)
    if not model_data:
        return
    model, data = model_data

    # 执行器ID
    act_ids = {
        "shoulder": model.actuator("shoulder").id,
        "elbow": model.actuator("elbow").id
    }
    joint_ids = {
        "shoulder": model.joint("shoulder").id,
        "elbow": model.joint("elbow").id
    }

    # 可视化
    try:
        viewer = mujoco.viewer.launch(model, data)
    except:
        viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0.2, 0, 0.4]

    print("===== 机械臂能耗模拟测试 =====")
    print("适配新mass参数 | 计算关节运动能耗 | ESC退出\n")

    total_energy = 0.0
    step = 0

    while viewer.is_running():
        # 控制关节运动
        data.ctrl[act_ids["shoulder"]] = 1.0 if step < 1000 else -1.0
        data.ctrl[act_ids["elbow"]] = -1.0 if step < 1000 else 1.0

        # 计算能耗（力矩×速度×时间步）
        for joint_name in ["shoulder", "elbow"]:
            torque = data.ctrl[act_ids[joint_name]]
            vel = data.qvel[joint_ids[joint_name]]
            energy = abs(torque * vel * 0.002)  # 0.002是模型timestep
            total_energy += energy

        # 每200步打印能耗
        if step % 200 == 0:
            print(f"累计能耗：{total_energy:.4f} J | 当前步数：{step}")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(STEP_DELAY)
        step += 1

    viewer.close()
    print(f"\n✅ 测试结束，总能耗：{total_energy:.4f} J")

if __name__ == "__main__":
    energy_consumption_test()