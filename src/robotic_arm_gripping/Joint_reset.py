import mujoco
import mujoco.viewer
import time
import os

MODEL_PATH = "arm_model.xml"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在：{MODEL_PATH}")
        return
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 先让机械臂随机动一下（模拟“非初始位置”）
    print("第一步：机械臂随机运动1000步...")
    for _ in range(1000):
        data.ctrl[model.actuator("shoulder").id] = 1.0
        data.ctrl[model.actuator("elbow").id] = -1.0
        data.ctrl[model.actuator("left").id] = 1.0
        mujoco.mj_step(model, data)
        time.sleep(0.001)

    # 启动可视化，显示“非初始位置”的机械臂
    with mujoco.viewer.launch(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.lookat = [0.2, 0, 0.4]

        print("\n第二步：按任意键让机械臂回到初始位置（5秒后自动复位）")
        time.sleep(5)  # 留5秒观察当前位置

        # 核心复位逻辑：调用mj_resetData回到初始状态
        mujoco.mj_resetData(model, data)
        print("✅ 机械臂已复位到初始位置！")

        # 保持可视化，方便观察
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    main()