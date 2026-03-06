import mujoco
import mujoco.viewer
import time
import os

MODEL_PATH = "arm_model.xml"

def main():
    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在：{MODEL_PATH}")
        return
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 执行器ID
    shoulder_act = model.actuator("shoulder").id
    elbow_act = model.actuator("elbow").id

    # 可视化
    viewer = mujoco.viewer.launch(model, data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0.35, 0, 0.2]

    print("===== 机械臂推动小球测试 =====")
    print("机械臂会缓慢移动，推动小球到右侧 | ESC退出")

    step = 0
    while viewer.is_running():
        # 缓慢调整肩关节，推动小球
        if step < 800:
            # 肩关节缓慢向右摆，推动小球
            data.ctrl[shoulder_act] = 0.5
            data.ctrl[elbow_act] = -0.8  # 保持末端在小球高度
        else:
            # 推到位后停止
            data.ctrl[shoulder_act] = 0.0
            data.ctrl[elbow_act] = -0.8
            print("小球已推动到位！")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
        step += 1

    viewer.close()

if __name__ == "__main__":
    main()