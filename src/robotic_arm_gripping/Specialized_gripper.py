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

    # 只关注夹爪执行器
    left_act = model.actuator("left").id
    right_act = model.actuator("right").id

    # 可视化
    viewer = mujoco.viewer.launch(model, data)
    viewer.cam.distance = 1.0  # 拉近相机，看清夹爪
    viewer.cam.azimuth = 60
    viewer.cam.elevation = -10
    viewer.cam.lookat = [0.4, 0, 0.4]

    print("===== 夹爪单独控制测试 =====")
    print("夹爪会循环：张开 → 闭合 → 保持 → 张开 | ESC退出")

    step = 0
    while viewer.is_running():
        # 阶段1：0-400步 → 张开夹爪
        if step < 400:
            data.ctrl[left_act] = -1.0
            data.ctrl[right_act] = -1.0
        # 阶段2：400-800步 → 闭合夹爪
        elif step < 800:
            data.ctrl[left_act] = 1.0
            data.ctrl[right_act] = 1.0
        # 阶段3：800-1000步 → 保持闭合
        elif step < 1000:
            data.ctrl[left_act] = 1.0
            data.ctrl[right_act] = 1.0
        # 阶段4：重置步数，循环
        else:
            step = 0

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
        step += 1

    viewer.close()

if __name__ == "__main__":
    main()