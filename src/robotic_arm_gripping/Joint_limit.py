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

    # 关节信息（ID+限位）
    joints = {
        "shoulder": {
            "act_id": model.actuator("shoulder").id,
            "min": -1.57,  # 模型中定义的range最小值
            "max": 1.57    # 模型中定义的range最大值
        },
        "elbow": {
            "act_id": model.actuator("elbow").id,
            "min": -2.0,
            "max": 2.0
        }
    }

    # 可视化
    viewer = mujoco.viewer.launch(model, data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0.2, 0, 0.4]

    print("===== 关节限位测试 =====")
    print("先测试肩关节限位 → 再测试肘关节限位 | ESC退出")

    step = 0
    current_joint = "shoulder"  # 当前测试的关节
    while viewer.is_running():
        # 测试肩关节（0-1200步）
        if current_joint == "shoulder":
            if step < 300:
                data.ctrl[joints["shoulder"]["act_id"]] = -2.0  # 左极限
                print("肩关节：左极限")
            elif step < 600:
                data.ctrl[joints["shoulder"]["act_id"]] = 0.0   # 回中
                print("肩关节：回中")
            elif step < 900:
                data.ctrl[joints["shoulder"]["act_id"]] = 2.0   # 右极限
                print("肩关节：右极限")
            elif step < 1200:
                data.ctrl[joints["shoulder"]["act_id"]] = 0.0   # 回中
                print("肩关节：回中")
            else:
                step = 0
                current_joint = "elbow"
                print("\n开始测试肘关节")
        # 测试肘关节（0-1200步）
        elif current_joint == "elbow":
            if step < 300:
                data.ctrl[joints["elbow"]["act_id"]] = -2.0  # 下极限（弯曲）
                print("肘关节：下极限（弯曲）")
            elif step < 600:
                data.ctrl[joints["elbow"]["act_id"]] = 0.0   # 回中
                print("肘关节：回中")
            elif step < 900:
                data.ctrl[joints["elbow"]["act_id"]] = 2.0   # 上极限（伸展）
                print("肘关节：上极限（伸展）")
            elif step < 1200:
                data.ctrl[joints["elbow"]["act_id"]] = 0.0   # 回中
                print("肘关节：回中")
            else:
                step = 0
                current_joint = "shoulder"
                print("\n重新测试肩关节")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # 放慢速度，看清限位
        step += 1

    viewer.close()

if __name__ == "__main__":
    main()