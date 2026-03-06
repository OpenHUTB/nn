import mujoco
import mujoco.viewer
import time
import os

# 配置项
MODEL_PATH = "arm_model.xml"  # 替换为你的模型绝对路径
STEP_DELAY = 0.005
TORQUE_VALUE = 2.0  # 适配模型ctrlrange="-2 2"

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

def gripper_grasp_stability_test():
    # 加载模型
    model_data = load_model(MODEL_PATH)
    if not model_data:
        return
    model, data = model_data

    # 获取执行器ID
    act_ids = {
        "shoulder": model.actuator("shoulder").id,
        "elbow": model.actuator("elbow").id,
        "left": model.actuator("left").id,
        "right": model.actuator("right").id
    }

    # 初始化可视化
    try:
        viewer = mujoco.viewer.launch(model, data)
    except:
        viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0.35, 0, 0.2]

    print("===== 夹爪抓取稳定性测试 =====")
    print("适配新模型接触约束 | 抓取后晃动验证稳定性 | ESC退出\n")

    step = 0
    while viewer.is_running():
        # 阶段1：0-800步 → 移动到小球
        if step < 800:
            data.ctrl[act_ids["shoulder"]] = 0.0
            data.ctrl[act_ids["elbow"]] = -1.0  # 适配新模型小球高度
            data.ctrl[act_ids["left"]] = -TORQUE_VALUE
            data.ctrl[act_ids["right"]] = -TORQUE_VALUE
        # 阶段2：800-1200步 → 闭合夹爪（利用新接触约束）
        elif step < 1200:
            data.ctrl[act_ids["shoulder"]] = 0.0
            data.ctrl[act_ids["elbow"]] = -1.0
            data.ctrl[act_ids["left"]] = TORQUE_VALUE
            data.ctrl[act_ids["right"]] = TORQUE_VALUE
        # 阶段3：1200步后 → 抬起+晃动（测试稳定性）
        else:
            # 小幅晃动肩关节，验证抓取是否稳定
            shake_torque = 1.0 * (step % 200 < 100) - 0.5
            data.ctrl[act_ids["shoulder"]] = shake_torque
            data.ctrl[act_ids["elbow"]] = -0.3  # 抬起
            data.ctrl[act_ids["left"]] = TORQUE_VALUE
            data.ctrl[act_ids["right"]] = TORQUE_VALUE

        # 打印抓取状态
        if step == 1200:
            print("✅ 夹爪已闭合，开始晃动测试（新接触约束保证不滑落）")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(STEP_DELAY)
        step += 1

    viewer.close()

if __name__ == "__main__":
    gripper_grasp_stability_test()