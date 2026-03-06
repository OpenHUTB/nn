import mujoco
import mujoco.viewer
import time
import os

MODEL_PATH = "arm_model.xml"
STEP_DELAY = 0.008  # 适配新阻尼，放慢速度更平滑

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

def joint_smooth_move_test():
    model_data = load_model(MODEL_PATH)
    if not model_data:
        return
    model, data = model_data

    # 执行器ID
    shoulder_act = model.actuator("shoulder").id
    elbow_act = model.actuator("elbow").id

    # 可视化
    try:
        viewer = mujoco.viewer.launch(model, data)
    except:
        viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0.2, 0, 0.4]

    print("===== 关节平滑运动测试 =====")
    print("适配新阻尼参数 | 匀速运动无抖动 | ESC退出\n")

    # 运动方向控制
    shoulder_dir = 1  # 1=右，-1=左
    elbow_dir = -1    # -1=弯曲，1=伸展
    step = 0

    while viewer.is_running():
        # 每800步切换方向
        if step % 800 == 0:
            shoulder_dir *= -1
            elbow_dir *= -1
            dir_desc = "右/弯曲" if shoulder_dir == 1 else "左/伸展"
            print(f"切换运动方向：{dir_desc}")

        # 匀速力矩（适配新阻尼）
        data.ctrl[shoulder_act] = shoulder_dir * 1.5
        data.ctrl[elbow_act] = elbow_dir * 1.5

        # 打印关节速度（验证平滑性）
        if step % 200 == 0:
            shoulder_vel = data.qvel[model.joint("shoulder").id]
            elbow_vel = data.qvel[model.joint("elbow").id]
            print(f"关节速度：肩关节={shoulder_vel:.2f}, 肘关节={elbow_vel:.2f}")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(STEP_DELAY)
        step += 1

    viewer.close()

if __name__ == "__main__":
    joint_smooth_move_test()