import mujoco
import mujoco.viewer
import time
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("arm6dof_final.xml")
data = mujoco.MjData(model)
model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EQUALITY

# 关节执行器
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
act = {}
for j in joint_names:
    act[j] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{j}")

# 每个关节运动的幅度（保证动作明显）
joint_amplitude = {
    "joint1": 1.2,
    "joint2": 1.0,
    "joint3": 1.0,
    "joint4": 1.0,
    "joint5": 0.8,
    "joint6": 0.8
}

# ----------------------------------
# 单关节依次运动测试
# ----------------------------------
def single_joint_sequence_test():
    print("=" * 60)
    print("🤖 6 轴机械臂 —— 单关节依次运动测试")
    print("每个关节单独正转 → 保持 → 反转 → 归零")
    print("=" * 60)

    # 遍历每个关节
    for joint_name in joint_names:
        print(f"\n👉 正在测试: {joint_name}")

        # 1. 正转
        print(f"   ├─ 正转")
        for _ in range(600):
            data.ctrl[:] = 0  # 其他关节全部清零
            data.ctrl[act[joint_name]] = joint_amplitude[joint_name]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        # 保持
        time.sleep(0.5)

        # 2. 反转
        print(f"   ├─ 反转")
        for _ in range(600):
            data.ctrl[:] = 0
            data.ctrl[act[joint_name]] = -joint_amplitude[joint_name]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        # 保持
        time.sleep(0.5)

        # 3. 归零
        print(f"   └─ 归零")
        for _ in range(400):
            data.ctrl[:] = 0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

        print(f"✅ {joint_name} 测试完成")

    print("\n🎉 全部 6 个关节单关节测试完成！")

# ------------------------------
# 运行
# ------------------------------
with mujoco.viewer.launch_passive(model, data) as v:
    global viewer
    viewer = v
    viewer.cam.distance = 2.2
    viewer.cam.lookat = [0.2, 0, 0.3]
    viewer.cam.azimuth = 50

    # 启动单关节依次运动
    single_joint_sequence_test()

    # 保持窗口打开
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()