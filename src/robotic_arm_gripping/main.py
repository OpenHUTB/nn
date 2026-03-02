import mujoco
import mujoco.viewer
import numpy as np
import time


def main():
    # 1. 加载模型和创建仿真数据
    model = mujoco.MjModel.from_xml_path("arm_model.xml")
    data = mujoco.MjData(model)

    # 2. 获取关节名称和数量（排除固定关节，用数值0判断固定关节，兼容3.1.6）
    joint_names = []
    n_joints = 0
    for i in range(model.njnt):
        # 关键修改：固定关节的数值常量是0，直接判断数值（所有MuJoCo版本通用）
        if model.joint(i).type != 0:  # 0 = mjJNT_FIXED（固定关节）
            joint_names.append(model.joint(i).name)
            n_joints += 1
    print(f"机械臂关节列表: {joint_names}")
    print(f"可控关节数量: {n_joints}")

    # 3. 初始化关节目标位置（初始姿态）
    target_pos = np.zeros(n_joints)
    # 设置初始目标（轻微抬起机械臂）
    target_pos[1] = np.pi / 4  # joint1 抬起45度
    target_pos[2] = np.pi / 6  # joint2 旋转30度

    # 4. 启动可视化窗口并运行仿真
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置窗口参数
        viewer.cam.azimuth = 45  # 相机方位角
        viewer.cam.elevation = -20  # 相机仰角
        viewer.cam.distance = 1.5  # 相机距离
        viewer.cam.lookat = [0, 0, 0.3]  # 相机看向的点

        # 仿真主循环
        start_time = time.time()
        while viewer.is_running():
            # 计算当前时间，用于生成关节轨迹
            current_time = time.time() - start_time

            # 让关节1和关节2缓慢摆动（演示用）
            target_pos[1] = np.pi / 4 + 0.2 * np.sin(current_time)  # 上下摆动
            target_pos[2] = np.pi / 6 + 0.3 * np.cos(current_time)  # 左右摆动

            # 设置执行器目标位置（控制关节）
            for i in range(n_joints):
                data.actuator(i).ctrl = target_pos[i]

            # 运行仿真步
            mujoco.mj_step(model, data)

            # 刷新可视化窗口
            viewer.sync()

            # 控制仿真帧率（约60fps）
            time.sleep(1 / 60)


if __name__ == "__main__":
    # 检查mujoco版本（确保兼容性）
    print(f"MuJoCo 版本: {mujoco.__version__}")
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        print("请确认已安装mujoco：pip install mujoco==3.1.6")