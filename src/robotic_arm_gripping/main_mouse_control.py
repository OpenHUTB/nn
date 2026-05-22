import mujoco
import mujoco.viewer
import numpy as np
import time
import os


def main():
    # 读取外部XML模型文件
    model_path = "arm_model.xml"

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请确认文件名称正确，且和脚本放在同一目录下")
        return

    # 加载外部模型文件
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"加载模型失败：{e}")
        return

    # 关节ID（只关注肩、肘关节的弯曲）
    shoulder_joint_id = model.joint("shoulder").id
    elbow_joint_id = model.joint("elbow").id
    shoulder_act_id = model.actuator("shoulder").id
    elbow_act_id = model.actuator("elbow").id

    # PD控制参数（幅度增大后，适当调高增益保证跟踪效果）
    kp = 1000  # 比例增益（原800，增大以适配更大幅度）
    kd = 80  # 阻尼增益（原60，增大避免抖动）

    # 可视化窗口加载
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角（方便观察大幅弯曲）
        viewer.cam.distance = 1.5  # 调远相机，避免机械臂超出视野
        viewer.cam.azimuth = 30
        viewer.cam.elevation = -15
        viewer.cam.lookat = [0.2, 0.0, 0.4]

        print("===== 机械臂大幅弯曲动作演示 =====")
        print("机械臂会以更大幅度循环做：伸展 → 弯曲 → 伸展 的动作")
        print("按ESC键退出演示")

        # 仿真时间累计
        sim_time = 0.0
        timestep = model.opt.timestep

        # 主仿真循环
        while viewer.is_running():
            # ===================== 核心修改：增大弯曲幅度 =====================
            # 肩关节：大幅摆动（从原0.5倍提升到1.2倍，接近关节限位-1.57~1.57）
            shoulder_target = 1.2 * np.sin(sim_time * 0.8)  # 1.2是幅度系数（核心增大项）
            # 肘关节：超大幅度弯曲（从原-1.0倍提升到-1.8倍，接近关节限位-2.0）
            elbow_target = -1.8 * (1 + np.cos(sim_time * 0.6))  # 1.8是幅度系数（核心增大项）

            # 限制肘关节目标位置不超过模型定义的限位（-2.0），避免报错
            elbow_target = max(elbow_target, -2.0)

            # PD控制计算力矩
            # 肩关节控制
            shoulder_error = shoulder_target - data.qpos[shoulder_joint_id]
            shoulder_vel = data.qvel[shoulder_joint_id]
            data.ctrl[shoulder_act_id] = kp * shoulder_error - kd * shoulder_vel
            data.ctrl[shoulder_act_id] = np.clip(data.ctrl[shoulder_act_id], -500, 500) 
            # 肘关节控制
            elbow_error = elbow_target - data.qpos[elbow_joint_id]
            elbow_vel = data.qvel[elbow_joint_id]
            data.ctrl[elbow_act_id] = kp * elbow_error - kd * elbow_vel
            data.ctrl[elbow_act_id] = np.clip(data.ctrl[elbow_act_id], -800, 800)

            # 夹爪保持张开状态
            data.ctrl[model.actuator("left").id] = 0.0
            data.ctrl[model.actuator("right").id] = 0.0

            # 运行一个仿真步
            mujoco.mj_step(model, data)

            # 更新可视化和时间
            viewer.sync()
            time.sleep(timestep)
            sim_time += timestep


if __name__ == "__main__":
    # 检查依赖
    try:
        import mujoco
    except ImportError:
        print("请先安装mujoco：pip install mujoco numpy")
    else:
        main()
