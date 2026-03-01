import mujoco
import mujoco_viewer
import numpy as np
import time


def main():
    # 1. 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path("arm.xml")
    data = mujoco.MjData(model)

    # 2. 创建可视化器
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 3. 设置仿真参数
    sim_duration = 10.0  # 仿真总时长（秒）
    dt = model.opt.timestep  # 仿真步长（0.005秒）
    total_steps = int(sim_duration / dt)

    # 4. 定义关节目标轨迹（简单的正弦运动）
    def get_joint_targets(t):
        """根据时间t生成关节目标角度"""
        joint1_target = np.sin(t * 0.5) * 1.0  # 基座旋转（-1~1 rad）
        joint2_target = np.cos(t * 0.7) * 0.8  # 大臂摆动（-0.8~0.8 rad）
        joint3_target = np.sin(t * 0.9) * 0.6  # 小臂摆动（-0.6~0.6 rad）
        return [joint1_target, joint2_target, joint3_target]

    # 5. 运行仿真循环
    try:
        for step in range(total_steps):
            # 获取当前时间
            t = step * dt

            # 设置关节目标位置
            targets = get_joint_targets(t)
            data.ctrl[:] = targets  # 将目标值赋值给控制器

            # 运行一步仿真
            mujoco.mj_step(model, data)

            # 更新可视化窗口
            viewer.render()

            # 可选：控制仿真速度（匹配实时）
            time.sleep(dt)

            # 打印关键信息（每100步）
            if step % 100 == 0:
                print(f"Time: {t:.2f}s | Joint angles: {data.qpos[:3]}")

    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    finally:
        # 关闭可视化器
        viewer.close()
        print("仿真结束")


if __name__ == "__main__":
    main()