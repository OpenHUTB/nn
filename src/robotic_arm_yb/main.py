import mujoco
import mujoco.viewer
import numpy as np
import time
import os


def load_arm_model(model_path):
    """加载机械臂模型，增加路径校验"""
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型和数据
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"✅ 成功加载模型")
        print(f"  - 关节数量: {model.njnt}")
        print(f"  - 电机数量: {model.nu}")
        return model, data
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")


def set_joint_positions(model, data, joint_positions):
    """设置机械臂关节初始位置"""
    if len(joint_positions) != model.njnt:
        raise ValueError(f"关节数量不匹配！需要{model.njnt}个，输入{len(joint_positions)}个")

    # 重置关节位置和速度
    data.qpos[:model.njnt] = joint_positions
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    # 更新仿真状态
    mujoco.mj_forward(model, data)


def pd_controller(model, data, target_pos, kp=300, kv=40):
    """PD控制器实现关节位置跟踪"""
    # 获取当前关节位置和速度
    current_pos = data.qpos[:model.njnt]
    current_vel = data.qvel[:model.njnt]

    # 计算关节误差
    pos_error = target_pos - current_pos

    # PD控制律
    ctrl = kp * pos_error - kv * current_vel

    # 限制控制输出
    ctrl = np.clip(ctrl, -500, 500)

    # 设置控制输入
    data.ctrl[:] = ctrl

    return pos_error


def print_simulation_status(data, step, pos_error):
    """打印仿真状态（每100步打印一次）"""
    if step % 100 == 0:
        avg_error = np.mean(np.abs(pos_error))
        print(f"\n📊 仿真步数: {step}")
        print(f"  - 平均关节误差: {avg_error:.4f} rad")


def main():
    # 1. 模型路径（终极兼容版XML）
    model_path = "arm6dof_final.xml"

    # 2. 加载模型
    try:
        model, data = load_arm_model(model_path)
    except Exception as e:
        print(f"❌ {e}")
        return

    # 3. 定义目标关节姿态（弧度）
    home_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lift_pose = np.array([0.0, -0.6, 0.9, 0.0, -0.4, 0.0])
    rotate_pose = np.array([0.7, -0.6, 0.9, 0.4, -0.4, 0.3])

    # 4. 初始化关节位置
    set_joint_positions(model, data, home_pose)

    # 5. 启动仿真可视化（适配旧版viewer接口）
    print("\n🚀 启动机械臂仿真...")
    print("   - 按ESC键退出仿真")
    print("   - 仿真时长: 20秒，自动切换3种姿态")

    # 兼容不同版本的viewer启动方式
    viewer = mujoco.viewer.launch(model, data)

    # 仿真参数
    sim_time = 0.0
    sim_duration = 20.0
    dt = model.opt.timestep
    step = 0

    # 姿态切换时间点
    phase1 = 5.0
    phase2 = 10.0
    phase3 = 15.0

    try:
        while sim_time < sim_duration:
            # 检查viewer是否关闭
            if not viewer.is_running():
                break

            # 选择目标姿态
            if sim_time < phase1:
                target = home_pose
            elif sim_time < phase2:
                target = lift_pose
            elif sim_time < phase3:
                target = rotate_pose
            else:
                target = home_pose

            # PD控制
            pos_error = pd_controller(model, data, target)

            # 运行仿真步
            mujoco.mj_step(model, data)

            # 更新可视化（适配旧版接口）
            viewer.sync()

            # 打印状态
            print_simulation_status(data, step, pos_error)

            # 控制仿真速率
            time.sleep(dt)

            # 更新时间和步数
            sim_time += dt
            step += 1
    finally:
        # 确保viewer正常关闭
        viewer.close()

    print("\n✅ 仿真结束！")


if __name__ == "__main__":
    # 检查MuJoCo是否安装
    try:
        import mujoco
    except ImportError:
        print("❌ 未安装MuJoCo！请执行：pip install mujoco -i https://pypi.tuna.tsinghua.edu.cn/simple")
    else:
        main()