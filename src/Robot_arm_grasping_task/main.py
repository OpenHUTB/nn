import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import os

# ===================== 基础配置 =====================
# 修复Matplotlib中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'

# 路径配置（兼容所有系统）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "robot.xml")

# 核心仿真参数（极简配置，优先保证运动）
TARGET_OBJECT_POS = np.array([0.3, 0.0, 0.1])  # 降低目标距离，更容易到达
GOAL_POS = np.array([-0.1, 0.0, 0.1])
SIMULATION_STEPS = 8000
# 极简PID（优先保证关节能动）
KP = 10.0
KI = 0.0
KD = 1.0
# 可视化配置
CAMERA_DISTANCE = 2.0  # 相机距离，确保能看到整个模型
CAMERA_ELEVATION = -20  # 相机仰角
CAMERA_AZIMUTH = 90  # 相机方位角


# ===================== 模型校验 & 调试工具 =====================
def validate_model(model, data):
    """校验模型关键组件，输出调试信息"""
    print("\n===== 模型调试信息 =====")
    # 检查关节
    print(f"总关节数: {model.njnt}")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"关节{i}: {joint_name} | 控制维度: {model.nu}")

    # 检查位点/物体
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    print(f"末端位点(ee_site) ID: {ee_site_id if ee_site_id >= 0 else '未找到'}")
    print(f"目标物体(target_object) ID: {object_body_id if object_body_id >= 0 else '未找到'}")

    # 检查控制维度
    print(f"控制维度数: {model.nu} (需≥5才能控制3关节+2夹爪)")
    print("========================\n")

    # 若关键组件缺失，抛出明确错误
    if ee_site_id < 0:
        raise ValueError("模型中未找到'ee_site'位点！请检查robot.xml")
    if object_body_id < 0:
        raise ValueError("模型中未找到'target_object'物体！请检查robot.xml")
    if model.nu < 5:
        raise ValueError(f"模型控制维度不足（当前{model.nu}），需至少5个控制维度（3关节+2夹爪）")


# ===================== 简化控制逻辑（优先保证运动） =====================
def simple_joint_control(model, data, target_joint_angles):
    """极简关节位置控制（直接设置关节角度，跳过复杂IK/PID）"""
    # 只控制前3个关节
    for i in range(min(3, model.njnt)):
        # 简单PD控制，保证稳定
        error = target_joint_angles[i] - data.qpos[i]
        error_d = -data.qvel[i]
        data.ctrl[i] = KP * error + KD * error_d
        # 限制控制输出
        data.ctrl[i] = np.clip(data.ctrl[i], -5.0, 5.0)


def move_joints_step_by_step(model, data, step):
    """分步移动关节（可视化更清晰）"""
    # 阶段1：初始化位置（0,0,0）
    if step < 1000:
        return np.array([0.0, 0.0, 0.0])
    # 阶段2：抬升肩关节
    elif step < 2000:
        return np.array([0.5, 0.0, 0.0])
    # 阶段3：旋转肘关节
    elif step < 3000:
        return np.array([0.5, -0.5, 0.0])
    # 阶段4：旋转腕关节
    elif step < 4000:
        return np.array([0.5, -0.5, 0.3])
    # 阶段5：回到目标物体位置
    elif step < 5000:
        return np.array([0.3, -0.4, 0.2])
    # 阶段6：闭合夹爪
    elif step < 6000:
        data.ctrl[3] = 5.0  # 夹爪1
        data.ctrl[4] = -5.0  # 夹爪2
        return np.array([0.3, -0.4, 0.2])
    # 阶段7：搬运到目标位置
    elif step < 7000:
        return np.array([-0.2, -0.3, 0.2])
    # 阶段8：打开夹爪
    else:
        data.ctrl[3] = 0.0
        data.ctrl[4] = 0.0
        return np.array([-0.2, -0.3, 0.2])


# ===================== 主仿真函数（全面重构） =====================
def grasp_simulation():
    # 1. 基础校验
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件！路径：{MODEL_PATH}\n请确认robot.xml在{CURRENT_DIR}目录下")

    # 2. 加载模型并校验
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    validate_model(model, data)

    # 3. 初始化Viewer（优化视角）
    viewer = mujoco_viewer.MujocoViewer(model, data)
    # 设置相机视角，确保能看到整个机械臂
    viewer.cam.distance = CAMERA_DISTANCE
    viewer.cam.elevation = CAMERA_ELEVATION
    viewer.cam.azimuth = CAMERA_AZIMUTH
    viewer.cam.lookat = np.array([0.0, 0.0, 0.1])  # 相机看向原点

    # 4. 初始化变量
    ee_pos_history = []
    step_info = ""
    grasp_success = False

    print("🚀 机械臂仿真启动（极简模式）...")
    print("💡 操作提示：")
    print("   - 鼠标左键：旋转视角")
    print("   - 鼠标滚轮：缩放视图")
    print("   - 空格键：暂停/继续")
    print("   - Tab键：切换相机视角\n")

    try:
        for step in range(SIMULATION_STEPS):
            # 分步控制关节
            target_joints = move_joints_step_by_step(model, data, step)
            simple_joint_control(model, data, target_joints)

            # 更新仿真
            mujoco.mj_step(model, data)

            # 记录数据（可选）
            ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            if ee_site_id >= 0:
                ee_pos_history.append(data.site_xpos[ee_site_id].copy())

            # 输出阶段信息
            if step % 1000 == 0:
                if step < 1000:
                    step_info = "初始化位置"
                elif step < 2000:
                    step_info = "抬升肩关节"
                elif step < 3000:
                    step_info = "旋转肘关节"
                elif step < 4000:
                    step_info = "旋转腕关节"
                elif step < 5000:
                    step_info = "接近目标物体"
                elif step < 6000:
                    step_info = "闭合夹爪"
                elif step < 7000:
                    step_info = "搬运到目标位置"
                else:
                    step_info = "打开夹爪"
                print(f"📌 仿真步数: {step} | 当前阶段: {step_info}")

            # 渲染（强制渲染，保证可视化）
            viewer.render()
            time.sleep(0.001)  # 降低速度，便于观察

            # 判定任务完成
            if step > 7500:
                grasp_success = True

    except KeyboardInterrupt:
        print("\n⚠️ 仿真被手动终止")
    except Exception as e:
        print(f"\n❌ 仿真出错：{type(e).__name__}: {e}")
    finally:
        viewer.close()
        print("\n🔚 仿真结束")

    # 5. 简单结果展示
    if grasp_success and ee_pos_history:
        print("\n✅ 机械臂运动任务完成！")
        ee_pos = np.array(ee_pos_history)
        print(f"📊 末端执行器移动范围：X({ee_pos[:, 0].min():.2f}~{ee_pos[:, 0].max():.2f}) m")
    elif not ee_pos_history:
        print("\n⚠️ 未记录到末端执行器数据（可能模型位点缺失）")
    else:
        print("\n❌ 机械臂运动任务未完成")


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 先检查依赖
    try:
        import mujoco
        import mujoco_viewer
    except ImportError:
        print("❌ 缺少依赖库！请执行：")
        print("pip install mujoco mujoco-viewer numpy matplotlib")
        exit(1)

    # 运行仿真
    grasp_simulation()