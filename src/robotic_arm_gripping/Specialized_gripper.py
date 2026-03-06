import mujoco
import mujoco.viewer
import time
import os

# ===================== 配置项（集中管理，一键修改） =====================
MODEL_PATH = "arm_model.xml"  # 模型文件路径（建议替换为绝对路径）
# 相机配置（聚焦夹爪，看清细节）
CAMERA_CONFIG = {
    "distance": 1.0,
    "azimuth": 60,
    "elevation": -10,
    "lookat": [0.4, 0, 0.4]
}
STEP_DELAY = 0.005  # 仿真步延迟（控制夹爪运动速度）
# 夹爪运动阶段配置（易扩展，可调整各阶段时长）
GRIPPER_PHASES = {
    "open": {"steps": 400, "left_torque": -1.0, "right_torque": -1.0, "desc": "张开夹爪"},
    "close": {"steps": 400, "left_torque": 1.0, "right_torque": 1.0, "desc": "闭合夹爪"},
    "hold": {"steps": 200, "left_torque": 1.0, "right_torque": 1.0, "desc": "保持闭合"}
}


# ===================== 工具函数（解耦核心逻辑，提升鲁棒性） =====================
def load_model(model_path):
    """
    加载MuJoCo模型，包含完整的错误处理
    :param model_path: 模型文件路径
    :return: (model, data) 或 None（加载失败）
    """
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 → {model_path}")
        print("  建议替换为绝对路径，例如：C:/Users/XXX/arm_model.xml")
        return None

    # 捕获模型加载异常
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"✅ 模型加载成功 → {model_path}")
        return model, data
    except Exception as e:
        print(f"❌ 模型加载失败 → {e}")
        return None


def init_viewer(model, data, camera_config):
    """
    初始化可视化窗口（兼容所有MuJoCo版本，避免类型注解报错）
    :param model: MuJoCo模型对象
    :param data: MuJoCo数据对象
    :param camera_config: 相机配置字典
    :return: 可视化窗口对象
    """
    # 兼容新旧版本的viewer启动方式
    try:
        viewer = mujoco.viewer.launch(model, data)
    except AttributeError:
        viewer = mujoco.viewer.launch_passive(model, data)

    # 配置相机参数，聚焦夹爪
    viewer.cam.distance = camera_config["distance"]
    viewer.cam.azimuth = camera_config["azimuth"]
    viewer.cam.elevation = camera_config["elevation"]
    viewer.cam.lookat = camera_config["lookat"]
    return viewer


# ===================== 核心逻辑（夹爪控制） =====================
def gripper_control_test():
    """夹爪单独控制测试主函数"""
    # 1. 加载模型
    model_data = load_model(MODEL_PATH)
    if not model_data:
        return
    model, data = model_data

    # 2. 获取夹爪执行器ID（增加异常处理）
    try:
        left_act_id = model.actuator("left").id
        right_act_id = model.actuator("right").id
    except Exception as e:
        print(f"❌ 获取夹爪执行器失败 → {e}")
        print("  请检查模型中actuator的name是否为'left'/'right'")
        return

    # 3. 初始化可视化窗口
    viewer = init_viewer(model, data, CAMERA_CONFIG)
    print("\n===== 夹爪单独控制测试 =====")
    print("操作说明：按ESC键退出测试")
    print("运动流程：", " → ".join([phase["desc"] for phase in GRIPPER_PHASES.values()]), "→ 循环\n")

    # 4. 阶段管理（简化循环逻辑，易扩展）
    phase_list = list(GRIPPER_PHASES.values())
    current_phase_idx = 0  # 当前阶段索引
    phase_step = 0  # 当前阶段内的步数

    while viewer.is_running():
        # 获取当前阶段配置
        current_phase = phase_list[current_phase_idx]

        # 打印阶段提示（仅在阶段开始时打印一次）
        if phase_step == 0:
            print(f"🔄 {current_phase['desc']}（剩余步数：{current_phase['steps']}）")

        # 控制夹爪力矩
        data.ctrl[left_act_id] = current_phase["left_torque"]
        data.ctrl[right_act_id] = current_phase["right_torque"]

        # 执行仿真步
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(STEP_DELAY)

        # 阶段步数递增 & 阶段切换
        phase_step += 1
        if phase_step >= current_phase["steps"]:
            phase_step = 0
            current_phase_idx = (current_phase_idx + 1) % len(phase_list)

    # 关闭可视化窗口
    viewer.close()
    print("\n✅ 夹爪控制测试结束")


# ===================== 程序入口 =====================
if __name__ == "__main__":
    gripper_control_test()