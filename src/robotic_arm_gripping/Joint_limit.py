import mujoco
import mujoco.viewer
import time
import os

# ===================== 配置项（集中管理，易修改） =====================
MODEL_PATH = "arm_model.xml"  # 模型文件路径
CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 45,
    "elevation": -20,
    "lookat": [0.2, 0, 0.4]
}
STEP_DELAY = 0.01  # 仿真步延迟（控制运动速度）
JOINT_STEP_DURATION = 300  # 每个限位阶段的步数
TORQUE_VALUE = 2.0  # 关节控制力矩（匹配模型ctrlrange）


# ===================== 工具函数（解耦核心逻辑） =====================
def load_mujoco_model(model_path: str):
    """
    加载MuJoCo模型，包含完整的错误处理
    :param model_path: 模型文件路径
    :return: (model, data) 或 None（加载失败）
    """
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 → {model_path}")
        print("  请检查文件路径是否正确，避免中文/空格")
        return None

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
    初始化可视化窗口，配置相机参数（移除类型注解，兼容所有版本）
    :param model: MuJoCo模型
    :param data: MuJoCo数据
    :param camera_config: 相机配置字典
    :return: 可视化窗口对象
    """
    viewer = mujoco.viewer.launch(model, data)
    viewer.cam.distance = camera_config["distance"]
    viewer.cam.azimuth = camera_config["azimuth"]
    viewer.cam.elevation = camera_config["elevation"]
    viewer.cam.lookat = camera_config["lookat"]
    return viewer


# ===================== 核心逻辑 =====================
def joint_limit_test():
    """关节限位测试主逻辑"""
    # 1. 加载模型
    model_data = load_mujoco_model(MODEL_PATH)
    if not model_data:
        return
    model, data = model_data

    # 2. 定义关节配置（结构化管理，易扩展）
    joint_configs = {
        "shoulder": {
            "name": "肩关节",
            "act_id": model.actuator("shoulder").id,
            "limits": (-TORQUE_VALUE, 0, TORQUE_VALUE, 0),  # 左极限→回中→右极限→回中
            "desc": ("左极限", "回中", "右极限", "回中")
        },
        "elbow": {
            "name": "肘关节",
            "act_id": model.actuator("elbow").id,
            "limits": (-TORQUE_VALUE, 0, TORQUE_VALUE, 0),  # 下极限→回中→上极限→回中
            "desc": ("下极限（弯曲）", "回中", "上极限（伸展）", "回中")
        }
    }
    joint_names = list(joint_configs.keys())
    current_joint_idx = 0  # 当前测试的关节索引
    current_phase_idx = 0  # 当前测试阶段索引（0-3）

    # 3. 初始化可视化
    viewer = init_viewer(model, data, CAMERA_CONFIG)
    print("\n===== 关节限位测试 =====")
    print(f"测试关节：{[joint_configs[j]['name'] for j in joint_names]}")
    print("操作说明：按ESC键退出测试\n")

    # 4. 测试主循环
    step = 0
    while viewer.is_running():
        # 获取当前测试的关节配置
        current_joint_name = joint_names[current_joint_idx]
        joint_cfg = joint_configs[current_joint_name]

        # 阶段切换逻辑（每JOINT_STEP_DURATION步切换一次阶段）
        if step % JOINT_STEP_DURATION == 0:
            # 打印当前阶段信息
            phase_desc = joint_cfg["desc"][current_phase_idx]
            print(f"{joint_cfg['name']}：{phase_desc}")

            # 更新关节控制力矩
            data.ctrl[joint_cfg["act_id"]] = joint_cfg["limits"][current_phase_idx]

            # 阶段索引递增（0→1→2→3→0）
            current_phase_idx += 1
            if current_phase_idx >= len(joint_cfg["limits"]):
                # 当前关节测试完成，切换到下一个关节
                current_phase_idx = 0
                current_joint_idx = (current_joint_idx + 1) % len(joint_names)
                print(f"\n开始测试{joint_configs[joint_names[current_joint_idx]]['name']}")

        # 执行仿真步
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(STEP_DELAY)
        step += 1

    # 关闭可视化窗口
    viewer.close()
    print("\n✅ 关节限位测试结束")


# ===================== 程序入口 =====================
if __name__ == "__main__":
    joint_limit_test()