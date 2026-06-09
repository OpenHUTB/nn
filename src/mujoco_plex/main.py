import os
import mujoco
import mujoco.viewer
import time
from typing import Optional


def load_mujoco_model(model_path: str) -> Optional[tuple[mujoco.MjModel, mujoco.MjData]]:
    if not isinstance(model_path, str):
        print(f"❌ 模型路径必须为字符串类型，当前类型：{type(model_path)}")
        return None

    abs_model_path = os.path.abspath(model_path)
    if not os.path.exists(abs_model_path):
        print(f"❌ 模型文件不存在：{abs_model_path}")
        return None

    try:
        model = mujoco.MjModel.from_xml_path(abs_model_path)
        data = mujoco.MjData(model)
        print(f"✅ 成功加载模型：{abs_model_path}")
        return model, data
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return None


def configure_robot_stability(model: mujoco.MjModel, data: mujoco.MjData):
    """
    新版 MuJoCo 完全兼容版：不会出现 shape 不匹配
    """
    # 安全获取 base ID
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    if base_id >= 0:
        # 正确写法：只改前3位，避免 shape 不匹配
        model.body_pos[base_id][:3] = [0, 0, 0.5]
        model.body_quat[base_id][:4] = [1, 0, 0, 0]

    # 仿真参数（标准写法）
    model.opt.timestep = 0.002
    model.opt.gravity[:] = [0, 0, -9.81]

    # 初始化关节控制量
    for i in range(model.nu):
        data.ctrl[i] = 0.0


def run_simulation(model: mujoco.MjModel, data: mujoco.MjData):
    print("✅ 仿真启动成功！机器人已稳定运行")
    print("❌ 关闭窗口即可退出")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_fps = 60
        frame_interval = 1.0 / target_fps

        while viewer.is_running():
            start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)


def main():
    MODEL_PATH = "anybotics_anymal_c/anymal_c.xml"
    model_data = load_mujoco_model(MODEL_PATH)
    if not model_data:
        return
    model, data = model_data

    configure_robot_stability(model, data)
    run_simulation(model, data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ 程序手动退出")
    except Exception as e:
        print(f"\n❌ 错误：{e}")