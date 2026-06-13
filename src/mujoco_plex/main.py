import os
import time
from typing import Optional, Tuple

import mujoco
import mujoco.viewer

# ===================== 配置中心（便于维护）=====================
CONFIG = {
    "model_path": "anybotics_anymal_c/anymal_c.xml",
    "base_body": "base",
    "base_pos": (0.0, 0.0, 0.5),
    "base_quat": (1.0, 0.0, 0.0, 0.0),
    "time_step": 0.002,
    "gravity": (0.0, 0.0, -9.81),
    "target_fps": 60,
    "print_interval": 1.0,  # 打印间隔(秒)
}

# ===================== 模型加载 =====================
def load_mujoco_model(model_path: str) -> Optional[Tuple[mujoco.MjModel, mujoco.MjData]]:
    """加载 MuJoCo 模型，带路径校验与异常捕获"""
    if not isinstance(model_path, str):
        print(f"❌ 模型路径必须为字符串，当前类型：{type(model_path)}")
        return None

    abs_path = os.path.abspath(model_path)
    if not os.path.isfile(abs_path):
        print(f"❌ 模型不存在：{abs_path}")
        return None

    try:
        model = mujoco.MjModel.from_xml_path(abs_path)
        data = mujoco.MjData(model)
        print(f"✅ 模型加载成功：{abs_path}")
        return model, data
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return None

# ===================== 机器人初始化 =====================
def configure_robot(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """配置机器人初始状态与仿真参数"""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CONFIG["base_body"])

    if base_id >= 0:
        model.body_pos[base_id][:3] = CONFIG["base_pos"]
        model.body_quat[base_id][:4] = CONFIG["base_quat"]

    # 仿真参数
    model.opt.timestep = CONFIG["time_step"]
    model.opt.gravity[:] = CONFIG["gravity"]

    # 控制量清零
    data.ctrl[:] = 0.0

# ===================== 仿真主循环 =====================
def run_simulation(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """运行稳定帧率仿真"""
    print("✅ 仿真启动成功 | 关闭窗口退出")
    frame_interval = 1.0 / CONFIG["target_fps"]
    print_timer = 0.0
    sim_start_time = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_start = time.perf_counter()
            current_sim_time = data.time

            # 仿真步进
            mujoco.mj_step(model, data)
            viewer.sync()

            # 定时打印仿真时长 + 关节角度
            if current_sim_time - print_timer >= CONFIG["print_interval"]:
                run_duration = time.perf_counter() - sim_start_time
                print(f"【运行时长】{run_duration:.2f}s | 仿真时间: {current_sim_time:.2f}s")
                joint_qpos = data.qpos[7:]  # 前7维为基座位姿，后续为关节
                print(f"【关节角度】{joint_qpos[:6].round(3)}")
                print_timer = current_sim_time

            # 帧率控制
            elapsed = time.perf_counter() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

# ===================== 主入口 =====================
def main() -> None:
    model_data = load_mujoco_model(CONFIG["model_path"])
    if not model_data:
        return

    model, data = model_data
    configure_robot(model, data)
    run_simulation(model, data)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ 程序手动退出")
    except Exception as e:
        print(f"\n❌ 运行错误：{e}")