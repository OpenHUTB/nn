import mujoco
import mujoco.viewer as viewer
import os

# 动态获取脚本所在目录，确保模型路径正确
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "humanoid.xml")

def main():
    if not os.path.exists(model_path):
        print(f"错误：未找到模型文件 {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"模型自由度总数: {model.nq}")

    # 1. 先把模型抬起来，离开地面
    if model.nq > 2:
        data.qpos[2] = 1.2

    # 2. 关键！强制模型直立（修正根关节四元数）
    # 这组数值代表模型正面朝上、垂直站立
    if model.nq > 6:
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0

    # 3. 重置所有关节角度，让模型保持“立正”姿势
    for i in range(7, model.nq):
        data.qpos[i] = 0.0

    print("模型加载成功，启动模拟...")

    # 优化相机视角，方便观察
    with viewer.launch_passive(model, data) as v:
        v.cam.distance = 5
        v.cam.elevation = -20
        v.cam.azimuth = 90

        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

    print("模拟结束")

if __name__ == "__main__":
    main()