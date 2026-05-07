import time
import mujoco
from mujoco import viewer

def main():
    # 模型路径（和文件同目录，直接写文件名）
    model_path = "d:/study/nn/src/muscle_humanoid/humanoid.xml"

    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 只设置初始高度，不设置任何关节姿态，避免维度不匹配
    data.qpos[2] = 1.3

    print("✅ 启动成功！")

    with viewer.launch_passive(model, data) as v:
        # 自动调好视角
        v.cam.distance = 4
        v.cam.elevation = -20
        v.cam.azimuth = 120

        # 物理循环
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.005)

if __name__ == "__main__":
    main()