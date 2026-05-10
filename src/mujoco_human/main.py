import os
import time
import mujoco
from mujoco import viewer


def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humanoid.xml")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 1)

    print("启动成功！人体模型仿真已启动")
    print("操作说明：")
    print("  - 鼠标左键拖拽：旋转视角")
    print("  - 鼠标右键拖拽：平移视角")
    print("  - 滚轮：缩放")
    print("  - 双击：选中物体")
    print("  - Ctrl+右键：施加力")

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
