import mujoco
import mujoco.viewer
import os

# 同目录下的 xml 路径
xml_path = os.path.join(os.path.dirname(__file__), "humanoid.xml")

print("正在加载模型：", xml_path)
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("模型加载成功，启动 viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("按关闭窗口 或 Ctrl+C 退出")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()