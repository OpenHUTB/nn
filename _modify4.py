
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "- **模拟环境**：**Carla 0.9.13+**。作为开源自动驾驶模拟器的事实标准，Carla 提供逼真的物理引擎、天气系统和传感器模型，其 Python API 赋予了研究者极高的可编程自由度。"
new = "- **模拟环境**：**Carla 0.9.13+**。作为开源自动驾驶模拟器的行业标准，Carla 提供逼真的物理引擎、天气系统和传感器模型，其 Python API 赋予了研究者极高的可编程自由度。"
t = t.replace(old, new, 1)

old2 = "- **神经网络框架**：**Ultralytics YOLOv8**。YOLOv8 是当前目标检测领域 SOTA（State-of-the-Art）级别的轻量化模型。"
new2 = "- **神经网络框架**：**Ultralytics YOLOv8**。YOLOv8 是目前目标检测领域 SOTA（State-of-the-Art）级别的轻量化模型。"
t = t.replace(old2, new2, 1)

old3 = "- **可视化与交互**：**OpenCV**。利用 OpenCV 的高效 I/O 与窗口回调机制，实现传感器数据流的实时渲染与键盘事件的非阻塞捕获，为\"人在回路\"验证提供直观的操作界面。"
new3 = "- **可视化与交互**：**OpenCV**。借助 OpenCV 的高效 I/O 与窗口回调机制，实现传感器数据流的实时渲染与键盘事件的非阻塞捕获，为\"人在回路\"验证提供直观的操作界面。"
t = t.replace(old3, new3, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
