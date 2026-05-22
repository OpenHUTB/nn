
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "当脚本运行后，OpenCV 窗口将弹出第一人称驾驶视角。驾驶车辆接近十字路口的停止标志时，系统表现如下："
new = "脚本运行后，OpenCV 窗口将弹出第一人称驾驶视角。驾驶车辆接近十字路口的停止标志时，系统表现如下："
t = t.replace(old, new, 1)

old2 = "- **端到端延迟**：稳定在 **15~25 ms**，远优于 L2 级辅助驾驶感知规范（<100 ms）。"
new2 = "- **端到端延迟**：稳定在 **15~25 ms**，明显优于 L2 级辅助驾驶感知规范要求（<100 ms）。"
t = t.replace(old2, new2, 1)

old3 = "若降级为 CPU 推理，延迟高达 80~120 ms。以车辆 60 km/h 行驶为例，本 GPU 加速方案将系统的\"感知盲区\"由 1.66 米大幅压缩至极其安全的 **0.4 米**以内。"
new3 = "若降级为 CPU 推理，延迟高达 80~120 ms。以车辆 60 km/h 行驶为例，本 GPU 加速方案将系统的\"感知盲区\"由 1.66 米大幅缩短至极其安全的 **0.4 米**以内。"
t = t.replace(old3, new3, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
