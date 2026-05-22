
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "在 Carla 中，传感器数据的获取不是标准的图像文件，而是经过序列化的一维字节流。"
new = "在 Carla 中，传感器数据的获取并非标准的图像文件，而是经过序列化的一维字节流。"
t = t.replace(old, new, 1)

old2 = "**原理**：Carla 传感器通常以 30-60 FPS 回调。若每帧都送入神经网络，主线程将被计算密集型的 CUDA 内核调用阻塞，导致渲染窗口卡顿（Stuttering）。"
new2 = "**原理**：Carla 传感器通常以 30-60 FPS 回调。若每帧都送入神经网络，主线程将被计算密集型的 CUDA 内核调用阻塞，导致渲染窗口卡顿（Stuttering）现象。"
t = t.replace(old2, new2, 1)

old3 = "- **渐进式油门/转向**：直接赋值 `control.throttle = 1.0` 会导致车辆瞬间窜出，违反物理直觉且易失控。"
new3 = "- **渐进式油门/转向**：直接赋值 `control.throttle = 1.0` 会导致车辆瞬间窜出，违背物理直觉且易于失控。"
t = t.replace(old3, new3, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
