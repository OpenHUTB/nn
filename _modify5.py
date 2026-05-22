
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "本模块严格遵循 Carla 官方推荐的 **Client-Server 异步架构**。图 2.1 展示了系统内部的数据流转闭环。"
new = "本模块严格遵循 Carla 官方推荐的 **Client-Server 异步架构**。下图展示了系统内部的数据流转闭环。"
t = t.replace(old, new, 1)

old2 = "Carla 服务端（通常通过 `CarlaUE4.exe` 运行）承担高负载的物理计算与图形渲染任务。"
new2 = "Carla 服务端（通常通过 `CarlaUE4.exe` 启动）承担高负载的物理计算与图形渲染任务。"
t = t.replace(old2, new2, 1)

old3 = "1. **会话管理**：通过 `carla.Client` 与模拟器握手并保持长连接。"
new3 = "1. **会话管理**：通过 `carla.Client` 与模拟器建立连接并维持长连接。"
t = t.replace(old3, new3, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
