
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "在现代自动驾驶系统（ADS，Autonomous Driving System）的感知-规划-控制链条中，**视觉感知模块**是连接物理世界与数字决策的首要接口。"
new = "在现代自动驾驶系统（ADS, Autonomous Driving System）的感知-规划-控制链条中，**视觉感知模块**是连接物理世界与数字决策的关键接口。"
t = t.replace(old, new, 1)

old2 = "交通标志识别（TSR，Traffic Sign Recognition）具有极高的合规性与安全性权重。为此，本模块精准切入这一架构空白，旨在构建一个低延迟、高鲁棒性的端到端视觉感知验证流水线，确保车辆能够在复杂光照与城市背景下准确捕获交通指示信息。"
new2 = "交通标志识别（TSR, Traffic Sign Recognition）具有极高的合规性与安全性权重。为此，本模块精准切入这一架构空白，旨在构建一套低延迟、高鲁棒性的端到端视觉感知验证流水线，确保车辆能够在复杂光照与城市背景下准确捕获交通指示信息。"
t = t.replace(old2, new2, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
