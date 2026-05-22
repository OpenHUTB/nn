
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "本项目源于《基础软件和开源系统》课程的实践要求，旨在通过向开源社区（`OpenHUTB/nn`）贡献高质量的代码与文档，完整走通 **\"Issue 分析 -> 分支管理 -> 功能开发 -> 测试验证 -> 文档撰写 -> Pull Request 提交\"** 的标准化软件工程流程。"
new = "本项目源于《基础软件和开源系统》课程实践要求，旨在通过向开源社区（`OpenHUTB/nn`）贡献高质量的代码与文档，完整走通 **\"Issue 分析 -> 分支管理 -> 功能开发 -> 测试验证 -> 文档撰写 -> Pull Request 提交\"** 的标准化软件工程全流程。"
t = t.replace(old, new, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
