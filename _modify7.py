
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "在实际部署过程中，遇到了 Windows 系统与 Git 底层交互的特定报错，以下是详细排雷过程，具有极高的教学参考价值。"
new = "在实际部署过程中，遇到了 Windows 系统与 Git 底层交互的特定报错，以下是详细排雷过程，具有较高的教学参考价值。"
t = t.replace(old, new, 1)

old2 = "- **现象**：执行 `git add` 时终端报错 `error: open(\"nul\"): Permission denied` 或 `unable to index file 'nul'`。"
new2 = "- **现象**：执行 `git add` 时终端报错 `error: open(\"nul\"): Permission denied` 或 `unable to index file 'nul'` 等异常信息。"
t = t.replace(old2, new2, 1)

old3 = "- **根因分析**：国内网络环境访问境外 Amazon AWS 服务器（GitHub 托管地）存在间歇性丢包或 DNS 污染。"
new3 = "- **根因分析**：国内网络环境访问境外 Amazon Web Services 服务器（GitHub 托管地）存在间歇性丢包或 DNS 污染现象。"
t = t.replace(old3, new3, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
