
import sys
md = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md, "rb") as f: raw = f.read()
t = raw.decode("utf-8")

old = "为了复现本实验，请确保开发环境满足以下最低配置："
new = "为了复现本实验，请确保开发环境满足以下最低配置要求："
t = t.replace(old, new, 1)

old2 = "# 1. 进入工作区并克隆仓库\ncd /d/workspace\ngit clone https://github.com/OpenHUTB/nn.git\ncd nn\ngit checkout dev-yss\n\n# 2. 安装 Python 依赖\n# 注意：carla 库需从本地 whl 安装\npip install ultralytics opencv-python numpy\npip install /path/to/carla/PythonAPI/carla/dist/carla-0.9.13-cp38-cp38-win_amd64.whl\n\n# 3. 启动 Carla 模拟器 (需单独打开终端运行 CarlaUE4.exe)\n\n# 4. 运行感知脚本\npython src/carla_traffic_sign_recognition/main.py"
new2 = "# 1. 进入工作区并克隆仓库\ncd /d/workspace\ngit clone https://github.com/OpenHUTB/nn.git\ncd nn\ngit checkout dev-yss\n\n# 2. 安装 Python 依赖\n# 注意：carla 库需从本地 whl 安装，请根据实际 Python 版本选择对应 whl\npip install ultralytics opencv-python numpy\npip install /path/to/carla/PythonAPI/carla/dist/carla-0.9.13-cp38-cp38-win_amd64.whl\n\n# 3. 启动 Carla 模拟器 (需单独打开终端运行 CarlaUE4.exe)\n\n# 4. 运行感知脚本\npython src/carla_traffic_sign_recognition/main.py"
t = t.replace(old2, new2, 1)
with open(md, "wb") as f: f.write(t.encode("utf-8"))
print("OK")
