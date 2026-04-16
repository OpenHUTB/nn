自动驾驶车道与路径检测
=========================================
## 在 GTAV 上运行 OpenPilot
本项目是对 littlemountainman/modeld项目的一个分支。
我们利用了他的工作，并将 DeepGTAV 和 VPilot 结合，从而能够将 comma.ai 的开源软件应用于 GTAV，并创建出由 openpilot 算法管理的自动驾驶车辆。

## 如何安装

为了能够运行此项目，我推荐使用 Python 3.7 或更高版本。

1. 安装所需依赖包

```
pip3 install -r requirements.txt
```

这将安装运行此项目所需的所有必要依赖。

2. 下载 [Vpilot with DeepGTAV](https://github.com/aitorzip/VPilot)

3. 下载 [ScriptHook](https://www.dev-c.com/gtav/scripthookv/)

4. 下载 [DeepGTA](https://github.com/aitorzip/DeepGTAV)

5. 将 ScriptHookV.dll、dinput8.dll、NativeTrainer.asi文件复制到游戏的主文件夹，即 GTA5.exe所在的目录。

6. 将 DeepGTAV/bin/Release文件夹下的所有内容复制并粘贴到你的 GTAV 游戏安装目录下。

7. 启动程序（确保 GTAV 已在运行）
```
python3 main.py
```


## 感谢

[littlemountainman/modeld](https://github.com/littlemountainman/modeld)
[aitorzip/DeepGTAV](https://github.com/aitorzip/DeepGTAV)
[aitorzip/VPilot](https://github.com/aitorzip/VPilot)


=========================================

## 在 CARLA 模拟器上运行自动驾驶

**如果你没有 GTAV，或者希望在更真实的自动驾驶仿真环境中进行开发，可以使用 CARLA 模拟器运行automatic_control.py**

CARLA 是一个开源的自动驾驶仿真器，提供更专业的交通场景、传感器模拟和车辆动力学。

### CARLA 版本要求

- **CARLA 0.9.15**
- Python 3.7 - 3.9

### 安装步骤

#### 1. 下载 CARLA 模拟器

从 [CARLA 官方 GitHub](https://github.com/carla-simulator/carla/releases/tag/0.9.15) 下载 `CARLA_0.9.15.zip`
1. 安装所需依赖包

```
pip3 install -r Requirements.txt
```
## 启动 CARLA 模拟器

# 进入 CARLA 安装目录
cd H:\carla0.9.15\WindowsNoEditor

# 启动模拟器
CarlaUE4.exe

# 运行自动驾驶程序
# 确保已激活虚拟环境
python code/automatic_control.py
本项目的 CARLA 版本支持以下功能：

🚦 交通信号灯检测
实时识别交通信号灯状态（红、黄、绿）

根据信号灯状态自动停车/通行

支持多路口复杂场景

🛣️ 车道线检测与保持
基于 OpenCV 的车道线识别

车道保持辅助（LKA）

弯道自适应速度控制

🚗 自动驾驶行为模式
谨慎模式 (Cautious)：保持安全距离，提前减速

正常模式 (Normal)：平衡效率与安全

激进模式 (Aggressive)：更快的加速和跟车

📊 实时可视化
摄像头画面实时显示

车道线检测结果叠加

交通信号灯状态标注

车辆速度、档位、转向角等信息

## 常见问题
Q: 连接 CARLA 时提示版本不匹配
A: 确保 Python API 版本与 CARLA 模拟器版本一致：
# 检查 Python API 版本
python -c "import carla; print(carla.__version__)"

# 检查模拟器版本（启动时控制台会显示）