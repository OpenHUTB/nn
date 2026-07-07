🚗 CARLA 0.9.15 自动驾驶与车道保持控制系统

基于 CARLA 仿真器（0.9.15 版本）实现的自动驾驶控制项目，涵盖交通灯识别与响应、自动路径规划与导航、实时天气切换、手动/自动驾驶模式切换等功能。项目提供了多个渐进式脚本，从最基础的自动驾驶到完整功能版，方便学习和二次开发。



目录

• 项目背景
• 功能特性
• 项目结构
• 脚本详解
• 核心架构
◦ 系统架构图
◦ 核心类说明
◦ 传感器体系
◦ 驾驶行为模式
◦ 天气系统
• 环境要求
• 安装
• 使用方法
◦ 方式一：一键启动（Windows）
◦ 方式二：手动启动
◦ 命令行参数
◦ 键盘操作
• 工作原理
◦ 交通灯检测与响应流程
◦ 自动导航流程
◦ 手动/自动切换流程
• 配置说明
• 常见问题
• 扩展开发指南
• 更新日志
• 许可证
• 致谢



项目背景

自动驾驶仿真是自动驾驶研发中不可或缺的环节。CARLA 是由 Intel Labs 主导开发的开源自动驾驶仿真平台，提供了丰富的城市环境、车辆模型、传感器和交通场景。

本项目基于 CARLA 0.9.15，从官方示例出发，逐步构建了一套实用的自动驾驶控制系统。项目特别针对以下痛点进行了适配：

• Python 3.9 兼容性：CARLA 旧版 .egg 导入机制在 Python 3.9+ 下存在冲突，项目提供了完整的修复方案
• 功能渐进式组合：不同脚本对应不同功能组合，可按需选用
• 轻量独立方案：world_Coordinate.py 提供了不依赖 CARLA 内置 Agent 的极简自动导航，方便理解和定制



功能特性

功能	说明
🗺️ 自动导航	基于 BehaviorAgent 实现路径规划与自动驾驶，到达目标后自动切换新目标
🚦 交通灯检测	通过 CARLA 内置目标包围框检测交通灯状态，红灯停车、绿灯通行
🌦️ 天气切换	运行时键盘切换 CARLA 内置天气预设（晴天、多云、雨天、雾天等数十种）
🔄 手动/自动切换	支持自动驾驶与手动键盘操控之间实时切换，无需重启
📡 多传感器融合	碰撞检测、车道偏离检测、GNSS 定位、RGB 摄像头
🖥️ HUD 实时信息	车速、坐标、航向角、驾驶模式、FPS 等实时显示
🐍 Python 3.9 适配	修复 egg 导入冲突，兼容 CARLA 0.9.15 最新 API
🪶 极简导航方案	SimpleAutoAgent 纯自实现导航，不依赖第三方 Agent 库



项目结构

carla_driving_car_lane/
├── main.py                       # 🔰 主入口：基础自动驾驶 + 天气切换
├── automatic_control.py          # 🐍 Python 3.9 适配版自动驾驶
├── Add_weather_switching.py      # 🌦️ 自动驾驶 + 天气实时切换
├── Automatic_model               # ⭐ 完整功能版：手动/自动 + HUD + 天气
├── Switch_automatic              # 🔄 纯自动驾驶 + 天气切换
├── world_Coordinate.py           # 🪶 极简版：自实现 SimpleAutoAgent，零外部依赖
├── automatic_control/
│   └── automatic_control.py      # 📦 自动驾驶控制（备选版本）
├── start_carla_sim.bat           # 🖥️ Windows 一键启动脚本
├── Requirements.txt              # 📋 Python 依赖
└── README.md                     # 📖 本文件




脚本详解

脚本功能对比

脚本	自动导航	天气切换	手动驾驶	HUD 面板	适配修复	导航方式	快捷键
main.py	✅	✅	❌	基础	❌	BehaviorAgent	TAB
automatic_control.py	✅	❌	❌	基础	✅	BehaviorAgent	—
Add_weather_switching.py	✅	✅	❌	基础	✅	BehaviorAgent	TAB
Automatic_model	✅	✅	✅	增强	✅	BehaviorAgent	C / TAB / WASD
Switch_automatic	✅	✅	❌	基础	✅	BehaviorAgent	TAB
world_Coordinate.py	✅	✅	❌	增强	✅	SimpleAutoAgent	PageUp/Down


💡 推荐使用顺序：world_Coordinate.py（理解原理）→ main.py（体验基础功能）→ Automatic_model（完整体验）
各脚本详细说明
`main.py` — 基础入门版
最接近 CARLA 官方示例的版本，集成了碰撞/车道/GNSS 传感器和天气切换功能。适合初次接触 CARLA 自动驾驶的用户。
• 传感器：碰撞、车道偏离、GNSS、RGB 摄像头
• 自动导航使用 BehaviorAgent，到达目标后自动换新目标
• 支持天气预设切换
`automatic_control.py` — Python 3.9 适配版
针对 Python 3.9 环境的适配版本，修复了 .egg 文件导入冲突问题。如果你的 Python 环境是 3.9+，建议优先使用此版本。
• 清理了 carla/dist 下的冲突 .egg 路径
• 传感器全部修复完成，运行稳定
• 纯自动驾驶，无天气切换
`Add_weather_switching.py` — 天气增强版
在 Python 3.9 适配版基础上增加了天气实时切换功能，按 TAB 键即可在不同天气预设间切换。
`Automatic_model` — ⭐ 完整功能版（推荐）
功能最完整的版本，包含手动/自动驾驶切换、增强 HUD 面板和天气切换。
增强功能：
• 按 C 键在手动/自动驾驶之间切换
• 手动模式下使用 WASD 控制车辆
• HUD 显示：驾驶模式、车速（km/h）、坐标（X/Y）、航向角
• 碰撞和车道偏离事件实时通知
• 默认加载 Town01 地图
`Switch_automatic` — 纯自动 + 天气版
自动驾驶 + 天气切换的精简组合，无手动驾驶功能。适合需要纯自动驾驶演示的场景。
`world_Coordinate.py` — 极简独立版
项目中最轻量的自动驾驶方案，完全不依赖 CARLA 内置的 Agent 库，自己实现了 SimpleAutoAgent。
特点：
• 自实现导航逻辑（基于向量点积和叉积计算转向角）
• 阻断所有可能冲突的第三方库（matplotlib/scipy/tensorflow/torch/keras）
• 摄像头位置根据车辆包围盒动态计算
• HUD 显示 FPS、车速和行驶状态
• 天气切换使用 PageUp/PageDown
SimpleAutoAgent 导航逻辑：
// python
# 计算车辆前方与目标方向的关系
dot = forward.x * target_dir.x + forward.y * target_dir.y    # 点积：判断前后
cross = forward.x * target_dir.y - forward.y * target_dir.x  # 叉积：判断左右

# 距离 > 5m：油门 0.5 + 微调转向；距离 ≤ 5m：刹车


核心架构
系统架构图
┌──────────────────────────────────────────────────────────────┐
│                        CARLA 仿真引擎                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  地图场景  │  │  交通系统  │  │  天气系统  │  │  物理引擎  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       │ TCP (2000 端口)
┌──────────────────────┼───────────────────────────────────────┐
│               Python 客户端                                   │
│  ┌───────────────────┼───────────────────┐                   │
│  │              World 场景管理             │                   │
│  │  ┌──────────┐  ┌──┴──┐  ┌──────────┐ │                   │
│  │  │ 传感器管理 │  │车辆  │  │ 相机管理  │ │                   │
│  │  └──────────┘  └─────┘  └──────────┘ │                   │
│  └───────────────────────────────────────┘                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  导航代理      │  │  控制器       │  │  HUD 显示     │        │
│  │ BehaviorAgent │  │ KeyboardCtrl │  │  FadingText  │        │
│  │ SimpleAgent   │  │  (WASD/切换) │  │  (信息面板)   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────────┐
│               Pygame 可视化界面                                │
│  ┌────────────────────────────────────────────────────┐       │
│  │              RGB 摄像头画面 (1280×720)               │       │
│  │  ┌──────────────────────────────────┐              │       │
│  │  │ 模式: 自动驾驶                     │              │       │
│  │  │ 车速: 45.2 km/h                   │              │       │
│  │  │ 坐标: X:123.4 Y:56.7              │              │       │
│  │  │ 航向: 89.2°                       │              │       │
│  │  └──────────────────────────────────┘              │       │
│  └────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘

核心类说明
类名	所属脚本	职责
World	全部	场景管理：车辆生成/销毁、传感器挂载、天气切换
KeyboardControl	全部	键盘事件处理：退出、天气切换、驾驶模式切换、手动驾驶
HUD	全部	HUD 信息渲染：FPS、车速、坐标、通知消息
FadingText	全部	渐隐文字效果：碰撞/车道偏离等事件通知
CollisionSensor	全部	碰撞传感器：检测与其他 Actor 的碰撞并通知
LaneInvasionSensor	全部	车道偏离传感器：检测压线/越线行为
GnssSensor	main/Automatic_model/Switch	GNSS 传感器：获取经纬度坐标
CameraManager	全部	摄像头管理：RGB 图像采集与 Pygame 渲染
BehaviorAgent	大部分脚本	CARLA 内置导航代理：路径规划 + 行为决策
SimpleAutoAgent	world_Coordinate.py	自实现导航代理：基于向量运算的极简导航

传感器体系
本项目使用了 CARLA 提供的多种传感器，构成完整的感知体系：
                    车辆 (Vehicle)
                        │
        ┌───────┬───────┼───────┬────────┐
        ▼       ▼       ▼       ▼        ▼
   ┌────────┐┌──────┐┌──────┐┌──────┐┌──────┐
   │RGB相机  ││碰撞   ││车道   ││GNSS  ││可扩展 │
   │        ││传感器  ││偏离   ││传感器││传感器 │
   │1280×720││       ││传感器 ││      ││(雷达) │
   └────────┘└──────┘└──────┘└──────┘└──────┘
       │        │        │        │
       ▼        ▼        ▼        ▼
   画面渲染   碰撞通知   越线通知   定位数据

传感器	Blueprint ID	挂载方式	输出
RGB 摄像头	sensor.camera.rgb	SpringArm（车后上方）	1280×720 图像 → Pygame 渲染
碰撞传感器	sensor.other.collision	附着车辆	碰撞事件 → HUD 通知
车道偏离	sensor.other.lane_invasion	附着车辆	越线事件 → HUD 通知
GNSS	sensor.other.gnss	车顶（z+2.0）	经纬度数据

驾驶行为模式
使用 BehaviorAgent 的脚本支持三种驾驶行为，通过 --behavior 参数指定：
模式	说明	油门	跟车距离	变道频率
normal	正常驾驶，遵守交通规则	适中	适中	适时变道
cautious	谨慎驾驶，大跟车距离	较低	较大	较少变道
aggressive	激进驾驶，快速行驶	较高	较小	频繁变道

// bash
# 谨慎模式
python Automatic_model --behavior cautious

# 激进模式
python Automatic_model --behavior aggressive

天气系统
CARLA 0.9.15 内置了数十种天气预设，项目通过 find_weather_presets() 函数自动枚举所有可用预设。常见预设包括：
类别	预设示例
☀️ 晴天	ClearNoon, ClearSunset, ClearNight
☁️ 多云	CloudyNoon, OvercastRain
🌧️ 雨天	SoftRainNoon, HardRainNoon, HeavyRainSunset
🌫️ 雾天	Fog
🌧️💨 暴风雨	HardRainNoon + 强风

切换方式因脚本而异：
• 大部分脚本：TAB 键循环切换
• world_Coordinate.py：PageUp / PageDown 正反向切换

环境要求
项目	最低要求	推荐配置
CARLA 仿真器	0.9.15	0.9.15
Python	3.9+	3.9.x
操作系统	Windows 10（仿真器端）	Windows 10/11
GPU	NVIDIA GTX 1060 6GB	NVIDIA RTX 3060 及以上
内存	16 GB	32 GB
磁盘	30 GB（CARLA 安装）	SSD
网络	本地回环（单机）	局域网（多机联仿）

⚠️ CARLA 仿真器仅支持 Windows/Linux，macOS 不支持。Python 客户端可在 macOS 上远程连接 CARLA 服务器。

安装
1. 安装 CARLA 0.9.15 仿真器
从 CARLA Release 页面 下载 0.9.15 版本，解压至默认路径：
D:\carla0.9.15\
├── CarlaUE4.exe          # 仿真器主程序
├── CarlaUE4\             # 引擎资源
├── PythonAPI\            # Python 接口
│   ├── carla\            # carla 模块
│   │   └── dist\         # .egg 文件（旧版用）
│   └── examples\         # 官方示例
├── Import\               # 资源导入目录
└── Util\                 # 工具脚本

2. 安装 Python 依赖
// bash
# 推荐使用虚拟环境
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux

pip install -r Requirements.txt

依赖详情：
包名	版本	必选	说明
carla	0.9.15	✅	CARLA PythonAPI 客户端库
pygame	2.1.0	✅	仿真界面渲染与键盘输入
numpy	1.23.5	✅	数值计算与图像数组处理
opencv-python	4.8.0.76	❌	图像处理（扩展功能）
matplotlib	3.7.1	❌	数据可视化（扩展功能）
Pillow	9.4.0	❌	图像处理（扩展功能）
torch	2.0.0	❌	深度学习检测模型（可选）
torchvision	0.15.1	❌	视觉模型（配合 torch 使用）


使用方法
方式一：一键启动（Windows）
双击运行 start_carla_sim.bat，脚本会自动完成以下步骤：
[1] 启动 CARLA 仿真器 → CarlaUE4.exe -quality-level=Low
[2] 等待 15 秒服务器就绪
[3] 运行自动驾驶脚本 → automatic_control.py

⚠️ 使用前需修改 .bat 文件中的路径，详见 配置说明。
方式二：手动启动
步骤 1：启动 CARLA 仿真器
// bash
# Windows — 低画质模式（推荐，性能友好）
CarlaUE4.exe -quality-level=Low

# Windows — 高画质模式
CarlaUE4.exe -quality-level=Epic

# Linux
./CarlaUE4.sh -quality-level=Low

# 指定窗口化运行
CarlaUE4.exe -windowed -ResX=1280 -ResY=720

步骤 2：等待仿真器就绪（控制台显示 "CarlaServer ready" 后继续）
步骤 3：运行自动驾驶脚本
// bash
# 完整功能版（推荐首次使用）
python Automatic_model

# 极简版（理解导航原理）
python world_Coordinate.py

# 基础版
python main.py

# 指定服务器地址和参数
python main.py --host 127.0.0.1 --port 2000 --res 1280x720 --behavior normal

# 连接远程 CARLA 服务器
python main.py --host 192.168.1.100 --port 2000

命令行参数
参数	默认值	说明
--host	127.0.0.1	CARLA 服务器地址（本机或远程 IP）
--port	2000	CARLA 服务器端口
--res	1280x720	渲染分辨率，格式：宽x高
--filter	vehicle.*	车辆蓝图过滤（如 vehicle.tesla.model3）
-b / --behavior	normal	驾驶行为模式：normal / cautious / aggressive

示例：
// bash
# 使用特定车辆模型
python Automatic_model --filter vehicle.tesla.model3

# 4K 分辨率 + 激进模式
python Automatic_model --res 3840x2160 --behavior aggressive

# 连接远程服务器 + 谨慎模式
python main.py --host 192.168.1.100 --behavior cautious

键盘操作
通用操作
按键	功能	适用脚本
ESC	退出程序	全部
Ctrl+Q	退出程序	全部
TAB	切换天气预设	main.py / Automatic_model / Switch_automatic
PageUp	上一个天气预设	world_Coordinate.py
PageDown	下一个天气预设	world_Coordinate.py

手动驾驶模式（仅 `Automatic_model`）
按键	功能
C	切换手动/自动驾驶模式
W	油门（加速）
S	刹车（减速）
A	左转向
D	右转向

💡 切换到手动模式后，HUD 会显示"手动驾驶模式"提示，此时 WASD 控制生效。

工作原理
交通灯检测与响应流程
      CARLA 仿真世界
           │
           ▼
  ┌──────────────────┐
  │ 获取世界中的交通灯 │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ 判断交通灯状态     │
  │ (红/黄/绿)        │
  └────────┬─────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  红灯 🛑      绿灯 🟢
     │           │
     ▼           ▼
  刹车=1.0    油门恢复
  停车等待     正常行驶

1. 客户端通过 CARLA API 获取世界中所有交通灯 Actor
2. 根据车辆位置和朝向，判断前方是否有交通灯
3. 检测交通灯的 state 属性（Red / Yellow / Green）
4. 红灯/黄灯时：设置 brake = 1.0，throttle = 0.0
5. 绿灯时：恢复正常行驶，由 BehaviorAgent 计算控制指令
自动导航流程
  ┌────────────────┐
  │  随机选择出生点  │
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │ 生成车辆 Actor   │ ← vehicle.* 蓝图过滤
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │ 设置导航目标     │ ← 随机选择 spawn point
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐     ┌────────────────┐
  │ BehaviorAgent   │────►│  计算控制指令    │
  │ run_step()      │     │ (throttle/steer │
  └───────┬────────┘     │  /brake)        │
          │              └────────────────┘
          ▼
  ┌────────────────┐
  │ 应用控制到车辆   │
  └───────┬────────┘
          │
          ▼
  ┌───────────────┐     否
  │ 是否到达目标？  │──────────┘
  └───────┬───────┘      ↑
      是  │              │
          ▼              │
  ┌────────────────┐     │
  │ 自动设置新目标   │─────┘
  └────────────────┘

手动/自动切换流程
                     ┌─────────────┐
                     │  程序启动    │
                     │  默认：自动  │
                     └──────┬──────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          ┌────────────┐        ┌────────────┐
          │  自动模式   │        │  手动模式   │
          │            │   C    │            │
          │ Agent 计算  │◄──────►│ WASD 控制  │
          │ 控制指令    │   C    │ 油门/刹车   │
          │            │◄──────►│ /转向      │
          └────────────┘        └────────────┘


配置说明
CARLA 安装路径
如 CARLA 安装路径非默认，需修改各 Python 脚本顶部的路径变量：
// python
CARLA_ROOT = r"D:\carla0.9.15"  # ← 修改为你的 CARLA 安装路径

涉及文件：main.py、automatic_control.py、Add_weather_switching.py、Automatic_model、Switch_automatic
一键启动脚本路径
修改 start_carla_sim.bat 中的以下路径：
// bat
:: CARLA 仿真器路径
start "" "D:\carla0.9.15\CarlaUE4.exe" -quality-level=Low

:: Python 解释器路径
"D:\carla_automatic1\.venv\Scripts\python.exe"

:: 自动驾驶脚本路径
"D:\carla_automatic1\automatic_control.py"

分辨率调整
可通过命令行参数或修改代码调整渲染分辨率：
// bash
# 命令行方式
python main.py --res 1920x1080

# 代码方式（修改 main() 函数中的默认值）
argparser.add_argument('--res', default='1920x1080')

CARLA 服务器性能调优
参数	说明	示例
-quality-level=Low	低画质，性能优先	CarlaUE4.exe -quality-level=Low
-quality-level=Epic	高画质，视觉优先	CarlaUE4.exe -quality-level=Epic
-world-port=2000	指定通信端口	CarlaUE4.exe -world-port=3000
-carla-rpc-port=3000	指定 RPC 端口	CarlaUE4.exe -carla-rpc-port=3000


常见问题
Q1: 启动脚本报错 `ModuleNotFoundError: No module named 'carla'`
原因：CARLA PythonAPI 路径未正确配置。
解决：
1. 确认 CARLA_ROOT 路径指向正确的 CARLA 安装目录
2. 确认目录下存在 PythonAPI/carla/ 文件夹
3. Python 3.9+ 用户使用 automatic_control.py 等已适配版本
Q2: 连接 CARLA 服务器超时 `RuntimeError: time-out`
原因：仿真器未启动或未就绪。
解决：
1. 先启动 CarlaUE4.exe，等待控制台显示 "ready"
2. 确认端口一致（默认 2000）
3. 可尝试增大超时时间：client.set_timeout(30.0)
Q3: Python 3.9 下 egg 导入报错
原因：CARLA 旧版 .egg 文件与 Python 3.9 不兼容。
解决：使用 automatic_control.py 等已适配版本，脚本会自动清理冲突路径：
// python
# 已在脚本中处理
for path in list(sys.path):
    if "carla/dist" in path and path.endswith(".egg"):
        sys.path.remove(path)

Q4: 画面卡顿 / FPS 过低
解决：
1. 使用低画质模式启动：CarlaUE4.exe -quality-level=Low
2. 降低分辨率：python main.py --res 800x600
3. 关闭不必要的后台程序
4. 确认 GPU 驱动已更新
Q5: 导入 BehaviorAgent 报错 `ModuleNotFoundError`
原因：缺少 CARLA Agent 模块。
解决：确认 CARLA 安装目录下存在 PythonAPI/carla/agents/ 目录。如不存在，可使用 world_Coordinate.py 中的 SimpleAutoAgent 替代。
Q6: 碰撞后车辆不恢复行驶
原因：碰撞传感器仅做通知，不处理恢复逻辑。
解决：可在碰撞回调中添加恢复逻辑，如短暂停车后重新规划路径。

扩展开发指南
添加新传感器
参照现有传感器类的实现模式，在 World.restart() 中挂载：
// python
class MyCustomSensor(object):
    def __init__(self, parent_actor, hud):
        bp = parent_actor.get_world().get_blueprint_library().find('sensor.other.obstacle')
        self.sensor = parent_actor.get_world().spawn_actor(
            bp, carla.Transform(), attach_to=parent_actor)
        self.sensor.listen(lambda event: self._on_event(event))

    def _on_event(self, event):
        # 处理传感器数据
        pass

可用的传感器 Blueprint：
Blueprint ID	说明
sensor.camera.rgb	RGB 摄像头
sensor.camera.depth	深度摄像头
sensor.camera.semantic_segmentation	语义分割摄像头
sensor.lidar.ray_cast	激光雷达
sensor.other.radar	毫米波雷达
sensor.other.imu	惯性测量单元
sensor.other.obstacle	障碍物检测

切换地图
修改 game_loop() 中的地图加载：
// python
# 加载指定地图
client.load_world('Town03')

# 可用地图：Town01 ~ Town10HD
# Town01: 简单小镇，适合测试
# Town03: 圆环交叉路口
# Town05: 高架桥与隧道
# Town10HD: 高清城市

添加更多车辆
修改 --filter 参数或在代码中指定：
// python
# 特定车型
argparser.add_argument('--filter', default='vehicle.tesla.model3')

# 所有两轮车
argparser.add_argument('--filter', default='vehicle.*.bike')

# 所有摩托车
argparser.add_argument('--filter', default='vehicle.*.motorcycle')

自定义 SimpleAutoAgent
world_Coordinate.py 中的 SimpleAutoAgent 是最易修改的导航方案，适合快速实验：
// python
class SimpleAutoAgent:
    def run_step(self):
        control = carla.VehicleControl()
        # ← 在这里修改你的控制逻辑
        # 可加入：PID 控制、纯跟踪算法、深度学习模型等
        return control


更新日志
v1.0.0
• 基础自动驾驶功能（main.py）
• 碰撞/车道偏离/GNSS 传感器集成
• 天气切换功能
v1.1.0
• Python 3.9 适配，修复 .egg 导入冲突
• 新增 automatic_control.py / Add_weather_switching.py
• 新增 Switch_automatic 脚本
v1.2.0
• 完整功能版 Automatic_model：手动/自动切换、增强 HUD
• 极简独立版 world_Coordinate.py：自实现 SimpleAutoAgent
• Windows 一键启动脚本 start_carla_sim.bat

许可证
MIT License — 基于 Intel Labs 原始代码适配修改。
详见 LICENSE

致谢
• CARLA Simulator — 开源自动驾驶仿真平台
• Intel Labs — CARLA 原始代码贡献
• 原始代码作者：German Ros (german.ros@gmail.com)
• CARLA 文档 — API 参考与教程

本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
