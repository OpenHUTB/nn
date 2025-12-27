# 智能感知自动驾驶

操作系统：Windows 10/11,

处理器：Intel i5 或同等性能以上

内存：8GB RAM 以上

显卡：支持DirectX 11的独立显卡

存储空间：至少10GB可用空间



软件依赖

Python 3.7+

CARLA 0.9.10+（自动驾驶仿真引擎）

Pygame 2.0+（游戏界面库）

NumPy（科学计算库）



### 📦 安装步骤

1\. 安装CARLA

bash

\# 从官网下载CARLA 0.9.10或更高版本

\# 解压到合适目录，如：D:/CARLA\_0.9.10/

2\. 设置Python环境

bash

\# 创建虚拟环境（推荐）

python -m venv carla\_env

source carla\_env/bin/activate  # Linux/macOS

\# 或

carla\_env\\Scripts\\activate     # Windows

\# 安装依赖

pip install pygame numpy

3\. 配置环境变量（可选）

bash

\# 设置CARLA路径环境变量（Windows）

set CARLA\_ROOT=D:\\CARLA\_0.9.10



\# 或直接修改main.py中的路径配置

### 🚀 运行程序

启动步骤

启动CARLA服务器



bash

\# Windows: 双击CarlaUE4.exe

\# Linux: ./CarlaUE4.sh

运行模拟器程序



bash

python main.py

预期输出

text

✅ 添加CARLA路径: D:/CARLA\_0.9.10/WindowsNoEditor/PythonAPI/carla

==================================================

🚗 CARLA 自动驾驶模拟器

==================================================

🔄 连接到Carla服务器...

✅ 已连接，当前地图: Town03

🎮 初始化Pygame界面...

🚘 生成自动驾驶车辆...

✅ 车辆已生成在位置: (161.9, 58.9)

🚦 自动驾驶已启用

✅ 激光雷达已安装

✅ 摄像头已安装

▶️ 启动自动驾驶...

📊 车辆速度和位置将显示在屏幕上

🚧 障碍物检测系统已启用

ℹ️  按ESC键退出程序

==================================================

### 🎯 功能详解

1\. 自动驾驶系统

车辆控制：基于Carla内置的自动驾驶系统

路径跟踪：车辆在环境中自主导航

智能避障：通过障碍物检测实现安全行驶

2\. 环境感知系统（obstacle\_detector.py）

这是项目的核心创新模块，实现了以下功能：

点云数据处理

python

\# 激光雷达数据流处理

点云数据 → 区域过滤 → 障碍物判断 → 威胁评估

四级预警机制

警告级别	颜色	距离阈值	含义

0-安全	🟢 绿色	>7米	前方区域清晰

1-注意	🟡 黄色	4-7米	注意前方障碍物

2-警告	🟠 橙色	2-4米	警告，建议减速

3-危险	🔴 红色	<2米	危险，需要紧急制动

3\. 可视化界面（drawer.py）

实时速度显示：左上角显示当前车速（km/h）

位置信息：显示车辆坐标

障碍物警告：分级颜色显示警告信息和状态条

视觉反馈：根据警告级别动态改变界面颜色

4\. 游戏循环系统（sync\_pygame.py）

事件处理：键盘输入（ESC退出）

帧率控制：稳定60FPS运行

状态同步：协调Carla世界和Pygame界面



### 🕹️ 使用说明

基本操作

程序启动后：车辆自动开始行驶

观察界面：关注左上角的速度和警告信息

退出程序：按ESC键或关闭窗口

调试信息

程序运行时会在控制台输出：

激光雷达点云数量（1%概率）

障碍物检测结果（5%概率）

错误和异常信息

数据流

text

Carla传感器 → 原始数据 → 数据处理 → 算法分析 → 决策输出 → UI显示

   ↓           ↓          ↓          ↓          ↓         ↓

激光雷达 → 点云数据 → 区域过滤 → 障碍检测 → 预警级别 → 颜色编码

摄像头 → 图像数据 → (预留接口) → (未来扩展) → (未来扩展) → 图像显示

🛠️ 开发指南

代码结构说明

main.py（主控制器）

职责：初始化所有模块，协调系统运行

核心类：Main

主要方法：

\_\_init\_\_(): 初始化系统和连接Carla

spawn\_vehicle(): 生成自动驾驶车辆

on\_tick(): 每帧更新，核心逻辑处理

cleanup(): 资源清理

obstacle\_detector.py（感知核心）

职责：环境感知和障碍物检测

核心类：ObstacleDetector

算法特点：

区域过滤算法

分级预警机制

异常处理机制

drawer.py（可视化）

职责：用户界面显示

核心类：PyGameDrawer

显示要素：

速度表

位置信息

警告系统

状态指示器

sync\_pygame.py（游戏引擎）

职责：事件循环和状态管理

核心类：SyncPyGame

特性：

60FPS稳定运行

事件驱动架构

资源管理

扩展开发

添加新传感器

python

\# 在main.py的setup\_sensors()方法中添加

def setup\_new\_sensor(self):

    sensor\_bp = self.world.get\_blueprint\_library().find("sensor.type")

    # ... 配置和安装传感器

修改检测算法

python

\# 在obstacle\_detector.py中修改detect()方法

def detect(self, point\_cloud):

    # 实现新的检测算法

    # 调整阈值参数

    # 添加新的预警逻辑

扩展UI界面

python

\# 在drawer.py中添加新的显示方法

def display\_new\_info(self, data):

    # 创建新的显示元素

    # 添加到现有界面中

### 📊 性能优化

当前性能

帧率：稳定60FPS



激光雷达处理：每秒50000个点



内存使用：约200-300MB



CPU占用：<30%



优化建议

降低点云密度：调整points\_per\_second参数



减少输出频率：调整控制台输出概率



简化检测算法：优化障碍物检测逻辑



启用同步模式：提高Carla仿真稳定性



### 故障排除

常见问题

1\. 连接Carla失败

text

❌ 未找到CARLA路径，请手动设置

解决方案：



检查CARLA安装路径



修改main.py中的possible\_paths列表



确保CarlaUE4.exe正在运行



2\. 导入模块错误

text

ModuleNotFoundError: No module named 'carla'

解决方案：



确保CARLA PythonAPI路径已添加到sys.path



检查Python版本兼容性（推荐Python 3.7）



3\. 窗口无响应

解决方案：



检查显卡驱动



降低Pygame窗口分辨率



关闭其他占用GPU的程序



4\. 障碍物检测不准确

解决方案：



调整obstacle\_detector.py中的阈值参数



检查激光雷达安装位置和参数



验证点云数据是否正确接收

添加新传感器

python

def setup\_radar(self):

&nbsp;   """安装雷达传感器"""

&nbsp;   radar\_bp = self.world.get\_blueprint\_library().find("sensor.other.radar")

&nbsp;   # 配置参数

&nbsp;   radar\_bp.set\_attribute("range", "100")

&nbsp;   radar\_bp.set\_attribute("points\_per\_second", "1500")

&nbsp;   # 安装和监听...

自定义检测算法

python

class AdvancedObstacleDetector(ObstacleDetector):

&nbsp;   def detect(self, point\_cloud):

&nbsp;       # 实现更复杂的检测逻辑

&nbsp;       # 例如：聚类分析、机器学习模型、多帧融合

&nbsp;       pass

扩展 UI 功能

python

def display\_radar(self, radar\_data):

&nbsp;   """显示雷达信息"""

&nbsp;   # 创建雷达图

&nbsp;   # 显示距离和速度信息

&nbsp;   pass

集成规划控制模块

python

def control\_vehicle(self, obstacles):

&nbsp;   """根据障碍物信息控制车辆"""

&nbsp;   if obstacles:

&nbsp;       # 减速或变道逻辑

&nbsp;       control = carla.VehicleControl()

&nbsp;       control.throttle = 0.3  # 减少油门

&nbsp;       control.brake = 0.2     # 轻微制动

&nbsp;       self.ego.apply\_control(control)

## 📈 性能测试

测试脚本

python

\# performance\_test.py

import time

import statistics



def test\_detection\_speed(detector, sample\_data):

&nbsp;   """测试检测速度"""

&nbsp;   times = \[]

&nbsp;   for \_ in range(100):

&nbsp;       start = time.time()

&nbsp;       detector.detect(sample\_data)

&nbsp;       times.append(time.time() - start)

&nbsp;   

&nbsp;   avg\_time = statistics.mean(times)

&nbsp;   print(f"平均检测时间: {avg\_time\*1000:.2f}ms")

&nbsp;   print(f"最大检测时间: {max(times)\*1000:.2f}ms")

&nbsp;   print(f"最小检测时间: {min(times)\*1000:.2f}ms")

基准测试结果

检测延迟：平均 5-15ms



内存峰值：~250 MB



CPU 使用率：15-25%



GPU 使用率：20-40%

