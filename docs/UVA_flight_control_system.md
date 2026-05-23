# 基于AirSim的无人机飞行控制系统
## 摘要
本项目设计并实现了一套基于AirSim仿真平台的无人机飞行控制系统。系统以**多旋翼无人机**为控制对象，通过Python接口实现键盘交互式飞行控制，并集成了**多种典型飞行模式**，包括环绕飞行、方形轨迹、螺旋上升和原地旋转等等。同时，针对传统手动控制中存在的安全性与可控性问题，系统引入了自动返航机制，可在任意时刻一键返回起飞点并安全悬停。

在实现过程中，通过**多线程机制**对各类飞行模式进行异步调度，避免阻塞主控制流程，从而提升系统响应性能；结合速度分级与平滑控制策略，优化了飞行过程的稳定性与流畅性。实验表明，该系统能够在仿真环境中**稳定运行，实现多模式切换与实时控制**，具备良好的可扩展性，可为后续无人机感知避障、路径规划等进阶开发提供稳定可靠的基础框架。

## 1. 项目背景
### 1.1 功能定位
UVA_flight_control_system是一个基于**AirSim仿真平台开发**的无人机核心控制模块。该模块作为无人机系统的中枢，负责处理从底层仿真通信到高层自主航迹生成的全流程控制逻辑，主要面向**无人机飞行控制算法验证与基础自主飞行任务开发**。
#### 1.1.1 系统层级架构
无人机飞行控制系统采用分层设计，在仿真环境中构建了清晰的技术栈，从物理仿真到指令执行形成完整的控制闭环，各层级功能如下表所示：

|名称|功能|
| ---- | ---- |
**环境层**|AirSim物理仿真引擎，提供真实的飞行反馈
**控制层**|键盘输入映射、多线程轨迹规划、飞行状态平滑处理、指令解析与调度
**执行层**|通过AirSim API下发`moveByVelocity`、`moveToPosition`控制指令，驱动无人机运动执行

#### 1.1.2 核心任务描述
无人机控制系统的核心任务围绕“**精准操控**、**自主运行**、**安全可靠**”三大目标展开，涵盖实时交互、轨迹规划、性能调节及应急保障四大核心维度，具体如下：

|名称|功能|
| ---- | ---- |
 **实时交互控制**|实现低延迟键盘遥控，支持前后、左右、垂直升降全维度操控
**自动化航迹算法**|内置多种数学模型，实现环绕、方形、螺旋上升等三维复杂轨迹飞行
**动态性能调节**|设计多档位速度切换机制，适配精细操控与快速巡航不同需求
**应急保障机制**|集成一键悬停、一键返航、安全降落逻辑，保障仿真飞行稳定性

### 1.2 原代码的设计问题及其优化
原生AirSim基础示例代码仅实现简单移动功能，未考虑实际操控体验与工程化应用需求，存在诸多设计短板，以下是针对各短板的优化说明：

#### 1.2.1 操控体验操控体验优化：引入平滑过渡与速度分级

原生代码直接下发固定速度指令，无平滑处理与速度分级，导致飞行卡顿、启停抖动，不符合真实无人机惯性特性。

```python
# 原生代码：无平滑处理的按键控制（已废弃）
def on_press(key):
    try:
        # 直接使用固定速度，无任何平滑过渡
        if key.char == 'w':
            client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 0.1)
        if key.char == 's':
            client.moveByVelocityBodyFrameAsync(-1.5, 0, 0, 0.1)
        if key.char == 'a':
            client.moveByVelocityBodyFrameAsync(0, -1.5, 0, 0.1)
        if key.char == 'd':
            client.moveByVelocityBodyFrameAsync(0, 1.5, 0, 0.1)
    except:
        pass

# 松开按键无制动处理，飞机易漂移
def on_release(key):
    pass
```    

#### 1.2.2 多任务并发优化：基于`threading`实现异步执行

原生代码所有逻辑耦合在主线程中，自动飞行与手动控制无法同时运行，扩展性差。

```python
# 原生代码：无多线程的耦合逻辑（已废弃）
while True:
    # 自动飞行与键盘监听在同一线程，会互相阻塞
    if orbit_mode_active:
        orbit_mode()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    time.sleep(0.1)
```

#### 1.2.3 功能完整性优化：新增档位调速与安全保障

原生代码缺少实用功能，无速度分级、一键悬停、自动返航等机制，无法适配不同操控场景，也无安全保障。

```python
# 原生代码：无调速与安全功能的主逻辑（已废弃）
# 无速度分级、无悬停、无返航功能
SPEED = 1.5
HEIGHT = -3

# 起飞后仅支持基础移动
client.takeoffAsync().join()
time.sleep(2)
client.moveToZAsync(HEIGHT, 1).join()

def on_press(key):
    try:
        # 仅支持固定速度移动，无其他功能
        if key.char == 'w':
            client.moveByVelocityBodyFrameAsync(SPEED, 0, 0, 0.1)
    except:
        pass
```

#### 1.2.4 代码架构优化：模块化拆分与稳定性提升

原生代码各功能逻辑耦合度高、无模块化拆分，核心控制代码与输入输出逻辑混杂，易出现报错、卡死等问题，环境复现性差。

```python
# 原生代码：结构混乱的耦合代码（已废弃）
import airsim
import time
from pynput import keyboard

# 所有逻辑混杂在一起，无模块化拆分
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()

def on_press(key):
    try:
        # 所有按键逻辑写在同一个函数中，难以维护
        if key.char == 'w': client.moveByVelocityBodyFrameAsync(1.5,0,0,0.1)
        if key.char == 's': client.moveByVelocityBodyFrameAsync(-1.5,0,0,0.1)
        if key.char == 'a': client.moveByVelocityBodyFrameAsync(0,-1.5,0,0.1)
        if key.char == 'd': client.moveByVelocityBodyFrameAsync(0,1.5,0,0.1)
        if key.char == 'z': client.moveToZAsync(-3.5,1)
        if key.char == 'x': client.moveToZAsync(-2.5,1)
    except:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:
    time.sleep(0.1)
```

本次优化统一了代码运行链路，解决程序报错、按键卡死、机身掉高漂移等问题，提升了环境复现性。

## 2. 核心技术栈与理论基础
### 2.1 核心技术栈
|技术/工具|用途|
| ---- | ---- |
 Python 3.12|核心开发语言，负责逻辑编写、多线程调度与指令处理 
AirSim API|无人机仿真通信核心接口，实现状态读取与运动控制
Unreal Engine|高保真三维物理场景渲染，提供真实物理环境模拟
pynput|全局无阻塞键盘监听，实时捕获用户操控指令
Math库|三角函数计算，支撑环绕、螺旋等自主轨迹生成
Threading|多线程并发处理，分离手动操控与自动轨迹任务

### 2.2 核心理论基础
#### 2.2.1 系统整体运行流程
<img width="450" alt="系统流程图" src="https://github.com/user-attachments/assets/8cfd4940-8560-4552-99a9-25f7f8b8edcf" />

#### 2.2.2 关键算法原理
**自主巡航轨迹数学模型**

- **方形巡航**：通过序列化坐标点位，分段规划直线路径，依次抵达四个顶点，分段衔接闭环飞行

- **螺旋上升轨迹**：融合极坐标变换与高度增量控制，定时更新平面坐标与垂直高度
  
$$
x=R \cos(\omega t),\quad y=R \sin(\omega t),\quad z=z_0-kt
$$
  
，其中R为盘旋半径，k为上升速率系数，实现匀速盘旋加稳定爬升三维运动。

- **NED坐标系映射**：系统严格遵循无人机标准 NED 坐标系规范：
  
     X轴：正北方向（机体前进正向）

     Y轴：正东方向（机体右移正向）

     Z轴：地向为正，负值对应高度上升

- **多档位调速原理**：通过全局档位变量搭配倍率数组，动态换算实时飞行速度，支持飞行中无感切换

```python
  speed_ratio = [0.6, 1.0, 1.6]
  now_speed = SPEED * speed_ratio[speed_level - 1] * smooth
```

## 3. 系统模块设计
本系统为了提升代码的可维护性与功能的可扩展性，采用模块化分层设计，构建了 “**通信交互层 - 核心控制层 - 功能实现层**” 三级架构：以飞行控制模块为核心，整合AirSim仿真通信与键盘交互能力，将各类轨迹与安全功能封装为独立可调用的模块，既实现了控制逻辑的解耦，也为后续功能迭代与问题调试提供了便利。
### 3.1 核心模块概览

<img width="2191" height="523" alt="核心模块图" src="https://github.com/user-attachments/assets/63232eb7-abb9-40ee-ba11-eb959c29df00" />

### 3.2 主要函数说明
  所有功能均封装为独立函数，每个函数实现单一功能，便于调试与维护：

| 模块/函数 | 功能描述 |
| ---- | ---- |
`auto_return_home()`|自动返航至起飞点并悬停
`orbit_mode()`|环绕轨迹飞行
`square_mode()`|方形轨迹飞行
`spiral_mode()`|螺旋上升轨迹飞行
`rotate_mode()`|原地旋转飞行
`on_press()` /`on_release()`|键盘实时交互控制
`start_*()`|启动对应轨迹模式的线程

## 4. 开发关键问题与技术解决方案
  在项目开发与调试过程中，遇到了多类典型技术问题，如异步指令执行时序异常、异常捕获机制屏蔽底层报错、控制权限生效时序不当等。本节对关键问题进行分析，并给出经调试验证的解决方案，为同类仿真控制项目提供参考。
### 4.1 无人机执行起飞指令无响应
- **根本原因**：

  原代码中起飞指令未添加`.join()`阻塞等待，异步指令未执行完成就直接运行后续代码，导致起飞流程中断；且外层包裹冗余`try-except`代码块，吞掉了底层报错信息，无法定位故障。

- **解决方案**：

  移除冗余的`try-except`异常捕获，避免屏蔽报错；给`takeoffAsync()`添加`.join()`，确保起飞指令完整执行后再运行后续高度调节代码；确认API控制与电机解锁步骤顺序无误。
  
```
   修复前
   try:
      client.takeoffAsync()
      time.sleep(2)
      client.moveToZAsync(HEIGHT, 1).join()
   except:
      pass
   
   修复后
   client.takeoffAsync().join()
   time.sleep(2)
   client.moveToZAsync(HEIGHT, 1).join()
```

   <img width="1280" height="642" alt="平稳飞行512" src="https://github.com/user-attachments/assets/c62730be-1c40-419d-9e4c-e94b18f254b2" />

### 4.2 飞行时抖动严重
- **根本原因**：
  
  原代码直接采用固定速度调用`moveByVelocityBodyFrameAsync`，无平滑系数处理，指令下发频率过高且速度突变；同时缺少速度分级机制，操控力度无法适配精细飞行需求。

- **解决方案**：

  新增平滑系数smooth与三级速度倍率数组speed_ratio，动态计算实时飞行速度，避免速度突变；优化指令下发时长参数，让速度过渡更平缓，模拟真实无人机惯性。
```
   核心修复代码片段：
   # 新增平滑与调速参数
   smooth = 0.5
   speed_level = 2
   speed_ratio = [0.6, 1.0, 1.6]
  
   # 优化后速度计算
   now_speed = SPEED * speed_ratio[speed_level - 1] * smooth
   client.moveByVelocityBodyFrameAsync(now_speed,0,0,0.1)
```
   <img width="1280" height="642" alt="平稳飞行512" src="https://github.com/user-attachments/assets/c62730be-1c40-419d-9e4c-e94b18f254b2" />  

### 4.3 按下自动返航键无人机无反应
- **根本原因**：

  自动返航函数`auto_return_home`直接在主线程中执行，返航过程会持续阻塞键盘监听与控制流程，导致界面卡死、指令无法响应，属于单线程架构缺陷。

- **解决方案**：

  将自动返航函数封装为独立守护线程，与主线程分离运行，避免阻塞键盘监听与主控制流程，实现返航过程中仍可正常响应操控指令。
```
  # 修复前
  if key.char == 'b':
      auto_return_home()
  
  # 修复后
  if key.char == 'b':
      threading.Thread(target=auto_return_home, daemon=True).start()
```

 <img width="1280" height="640" alt="返航512" src="https://github.com/user-attachments/assets/c8f5cbc9-9af9-4e8e-9063-4eb2df2b62b4" />

  
### 4.4 自动轨迹模式运行时，键盘操控失灵无响应
- **根本原因**：

  原自动轨迹函数直接在主线程运行，持续占用主线程资源，阻塞键盘监听逻辑，导致按键指令无法被正常捕获与执行，属于单线程架构缺陷。
  
- **解决方案**：

  采用多线程异步架构，将所有自动轨迹函数封装为独立守护线程，与键盘监听主线程分离运行，实现自动飞行与手动操控无缝切换，手动指令可随时打断自动轨迹。
```
  # 独立线程启动自动轨迹
  def start_orbit():
      threading.Thread(target=orbit_mode, daemon=True).start()
  
  # 按键触发线程
  if key.char == 'o': start_orbit()
  if key.char == 'm': start_square()
```

 <img width="1280" height="640" alt="环形512" src="https://github.com/user-attachments/assets/5eda9d8f-a704-414f-b4e1-e77f8da2e564" />

## 5. 系统运行效果
### 5.1 运行环境
|项目|配置参数|
| ---- | ---- |
操作系统|Windows 10 / Windows 11
运行语言|Python 3.12
仿真平台|AirSim 
核心依赖|airsim、pynput、numpy、threading
### 5.2 运行部署方式
1.启动AirSim仿真程序，等待三维场景加载完成

2.安装项目所需依赖库

3.运行主程序`main.py`，无人机自动起飞至安全高度，进入待命操控状态

### 5.3 按键操作说明
| 按键    | 功能          |
| ----- | ----------- |
| W / S | 前进 / 后退     |
| A / D | 左 / 右       |
| Z / X | 上升 / 下降     |
| H     | 悬停          |
| B     | 自动返航        |
| O     | 环绕轨迹 Orbit  |
| M     | 方形轨迹 Square |
| N     | 原地旋转 Rotate |
| L     | 螺旋上升 Spiral |
| P     | 切换飞行速度档位    |
| ESC   | 退出系统        |

### 5.4 运行效果
下面是部分功能的运行效果展示：
#### 5.4.1 螺旋上升

<img width="1280" height="564" alt="螺旋上升512" src="https://github.com/user-attachments/assets/292797e2-06bd-4199-9432-1d5999fb09b5" />


#### 5.4.2 原地旋转

<img width="1280" height="632" alt="原地旋转512" src="https://github.com/user-attachments/assets/15a7ae42-f60b-413e-9dc9-b26501fd7218" />


#### 5.4.3 切换飞行速度

<img width="1280" height="632" alt="变速512" src="https://github.com/user-attachments/assets/3c98c691-9f1c-40e2-a330-a3a0ffb1512c" />


## 6. 功能扩展与未来规划
  在现有基础飞行控制能力之上，本系统仍有较大的扩展空间。未来将围绕环境感知、路径规划、视觉任务与数据可视化四个方向持续迭代，逐步构建更智能、更稳定的无人机控制体系，为后续复杂场景下的算法验证与应用开发提供更完善的平台支撑。

- **环境感知融合**：接入AirSim激光雷达、视觉传感器，开发障碍物检测与自动避障功能
- **智能路径规划**：引入A*、RRT等路径规划算法，实现自定义航点自主飞行
- **视觉跟踪增强**：结合OpenCV图像识别，实现目标跟随、区域巡航等智能任务
- **数据可视化**：新增飞行日志记录，实时展示高度、速度、位置等运行参数

## 7. 总结
  本次基于AirSim平台完成无人机智能控制系统的全面优化与功能迭代，在原生基础功能之上，完成飞行平滑优化、多线程任务改造、三级速度调节、多类自主轨迹开发与全套安全保护机制搭建。
  
优化后系统解决了原生版本操控生硬、运行不稳定、功能单一等问题，实现手动遥控+自动巡航双模式协同工作，代码结构清晰、运行可复现、操作简洁直观。

既满足了模拟无人机正常飞行的要求，也为后续无人机避障、路径规划、智能控制等进阶开发提供完整基础框架。
