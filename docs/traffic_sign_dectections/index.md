# CARLA 自动驾驶仿真客户端

基于 **CARLA 0.9.13** 开发的轻量化自动驾驶仿真控制程序，支持自动巡航、多传感器数据采集、双格式日志记录与车辆一键重置。

---

## 📑 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [环境要求](#环境要求)
- [项目结构](#项目结构)
- [快速启动](#快速启动)
- [命令行参数](#命令行参数)
- [操作快捷键](#操作快捷键)
- [日志说明](#日志说明)
- [代码优化说明](#代码优化说明)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

---

## 项目简介

本项目基于 **CARLA 0.9.13** 与 **Python 3.7** 开发，提供完整的自动驾驶仿真客户端功能。集成智能驾驶智能体、多传感器可视化、行驶数据记录等核心能力，无需复杂配置即可快速运行。

适用于：

- 自动驾驶算法验证
- 仿真场景测试
- 数据采集与轨迹分析
- 智能驾驶控制逻辑开发
- 教学与科研实验

---

## 核心功能

### 🚗 智能驾驶

- 支持 `BehaviorAgent` 与 `BasicAgent`
- 支持多种驾驶风格切换
  - Normal（正常）
  - Aggressive（激进）
  - Cautious（谨慎）
- 自动路径规划
- 自动避障与交通规则遵循
- 循环自动巡航模式

### 🔄 自动巡航

- 自动生成随机目标点
- 到达终点后自动规划下一段路线
- 支持长时间无人值守运行

### 🎮 车辆控制

- 一键重置车辆位置
- 自动寻找最近合法生成点
- 避免车辆卡死或翻车后无法继续运行

### 📡 多传感器集成

已集成：

- RGB Camera
- LiDAR
- Collision Sensor
- Lane Invasion Sensor
- GNSS Sensor

支持实时数据获取与后续扩展开发。

### 📝 双格式日志系统

支持：

#### CSV 行驶日志

记录：

- 时间戳
- 车辆位置
- 当前速度
- 碰撞状态
- 到达目标次数

#### JSON 全量轨迹日志

记录：

- Frame ID
- Location
- Rotation
- Velocity
- Timestamp
- Metadata

便于后续：

- 数据分析
- 轨迹回放
- 可视化展示
- 机器学习训练

### 📊 实时 HUD

显示：

- FPS
- 当前车速
- 世界坐标
- 碰撞状态
- 自动驾驶状态
- 巡航统计信息

---

## 环境要求

| 项目 | 版本 |
|--------|--------|
| 操作系统 | Windows 10 / 11 |
| CARLA | 0.9.13 |
| Python | 3.7.x |
| pygame | 最新兼容版本 |
| numpy | 最新兼容版本 |

---

## 项目结构

```text
CARLA-Autonomous-Driving/
├── main.py
├── WindowsNoEditor/
├── logs/
│   ├── driving_log_*.csv
│   └── trajectory_log_*.json
└── README.md
```

### 目录说明

| 文件/目录 | 说明 |
|------------|------------|
| main.py | 主程序入口 |
| WindowsNoEditor | CARLA 服务端目录 |
| logs | 日志输出目录 |
| README.md | 项目文档 |

---

## 快速启动

### 1. 安装依赖

```bash
pip install pygame numpy
```

### 2. 启动 CARLA 服务端

```bash
WindowsNoEditor/CarlaUE4.exe
```

等待地图加载完成。

### 3. 启动客户端

基础运行：

```bash
py -3.7 main.py
```

循环巡航模式：

```bash
py -3.7 main.py --loop
```

激进驾驶模式：

```bash
py -3.7 main.py --loop --behavior aggressive
```

谨慎驾驶模式：

```bash
py -3.7 main.py --loop --behavior cautious
```

使用 BasicAgent：

```bash
py -3.7 main.py --agent Basic
```

---

## 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|--------|--------|--------|--------|
| --host | - | CARLA 服务端 IP | 127.0.0.1 |
| --port | -p | 服务端端口 | 2000 |
| --loop | -l | 启用循环自动巡航 | False |
| --behavior | -b | 驾驶风格 | normal |
| --agent | -a | 智能体类型 | Behavior |
| --res | - | 窗口分辨率 | 1280x720 |

### 示例

```bash
py -3.7 main.py --loop --behavior aggressive
```

```bash
py -3.7 main.py --agent Basic
```

```bash
py -3.7 main.py --host 192.168.1.100 --port 2000
```

---

## 操作快捷键

| 按键 | 功能 |
|--------|--------|
| ESC | 退出程序 |
| Ctrl + Q | 强制退出 |
| R | 重置车辆 |
| H | 显示/隐藏帮助 |
| 鼠标滚轮 | 切换摄像机视角 |

---

## 日志说明

### CSV 行驶日志

输出路径：

```text
logs/driving_log_时间戳.csv
```

示例：

```csv
timestamp,x,y,speed,collision,target_count
1711111111,23.5,12.4,35.6,False,5
```

---

### JSON 轨迹日志

输出路径：

```text
logs/trajectory_log_时间戳.json
```

示例：

```json
{
  "frame": 100,
  "location": {
    "x": 12.3,
    "y": 45.6,
    "z": 0.2
  },
  "speed": 32.5
}
```

---

## 代码优化说明

### 兼容性优化

- 移除高版本 API 依赖
- 完全适配 CARLA 0.9.13
- 兼容 Python 3.7

### 模块化重构

将系统拆分为：

- Agent 模块
- Sensor 模块
- Logging 模块
- HUD 模块
- Vehicle Controller 模块

降低代码耦合度，提高可维护性。

### 稳定性增强

- 车辆生成重试机制
- 自动检测空闲生成点
- 异常统一捕获
- 安全释放 Actor
- 安全销毁 Sensor
- 防止 CARLA 残留进程

### 性能优化

- 固定 60 FPS
- 日志流式写入
- 减少内存占用
- 降低传感器回调阻塞
- 优化 HUD 刷新频率

### 可扩展性优化

支持快速扩展：

- 新传感器
- 新智能体
- 自定义控制器
- 自定义地图测试

### 调试能力增强

- 更详细的控制台日志
- 实时运行状态监控
- HUD 信息展示
- 异常堆栈输出

---

## 常见问题

### CARLA API 导入失败

**原因：**

CARLA Python API 未正确加载。

**解决方案：**

确保目录结构如下：

```text
project/
├── main.py
└── WindowsNoEditor/
```

---

### 车辆生成失败

**原因：**

地图生成点被占用。

**解决方案：**

- 重启 CARLA
- 更换地图
- 检查生成点状态

---

### 无法连接服务器

**原因：**

2000 端口未监听。

**解决方案：**

确认：

```bash
CarlaUE4.exe
```

已经启动并完成加载。

---

### 日志未生成

检查：

- logs 文件夹权限
- 程序是否正常退出
- 磁盘空间是否充足

---

### Git 推送失败

国内网络环境下可能出现超时。

建议：

```bash
git push
```

多尝试几次，或配置代理。

---

## 更新日志

### v1.2.0

#### 新增

- JSON 全量轨迹记录
- 轨迹元数据统计

#### 优化

- 车辆重置逻辑
- 日志系统结构

#### 修复

- CARLA 0.9.13 兼容性问题

---

### v1.1.0

#### 新增

- CSV 行驶日志
- HUD 可视化界面
- 循环自动巡航

---

### v1.0.0

#### 初始版本

- 自动驾驶控制
- 多传感器采集
- 车辆控制功能
- 基础可视化界面

---

## License

仅用于学习研究与自动驾驶仿真测试。

如需商业使用，请遵循 CARLA 官方许可证要求。