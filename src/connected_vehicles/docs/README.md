---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 142305146964284_0-data_volume/7649677419162616127-files/所有对话/主对话/网联车仿真README/README.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 142305146964284#1781080314578
    ReservedCode2: ""
---
# 🚗 网联车智慧城市仿真系统

> 基于 **CARLA 0.9.14+** 仿真平台，集成车辆控制、碰撞检测、红绿灯违规识别、动态天气模拟、驾驶员生命体征监测及可视化 GUI 监控的自动驾驶仿真系统。
>
> 适用于自动驾驶仿真、驾驶员状态分析、智能交通系统验证等研究与教学场景。

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](#)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.14%20%7C%200.9.15-green)](#)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Ubuntu-lightgrey)](#)
[![License](https://img.shields.io/badge/License-Research%20%26%20Education-orange)](#)

---

## 📑 目录

- [项目概览](#-项目概览)
- [核心功能](#-核心功能)
- [系统架构](#-系统架构)
- [数据流与状态机](#-数据流与状态机)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [操作说明](#️-完整操作说明)
- [驾驶员体征规则](#-驾驶员体征监测规则)
- [实时监控窗口](#-实时监控窗口)
- [配置参考](#️-配置参考)
- [模块 API 速查](#-模块-api-速查)
- [常见问题](#-常见问题)
- [开发与扩展](#-开发与扩展)
- [更新日志](#-更新日志)
- [路线图](#-路线图)

---

## 🌟 项目概览

本项目在 CARLA 仿真环境中构建了一个**完整的智能交通仿真闭环**，核心特点：

- 🧩 **模块化设计**：车辆控制、碰撞、天气、交通灯、体征、GUI 六大模块完全解耦，可独立替换与测试
- 🧵 **多线程架构**：主循环 + GUI 独立线程 + 碰撞监听，事件驱动 + 状态轮询双模式
- 🔁 **状态联动**：车辆状态、天气变化、碰撞事件、闯红灯行为都会**实时反作用于驾驶员体征**
- 🛡️ **健壮性**：信号优雅退出、资源自动回收、线程安全字典、防抖按键、异常降级日志
- 📊 **可观测性**：双视图监控（终端周期输出 + Tkinter 实时窗口），碰撞日志落盘

---

## 🎯 核心功能

| 模块 | 能力要点 |
| --- | --- |
| 🚗 **车辆控制** | 键盘操控（前进/倒车/转向/急刹/普通刹车）、自动限速、视角跟随、初始位置重置 |
| 🌤 **环境模拟** | 晴天 / 雨天 / 雾天 / 夜间 四态切换，能见度、降水、雾密度、太阳高度角联动调整 |
| 🚨 **碰撞监测** | 碰撞传感器实时检测 + 手动模拟碰撞，自动写入 `collision_logs.txt`（含时间、对象、位置、速度） |
| 🚦 **交通违规检测** | 闯红灯识别（带距离过滤、缓存优化、状态变化告警） |
| 🧑‍⚕️ **驾驶员体征** | 心率、血压、疲惫度多因素耦合计算，**疲惫度时间流速 ×10** |
| 📊 **可视化 GUI** | 独立 Tkinter 窗口（置顶、非阻塞、线程安全），异常高亮预警 |
| 🧹 **资源管理** | 退出时自动销毁车辆、传感器、GUI 线程，零资源残留 |

---

## 🏗️ 系统架构

### 模块依赖关系

```
                ┌──────────────────────────────────────────────┐
                │                  drive.py                    │
                │  (主入口 / 主循环 / 信号处理 / 资源清理)      │
                └──────────────┬───────────────────────────────┘
                               │
       ┌───────────────┬───────┼───────┬────────────────┐
       ▼               ▼       ▼       ▼                ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  collision  │ │ environment │ │traffic_light│ │   driver    │ │vehicle_status│
│  _monitor   │ │ _controller │ │_controller  │ │   _vitals   │ │    _gui     │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │               │               │
       └───────────────┴───────────────┼───────────────┘               │
                                       ▼                               │
                                 ┌──────────┐                         │
                                 │   utils  │ ◀───────────────────────┘
                                 │ (工具层) │
                                 └────┬─────┘
                                      ▼
                              ┌──────────────┐
                              │   config.py  │
                              │ (全局配置)   │
                              └──────────────┘
```

### 主循环执行流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ ① 计算车速  │───▶│ ② 周期打印  │───▶│ ③ 车辆控制  │
│   (10ms)    │    │   (状态行)   │    │  (键盘映射) │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                            │
       ┌────────────────────────────────────┘
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ ④ 视角跟随  │───▶│ ⑤ 碰撞处理  │───▶│ ⑥ 模拟碰撞  │
│  (spectator)│    │  (状态复位)  │    │   (C 键)   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                            │
       ┌────────────────────────────────────┘
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ ⑦ 天气切换  │───▶│ ⑧ 红绿灯    │───▶│ ⑨ 重置车辆  │
│   (W 键)   │    │   违规检测  │    │   (R 键)   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                            │
       ┌────────────────────────────────────┘
       ▼
┌─────────────┐    ┌─────────────┐
│ ⑩ 体征更新  │───▶│ ⑪ 退出检测  │─── (回到 ①)
│  (联动)    │    │   (ESC)     │
└─────────────┘    └─────────────┘
```

---

## 🔄 数据流与状态机

### 数据流向

```
┌──────────────┐                              ┌──────────────┐
│   CARLA      │   vehicle/weather/light      │   主循环     │
│  World API   │ ──────────────────────────▶  │  drive.py    │
└──────────────┘                              └──────┬───────┘
                                                      │
                ┌─────────────────────────────────────┤
                │                                     │
                ▼                                     ▼
        ┌──────────────┐                      ┌──────────────┐
        │  状态监测器  │   speed/collision/   │  GUI 实例    │
        │ (collision / │ ─── violation/... ──▶│ (共享状态字典)│
        │  tl / env)  │                      └──────┬───────┘
        └──────┬───────┘                             │
               │ heart_rate/bp/fatigue               │ 线程安全
               ▼                                     ▼
        ┌──────────────┐                      ┌──────────────┐
        │  体征监测器  │                      │  Tkinter     │
        │(vitals_mon)  │                      │  渲染循环    │
        └──────────────┘                      └──────────────┘
```

### 疲惫度状态机

```
        ┌──────────┐    疲惫度<30     ┌──────────┐
        │  启动    │ ────────────────▶ │  normal  │
        │  0.0     │                  └────┬─────┘
        └──────────┘                       │ 持续驾驶
                                           ▼
                                    ┌──────────┐
                                    │  tired   │ (30~60)
                                    └────┬─────┘
                                         │ 时间增长
                                         ▼
                                    ┌──────────┐
                                    │ fatigued │ (60~80)
                                    └────┬─────┘
                                         │
                                         ▼
                                    ┌──────────────┐
                                    │extreme_fatigue│ (≥80) ⚠️
                                    └──────────────┘

注：R 键重置车辆时 → 回到 normal
```

### 天气状态机

```
  ┌────────┐    W 键    ┌────────┐    W 键    ┌────────┐    W 键    ┌────────┐
  │ clear  │ ─────────▶ │  rain  │ ─────────▶ │  fog   │ ─────────▶ │ night  │
  │100% 能 │            │ 60% 能 │            │ 30% 能 │            │ 80% 能 │
  │见度 12:00          │见度 14:00          │见度 08:00          │见度 22:00
  └────────┘ ◀───────── └────────┘ ◀───────── └────────┘ ◀───────── └────────┘
                       W 键循环回到 clear
```

---

## 📁 项目结构

```
.
├── drive.py                     # 主程序入口（CarlaDriver 类 + 主循环）
├── config.py                    # 全局参数配置
├── vehicle_status_gui.py        # 可视化监控窗口（Tkinter + 独立线程）
├── driver_vitals_monitor.py     # 驾驶员体征计算模块
├── collision_monitor.py         # 碰撞传感器与日志模块
├── environment_controller.py    # 天气 / 能见度 / 模拟时间控制
├── traffic_light_controller.py  # 红绿灯缓存 + 闯红灯检测
├── utils.py                     # 工具函数（车速计算、防抖、线程安全字典）
├── collision_logs.txt           # 碰撞日志（运行时自动生成）
├── images/                      # 截图 / 文档图片
└── README.md                    # 本说明文档
```

| 文件 | 行数级别 | 核心职责 |
| --- | --- | --- |
| `drive.py` | 250+ | 入口、信号处理、主循环、按键映射、视角跟随 |
| `config.py` | 60 | 集中所有可调参数（连接、控制、GUI、天气、体征） |
| `vehicle_status_gui.py` | 250+ | Tkinter 窗口构建、线程安全数据更新、样式异常高亮 |
| `driver_vitals_monitor.py` | 150+ | 体征计算（多因素耦合）、状态机、随机波动、异常日志 |
| `collision_monitor.py` | 80 | 传感器创建、回调处理、日志落盘、状态管理 |
| `environment_controller.py` | 60 | WeatherParameters 配置、能见度计算 |
| `traffic_light_controller.py` | 80 | 红绿灯缓存、距离过滤、状态查询、违规判定 |
| `utils.py` | 50 | 车速计算、防抖检查、加锁字典更新 |

---

## 🚀 快速开始

### 1. 环境要求

| 项目 | 版本 / 要求 |
| --- | --- |
| 操作系统 | Windows 10+ / Ubuntu 20.04+ |
| CARLA | 0.9.14 ~ 0.9.15 |
| Python | 3.8 ~ 3.11 |
| 推荐 IDE | PyCharm（社区版 / 专业版均可） |

### 2. 安装依赖

```bash
pip install carla keyboard
```

> ⚠️ Linux 下 `keyboard` 需要 root 权限。
> ⚠️ Windows 下 `keyboard` 若遇权限问题，以管理员身份运行终端。

### 3. 启动步骤

**① 启动 CARLA 模拟器**

```bash
# Windows
CarlaUE4.exe
# Linux
./CarlaUE4.sh
```

**② 运行主程序**

```bash
python main.py
```

程序启动后自动完成：

1. 连接 CARLA `127.0.0.1:2000`，生成主车辆
2. 挂载碰撞传感器，初始化天气为「晴天」
3. 弹出独立的「车辆实时状态」监控窗口（置顶）
4. 在终端打印操作说明
5. 进入主循环（10ms tick）

---

## ⌨️ 完整操作说明

### 基础操控

| 按键 | 功能 | 备注 |
| --- | --- | --- |
| ↑ | 前进 | throttle=1.0 |
| ↓ | 倒车 | reverse=True, gear=-1 |
| ← | 左转 | steer=-0.5 |
| → | 右转 | steer=+0.5 |
| 空格 | 急刹 | brake=1.0, hand_brake=True，控制台打印当前车速 |
| S | 普通刹车 | brake=1.0 |

### 功能键（全部带防抖）

| 按键 | 功能 | 触发条件 |
| --- | --- | --- |
| **C** | 模拟碰撞 | 车速 > 0 时记录碰撞车速并标记碰撞状态 |
| **W** | 循环切换天气 | 晴天 → 雨天 → 雾天 → 夜间 → 晴天 |
| **R** | 重置车辆 | 回到初始生成点，速度/碰撞/体征全部清零 |
| **ESC** | 退出程序 | 优雅释放所有资源 |

### 自动化行为

- **限速保护**：当前车速 > `MAX_SPEED_KMH`（默认 100）时，throttle 自动降到 0.2
- **视角跟随**：spectator 始终位于车辆后上方（-10m, +4m），俯角 -20°

---

## 🧑‍⚕️ 驾驶员体征监测规则

### 1. 监测指标

| 指标 | 影响因素 | 上限 | 随机扰动 |
| --- | --- | --- | --- |
| **心率** | 驾驶时长 + 车速 + 天气 + 碰撞 | 180 次/分钟 | ±2 |
| **血压** | 车速 + 天气 + 碰撞 | 收缩压 180 / 舒张压 120 | — |
| **疲惫度** | 驾驶时长 + 天气（**与车速完全解耦**） | 100 | — |

> 🕐 **时间流速 ×10**：疲惫度增长 = 真实时间 × 10，便于快速观察状态变化。

### 2. 计算公式（简版）

```
心率 = 基础心率
     + 车速 × 0.3
     + 驾驶时长(分) × 0.1 × 天气因子
     + 末次碰撞车速 × 0.8
     ± 随机扰动

收缩压 = 基础收缩压
       + 车速 × 0.2 × 天气因子
       + 末次碰撞车速 × 0.5

舒张压 ≈ 收缩压 × 0.5（按比例联动）

疲惫度 = max(当前疲惫度, 驾驶时长(分) × 0.2 × 10 × 天气因子)
```

### 3. 疲惫度等级

| 数值范围 | 状态 | GUI 样式 | 建议 |
| --- | --- | --- | --- |
| 0 ~ 30 | `normal` | 默认 | 正常驾驶 |
| 30 ~ 60 | `tired` | 黄色警告 | 适当休息 |
| 60 ~ 80 | `fatigued` | 黄色警告 | 计划停车 |
| ≥ 80 | `extreme_fatigue` | 🔴 红色加粗 | 立即停车 |

### 4. 异常预警

- 心率 > 120 次/分钟 → GUI 心率标红 + 控制台 WARN 日志
- 疲惫度 > 80 → GUI 等级标红 + 控制台 WARN 建议

---

## 📊 实时监控窗口

### 终端周期输出（每 200ms）

```
速度：45.3 km/h | 天气：rain | 能见度：60% | 碰撞车速：0.0 km/h | 心率：88.5 | 疲惫度：12.4% | 闯红灯：否
```

### Tkinter GUI 窗口（独立线程）

布局如下：

```
┌──────────────────────────────────────┐
│      🚗 车辆实时状态                  │
├──────────────────────────────────────┤
│ 当前车速：     45.3 km/h             │
│ 当前天气：     rain                  │
│ 能见度：       60%                   │
│ 碰撞车速：     0.0 km/h              │
│ 碰撞状态：     未碰撞                 │
│ 闯红灯：       否                    │
├──────────────────────────────────────┤
│      🧑‍⚕️ 驾驶员体征监测               │
├──────────────────────────────────────┤
│ 心率：         88.5 次/分钟           │
│ 血压：         128/85 mmHg           │
│ 疲惫度：       12.4 %                │
│ 疲惫等级：     normal                │
└──────────────────────────────────────┘
```

- 窗口属性：置顶（`-topmost`）、工具窗口（`-toolwindow`）、不可缩放
- 更新频率：50ms
- 异常高亮：自动切换为黄色 / 红色加粗样式

---

## ⚙️ 配置参考

所有可调参数集中在 [`config.py`](../src/config.py)：

### CARLA 连接

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `CARLA_HOST` | `127.0.0.1` | 服务器地址 |
| `CARLA_PORT` | `2000` | 服务器端口 |
| `CARLA_TIMEOUT` | `10.0` | 连接超时（秒） |

### 车辆控制

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_SPEED_KMH` | `100.0` | 最大车速限制 |
| `SPAWN_POINT_OFFSET` | `10.0` | 车辆生成位置后退距离（米） |
| `STEER_ANGLE` | `0.5` | 转向角度（-1~1） |
| `BRAKE_INTENSITY` | `1.0` | 刹车强度 |

### GUI

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `GUI_WINDOW_SIZE` | `"500x450+20+20"` | 窗口尺寸与位置 |
| `GUI_UPDATE_INTERVAL_MS` | `50` | GUI 刷新间隔（毫秒） |
| `GUI_TITLE` | `"车辆实时状态监控 - 含驾驶员体征"` | 窗口标题 |

### 天气

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `WEATHER_LIST` | `["clear", "rain", "fog", "night"]` | 天气循环列表 |
| `DEFAULT_WEATHER` | `"clear"` | 初始天气 |

### 红绿灯

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `TRAFFIC_LIGHT_DETECT_DISTANCE` | `50.0` | 检测最大距离（米） |
| `TRAFFIC_LIGHT_FILTER` | `"traffic.traffic_light"` | Actor 过滤 |

### 碰撞

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `COLLISION_SENSOR_BP` | `"sensor.other.collision"` | 传感器蓝图 |
| `COLLISION_LOG_FILE` | `"collision_logs.txt"` | 日志输出文件 |

### 驾驶员体征

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `VITALS_BASE_HEART_RATE` | `75.0` | 基础心率（次/分） |
| `VITALS_BASE_BLOOD_PRESSURE` | `(120, 80)` | 基础血压（收缩/舒张） |
| `VITALS_BASE_FATIGUE` | `0.0` | 基础疲惫度（0-100） |
| `VITALS_SPEED_FACTOR` | `{heart_rate: 0.3, blood_pressure: 0.2}` | 车速影响因子 |
| `VITALS_TIME_FACTOR` | `{heart_rate: 0.1, fatigue: 0.2}` | 时间影响因子 |
| `VITALS_WEATHER_FACTORS` | `{clear: 1.0, rain: 1.2, fog: 1.3, night: 1.15}` | 天气影响因子 |
| `VITALS_COLLISION_FACTOR` | `{heart_rate: 0.8, blood_pressure: 0.5}` | 碰撞影响因子 |

---

## 🧩 模块 API 速查

### `CollisionMonitor`

```python
monitor = CollisionMonitor()
monitor.create_collision_sensor(world, vehicle)  # 创建并绑定
monitor.get_collision_occurred() -> bool          # 查询状态
monitor.reset_collision_occurred()                 # 复位
monitor.stop()                                     # 销毁传感器
```

### `EnvironmentController`（全局单例 `env_controller`）

```python
env_controller.set_weather(world, "rain")            # 切换天气
state = env_controller.get_current_environment_state()  # {'weather_type', 'visibility', 'current_hour'}
```

### `TrafficLightController`（全局单例 `tl_controller`）

```python
light = tl_controller.get_vehicle_traffic_light(world, vehicle)  # 最近红灯
state = tl_controller.get_traffic_light_state(light)              # 'red'/'yellow'/'green'/'off'
violation = tl_controller.check_red_light_violation(world, vehicle)  # bool
```

### `DriverVitalsMonitor`（全局单例 `vitals_monitor`）

```python
update_driver_vitals(vehicle, weather_type, collision_occurred)
data = get_driver_vitals()           # {'heart_rate', 'blood_pressure', 'fatigue', 'fatigue_level'}
reset_driver_vitals()                # 重置（车辆重置时调用）
```

### `VehicleStatusGUI`（全局单例 `gui_instance`）

```python
create_status_window()               # 启动 GUI 线程
update_vehicle_status(key, value)    # 线程安全更新
stop_gui()                           # 关闭 GUI
```

### `utils`

```python
calculate_vehicle_speed_kmh(vehicle) -> float  # 车速（km/h）
debounce_check(pressed, flag_list) -> bool     # 按键防抖
safe_update_dict(d, k, v)                      # 加锁字典更新
```

---

## 🛠️ 常见问题

**Q1. 连接 CARLA 失败？**
- 确认 CARLA 模拟器已启动
- 确认端口为 2000（与 `config.py` 一致）
- 关闭防火墙或放行 2000 端口
- Linux 确认 2000 端口未被占用：`lsof -i :2000`

**Q2. 车辆无法移动？**
- 检查终端窗口是否获得键盘焦点
- 确认 `pip install keyboard` 已成功执行
- Linux 需要 root 权限或安装 `setcap cap_net_bind_service=+ep /usr/bin/python3`

**Q3. 监控窗口没出来？**
- 确认 Tkinter 可用：`python -c "import tkinter"`
- 留意终端是否有异常堆栈
- 某些远程桌面/SSH 环境不支持 GUI，需要本地运行

**Q4. 疲惫度只涨到 ~20 就卡住？**
- 旧版本逻辑有 bug，请升级到 V2.0+（已使用 `max(当前, 目标)` 强制递增）

**Q5. 多个红绿灯/车辆导致卡顿？**
- 红绿灯列表已加缓存（`traffic_lights_cache`），仅在空时刷新
- 减少生成车辆数量，关闭非必要传感器

**Q6. PyCharm 中 README 图片不显示？**
将图片放入 `images/` 文件夹，使用：
```markdown
![图片名](images/xxx.png)
```

---

## 🔧 开发与扩展

### 二次开发入口

| 想做什么 | 从哪里改 |
| --- | --- |
| 加新天气类型 | `config.py::WEATHER_LIST` + `environment_controller.py::set_weather` |
| 调体征灵敏度 | `config.py::VITALS_*_FACTOR` |
| 加新的体征指标（如血氧） | `driver_vitals_monitor.py` + `vehicle_status_gui.py` |
| 加新按键功能 | `drive.py::main_loop`（参考 C/W/R 键的 debounce_check 用法） |
| 替换仿真平台 | 各 controller 类的接口保持稳定，只需替换 CARLA 调用 |

### 测试建议

```bash
# 1. 静态检查
python -m py_compile *.py

# 2. 模块独立测试（无需 CARLA）
python -c "from utils import calculate_vehicle_speed_kmh; print(calculate_vehicle_speed_kmh(None))"

# 3. GUI 单独测试
python -c "from vehicle_status_gui import create_status_window, stop_gui, update_vehicle_status; import time; create_status_window(); time.sleep(2); update_vehicle_status('fatigue', 75); time.sleep(5); stop_gui()"
```

### 常见扩展方向

- 🚛 **多车协同仿真**：基于 `TrafficManager` 自动驾驶
- 📡 **V2X 车联网通信**：通过 CARLA ROS Bridge 对接 ROS2
- 🤖 **强化学习训练**：用本项目作为 Gym 环境封装
- 📼 **真实数据回放**：将体征数据导出为 CSV/Parquet
- 🌐 **Web 可视化**：用 Flask/FastAPI 暴露体征数据 Web 面板
- 📦 **Docker 化部署**：CARLA + 本项目一键拉起

---

## 📝 更新日志

### V2.0（当前版本 · 驾驶员体征增强版）

- ✅ 心率/血压/疲惫度**多因素耦合**计算
- ✅ 疲惫度与车速**完全解耦**，仅随时间增长
- ✅ 疲惫度时间流速 **×10 倍**，仿真更直观
- ✅ 心率随机扰动（±2），更真实
- ✅ GUI 异常高亮（心率/疲惫等级）
- ✅ 资源自动管理，退出自动销毁
- ✅ 完全兼容原有所有功能

### V1.0（基础版）

- ✅ 车辆控制（前进/倒车/转向/刹车/限速）
- ✅ 天气切换（晴天/雨天/雾天/夜间）
- ✅ 碰撞检测 + 模拟碰撞 + 日志
- ✅ 红绿灯违规检测
- ✅ Tkinter 状态监控窗口

---

## 🚀 路线图

- [ ] 主车辆自动驾驶模式（P 键切换）
- [ ] A 键一键生成 N 辆 AI 自动驾驶车辆
- [ ] V2X 车与车通信仿真
- [ ] 强化学习 Gym 环境封装
- [ ] 真实驾驶员数据驱动（CSV 回放）
- [ ] Web 可视化面板
- [ ] Docker 一键部署
- [ ] 单元测试覆盖

---

## 📞 引用信息

| 项 | 说明 |
| --- | --- |
| 项目版本 | V2.0（驾驶员体征增强版） |
| 仿真平台 | [CARLA Simulator](https://carla.org/) |
| 核心依赖 | `carla`、`keyboard`、`tkinter`（Python 内置） |
| 适用工具 | PyCharm 全版本 |

> ✨ 如果这个项目对你有帮助，欢迎点个 Star 鼓励一下～

---

**🤝 贡献指南**：欢迎提交 Issue 和 PR，建议在提交前运行 `python -m py_compile *.py` 确保无语法错误。

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
