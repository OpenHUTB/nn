# 系统架构

## 整体架构

```
┌──────────────────────────────────────────────┐
│                  main.py                      │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ CARLA    │  │ 摄像头   │  │ OpenCV HUD │  │
│  │ Client   │  │ Pipeline │  │ 渲染引擎   │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │              │               │         │
│  ┌────┴──────────────┴───────────────┴──────┐ │
│  │              主循环 (while True)          │ │
│  │  world.tick() → 更新视角 → 更新泊车 →    │ │
│  │  更新轨迹 → 渲染HUD → 处理按键           │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

## CARLA 同步模式

程序采用 CARLA 的 **同步模式（Synchronous Mode）**，保证仿真与渲染步调一致：

```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)
```

主循环中每帧调用 `world.tick()` 推进仿真一步。退出时恢复异步模式避免阻塞模拟器。

## 摄像头管线

```
CARLA 摄像头传感器
       │
       ▼ (raw_data: BGRA uint8 数组)
  NumPy 解码
       │
       ▼ (reshape → (480, 640, 4) → [:,:,:3])
  BGR 图像 (480×640×3)
       │
       ▼
  OpenCV 叠加 HUD 信息
       │
       ▼
  cv2.imshow("HUD", frame)
```

- 摄像头蓝图：`sensor.camera.rgb`
- 安装位置：`carla.Location(x=2.5, z=1.5)` — 车辆前方 2.5m，高度 1.5m
- 通过 `camera.listen(callback)` 异步接收数据
- `images['front']` 字典存储最新一帧

## 视角系统

5 种视角通过操作 Spectator（观察者）的 Transform 实现：

| 视角 | Location 偏移 | Rotation |
|------|--------------|----------|
| top | `+ Location(z=50)` | `pitch=-90` |
| follow | `+ Location(x=-5, z=3)` | `pitch=-15` (跟随车辆 yaw) |
| chase | `+ Location(x=-5, z=2)` | `pitch=-10` (跟随车辆 yaw) |
| side | `+ Location(y=5, z=3)` | `pitch=-15, yaw+90` |
| close | `+ Location(x=-5, z=4)` | `pitch=-25` (跟随车辆 yaw) |

所有偏移基于车辆当前 Transform 计算，Spectator 通过 `world.get_spectator().set_transform()` 更新。

## 轨迹渲染系统

### 数据采集

```python
trail_points = []            # 轨迹点队列
max_trail_length = 200       # 最大容量
```

每帧将车辆当前位置 `(x, y)` 追加到队列，超出上限时从队首移除（FIFO）。

### 坐标转换

世界坐标 → 图像坐标的简化映射：

```
img_x = 320 + (world_x - current_x) × 5
img_y = 240 + (world_y - current_y) × 5
```

以画面中心 (320, 240) 为基准，缩放因子 5 控制轨迹显示范围。仅绘制在画面范围内的线段。

### 颜色渐变

```python
def get_color_by_time(index, total):
    ratio = index / total        # 0.0(最旧) → 1.0(最新)
    r = int(255 * ratio)         # 旧→红: 0→255
    g = int(100 * (1 - ratio))   # 过渡绿
    b = int(255 * (1 - ratio))   # 新→蓝: 255→0
    return (b, g, r)             # OpenCV BGR 格式
```

渐变效果：蓝色（最新轨迹）→ 紫色 → 红色（最旧轨迹）。

## 天气系统

5 种天气模式由 `carla.WeatherParameters` 的 3 个核心参数控制：

```python
# 晴天: cloudiness=0,  precipitation=0,  fog_density=0
# 多云: cloudiness=80, precipitation=0,  fog_density=20
# 雨天: cloudiness=100, precipitation=80, fog_density=30
# 雾天: cloudiness=90, precipitation=20, fog_density=60
# 雪天: cloudiness=100, precipitation=100,fog_density=40
```

太阳高度角计算公式：`sun_altitude_angle = (current_hour - 6) × 15`

- 6:00 → 0°（日出）
- 12:00 → 90°（正午）
- 18:00 → 180°（日落）
- 0:00 → -90°（深夜）

## 泊车状态机

自动泊车由 6 阶段有限状态机驱动，每帧执行当前阶段操作：

```
  [P键触发]
      │
      ▼
  阶段0: 制动 1.0 ──→ 阶段1: 前进左转 ──→ 阶段2: 制动 1.0
                                                      │
                                                      ▼
  阶段5: 制动+恢复AP ←── 阶段4: 倒车直行 ←── 阶段3: 倒车右转
```

- 触发时 `vehicle.set_autopilot(False)` 禁用自动驾驶
- 阶段 5 完成后恢复 `vehicle.set_autopilot(True)`

## 车辆切换

切换车辆类型时销毁旧车辆并在相同位置生成新车辆。CARLA 车辆蓝图 ID：

```
vehicle.tesla.model3    → vehicle.mercedes.coupe  → vehicle.audi.a2
→ vehicle.bmw.isetta    → vehicle.ford.crown      → (循环)
```

## 资源清理

程序退出（ESC 或关闭窗口）时执行 `finally` 块：

```python
finally:
    settings.synchronous_mode = False  # 恢复异步模式
    world.apply_settings(settings)
    camera.destroy()                   # 销毁摄像头
    vehicle.destroy()                  # 销毁车辆
    cv2.destroyAllWindows()            # 关闭 OpenCV 窗口
```
