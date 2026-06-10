# CARLA 自动驾驶控制程序使用文档
## 1. 程序概述
该Python程序基于CARLA 0.9.15仿真平台实现了自动驾驶车辆的控制功能，支持自动驾驶/手动驾驶切换、传感器数据采集、交通参与者生成与管理、天气切换等核心功能，适用于自动驾驶算法测试、交通场景仿真等应用场景。

### 1.1 核心特性
- 支持自动驾驶（BehaviorAgent/BasicAgent）与手动驾驶模式切换
- 集成碰撞、车道入侵、GNSS等传感器，实时采集车辆状态数据
- 动态生成行人、NPC交通车辆，支持批量生成与清理
- 支持天气预设切换、摄像头视角调整
- 实时HUD显示车辆状态（速度、位置、里程、碰撞信息等）
![red_light_stop](images/redlight.png)
![green_light_run](images/greenlight.png)

## 2. 环境准备
### 2.1核心运行依赖
本项目基于 CARLA 0.9.15 开发，核心依赖及推荐版本如下：

| 依赖库 | 版本要求 |
| ---- | ---- |
| carla | 0.9.15 |
| numpy | ≥1.21.6 |
| pygame | ≥2.1.0 |

一键安装命令：
```bash
pip install pygame numpy
```

### 2.2 路径配置
程序中已配置CARLA PythonAPI路径，需根据实际安装路径调整：
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'CARLA_0.9.15', 'WindowsNoEditor', 'PythonAPI', 'carla'))
```

## 3. 核心功能说明
### 3.1 驾驶模式控制
| 按键 | 功能 |
|------|------|
| F | 切换自动驾驶/手动驾驶模式 |
| WASD/方向键 | 手动驾驶（仅自动驾驶关闭时生效）：<br>- W/上方向键：油门（0.8）<br>- S/下方向键：刹车（0.8）<br>- A/左方向键：左转（-0.6）<br>- D/右方向键：右转（0.6） |

![f键切换驾驶模式](images/f.png)

### 3.2 场景管理
| 按键 | 功能 |
|------|------|
| G | 生成随机行人（优先在人行道生成，无人行道则在道路生成） |
| T | 生成5辆NPC交通车辆（自动开启自动驾驶） |
| Y | 批量清理所有生成的交通车辆 |
| N | 切换下一个天气预设 |
| M | 切换上一个天气预设 |
| C | 切换摄像头视角 |

![g键生成随机行人](images/g.png)
![t键生成npc交通车辆](images/t.png)
![切换天气](images/wetherchange.png)
![切换视角](images/c1.png)
![切换视角](images/c2.png)
![切换视角](images/c3.png)

### 3.3 基础操作
| 按键 | 功能 |
|------|------|
| H | 显示/隐藏帮助信息 |
| ESC/Ctrl+Q | 退出程序 |

![helptext_show](images/h.png)

## 4. 核心类说明
### 4.1 World类
- 作用：管理CARLA仿真世界，包括车辆生成、传感器初始化、天气切换等
- 核心方法：
  - `restart()`：重新生成车辆并初始化传感器
  - `next_weather()`：切换天气预设
  - `destroy()`：销毁所有演员（车辆、传感器等）

### 4.2 KeyboardControl类
- 作用：处理键盘事件，实现驾驶控制、场景交互
- 核心方法：
  - `parse_events()`：解析键盘输入事件
  - `spawn_random_pedestrian()`：生成随机行人
  - `spawn_traffic_vehicles()`：生成NPC交通车辆
  - `clear_traffic_vehicles_batch()`：批量清理交通车辆

### 4.3 传感器类
| 类名 | 作用 |
|------|------|
| CollisionSensor | 碰撞传感器，检测车辆碰撞事件并记录碰撞强度 |
| LaneInvasionSensor | 车道入侵传感器，检测车辆压线行为 |
| GnssSensor | GNSS传感器，采集车辆经纬度信息 |
| CameraManager | 摄像头管理器，控制摄像头视角与数据采集 |

### 4.4 HUD类
- 作用：实时显示车辆状态信息，包括：
  - 服务器/客户端帧率
  - 车辆速度、航向、位置、GNSS坐标
  - 行驶里程（里程表）
  - 碰撞历史、附近车辆数量
  - 油门/刹车/转向参数

![hud_show](images/hud.png)

## 5. 运行说明
### 5.1 启动CARLA服务器
```bash
# Windows
cd CARLA_0.9.15/WindowsNoEditor
CarlaUE4.exe
# Linux
cd CARLA_0.9.15/NoEditor
./CarlaUE4.sh
```

### 5.2 运行程序
```bash
python automatic_control.py [可选参数]
```

### 5.3 可选参数
| 参数 | 说明 |
|------|------|
| --filter | 车辆蓝图过滤（如：vehicle.*） |
| --gamma | 摄像头伽马校正值 |
| --seed | 随机种子 |
| --agent | 自动驾驶代理类型（Basic/Behavior） |
| --loop | 循环模式 |
| --behavior | 驾驶风格（normal/cautious/aggressive） |

## 6. 扩展开发
### 6.1 新增传感器
可参考现有传感器类（如CollisionSensor）实现新传感器集成：
```python
class CustomSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        # 加载传感器蓝图
        blueprint = world.get_blueprint_library().find('sensor.xxx.xxx')
        # 生成传感器并绑定到车辆
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # 监听传感器数据
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: self._on_data(weak_self, event))
    
    @staticmethod
    def _on_data(weak_self, event):
        self = weak_self()
        if not self:
            return
        # 处理传感器数据
        pass
```

### 6.2 自定义驾驶策略
修改`KeyboardControl`类的`parse_events`方法，或扩展BehaviorAgent/BasicAgent实现自定义自动驾驶逻辑：
```python
# 示例：修改手动驾驶参数
elif event.key == pygame.K_UP or event.key == pygame.K_w:
    self.throttle = 1.0  # 调整油门至100%
    self.brake = 0.0
```

## 7. 注意事项
1. 确保CARLA服务器版本与PythonAPI版本一致（0.9.15）
2. 生成大量交通车辆时，建议降低仿真帧率以保证性能
3. 清理交通车辆时，程序会自动检查车辆状态，避免销毁已失效的演员
4. 行人生成优先选择人行道，若无人行道会降级到道路生成，需注意碰撞风险
5. 自动驾驶模式下，BehaviorAgent支持交通灯识别，BasicAgent仅实现基础路径跟踪

## 8. 故障排除
| 问题 | 解决方案 |
|------|----------|
| 无法导入carla模块 | 检查PythonAPI路径配置是否正确 |
| 车辆无法生成 | 确认地图存在Spawn Point，或检查过滤参数是否正确 |
| 传感器数据无响应 | 检查传感器是否成功生成，或CARLA服务器是否正常运行 |
| 批量清理车辆失败 | 确认client对象已初始化，或手动调用clear_traffic_vehicles()方法 |