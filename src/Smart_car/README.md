---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: 77d13f0624762f2de86f1694a087dcde
    PropagateID: 77d13f0624762f2de86f1694a087dcde
    ReservedCode1: 30450220450de8ed6ebcc59d795ca07193a8349cdff122af3914c845bc06a5eccd697a86022100801cefd02340cae09cb226bc46bd40a26c0a6a3fa0b44e7f34697b6d73b2d4f2
    ReservedCode2: 3045022100f50987c5dc34b38c3c06803f29fc5d64f7d0a1d3dc0ea74c7108f5005deb029e02203a74966d98dd7f581473e2263aa5c776873a84476b30fffe04599a50827e0d83
---

# 无人车项目

一个基于计算机视觉的简单无人车导航系统，支持车道线检测、目标识别和自动控制。

## 功能特性

1.**车道线检测**: 实时识别道路车道线
2.**目标识别**: 检测车辆、行人和交通标志
3.**路径规划**: 自动规划安全行驶路径
4.**智能控制**: PID控制器实现精确转向和速度控制
5.**实时显示**: 可视化检测结果和控制状态

## 快速开始

### 方式一：一键启动（推荐）

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```cmd
start.bat
```

### 方式二：手动启动

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行程序**
   ```bash
   python main.py
   ```

## 运行模式

`python main.py` - 完整模式（所有功能）
`python main.py --mode perception` - 仅感知测试
`python main.py --debug` - 调试模式（显示详细信息）

## 系统要求

**操作系统**: Windows/Linux/macOS
**Python**: 3.8+
**内存**: 4GB+
**摄像头**: USB摄像头（720p以上）

## 项目结构

```
无人车项目/
├── main.py                 # 主程序
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖包
├── start.sh / start.bat    # 启动脚本
├── src/                    # 源代码
│   ├── perception/         # 感知模块
│   ├── planning/           # 规划模块
│   └── control/            # 控制模块
└── docs/                   # 文档
```

## 核心模块

### 感知模块 (perception)
**车道线检测**: 使用OpenCV识别车道线
**目标检测**: 检测车辆、行人等目标

### 规划模块 (planning)
**路径生成**: 基于感知结果生成行驶路径
**障碍物避让**: 智能避障算法

### 控制模块 (control)
**PID控制**: 精确的转向和速度控制
**轨迹跟踪**: 沿着规划路径行驶

## 配置说明

主要配置参数在 `config.yaml` 文件中：

```yaml
# 相机设置
camera:
  resolution: [1280, 720]   # 分辨率
  fps: 30                   # 帧率

# 检测设置
lane_detection:
  min_line_length: 50       # 最小线段长度

control:
  target_speed: 25          # 目标速度 (km/h)
  kp: 1.2                   # PID参数
```

## 运行效果

程序运行时会显示：
**实时视频**: 来自摄像头的画面
**检测结果**: 车道线和目标识别结果
**控制信息**: 转向角度和速度控制输出
**状态日志**: 系统运行状态

示例输出：
```
[INFO] 启动无人车导航系统
[INFO] 相机初始化完成: 1280x720@30fps
[INFO] 车道线检测: 找到2条车道线
[INFO] 检测到目标: 车辆(距离12.5m) 行人(距离8.2m)
[INFO] 路径规划完成，长度: 45个点
[INFO] 控制输出: 转向=2.3度, 速度=24.8km/h
```

## 常见问题

**Q: 摄像头无法启动？**
A: 检查摄像头连接，或尝试 `--camera 1` 指定其他设备

**Q: 程序运行很慢？**
A: 降低相机分辨率或关闭调试模式

**Q: 车道线检测不准确？**
A: 检查光线条件，调整配置文件中的参数

## 故障排除

1. **依赖安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **权限问题 (Linux)**
   ```bash
   sudo usermod -a -G video $USER
   # 重新登录或重启
   ```

3. **调试模式**
   ```bash
   python main.py --debug --save-images
   ```

## 性能调优

提升性能的方法：
使用更高配置的硬件
降低图像分辨率
关闭不必要的可视化
调整检测参数
## 技术支持

查看 `docs/usage_guide.md` 了解详细使用说明
运行 `python tests/test_basic.py` 进行功能测试
检查日志文件了解运行状态
