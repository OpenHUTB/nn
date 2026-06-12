# 快速开始

## 环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| CARLA Simulator | 0.9.14 或更高 | 自动驾驶模拟引擎 |
| Python | 3.7 或更高 | 主程序运行环境 |
| OpenCV | 最新稳定版 | 图像渲染与 HUD 显示 |
| NumPy | 最新稳定版 | 摄像头原始数据解码 |

## 安装步骤

### 1. 安装 CARLA

从 [CARLA 官方 GitHub](https://github.com/carla-simulator/carla/releases) 下载对应平台的版本，解压到本地目录。

### 2. 安装 Python 依赖

```bash
pip install opencv-python numpy
```

如需使用清华镜像加速：

```bash
pip install opencv-python numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 获取项目代码

将 `vehicle_HUD/` 目录复制到本地任意位置。

## 运行程序

### 第一步：启动 CARLA 模拟器

进入 CARLA 安装目录，启动模拟器：

```bash
# Windows
CarlaUE4.exe

# Linux
./CarlaUE4.sh
```

模拟器启动后默认监听 `localhost:2000`。

### 第二步：运行 HUD 主程序

在另一个终端中进入 `vehicle_HUD/` 目录：

```bash
cd vehicle_HUD
python main.py
```

程序启动后：
1. 自动连接 CARLA（超时 20 秒）
2. 在随机生成点创建 Tesla Model 3
3. 开启自动驾驶模式
4. 弹出 HUD 可视化窗口，显示实时画面与状态信息

## 操作指南

程序通过键盘控制，所有功能键大小写不敏感：

| 类别 | 按键 | 功能 |
|------|------|------|
| 视角 | `T` `F` `C` `S` `X` | 5 种视角切换 |
| 车辆 | `K` | 循环切换车型 |
| 泊车 | `P` | 启动自动泊车 |
| 天气 | `V` | 循环切换天气 |
| 时间 | `U` | 切换时间（+3 小时） |
| 导航 | `N` | 设置/重置随机目的地 |
| 限速 | `+` `-` | 调整超速阈值 |
| 警告 | `W` | 开关速度警告 |
| 轨迹 | `R` | 重置行驶轨迹 |
| 退出 | `ESC` | 退出程序 |

## 退出程序

按 `ESC` 键或关闭 HUD 窗口即可退出。程序会自动执行资源清理：

- 恢复 CARLA 异步模式
- 销毁摄像头与车辆 Actor
- 关闭所有 OpenCV 窗口

## 常见启动问题

**"连接超时" 错误**

确认 CARLA 模拟器已完全启动且监听在 `localhost:2000`。可在模拟器窗口看到 3D 场景即表示就绪。

**ImportError: No module named 'carla'**

需要安装 CARLA Python API（`.egg` 或 `.whl` 文件位于 CARLA 安装目录的 `PythonAPI/carla/dist/` 下）：

```bash
pip install <CARLA目录>/PythonAPI/carla/dist/carla-*.whl
```

**OpenCV 窗口无响应**

确认 `main.py` 在 CARLA 模拟器启动后运行，且 CARLA 未处于无渲染模式（`-RenderOffScreen`）。
