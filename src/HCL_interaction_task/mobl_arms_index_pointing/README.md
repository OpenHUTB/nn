# 生物模型交互任务--食指追踪小球

- 一个基于 MuJoCo 的人体手臂和手部运动仿真项目，专注于食指指向功能。该项目集成了手部追踪与基于物理的仿真，以复现逼真的类人指向手势。

## 概述

- 本仓库包含人体上肢的物理仿真，专门设计用于模拟食指指向动作。仿真使用MuJoCo作为物理引擎，并并结合MediaPipe进行实时手部追踪，能够将现实世界的手部手势映射到虚拟化身的动作中。

- **运行视频**

https://private-user-images.githubusercontent.com/221759988/494325644-39245f50-7e03-46c3-b7d9-c451c5a28113.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg4Nzc0NjksIm5iZiI6MTc1ODg3NzE2OSwicGF0aCI6Ii8yMjE3NTk5ODgvNDk0MzI1NjQ0LTM5MjQ1ZjUwLTdlMDMtNDZjMy1iN2Q5LWM0NTFjNWEyODExMy5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwOTI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDkyNlQwODU5MjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xYmIwMWZhYjE2MWU4NGI0ZWMzZTBhNzEyZGQ4ZGNiNzFkZjQwZmNkYzBjMzllNTViZTUwNmFjNmIzOTE3OTZjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.EJoN_MZUfH3cgyccDSDbuhTBrsJBkT7wSPSV835Gl7w

## 主要特点

- **高精度骨骼模型**：在原框架基础上扩展了手部骨骼细节，包含掌骨、指骨（近节/中节/远节）的完整结构
- **增强型肌肉驱动**：新增食指相关肌肉群（如FDPI、EDM等）的肌腱路径定义，提升指向动作真实性
- **实时手势映射**：集成MediaPipe手部追踪，可将真实手势实时映射到虚拟模型
- **目标追踪功能**：支持配置目标坐标，实现食指自动指向指定位置
- **兼容性设计**：保持与[User-in-the-Box](https://github.com/User-in-the-Box/user-in-the-box)原项目的模型格式与仿真逻辑兼容

## 文件说明
| 文件名 | 描述 |
|--------|------|
| **config.yaml**（仿真参数配置文件） | 包含仿真步长（dt）、渲染模式（render_mode）、窗口分辨率（resolution）及目标追踪位置（target_pos）等关键参数，可直接修改以调整仿真行为 |
| **simulation.xml**（MuJoCo 模型定义文件） | 包含完整的上肢骨骼结构（锁骨、肱骨、尺骨、手部掌骨及指骨等）、肌肉与肌腱参数（如长度范围、增益参数）、标记点（用于肌肉附着与关键位置定位）及关节活动范围等核心物理仿真信息，是整个仿真的几何与物理基础 |
| **evaluator.py**（程序入口脚本） | 通过命令行参数接收配置文件（--config）和模型文件（--model）路径，初始化仿真器并启动仿真循环，按 ESC 键可退出仿真 |
| **simulator.py**（仿真器核心逻辑实现） | 包含 MuJoCo 环境初始化、Viewer（可视化窗口）适配（兼容不同版本 MuJoCo API）、仿真循环控制及手势映射等关键功能，是连接模型与交互逻辑的核心模块 |
| **assets/**（模型资源文件夹） | 存放模型所需的网格文件（.stl）和纹理文件，用于定义骨骼、手部等组件的几何形状与外观，是 simulation.xml 中引用的可视化资源基础 |

## 模型结构

仿真模型定义在simulation.xml中，包含以下核心组件：
- **骨骼结构**：详细的上肢骨骼模型，包括锁骨、尺骨、手部掌骨及指骨（如 index0、index1 对应食指的不同节段），并通过网格文件（.stl）定义几何形状。
- **标记点（sites）**：用于定位肌肉附着点和关键位置，如手指关节（FDPI-P3 至 FDPI-P9 对应食指相关点位）、肌肉路径点（BIClong-P1 至 BIClong-P11 对应肱二头肌长头）。
- **肌腱与肌肉**：定义了主要肌肉的路径（如三角肌 TRI、肱二头肌 BIC、肱三头肌 TRI 等）及物理参数（长度范围、增益参数等），实现逼真的肌肉驱动效果。
- **关节**：定义了各关节的活动范围和轴方向，如肘关节弯曲（elbow_flexion）的活动角度限制。

## 系统要求

- ubuntu 22.04(humble)
- Python 3.8+
- MuJoCo 2.3.0+
- OpenCV
- MediaPipe
- NumPy
- PyYAML

## 安装

1. **克隆本仓库**
```python
# 克隆主项目
git clone https://github.com/yourusername/mobl-arms-index-pointing.git
cd mobl-arms-index-pointing

# 克隆 User-in-the-Box 核心依赖（如需要）
git clone https://github.com/User-in-the-Box/user-in-the-box.git
```

2. **安装依赖**

- 创建conda环境(本地虚拟环境名为*mjoco*)，根据需要的包(mujoco, mediapipe, numpy, pyyaml等)下载依赖
- **推荐**：或者在[User-in-the-Box](https://github.com/User-in-the-Box/user-in-the-box)原项目的[安装/设置](https://github.com/User-in-the-Box/user-in-the-box?tab=readme-ov-file#installation--setup)配置
```python
#需要先激活虚拟环境
pip install -e .
```

## 配置说明

配置文件config.yaml用于设置仿真参数，与原项目格式保持一致,主要包含以下选项：
```python
dt: 0.05                # 仿真步长
render_mode: "human"    # 显示可视化窗口（可选值："human" 显示窗口，"offscreen" 无窗口运行）
resolution: [1280, 960] # 窗口分辨率 [宽度, 高度]
target_pos: [0.4, 0, 0.7] # 追踪目标位置（可选，用于指定食指指向的目标坐标）
``` 
可根据需求修改上述参数，例如调整仿真精度（dt 越小精度越高但速度越慢）或窗口大小。

## 使用/运行

运行仿真程序：
```python
python evaluator.py --config config.yaml --model simulation.xml
```
启动后，程序将初始化仿真环境并进入循环，实时渲染手臂运动。按 ESC 键可退出仿真。

## 扩展与定制

- 模型扩展：可通过修改 simulation.xml 调整骨骼结构、肌肉参数或添加新的标记点。
- 手势追踪：如需自定义追踪逻辑，可修改仿真器核心代码（simulator.py）中的手势映射部分。
- 目标设置：通过修改 config.yaml 中的 target_pos 可让食指指向不同的三维坐标。

## 项目来源

[User-in-the-Box](https://github.com/User-in-the-Box/user-in-the-box)