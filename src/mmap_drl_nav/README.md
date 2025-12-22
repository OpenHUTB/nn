# CARLA 智能驾驶导航仿真系统

## 项目介绍
本项目基于 CARLA 模拟器与 PyTorch 框架，实现了一套智能驾驶导航仿真系统。核心功能包括：CARLA 环境连接与车辆生成、多传感器数据（图像/激光雷达/IMU）模拟、跨域注意力特征融合、智能决策输出与车辆控制，可直接运行并直观观察车辆在虚拟场景中的行驶状态。

项目解决了新手常见的模块导入、张量维度适配、CARLA 连接配置等问题，形成完整的仿真闭环，可作为智能驾驶算法开发与验证的基础框架。

## 核心功能
- CARLA 环境自动化初始化（连接、车辆生成、资源清理）
- 多源传感器数据模拟与输入适配
- 跨域注意力模块实现多模态特征融合
- 决策模块输出车辆控制指令（油门、转向）
- 持续仿真循环，支持直观观察车辆运动状态

## 环境依赖
### 1. 基础环境
- 操作系统：Windows 10/11（64位）
- Python 版本：3.7.x（推荐 3.7.9）
- 虚拟环境：venv（项目内置，或重新创建）

### 2. 核心依赖库
| 库名称       | 版本要求                  | 安装命令                                                                 |
|--------------|---------------------------|--------------------------------------------------------------------------|
| PyTorch      | 1.10.2+cpu                | pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple |
| CARLA        | 0.9.13（与模拟器版本一致） | pip install carla==0.9.13 -i https://pypi.tuna.tsinghua.edu.cn/simple    |
| NumPy        | 任意兼容版本              | pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple            |
| OpenCV-Python| 任意兼容版本              | pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple    |

### 3. CARLA 模拟器
- 版本：CARLA 0.9.13（Windows 版本）
- 下载地址：[CARLA 0.9.13 官方下载](https://github.com/carla-simulator/carla/releases/tag/0.9.13)
- 存放路径：推荐 D:\carla\WindowsNoEditor（需与后续启动命令适配）

## 项目结构
```
robot_nav_sys/
├─ envs/                # 环境模块（CARLA 交互核心）
│  ├─ __init__.py       # 模块标识文件（空文件即可）
│  └─ carla_environment.py  # CARLA 环境初始化、车辆生成与控制
├─ models/              # 模型模块（特征融合与决策）
│  ├─ __init__.py       # 模块标识文件（空文件即可）
│  ├─ attention_module.py  # 跨域注意力模块（多模态特征融合）
│  └─ decision_module.py   # 决策模块（输出油门、转向控制指令）
├─ venv/                # 虚拟环境（存放依赖库）
├─ run_simulation.py    # 仿真入口（核心运行文件）
└─ README.md            # 项目说明文档（本文档）
```

## 快速运行步骤
⚠️ 关键顺序：必须先启动 CARLA 模拟器，再运行项目代码！

### 步骤 1：启动 CARLA 模拟器（端口 2000）
1. 打开 Windows 命令提示符（CMD），执行以下命令切换到 CARLA 目录：
   ```bash
   # 切换到 D 盘
   D:
   # 进入 CARLA 模拟器目录（根据实际路径调整）
   cd carla\WindowsNoEditor
   ```
2. 执行命令启动模拟器（强制指定端口 2000，确保与代码连接匹配）：
   ```bash
   CarlaUE4.exe -carla-port=2000
   ```
3. 等待 5-10 秒，直到弹出 CARLA 3D 窗口（显示 Town01 城镇场景，无「未响应」标识）。

### 步骤 2：启动项目仿真
1. 新开一个 CMD 窗口（不要关闭模拟器窗口），执行以下命令切换到项目目录：
   ```bash
   # 切换到 D 盘
   D:
   # 进入项目根目录（根据实际路径调整）
   cd carla_work\robot_nav_sys
   ```
2. 激活虚拟环境：
   ```bash
   venv\Scripts\activate.bat
   ```
   ✅ 激活成功标志：窗口开头出现 `(venv)` 前缀
3. 安装依赖库（若未安装）：
   ```bash
   # 执行「环境依赖」章节中的所有 pip 安装命令
   ```
4. 运行仿真代码：
   ```bash
   python run_simulation.py
   ```

### 步骤 3：观察仿真效果
1. 切换到 CARLA 模拟器窗口，按住鼠标右键拖动调整视角，滚轮缩放，找到白色 Toyota 车辆（生成位置默认：x=100, y=100, z=2）。
2. 观察到车辆以 0.5 油门持续前进，小幅转向，完成 50 步仿真循环。
3. 仿真结束后，终端会输出「✅ 仿真结束，CARLA 环境已清理」，车辆自动销毁，资源释放。

## 关键功能说明
### 1. 环境模块（envs/carla_environment.py）
- 初始化：连接 CARLA 模拟器（超时时间 20 秒），获取虚拟世界。
- 车辆生成：使用 Toyota 车型蓝图，支持 3 次重试机制，避免生成失败；无预设生成点时自动创建手动生成点。
- 车辆控制：接收油门、转向指令，裁剪到合法范围（油门 0~1，转向 -1~1）。
- 资源清理：仿真结束后自动销毁车辆，关闭模拟器连接。

### 2. 模型模块
- attention_module.py：跨域注意力模块，统一多模态张量维度（2 维/4 维 → 2 维），通过多块注意力层融合特征。
- decision_module.py：决策模块，将融合后的 1024 维特征映射到 256 维，最终输出 2 维控制指令（油门、转向）和 1 维价值估计。

### 3. 仿真入口（run_simulation.py）
- 集成环境与模型，构建完整仿真闭环。
- 支持 50 步持续仿真，固定油门 0.5 确保车辆运动，小幅转向避免跑偏。
- 实时打印仿真状态（每 10 步输出一次油门、转向、价值），方便观察运行情况。

## 常见问题解决
### 1. 连接 CARLA 失败：time-out of 20000ms
- 原因：CARLA 模拟器未启动、端口不匹配、端口被占用。
- 解决：
  1. 确认 CarlaUE4.exe 已启动，且窗口显示城镇场景。
  2. 重新执行「步骤 1」，确保用命令 `CarlaUE4.exe -carla-port=2000` 强制指定端口。
  3. 检查端口 2000 是否被占用，执行命令 `netstat -ano | findstr :2000`，结束占用进程（`taskkill /F /PID 进程号`）。

### 2. 虚拟环境激活失败：系统找不到指定的路径
- 原因：虚拟环境文件夹名错误（非 venv）、路径拼写错误。
- 解决：
  1. 执行 `dir` 命令查看项目目录下的虚拟环境文件夹名（可能为 env、environment 等）。
  2. 根据实际文件夹名修改激活命令，例如：`env\Scripts\activate.bat`。
  3. 若无虚拟环境，执行 `python -m venv venv` 重新创建。

### 3. 车辆生成失败 / 找不到车辆
- 原因：车辆蓝图不兼容、生成点坐标不合理。
- 解决：
  1. 修改 envs/carla_environment.py 中的车辆蓝图，将 `toyota` 替换为 `model3` 或 `carlacola`。
  2. 调整手动生成点坐标，例如：`spawn_points = [carla.Transform(carla.Location(x=200, y=200, z=2))]`。
  3. 在 CARLA 窗口中按住鼠标右键拖动视角，缩放查找车辆。

### 4. Lazy modules 警告（非报错）
- 警告信息：`UserWarning: Lazy modules are a new feature under heavy development...`
- 解决：可忽略，不影响运行；若想关闭，在 run_simulation.py 开头添加：
  ```python
  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)
  ```

## 扩展方向
- 替换传感器数据：将 run_simulation.py 中的示例数据（image、lidar_data 等）替换为 CARLA 真实传感器采集的数据（需添加传感器配置代码）。
- 优化决策算法：在 decision_module.py 中集成 DQN、PPO 等强化学习算法，实现自主导航训练。
- 丰富场景配置：添加障碍物、交通灯、不同城镇地图（修改 CARLA 环境加载逻辑）。
- 增加可视化：添加传感器数据可视化、车辆轨迹绘制功能。

## 注意事项
- Python 版本必须为 3.7.x，否则可能导致 PyTorch 或 CARLA 库安装失败。
- CARLA 模拟器与 carla 库版本必须一致（均为 0.9.13），否则会出现兼容性错误。
- 仿真过程中不要关闭 CARLA 窗口或 CMD 窗口，否则会导致资源泄漏（车辆无法自动销毁）。
- 若 CARLA 窗口卡顿，可关闭其他占用显卡的程序，或降低模拟器画质（在 CARLA 窗口设置中调整）。

## 致谢
- CARLA 官方文档：[CARLA 官方文档](https://carla.readthedocs.io/)
- PyTorch 官方文档：[PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
