# 自动驾驶车道与路径检测
=========================================

## 项目简介

本项目实现了基于深度学习的自动驾驶车道检测与路径规划功能，支持以下两种仿真平台：

| 平台 | 特点 | 适用场景 |
|------|------|----------|
| **GTAV + OpenPilot** | 基于游戏引擎，画面逼真 | 快速原型验证 |
| **CARLA 模拟器** | 专业自动驾驶仿真平台 | 学术研究、算法评估 |

---

## 在 GTAV 上运行 OpenPilot

本项目是对 [littlemountainman/modeld](https://github.com/littlemountainman/modeld) 项目的一个分支。

我们利用了他的工作，并将 DeepGTAV 和 VPilot 结合，从而能够将 comma.ai 的开源软件应用于 GTAV，并创建出由 openpilot 算法管理的自动驾驶车辆。

### GTAV 安装步骤

**环境要求：** Python 3.7 或更高版本

1. **安装所需依赖包**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **下载 VPilot with DeepGTAV**
   - 下载地址：[VPilot](https://github.com/aitorzip/VPilot)

3. **下载 ScriptHookV**
   - 下载地址：[ScriptHookV](https://www.dev-c.com/gtav/scripthookv/)

4. **下载 DeepGTAV**
   - 下载地址：[DeepGTAV](https://github.com/aitorzip/DeepGTAV)

5. **安装到 GTAV 目录**
   - 将 `ScriptHookV.dll`、`dinput8.dll`、`NativeTrainer.asi` 复制到游戏主文件夹（即 `GTA5.exe` 所在目录）
   - 将 `DeepGTAV/bin/Release/` 文件夹下的所有内容复制到 GTAV 游戏安装目录

6. **启动程序**
   ```bash
   # 确保 GTAV 已在运行
   python3 main.py
   ```

---

## 在 CARLA 模拟器上运行自动驾驶

**如果你没有 GTAV，或者希望在更真实的自动驾驶仿真环境中进行开发，可以使用 CARLA 模拟器。**

CARLA 是一个开源的自动驾驶仿真器，提供更专业的交通场景、传感器模拟和车辆动力学。

### CARLA 环境要求

| 组件 | 版本要求 |
|------|----------|
| CARLA 模拟器 | 0.9.15 |
| Python | 3.7 - 3.9 |
| 操作系统 | Windows 10/11 |

### CARLA 安装步骤

#### 1. 下载 CARLA 模拟器

从 [CARLA 官方 GitHub](https://github.com/carla-simulator/carla/releases/tag/0.9.15) 下载 `CARLA_0.9.15.zip` 并解压到指定目录（例如 `H:\carla0.9.15\`）。

#### 2. 安装 Python 依赖

```bash
pip3 install -r requirements.txt
```

#### 3. 安装 CARLA Python API

```bash
# 进入 CARLA Python API 目录
cd H:\carla0.9.15\WindowsNoEditor\PythonAPI\carla\dist

# 安装 carla 包（根据你的 Python 版本选择）
pip install carla-0.9.15-cp38-cp38-win_amd64.whl   # Python 3.8 示例
```

---

## 🚀 一键启动（推荐）

本项目提供了 Windows 批处理脚本 `start_carla.bat`，可以**一键启动 CARLA 模拟器和自动驾驶程序**，无需手动执行多条命令。

### 使用方法

1. **确保目录结构正确**
   ```
   H:\
   ├── carla0.9.15\
   │   └── WindowsNoEditor\
   │       └── CarlaUE4.exe
   └── openhutb\
       └── openhutb\
           └── src\
               └── driveSim-enhanced\
                   └── automatic_control.py
                   └── drive.py
                   └── rl_agent.py
                   └── map_swithcer.py
                   └── main.py
                   └── README.py
   ```

2. **双击运行启动脚本**
   ```
   start_carla.bat
   ```

3. **脚本会自动完成以下操作：**
   - ✅ 检测 CarlaUE4.exe 是否存在
   - ✅ 启动 CARLA 模拟器（带 `-quality-level=Low` 优化参数）
   - ✅ 等待 15 秒让服务器完全启动
   - ✅ 自动检测 Python 脚本路径
   - ✅ 运行自动驾驶程序

### 手动启动方式

如果不使用一键脚本，也可以手动执行：

```bash
# 进入 CARLA 安装目录
cd H:\carla0.9.15\WindowsNoEditor

# 启动模拟器
CarlaUE4.exe

# 新开一个终端，运行自动驾驶程序
python code/automatic_control.py
```

---

## CARLA 功能特性

本项目的 CARLA 版本支持以下功能：

| 功能 | 说明 |
|------|------|
| 🚦 交通信号灯检测 | 实时识别红、黄、绿灯状态 |
| 🚗 车道线检测 | 基于深度学习的车道线识别 |
| 📍 路径规划 | 自动规划行驶路线 |
| 📊 车辆信息显示 | 速度、档位、转向角等实时信息 |
| 🎮 多视角切换 | 支持第一人称、第三人称、俯视等视角 |

---

## 常见问题

### Q1: 连接 CARLA 时提示版本不匹配

**A:** 确保 Python API 版本与 CARLA 模拟器版本一致：

```bash
# 检查 Python API 版本
python -c "import carla; print(carla.__version__)"
```

如果版本不匹配，重新安装对应版本的 `carla` 包。

### Q2: 启动脚本提示找不到 Python

**A:** 请确保 Python 已添加到系统环境变量：

```bash
# 测试 Python 是否可用
python --version
```

如果提示找不到命令，请重新安装 Python 并勾选 "Add Python to PATH"。

### Q3: 启动脚本中文乱码或闪退

**A:** 
- Windows 11 用户请确保脚本保存为 **UTF-8 with BOM** 编码
- 或者使用 ANSI 编码保存

### Q4: CARLA 模拟器启动后画面卡顿

**A:** 脚本默认使用 `-quality-level=Low` 参数降低画质，如果需要更流畅的体验，可以：

1. 关闭其他占用 GPU 的程序
2. 在 CARLA 设置中降低分辨率
3. 使用 DirectX 11 模式运行（默认）

---

## 项目结构

```
openhutb/
├── src\
    └── driveSim-enhanced\
        └── automatic_control.py
        └── drive.py
        └── rl_agent.py
        └── map_swithcer.py
        └── main.py
        └── README.py
```

---

## 致谢

- [littlemountainman/modeld](https://github.com/littlemountainman/modeld)
- [aitorzip/DeepGTAV](https://github.com/aitorzip/DeepGTAV)
- [aitorzip/VPilot](https://github.com/aitorzip/VPilot)
- [CARLA Simulator](https://carla.org/)
- [comma.ai/openpilot](https://github.com/commaai/openpilot)

---

## 许可证

本项目遵循原项目的开源许可证。

```

这个 README 主要完善了：
1. ✅ 添加了一键启动 `start_carla.bat` 的使用说明
2. ✅ 完善了 CARLA 安装步骤（包括 API 安装）
3. ✅ 添加了常见问题解答（Q&A）
4. ✅ 添加了项目结构说明
5. ✅ 优化了排版，增加了表格对比
6. ✅ 添加了目录结构示意
