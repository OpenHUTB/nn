# 🚗 DriveSim-Enhanced：自动驾驶车道检测与路径规划仿真平台

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/) [![CARLA 0.9.15](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)

## 📑 目录
1. [产品概述](#1-产品概述)
2. [核心特性](#2-核心特性)
3. [仿真平台支持](#3-仿真平台支持)
4. [环境准备与快速部署](#4-环境准备与快速部署)
5. [系统模块架构](#5-系统模块架构)
6. [常见问题 (FAQ)](#6-常见问题-faq)
7. [演进路线图](#7-演进路线图)
8. [致谢与许可](#8-致谢与许可)

---

## 1. 产品概述

**DriveSim-Enhanced** 是一套为自动驾驶算法研究与快速原型验证而设计的综合仿真平台方案。本项目集成了基于深度学习的车道线检测、交通信号灯识别与智能路径规划模块，旨在帮助研究人员和开发者在虚拟环境中安全、低成本地验证自动驾驶算法。

本项目在 [modeld](https://github.com/littlemountainman/modeld) 的基础上进行了深度演进，打通了 comma.ai 的 openpilot 算法与多种仿真引擎的连接。

**🎯 适用人群：** 自动驾驶算法工程师、具身智能研究者、高校学术团队及极客开发者。

---

## 2. 核心特性

*   **🧠 深度学习视觉感知：** 实时处理摄像头输入，实现高精度的车道线提取与跟踪。
*   **🚦 智能交通交互：** 支持对红、黄、绿交通信号灯状态的精准检测与响应。
*   **🗺️ 全局与局部路径规划：** 动态计算最优行驶路线，支持复杂路况下的自主决策。
*   **📊 实时数据遥测屏：** 仪表盘式可视化界面，毫秒级同步显示车速、档位、转向角等关键车辆动力学参数。
*   **🎮 多模态视角交互：** 提供第一人称（沉浸式驾驶）、第三人称及俯视全局等多视角无缝切换体验。
*   **⚡ 极简自动化部署：** 提供开箱即用的自动化部署脚本，消除繁琐的环境配置烦恼。

---

## 3. 仿真平台支持

为满足不同维度的研发需求，本项目提供两种仿真引擎支持方案：

| 平台特性 | 🎮 GTAV + OpenPilot 联合仿真 | 🏢 CARLA 专业仿真器 (推荐) |
| :--- | :--- | :--- |
| **定位与场景** | 侧重视觉逼真度，适合端到端视觉算法的快速原型验证 | 侧重物理规则与传感真实性，适合严谨的学术研究与算法评估 |
| **核心优势** | 光影渲染极其逼真，天气系统丰富，NPC 行为具有高随机性 | 提供专业的 Python API 控制，支持激光雷达等多种传感器模拟，符合行业标准 |
| **系统要求** | 需要安装 GTA5 游戏本体，资源消耗中等 | 需要下载并部署 CARLA 引擎，对显卡性能要求较高 |

---

## 4. 环境准备与快速部署

### 方案 A：在 CARLA 模拟器上运行（专业研究推荐）

**环境依赖：** Windows 10/11，Python 3.7 - 3.9，[CARLA 0.9.15 客户端](https://github.com/carla-simulator/carla/releases/tag/0.9.15)。

#### 1. 环境初始化
请确保将 CARLA 解压至本地磁盘（例如 `H:\carla0.9.15\`），并安装 Python 依赖库与 CARLA API 包：

```bash
# 安装基础依赖
pip install -r requirements.txt

# 进入 CARLA Python API 目录并安装 whl 文件（以 Python 3.8 为例）
cd H:\carla0.9.15\WindowsNoEditor\PythonAPI\carla\dist
pip install carla-0.9.15-cp38-cp38-win_amd64.whl
```

#### 2. 🚀 极简启动（One-Click Start）
为了提供最佳的用户体验，我们封装了 `start_carla.bat` 批处理脚本。您只需**双击运行**该脚本，系统将自动化完成以下编排：
*   ✅ 环境预检（定位 `CarlaUE4.exe`）
*   ✅ 启动仿真引擎（默认注入 `-quality-level=Low` 以保障帧率）
*   ✅ 服务就绪检测（等待 15 秒同步时间）
*   ✅ 自动挂载并拉起核心驾驶算法模块

*如果您希望手动启动，请依次执行 `CarlaUE4.exe` 并新开终端运行 `python src/driveSim-enhanced/automatic_control.py`。*

### 方案 B：在 GTAV 上运行 OpenPilot（视觉感知推荐）

1. 安装依赖：`pip install -r requirements.txt`
2. 资源下载：获取 [VPilot](https://github.com/aitorzip/VPilot)、[ScriptHookV](https://www.dev-c.com/gtav/scripthookv/) 与 [DeepGTAV](https://github.com/aitorzip/DeepGTAV)。
3. 引擎注入：将 `ScriptHookV.dll`、`dinput8.dll`、`NativeTrainer.asi` 及 `DeepGTAV/bin/Release/` 内的文件放置于 `GTA5.exe` 同级目录。
4. 启动服务：在游戏运行状态下，执行 `python main.py`。

---

## 5. 系统模块架构

项目采用模块化设计，高内聚低耦合，核心代码结构如下：

```text
openhutb/
└── src/
    └── driveSim-enhanced/
        ├── main.py                # 系统主入口，负责模块初始化与生命周期管理
        ├── automatic_control.py   # 核心业务逻辑：自动驾驶策略控制中心
        ├── drive.py               # 车辆底盘与动力学控制接口适配
        ├── rl_agent.py            # 强化学习/决策代理模块（强化学习训练入口）
        └── map_swithcer.py        # 仿真环境地图管理与动态切换模块
```

---

## 6. 常见问题 (FAQ)

**Q: 连接 CARLA 时，终端抛出 `carla` 模块导入错误或版本不匹配？**
> **A:** 这通常是由于环境 Python API 与仿真器服务端版本脱节导致。请通过命令 `python -c "import carla; print(carla.__version__)"` 验证版本。若不为 `0.9.15`，请返回部署步骤重新安装对应的 `.whl` 文件。

**Q: 运行 `start_carla.bat` 出现中文乱码或闪退现象？**
> **A:** 请使用文本编辑器（如 VS Code）打开 `.bat` 文件，确保其编码格式保存为 **UTF-8 with BOM** 或系统默认的 **ANSI** 编码。

**Q: CARLA 模拟器运行中画面帧率极低，导致算法控制延迟？**
> **A:** 仿真引擎对 GPU 算力要求较高。系统默认已加入 `-quality-level=Low` 进行降级优化。若仍卡顿，建议：1) 在 CARLA 内部设置降低渲染分辨率；2) 确保显卡驱动已更新且后台无高负载图形应用。

---

## 7. 演进路线图
- [x] 完成 GTAV 环境与 OpenPilot 的基础对接
- [x] 引入 CARLA 仿真引擎支持与 API 封装
- [x] 开发一键启动与部署脚本
- [ ] **Next:** 引入基于强化学习（RL）的跟车模型
- [ ] **Next:** 支持自定义交通流与极端天气测试场景生成
- [ ] **Next:** 完善数据采集脚本，支持自动生成训练数据集

---

## 8. 致谢与许可

本产品的诞生离不开开源社区的伟大贡献，特别鸣谢以下项目及团队：
*   [littlemountainman/modeld](https://github.com/littlemountainman/modeld) - 核心框架参考
*   [aitorzip/DeepGTAV](https://github.com/aitorzip/DeepGTAV) & [VPilot](https://github.com/aitorzip/VPilot) - GTAV 环境插件
*   [CARLA Simulator](https://carla.org/) - 卓越的开源自动驾驶仿真器
*   [comma.ai/openpilot](https://github.com/commaai/openpilot) - 开源智驾先驱

**许可证：** 本项目代码遵循原项目的开源许可证协议。请在商业用途前仔细核对相关依赖包的授权条款。