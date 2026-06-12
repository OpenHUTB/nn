# 基于 CARLA 的 YOLOv8 多类目标定位数据集生成器

![](example_labels.png)

本仓库提供了一系列实用工具，用于在 [CARLA](https://carla.org//) 模拟器（v0.9.15）中生成带有标注的数据集，以训练 YOLO 目标检测模型。模型训练由 [Ultralytics](https://docs.ultralytics.com) API 提供支持。

## 目录

- [安装指南](#安装指南)  
- [快速开始](#快速开始)  
- [数据集自定义](#数据集自定义)  
- [🚀 开发路线图与进阶优化](#开发路线图与进阶优化) *(新增)*
- [鸣谢](#鸣谢)  

## 安装指南

本代码已在以下环境测试通过：
- **操作系统**: Ubuntu 18.04 / 22.04
- **CARLA 版本**: 0.9.15
- **Python 版本**: 3.10.x

### 安装 CARLA
请参考 [官方说明](https://carla.readthedocs.io/en/0.9.15/start_quickstart/) 安装 CARLA 模拟环境。如需快速部署，也可以尝试 [通过 Docker 运行 CARLA](https://carla.readthedocs.io/en/0.9.15/build_docker/)。

### 环境配置
```bash
git clone [https://github.com/RyangDiaz/carla-yolo-dataset-generator.git](https://github.com/RyangDiaz/carla-yolo-dataset-generator.git)
cd carla-yolo-dataset-generator
conda env create -f environment.yml
conda activate carla

如果不使用 conda，请执行：
pip install -r requirements.txt
使用说明
首先，修改 utils/server_utils.py 中的 LAUNCH_STRING 变量，填入正确的启动路径，以便脚本能自动为你启动 Carla 服务端。

快速开始
如需自动跨 5 张地图（Town01 到 Town05）生成包含 1250 张图像的数据集，并自动开始训练 YOLOv8 模型，请运行：

Bash
bash collect_data_and_train.sh
训练完成后，你可以在指定地图上运行实时推理，并查看或保存视频结果：

Bash
python yolo_realtime_inference.py --model 路径/到/你的模型.pt --num_steps 步数 --map Town05 --show --save_video
该检测器目前支持四类目标：vehicle（车辆）、pedestrian（行人）、traffic_light（红绿灯）和 traffic_sign（交通标志）。

数据集自定义
采集参数配置

你可以通过向 collect_yolo_data.py 传递参数来调整数据集特性：

--map {MAP}: 指定采集地图。

--constant_weather: 保持固定天气。默认情况下，采集器每隔几帧会切换一次天气。

--num_save {N}: 配合 --save 使用，达到保存 N 张目标图像后停止。

--num_detections_save {N}: 每一张保存的图像至少包含 N 个有效标注框。

开发路线图与进阶优化
为了使该数据集生成器更具鲁棒性、达到工业级标准并对开发者更加友好，本项目计划进行一系列进阶优化。这些改进专注于解决 CARLA 底层 C++ API 的边界错误、优化内存安全及标准化工程代码。

非常欢迎社区对以下即将提交的 Pull Request (PR) 方向进行代码审查与贡献：

阶段 1：核心稳定性与崩溃防护（开发中）
修复异步 I/O 导致的静默崩溃: 废弃原生的 image.save_to_disk() 方法，改用带有内存深拷贝的同步 cv2.imwrite。此举旨在消除 Python 垃圾回收机制与 CARLA C++ 异步写盘线程之间的内存竞态（Race Condition），解决无报错闪退问题。

健壮的 Actor 生成逻辑: 在 reset() 生命周期中引入 while-try-except 重试模式。这能有效处理在拥挤地图中生成点冲突导致的 NoneType 报错，确保自动驾驶逻辑（Autopilot）始终能成功分配。

仿真生命周期管理: 重新梳理全局 try-except-finally 结构。确保在脚本因任何原因中断时，都能强制执行资源清理逻辑，避免产生僵尸进程或显存泄漏。

阶段 2：架构重构与性能优化（计划中）
作用域逻辑重构: 修正主执行循环的缩进与作用域错误。确保在环境重置（Reset）后，数据采集、天气切换及边界框计算逻辑能正确恢复运行。

安全存档机制: 重构 checkpoint 读写逻辑，引入原子化写入（Atomic Write）并精确捕获文件异常，防止因意外断电或程序崩溃导致的进度文件损坏。

服务端冲突处理: 增加 RPC 端口（2000/2001）占用检测。自动识别已运行的仿真器实例，避免多开 CarlaUE4.exe 导致的显存溢出（OOM）。

阶段 3：开发者体验 (DX) 与标准化（计划中）
动态采样率配置: 将硬编码的 save_every 等核心变量提取至命令行参数（argparse）。开发者无需修改源码即可灵活调整测试与正式采集的频率。

环境预检机制: 增加启动前的依赖自检模块。在加载沉重的 Unreal 资源前，提前验证 CARLA 客户端版本及核心 Python 库的兼容性。

代码工程化标准: 集成 Black 格式化工具并提供推荐的 .vscode 设置，从工程基建层面杜绝因缩进问题导致的逻辑 Bug。

鸣谢
本项目工具基于以下 CARLA 数据集采集领域的优秀工作构建：

CARLA-2DDBBox

CARLA-Automatic-Dataset-Collector

CARLA 官方边界框教程