# airsim control

## 模块简介

本目录包含若干 AirSim 无人机控制相关脚本与实验工程，主要用于数据采集、基础控制与（迷宫导航等）任务探索。

## 目录结构

* AirSim_Maze_Navigation/：AirSim 迷宫导航实验（目录内已包含 README）
* Gazebo_Maze_Navigation _with_ROS/：Gazebo + ROS 迷宫导航实验（目录内已包含 README）
* main.py / drone_client.py / keyboard_controller.py / train.py / run_inference.py：历史脚本与入口集合
* src/airsim_control/：同名脚本的另一份目录结构

## 依赖说明

该模块通常需要：

* AirSim 仿真环境（以及其 Python API）
* Python 依赖：airsim、msgpack-rpc-python（以及脚本中引用的其它第三方库）

## 运行提示

* 如果你要运行迷宫导航相关示例，优先参考子目录的 README（AirSim_Maze_Navigation/ 与 Gazebo_Maze_Navigation _with_ROS/）。
* 当前目录下的 main.py 会导入 `agents.*` 与 `client.*` 命名空间，但仓库内未提供对应包结构；如需运行该入口，需要先补齐/调整导入与目录结构。
