# 快速开始

## 1. 环境要求

| 平台 | 版本 | 说明 |
|---|---|---|
| Windows 10/11 | Python 3.10 – 3.12 | 原生运行，无需 ROS；开发与训练均在此完成 |
| Ubuntu 20.04 | Python 3.8 + ROS Noetic | 课程虚拟机，运行 `ros1/` 的 launch |
| Ubuntu 22.04 / Docker | ROS 2 Humble | 运行 `ros2/` 的 launch |

依赖仅 5 个：`mujoco`（物理与渲染）、`torch`（CPU 版即可）、`numpy`、`pillow`、`matplotlib`。

```bash
pip install -r requirements.txt
# 或者在无 GPU 的服务器上安装 CPU 版 torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

!!! note "Linux 无显示器服务器"
    MuJoCo 默认使用 GLFW 创建离屏上下文。无显示器的 Ubuntu 服务器需要
    `export MUJOCO_GL=egl`（或 `osmesa`），并安装 `libegl1`；Windows 原生无需设置。

!!! tip "`python` 命令不可用？"
    Windows 上如果 `python` 打开的是应用商店，说明没有安装官方 Python。请安装
    Python 3.11 / 3.12 并勾选 **Add python.exe to PATH**；或者直接用 `py -3 main.py`
    （`main.bat` 会优先调用 `py` 启动器）。

!!! tip "Docker 运行（免安装 ROS）"
    仓库自带 `Dockerfile`（`ros:humble-ros-base` + CPU 版 MuJoCo）：

    ```bash
    docker build -t car_nn .
    docker run --rm car_nn python3 main.py eval --model runs/cnn_dagger.pt --laps 1
    docker run --rm -it car_nn        # 进入容器后可直接 ros2 launch car_nn_sim ...
    ```

    镜像内设置 `MUJOCO_GL=egl`，无需 X server，适合服务器/CI 环境。

## 2. 三条命令跑通全流程

```bash
# 1) 用 pure-pursuit 专家采集 5 圈、两种光照的模仿学习数据（约 10 s）
python main.py collect --laps 5 --scenes day,dusk

# 2) 行为克隆 + DAgger 训练端到端 CNN（约 2 min，CPU）
python main.py train --model cnn --epochs 24 --dagger

# 3) 闭环评测：跑 2 圈，输出成功率、横向误差、单圈时间
python main.py eval --model runs/cnn_dagger.pt --scene day --laps 2 --record
```

第三条命令会在 `docs/gifs/` 下生成一段带 HUD 的评测动图，同时打印：

```text
cnn_dagger.pt  day   laps=2.00 success=Y coll= 0 lat_rmse=0.47 lat_max=0.74 v_mean=10.29 lap= 22.4s infer= 1.5ms
```

## 3. 全部命令

| 命令 | 作用 | 主要参数 |
|---|---|---|
| `python main.py collect` | 专家驾驶采集 (image, lidar, state, action) | `--laps --scenes --seed --out` |
| `python main.py train` | 行为克隆；加 `--dagger` 启用 DAgger | `--model cnn/mlp --epochs --dagger-iters` |
| `python main.py eval` | 闭环评测，可录制 GIF | `--model --scene --laps --record --json` |
| `python main.py benchmark` | 结构/算法对比实验，生成表格与图表 | `--epochs --laps --scenes` |
| `python main.py demo` | 录制带 HUD 与第一视角小窗的演示 GIF | `--model --seconds --fps` |
| `python main.py slam` | 建图 + 扫描匹配，输出 ATE 与地图 GIF | `--laps --every --json` |
| `python main.py teleop` | 键盘遥控（MuJoCo 原生窗口） | `--scene` |

## 4. 脚本入口约定

作业要求"每个模块必须支持 launch 启动，入口以 main 开头"。本仓库提供：

| 文件 | 平台 | 用法 |
|---|---|---|
| `main.py` | 全平台 | `python main.py <cmd>` |
| `main.bat` | Windows | `main.bat <cmd>` |
| `main.sh` | Ubuntu | `./main.sh <cmd>` |

ROS 侧则通过 `roslaunch car_nn_sim car_nn.launch` / `ros2 launch car_nn_sim car_nn.launch.py`
启动，详见 [ROS 部署](ros.md)。

## 5. 目录约定

```text
data/     采集得到的 npz 数据集（自动生成，约 4.6 MB / 4000 帧）
runs/     模型权重与训练历史（自动生成）
docs/     mkdocs 文档、figures/ 实验结果图、gifs/ 演示动图
site/     mkdocs build 输出（GitHub Pages 发布目录）
```

数据与权重体积较大，已加入 `.gitignore`；任何一台机器都可以用上面三条命令重新生成，
保证仓库里只保存文本与少量图片。

## 6. 常见问题

??? question "`mujoco` 报 `GLFW error` / 窗口无法创建"
    Windows 上确认没有在无桌面会话（如 ssh）中运行；Ubuntu 服务器设置
    `MUJOCO_GL=egl`。本项目所有评测与录制都走离屏渲染，不需要打开窗口。

??? question "训练很慢"
    CPU 上 4000 帧、20 epoch 约 40 s。若安装了 CUDA 版 torch，把 `--device cuda`
    传入 `TrainConfig` 即可，代码默认 `cpu` 以保证任何机器可复现。

??? question "换了赛道会不会失效"
    不会。`track.py` 中赛道由参数方程生成，改 `R0/AMP2/AMP3` 即可换形状；
    相机、雷达、专家、网络输入均自动跟随，重新 `collect` 一次即可。
