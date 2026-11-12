# MuJoCo 无人车神经网络驾驶（开源技术与实践）

一套可在 **Windows / Ubuntu 20.04** 上直接跑通的无人车仿真课程作业：MuJoCo 物理仿真、
64 线激光雷达、端到端卷积网络驾驶（行为克隆 + DAgger）、占据栅格建图与扫描匹配 SLAM，
并提供 ROS 1（Noetic）与 ROS 2（Humble）两套 launch。

![演示](docs/gifs/nn_drive.gif)

## 快速开始

```bash
pip install -r requirements.txt

python main.py collect --laps 5 --scenes day,dusk     # 专家采集数据（约 10 s）
python main.py train   --model cnn --epochs 24 --dagger  # 训练（约 2 min，CPU）
python main.py eval    --model runs/cnn_dagger.pt --laps 2 --record
python main.py slam    --laps 1                       # 建图 + 定位
```

Windows 可双击 `main.bat`，Ubuntu 用 `./main.sh`。ROS：

```bash
roslaunch car_nn_sim car_nn.launch policy:=/abs/path/runs/cnn_dagger.pt
ros2 launch car_nn_sim car_nn.launch.py policy:=/abs/path/runs/cnn_dagger.pt
```

## 结果

| 策略 | 平均速度 (m/s) | 单圈 (s) | 横向 RMSE (m) | 碰撞 |
|---|---|---|---|---|
| 专家 pure-pursuit | 10.41 | 22.1 | 0.49 | 0 |
| 端到端 CNN（BC） | 8.03 | 28.6 | 0.45 | 0 |
| 端到端 CNN（BC+DAgger） | **10.29** | **22.4** | 0.47 | 0 |
| 端到端 MLP（BC） | 6.75 | 34.0 | 0.38 | 0 |

三种光照（day/dusk/fog）下均 0 碰撞、100% 完成；fog 为零样本场景。

## 文档

完整文档（仿真原理、公式、代码解析、实验、ROS 部署）位于 `docs/`，使用 mkdocs 构建：

```bash
pip install mkdocs mkdocs-material
mkdocs serve     # http://127.0.0.1:8000
mkdocs build     # 生成 site/，可发布 GitHub Pages
```

## 目录

```text
main.py                  # 入口
src/mujoco_car_nn/       # 仿真、数据、模型、训练、评测、SLAM
ros1/ ros2/              # ROS 包与 launch
docs/                    # mkdocs 文档与图片/GIF
scripts/                 # 生成图表、消融、演示的辅助脚本
tests/                   # 冒烟测试
```

## 许可证

MIT
