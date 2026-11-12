## 做了什么

新增 `mujoco_car_nn` 模块：MuJoCo 无人车物理仿真 + 端到端卷积网络驾驶（BC / DAgger）+
栅格建图与扫描匹配 SLAM，并提供 ROS 1 Noetic 与 ROS 2 Humble 两套包与 launch 文件。

## 对应课程任务

| 任务 | 实现 |
|---|---|
| （1）物理仿真 + 键盘控制 | MuJoCo 平面刚体车辆（自行车模型 + 真实碰撞）、`main.py teleop`、`car_teleop.launch` |
| （2）传感器感知 + 运动控制 | 64×64 车载相机、64 束激光雷达、里程计；感知→控制闭环 |
| （3）建图与导航、SLAM | log-odds 占据栅格 + 相关法扫描匹配，`/map` 1 Hz，ATE 2.37 m |
| （4）端到端模型 | `PolicyCNN`：图像 + 车速 → 转向/油门，CPU 1.5 ms/帧 |

## 怎么验证

```bash
pip install -r requirements.txt
python main.py collect --laps 5 --scenes day,dusk
python main.py train --model cnn --epochs 24 --dagger
python main.py eval --model runs/cnn_dagger.pt --laps 2 --record
python -m pytest tests -q          # 6 passed
```

ROS 2：

```bash
colcon build --symlink-install
ros2 launch car_nn_sim car_nn.launch.py
ros2 topic list -t                 # image_raw / scan / odom / cmd_vel
```

## 结果

| 策略 | 平均速度 (m/s) | 单圈 (s) | 横向 RMSE (m) | 碰撞 |
|---|---|---|---|---|
| 专家 pure-pursuit | 10.41 | 22.1 | 0.49 | 0 |
| CNN（BC） | 8.03 | 28.6 | 0.45 | 0 |
| CNN（BC + DAgger） | **10.29** | **22.4** | 0.47 | 0 |
| MLP（BC） | 6.75 | 34.0 | 0.38 | 0 |

day / dusk / fog 三种光照共 12 组实验全部 0 碰撞完成，fog 为零样本场景。

## 动图

- `docs/gifs/nn_drive.gif`（2.5 MB）：端到端驾驶，第三人称 + 第一视角 + HUD
- `docs/gifs/teleop.gif`（3.1 MB）：键盘控制
- `docs/gifs/slam_mapping.gif`（0.24 MB）：建图过程

## 影响面

仅新增文件，不修改既有模块；统一入口 `main.py`，Windows / Ubuntu 均可运行。
