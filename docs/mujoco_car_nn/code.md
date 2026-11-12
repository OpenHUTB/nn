# 代码解析

本章按"文件 → 关键函数 → 数据流"的顺序解析实现，便于 code review 与二次开发。

## 1. 目录与职责

| 文件 | 行数 | 职责 |
|---|---|---|
| `main.py` / `main.bat` / `main.sh` | 20 | 统一入口，把仓库 `src/` 加入 `sys.path` 后转交 CLI |
| `src/mujoco_car_nn/cli.py` | ~480 | 7 个子命令：collect / train / eval / benchmark / demo / slam / teleop |
| `src/mujoco_car_nn/sim.py` | ~430 | 场景 XML 生成、车辆动力学、相机、雷达、离屏渲染 |
| `src/mujoco_car_nn/track.py` | ~200 | 参数化赛道、曲率、墙体与路面扫掠 |
| `src/mujoco_car_nn/expert.py` | ~80 | pure-pursuit 专家与速度规划 |
| `src/mujoco_car_nn/dataset.py` | ~130 | 数据采集、npz 读写、Torch Dataset |
| `src/mujoco_car_nn/models.py` | ~150 | PolicyCNN / PolicyMLP / PolicyRunner |
| `src/mujoco_car_nn/train.py` | ~200 | BC 训练循环、DAgger 迭代 |
| `src/mujoco_car_nn/evaluate.py` | ~120 | 闭环 rollout 与指标统计 |
| `src/mujoco_car_nn/slam.py` | ~250 | 栅格地图、射线投射、扫描匹配、ATE/IoU |
| `src/mujoco_car_nn/viz.py` | ~170 | HUD、GIF 编码、地图渲染 |
| `ros1/car_nn_sim/scripts/sim_node.py` | ~130 | ROS 1 仿真节点 |
| `ros1/car_nn_sim/scripts/slam_node.py` | ~100 | ROS 1 建图节点 |
| `ros2/car_nn_sim/car_nn_sim/*.py` | ~220 | ROS 2 等价实现 |

## 2. 数据流

```text
track.build_track()
      │  points / tangents / curvature
      ▼
sim.CarSim.__init__  ──►  model XML (road, walls, car, camera)
      │
      ├─ reset() ──► CarState
      ├─ camera_image() ──► 64×64 uint8
      ├─ lidar() ──► 64 float32
      └─ step([steer, throttle]) ──► CarState
                    │
      ┌─────────────┼──────────────┐
      ▼             ▼              ▼
  dataset.py    expert.py       slam.py
  (采集数据)     (专家标签)      (建图/定位)
      │
      ▼
  train.py ──► runs/*.pt ──► evaluate.py / cli.py demo
```

## 3. 关键实现

### 3.1 `CarSim._xml()`：用 f-string 生成 MuJoCo 场景

```python
walls = []
for xy, yaw, hl in self.track.wall_poses(spacing=2.0):
    for (x, y), angle in zip(xy, yaw):
        walls.append(
            f'<geom type="box" pos="{x:.3f} {y:.3f} {WALL_HEIGHT/2:.3f}" '
            f'euler="0 0 {angle:.4f}" size="{hl:.3f} {WALL_THICKNESS/2:.3f} {WALL_HEIGHT/2:.3f}" '
            f'material="wall" group="1"/>')
```

路面不是贴图，而是把中心线左右偏移后用 OBJ 四边形条带生成：

```python
def _strip_obj(self, offset, half_width, z=0.012):
    left  = self.track.points[idx] + (offset + half_width) * self.track.normals[idx]
    right = self.track.points[idx] + (offset - half_width) * self.track.normals[idx]
    ...  # 输出 v / f 顶点与面
```

!!! tip "踩坑记录"
    1. 扁平 mesh 的惯量张量为 0，MuJoCo 会报 `mesh volume is too small`，需要
       `<mesh ... inertia="shell"/>`；
    2. MuJoCo 的 `vis.map.znear/zfar` 是**相对场景尺寸**的，场景 120 m 时
       `znear=0.05` 会变成 6 m，导致近处地面被裁掉——本项目取 `0.0005`；
    3. 默认 `geomgroup` 只显示 0/1/2 组，车身放在第 3 组会"隐身"。

### 3.2 `CarSim._apply_control()`：自行车模型

```python
v_fwd =  c * qvel[0] + s * qvel[1]
v_lat = -s * qvel[0] + c * qvel[1]
omega_des = v_fwd * np.tan(delta) / WHEELBASE
omega_des = np.clip(omega_des, -MAX_LAT_ACCEL / speed, MAX_LAT_ACCEL / speed)
tau = np.clip(INERTIA_Z * (omega_des - omega) * YAW_SERVO, -TAU_MAX, TAU_MAX)
f_long = MASS * accel - DRAG_C * v_fwd * abs(v_fwd) - ROLL_RESIST * MASS * 9.81 * np.sign(v_fwd)
f_lat = np.clip(-MASS * v_lat * 9.0, -MASS * 9.81, MASS * 9.81)
data.qfrc_applied[0] = c * f_long - s * f_lat
data.qfrc_applied[1] = s * f_long + c * f_lat
data.qfrc_applied[2] = tau
```

### 3.3 `PurePursuitExpert.act()`：专家

```python
target = self.track.point_at(s + lookahead)
alpha  = atan2(y_f, x_f)
steer  = clip(atan(2 * sin(alpha) / lookahead * WHEELBASE) / MAX_STEER, -1, 1)
v_tgt  = clip(sqrt(lat_accel / max(kappa_ahead, 1e-4)), v_min, v_max)
thr    = clip(0.3 + 0.45 * (v_tgt - speed), -1, 1)
```

### 3.4 `train_policy()`：训练循环

```python
for epoch in range(epochs):
    for x, s, y in train_loader:
        pred = net(x, s)
        loss = (((pred - y) ** 2) * weights).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        clip_grad_norm_(net.parameters(), 5.0); opt.step()
    sched.step()
```

### 3.5 `run_slam()`：建图主循环

```python
ranges = sim.lidar()                       # 1) 当前帧
matched = scan_match(local, grid, pose)    # 2) 相关匹配
pose = lateral_only_correction(pose, matched, alpha=0.7)
grid.update(pose, ranges, angles)          # 3) log-odds 更新
st = sim.step(action)                      # 4) 控制 + 物理
pose = dead_reckoning(pose, v, w, dt)      # 5) 里程计预测
```

## 4. 如何扩展

| 目标 | 修改点 |
|---|---|
| 换赛道 | `track.py` 中 `R0 / AMP2 / AMP3`，或改 `CONTROL_POINTS` 后重新采样 |
| 换车型（如阿克曼/差速） | `sim.py` 的 `_apply_control()` 与 `_xml()` 中的几何 |
| 加传感器（深度相机） | `sim.py` 新增 `<camera>` + `Renderer` 输出深度图，`dataset.py` 增加字段 |
| 换网络 | `models.build_model()` 注册新模型；训练/评测无需改动 |
| 换 RL 算法 | 在 `train.py` 中实现 PPO：用 `rollout()` 采样，奖励取 \(v\cos e_\psi-\lambda|e_y|\) |
| 加回环检测 | `slam.py` 增加 Scan Context 描述子与位姿图优化 |

## 5. 单元测试

```bash
python -m pytest tests -q
```

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_smoke.py` | 赛道闭环/弧长单调/最小转弯半径、仿真复位与专家加速、相机与雷达形状、网络输出范围、栅格占据/自由更新 |
| `tests/test_ros_nodes.py` | 用 `rospy`/`rclpy` 桩模块**真实执行** ROS 1 与 ROS 2 节点：发布 `/car/camera/image_raw`、`/car/scan`、`/car/odom`，校验图像字节数、雷达束数、里程计 frame，并验证遥操作指令能让车加速 |

后者在**没有安装 ROS 的 Windows CI** 上也能运行，用于提前发现消息字段拼写错误；
真正的 ROS 集成测试仍需在课程虚拟机上执行 `roslaunch`。
