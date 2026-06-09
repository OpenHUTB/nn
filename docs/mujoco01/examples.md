# 示例代码

## 基础用法

### 直接运行

```bash
cd src/mujoco01
python main.py
```

启动后 MuJoCo 窗口会显示人形机器人平躺姿态，右肘关节周期性抬手。

### 模块调用

```bash
python -m mujoco01.main
```

## 进阶用法

### 修改抬手幅度

编辑 `main.py`，增大 `elbow_lift_max` 使抬手更明显：

```python
elbow_lift_max = 0.5   # 原值 0.25，加大到 0.5
```

### 修改抬手速度

调整 `cycle_period` 改变抬手周期：

```python
cycle_period = 3.0     # 原值 6.0，加快到 3 秒一个周期
```

### 调整 PD 刚度

使姿态保持更"硬"：

```python
kp = 50.0              # 原值 35.0
kd = 12.0              # 原值 8.0
```

## 编程调用

### 在脚本中导入运行

```python
from mujoco01.main import main

# 直接调用，阻塞运行直到关闭窗口
main()
```

### 自定义控制逻辑

以下示例展示如何基于现有代码实现自定义控制：

```python
import mujoco
import numpy as np
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("src/mujoco01/humanoid.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qvel[:] = 0

kp, kd = 35.0, 8.0
qpos_ref = data.qpos.copy()

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        ctrl = np.zeros(model.nu)

        # 对所有关节施加 PD 控制
        for i in range(model.nu):
            joint_id = int(model.actuator_trnid[i][1])
            qidx = int(model.jnt_qposadr[joint_id])
            if qidx >= 0:
                torque = kp * (qpos_ref[qidx] - data.qpos[qidx]) - kd * data.qvel[qidx]
                ctrl[i] = np.clip(torque, -1.0, 1.0)

        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        v.sync()
```

## 数据后处理

### 读取仿真数据

仿真结束后，`simulation_data.csv` 包含完整的关节状态数据：

```python
import pandas as pd

df = pd.read_csv("simulation_data.csv")
print(f"数据行数: {len(df)}")
print(f"仿真时长: {df['time'].iloc[-1]:.1f}s")
print(f"列名: {list(df.columns)}")
```

### 绘制肘关节角度

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["qpos_18"], label="右肘关节角度")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("右肘关节角度随时间变化")
plt.legend()
plt.grid(True)
plt.show()
```

### 绘制肘关节力矩

```python
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["qvel_18"], label="右肘关节角速度")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("右肘关节角速度随时间变化")
plt.legend()
plt.grid(True)
plt.show()
```

### 绘制根节点高度

```python
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["qpos_2"], label="根节点 Z 坐标")
plt.axhline(y=0.3, color='r', linestyle='--', label="初始高度 0.3m")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("机器人根节点高度变化")
plt.legend()
plt.grid(True)
plt.show()
```

## 常见问题

### 如何让机器人站立而非平躺？

修改 `humanoid.xml` 中 keyframe 的 `qpos` 值，将根节点高度设为站立高度（约 1.282m），并调整各关节至站立姿态。

### 如何同时控制双臂？

在 `main.py` 的控制循环中添加左臂执行器的力矩计算，参考右肘的抬手逻辑。

### 如何添加外力扰动？

在 `mujoco.mj_step()` 之前设置 `data.xfrc_applied`：

```python
# 对 torso 施加一个向上的力
data.xfrc_applied[1, 2] = 10.0  # body_id=1, Z方向, 10N
```
