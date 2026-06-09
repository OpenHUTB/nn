# API 参考

## 模块结构

```
mujoco01/
├── main.py              # 主仿真控制循环
├── run_robot_control.py # 简易控制示例
└── humanoid.xml         # MJCF 模型定义
```

---

## `main()`

```python
def main() -> None
```

启动 MuJoCo 人形机器人仿真控制主循环。

**功能：** 加载模型 → PD 控制维持平躺姿态 → 右肘关节周期抬手 → 数据记录 → 可视化。

**控制参数（函数内硬编码，修改源码调整）：**

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `kp` | `35.0` | PD 比例增益 |
| `kd` | `8.0` | PD 微分增益 |
| `max_torque` | `1.0` | 最大力矩限制 (N·m) |
| `smooth_alpha` | `0.06` | EMA 平滑系数 |
| `elbow_lift_max` | `0.25` | 肘关节最大力矩 (N·m) |
| `cycle_period` | `6.0` | 抬手周期 (秒) |

**调用示例：**

```python
from mujoco01.main import main
main()
```

---

## 关节与执行器

`humanoid.xml` 定义 21 自由度人形机器人，执行器映射：

| 执行器索引 | 名称 | 说明 |
|:---|:---|:---|
| 0-2 | `abdomen_*` | 躯干 3 DOF |
| 3-8 | `hip/knee/ankle_*_right` | 右腿 6 DOF |
| 9-14 | `hip/knee/ankle_*_left` | 左腿 6 DOF |
| 15-17 | `shoulder*_right, elbow_right` | 右臂 3 DOF |
| 18-20 | `shoulder*_left, elbow_left` | 左臂 3 DOF |

---

## 数据输出

- **终端**：每 0.5s 打印仿真时间、抬手阶段、肘关节力矩、机器人高度
- **CSV**：`simulation_data.csv`，列：`time, qpos_0..nq-1, qvel_0..nv-1`

---

## 扩展示例

### 修改控制参数

```python
# main.py 第 175 行附近
kp = 50.0   # 增大刚度
```

### 编程方式调用

```python
import mujoco

model = mujoco.MjModel.from_xml_path("src/mujoco01/humanoid.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_step(model, data)
print(f"根部高度: {data.qpos[2]:.3f}m")
```
