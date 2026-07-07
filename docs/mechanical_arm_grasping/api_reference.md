# API 参考

## 主入口函数

### `robot_arm_auto_reset_demo()`

程序主入口，负责加载 MuJoCo 模型、初始化仿真、解析命令行参数并启动取放流程。

```python
def robot_arm_auto_reset_demo():
    """主入口函数，返回 EXIT_SUCCESS 或 EXIT_FAILURE"""
```

**返回值：**

| 值 | 说明 |
|----|------|
| `EXIT_SUCCESS (0)` | 程序正常完成 |
| `EXIT_FAILURE (1)` | 成功率不满足最低要求 |

---

## 核心控制函数

### `joint_move(joint_name, target_val, duration, viewer, step_desc)`

单关节精准移动控制，带实时进度条和力矩显示。

```python
def joint_move(joint_name, target_val, duration, viewer, step_desc):
    """
    参数:
        joint_name (str): 关节名称，可选 "joint1" / "joint2" / "joint3"
        target_val (float): 目标位置值（弧度或米）
        duration (float): 运动持续时间（秒）
        viewer: MuJoCo viewer 实例
        step_desc (str): 步骤描述文本
    返回:
        bool: 始终返回 True
    """
```

**关节名称映射：**

| joint_name | 对应执行器 | 关节类型 |
|------------|-----------|----------|
| `"joint1"` | `joint1_act` | 基座旋转 |
| `"joint2"` | `joint2_act` | 大臂俯仰 |
| `"joint3"` | `joint3_act` | 小臂伸缩 |

---

### `gripper_close(viewer, desc="目标", force="medium")`

软接触闭合夹爪抓取目标，带进度条显示，支持力度调节。

```python
def gripper_close(viewer, desc="目标", force="medium"):
    """
    参数:
        viewer: MuJoCo viewer 实例
        desc (str): 抓取目标描述
        force (str): 力度模式，可选 "light" / "medium" / "heavy"
    返回:
        bool: 始终返回 True
    """
```

**力度对应闭合速度：**

| force | 闭合速度 |
|-------|---------|
| `"light"` | -0.15 |
| `"medium"` | -0.25 |
| `"heavy"` | -0.35 |

---

### `gripper_open(viewer, desc="目标")`

张开夹爪放置目标。

```python
def gripper_open(viewer, desc="目标"):
    """
    参数:
        viewer: MuJoCo viewer 实例
        desc (str): 放置目标描述
    返回:
        bool: 始终返回 True
    """
```

---

### `robot_auto_reset(viewer)`

机械臂自动复位到初始位置（所有关节归零）。

```python
def robot_auto_reset(viewer):
    """
    复位顺序:
        1. joint2 归零（抬升大臂）
        2. joint3 归零（收缩小臂）
        3. joint1 归零（基座回正）
    返回:
        bool: 始终返回 True
    """
```

---

### `target_auto_reset(viewer)`

目标物体随机重置到工作平台上的新位置。

```python
def target_auto_reset(viewer):
    """
    随机范围:
        X: [0.5, 1.2]
        Y: [0.3, 1.2]
        Z: 固定 0.0
    """
```

---

### `grab_and_place(viewer, retry_max=2, speed="medium", grip_force="medium")`

完整取放流程，含自动重试、连续失败提醒、速度控制、力度控制和计时功能。

```python
def grab_and_place(viewer, retry_max=2, speed="medium", grip_force="medium"):
    """
    参数:
        viewer: MuJoCo viewer 实例
        retry_max (int): 最大重试次数，默认 2
        speed (str): 速度模式，"slow" / "medium" / "fast"
        grip_force (str): 力度模式，"light" / "medium" / "heavy"
    返回:
        bool: 抓取是否成功
    """
```

**执行流程：**

| 阶段 | 步骤 | 函数调用 |
|------|------|----------|
| 对准 | 1. 旋转基座对准目标 | `joint_move("joint1", 0.0, ...)` |
| 对准 | 2. 俯仰大臂接近目标 | `joint_move("joint2", -0.7, ...)` |
| 对准 | 3. 伸缩小臂对准目标 | `joint_move("joint3", 0.35, ...)` |
| 抓取 | 4. 闭合夹爪 | `gripper_close(viewer, ...)` |
| 转移 | 5. 抬升目标 | `joint_move("joint2", 0.0, ...)` |
| 转移 | 6. 旋转到放置区 | `joint_move("joint1", 3.14, ...)` |
| 转移 | 7. 降低到放置区 | `joint_move("joint2", -0.7, ...)` |
| 放置 | 8. 张开夹爪 | `gripper_open(viewer, ...)` |

---

## 全局配置

### `SPEED_CONFIG`

速度模式配置字典：

```python
SPEED_CONFIG = {
    "slow":   {"joint": 3.0, "grip": 1.5, "pause": 0.02},
    "medium": {"joint": 2.0, "grip": 1.0, "pause": 0.001},
    "fast":   {"joint": 1.0, "grip": 0.5, "pause": 0.0005}
}
```

### `GRIP_CONFIG`

夹爪力度配置字典：

```python
GRIP_CONFIG = {
    "light":  0.15,
    "medium": 0.25,
    "heavy":  0.35
}
```

### 退出码

```python
EXIT_SUCCESS = 0  # 程序正常完成
EXIT_FAILURE = 1  # 成功率不满足最低要求
```
