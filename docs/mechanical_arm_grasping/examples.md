# 示例代码

## 基础用法

### 默认运行

最简单的运行方式，使用所有默认参数：

```bash
cd src/mechanical_arm_grasping
python main.py
```

等价于：

```bash
python main.py 5 medium 0 medium
```

---

## 进阶用法

### 批量快速测试

快速抓取 20 次，检查系统稳定性：

```bash
python main.py 20 fast
```

### 质量验收测试

抓取 50 次，要求 95% 以上成功率，使用重力度确保抓稳：

```bash
python main.py 50 medium 95 heavy
```

### 调试慢速观察

慢速模式便于观察每个关节的运动轨迹：

```bash
python main.py 3 slow
```

### 轻力度抓取易碎物

```bash
python main.py 5 slow 0 light
```

---

## 编程调用

### 在 Python 脚本中调用

```python
import sys
sys.path.insert(0, "src/mechanical_arm_grasping")
from main import robot_arm_auto_reset_demo

# 直接调用主函数
exit_code = robot_arm_auto_reset_demo()
print(f"程序退出码: {exit_code}")
```

!!! warning "注意"
    `robot_arm_auto_reset_demo()` 内部通过 `sys.argv` 读取命令行参数，直接调用时需要在调用前设置 `sys.argv`：

```python
import sys
sys.argv = ["main.py", "10", "fast", "80", "heavy"]

from main import robot_arm_auto_reset_demo
robot_arm_auto_reset_demo()
```

---

## 自定义扩展

### 修改抓取目标位置

编辑 `main.py` 中 `grab_and_place()` 函数内的目标关节值：

```python
# 原始抓取位置（目标在 (0.9, 0.6) 附近）
joint_move("joint1", 0.0, ...)    # 基座旋转角度
joint_move("joint2", -0.7, ...)   # 大臂俯仰角度
joint_move("joint3", 0.35, ...)   # 小臂伸缩距离
```

### 修改放置位置

编辑放置阶段的关节目标值：

```python
# 旋转到放置区（绿色区域在 (-0.9, 0.6) 附近）
joint_move("joint1", 3.14, ...)   # 基座旋转 180°
joint_move("joint2", -0.7, ...)   # 大臂俯仰角度
```

### 添加新的力度等级

在 `main.py` 的 `GRIP_CONFIG` 字典中添加新条目：

```python
GRIP_CONFIG = {
    "light":  0.15,
    "medium": 0.25,
    "heavy":  0.35,
    "extreme": 0.50,  # 新增：极重力度
}
```

### 修改 MuJoCo 模型

编辑 `robot_arm.xml` 文件可以修改机械臂的物理结构：

- 调整关节范围：修改 `range` 属性
- 调整几何尺寸：修改 `size` 属性
- 修改颜色：修改 `material` 引用或 `rgba` 值
- 添加新物体：在 `<worldbody>` 中添加新的 `<body>` 元素

---

## 常见问题

### 机械臂抓不到目标

1. 检查目标位置是否在机械臂工作范围内
2. 尝试降低速度（`slow` 模式）
3. 增加夹爪力度（`heavy` 模式）

### 目标被弹飞

1. 减小夹爪力度（`light` 模式）
2. 降低运动速度
3. 检查目标物体初始位置是否合理

### 可视化窗口卡顿

1. 使用 `fast` 模式减少渲染帧数
2. 关闭其他占用 GPU 的程序
3. 检查 MuJoCo 是否正确安装 GPU 加速
