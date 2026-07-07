# 机械臂抓取模块

## 学生作业提交

- **学生**：minshu22
- **日期**：2026-04-20

## 项目简介

基于 MuJoCo 3.4.0 物理仿真引擎的 **3自由度机械臂精准取放系统**，支持自动复位、抓取计时、速度调节和力度控制等功能。

机械臂由三个关节组成：
- **关节1（基座旋转）**：绕 Z 轴旋转，范围 [-3.14, 3.14] 弧度
- **关节2（大臂俯仰）**：绕 Y 轴俯仰，范围 [-1.5, 1.5] 弧度
- **关节3（小臂伸缩）**：沿 X 轴滑动，范围 [0, 0.4] 米

末端配备平行夹爪，用于抓取蓝色球体目标并放置到绿色目标区域。

## 修改内容

将机械臂的 MuJoCo XML 模型从 `main.py` 中分离到独立的 `robot_arm.xml` 文件，提高代码可维护性。

### 修改前

XML 模型以字符串形式内嵌在 `main.py` 的 `robot_xml` 变量中：

```python
def robot_arm_auto_reset_demo():
    robot_xml = """
    <mujoco model="3-DOF Robot Arm with Auto Reset">
      ...
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(robot_xml)