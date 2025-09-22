# 人形机器人
# MuJoCo Python 模拟示例

这是一个使用MuJoCo物理引擎的Python示例项目，展示了如何加载人形机器人模型、运行物理模拟并实时可视化结果。示例模拟了人形机器人在预设初始姿势下的物理运动，支持实时查看关节状态与躯干位置。

## 项目结构
humanoid_mujoco_py/
├─ humanoid.xml       # MJCF模型文件，定义人形机器人结构与物理参数
├─ main.py            # Python主程序，加载模型、运行模拟与可视化
└─ README.md          # 项目说明文档

## 使用方法

1. 确认文件路径：确保`humanoid.xml`与`main.py`在同一目录下（或在`main.py`中修改模型路径为绝对路径）。

2. 运行模拟程序

   ```bash
   python main.py
   ```

3. 程序将：
   - 加载`humanoid.xml`中定义的人形机器人模型（含躯干、四肢、关节与执行器）
   - 初始化模拟数据（关节位置、速度、物理状态）
   - 设置初始姿势为「深蹲（squat）」（可修改代码切换其他姿势）
   - 启动可视化窗口，实时渲染机器人运动
   - 在终端输出实时仿真时间与躯干位置（格式：`时间: X.XX, 躯干位置: (X.XX, X.XX, X.XX)`）
   - 无限循环启动模拟

## 交互操作

在可视化窗口中，可以使用以下鼠标操作调整视角或交互：
- 左键拖动：旋转场景视角
- 右键拖动：平移场景位置
- 滚轮：缩放视图（拉近/拉远）
- Shift+左键：拖动场景中的刚体（如机器人躯干、四肢）

## 代码说明

- `humanoid.xml`：使用MJCF（MuJoCo XML格式）定义人形机器人的完整物理场景，核心内容包括：
  - **机器人结构**：躯干（torso）、头部（head）、四肢（大腿、小腿、手臂、手掌）的几何形状与连接关系
  - **关节参数**：各关节（髋关节、膝关节、肩关节等）的旋转轴、角度范围（如髋关节Y轴范围`-150~20°`）、阻尼与刚度
  - **物理属性**：材料（身体肤色、地面网格）、摩擦系数（`0.7`）、碰撞排除规则（避免腰部与大腿误碰撞）
  - **执行器**：21个电机（控制关节运动），含齿轮比（如髋关节电机`gear=120`）与控制范围（`-1~1`）
  - **初始关键帧**：4种预设姿势（深蹲`squat`、单左腿站立`stand_on_left_leg`、俯卧`prone`、仰卧`supine`）

- `main.py`：Python程序实现模拟全流程控制，核心功能包括：
  - 模型加载：通过`mujoco.MjModel.from_xml_path()`读取`humanoid.xml`，捕获加载错误（如路径错误）
  - 数据初始化：通过`mujoco.MjData(model)`创建仿真状态容器（存储关节位置`qpos`、时间`time`等）
  - 初始姿势设置：通过`mujoco.mj_resetDataKeyframe(model, data, 0)`设置为深蹲姿势（索引`0~3`对应4种预设姿势）
  - 可视化启动：通过`viewer.launch_passive()`启动被动式Viewer（手动控制仿真步长与视图更新）
  - 仿真循环：循环中，通过`mujoco.mj_step()`执行物理计算，`viewer.sync()`更新视图，`time.sleep(0.005)`控制帧率

## 扩展方向

1. **修改初始姿势与模拟时长**：
   - 切换姿势：修改`main.py`中`mujoco.mj_resetDataKeyframe()`的第三个参数（`0=深蹲`、`1=单左腿站立`、`2=俯卧`、`3=仰卧`）
   - 延长/缩短模拟：调整`for _ in range(1200)`的循环次数（公式：循环次数 = 目标时长（秒）/ 0.005）

2. **自定义机器人物理属性**：
   - 编辑`humanoid.xml`：调整关节角度范围（如`joint range="-30 10"`）、电机齿轮比（`actuator gear="40"`）、地面摩擦系数（`geom friction=".7"`）
   - 添加新部件：在`<worldbody>`中新增`<body>`节点，定义额外几何形状（如机器人手持物体）

3. **增强模拟功能**：
   - 施加控制信号：在仿真循环中修改`data.ctrl`数组（如`data.ctrl[0] = 0.5`控制腹部Z轴电机），实现行走、抬手等动作
   - 记录仿真数据：添加代码将`data.qpos`（关节位置）、`data.qvel`（关节速度）写入CSV文件，用于后续分析
   - 添加传感器：在`humanoid.xml`中添加`<sensor>`节点（如力传感器、加速度传感器），在`main.py`中读取`data.sensordata`获取数据

4. **优化可视化**：
   - 调整相机视角：修改`humanoid.xml`中`<camera>`节点的`pos`（位置）、`xyaxes`（朝向）参数
   - 自定义颜色：修改`<asset>`中`texture`的`rgb1`/`rgb2`值，更改机器人身体或地面颜色

## 参考资料

- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)：学习MJCF格式、API接口与物理仿真原理
- [MuJoCo GitHub 仓库](https://github.com/google-deepmind/mujoco)：获取最新版MuJoCo库与官方示例模型
- [MJCF 模型格式参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html)：详细了解`humanoid.xml`各节点配置规则
- [MuJoCo Python 教程](https://mujoco.readthedocs.io/en/stable/python.html)：学习Python调用MuJoCo的进阶用法（如控制、传感器、数据记录）