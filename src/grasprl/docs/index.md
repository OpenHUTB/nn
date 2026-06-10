# GraspRL：基于 DQN 与视觉感知的机械臂抓取强化学习项目
## 一、项目概述
本项目基于 **MuJoCo 物理引擎**与 **PyTorch** 深度学习框架，围绕 **UR5e 机械臂 + AG95 二指夹爪** 的视觉抓取任务，搭建并持续改进 DQN 强化学习训练框架。目标是将机械臂抓取策略的学习过程从"能跑通"提升到"可复现、可对比、可解释、可迭代"。

项目核心特点：
- **基于视觉的端到端抓取**：输入 RGB-D 图像，经过 ResNet 编码器-解码器网络，输出逐像素 Q 值，选择最优抓取像素位置并映射到世界坐标
- **多阶段抓取执行管线**：打开夹爪→粗定位→精准下降→夹取闭合→抬起→结果验证→放置，每阶段有独立的运动控制策略
- **操作空间控制（OSC）**：基于质量矩阵和雅可比矩阵的操作空间控制器，实现末端执行器平滑、稳定的位姿跟踪
- **工程化训练增强**：经验回放、目标网络、梯度裁剪、指令引导探索（Instruction-based Exploration）、密集+稀疏混合奖励
- **MuJoCo 物理仿真优化**：关节限位保护 IK、actuator 与 OSC 解耦同步、接触力检测验证抓取

本项目基于 [AvalonGuo/grasprl](https://github.com/AvalonGuo/grasprl) 进行深度改进，可用于机器人抓取策略验证、强化学习算法研究、机械臂控制仿真调试等场景。
---
## 二、开发环境与技术栈
### 2.1 核心技术栈
| 技术 | 用途 |
|------|------|
| Python 3.8+ | 主开发语言 |
| MuJoCo 2.3.0+ | 高精度物理仿真引擎 |
| NumPy | 数值计算（线性代数、矩阵运算） |
| PyTorch | 深度学习框架（Q 网络定义与训练） |
| Gymnasium | RL 环境标准接口封装 |
| OpenCV | 图像处理与数据集保存 |
| TensorBoard | 训练指标可视化 |

### 2.2 运行环境
- **操作系统**：Linux / Windows 均可
- **Python 版本**：3.8 ~ 3.10
- **仿真引擎**：MuJoCo 2.3 及以上
- **硬件**：CPU 可运行，GPU 推荐（加速网络训练）
### 2.3 依赖安装
```powershell
pip install numpy mujoco torch gymnasium opencv-python tensorboard tqdm
```
> **注意**：Windows 下若遇到 OpenMP 冲突，运行前设置环境变量：
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK="TRUE"
> ```
---
## 三、项目结构
```
nn/src/grasprl/
├── index.md                         # 项目说明（本文档）
├── docs/
│   ├── index.md                      # 文档主页
│   └── images/                       # 文档图片资源
│       ├── 1.png                      # MuJoCo仿真场景截图
│       ├── 2.png                      # Top-down相机观测视角
│       ├── 3.png                      # 训练终端输出日志
│       ├── 4.png                      # TensorBoard训练曲线
│       └── 5.png                      # 硬编码抓取运行截图
├── run.py                            # 训练入口脚本
├── debug_step.py                     # 端到端硬编码抓取测试脚本
├── count_dataset.py                  # 数据集统计工具
│
├── grasprl/
│   ├── run.py                        # 训练入口（重定向到 trainer）
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── joint_effort_controller.py    # 关节力矩控制器 + 夹爪力矩控制器
│   │   └── operational_space_controller.py # 操作空间控制器（OSC）
│   ├── envs/
│   │   └── grasp.py                      # ★ 核心环境：GraspRobot 类
│   ├── modules/
│   │   └── qnet.py                       # ★ Q 网络模块：ResNet 编码器-解码器
│   ├── renderer/
│   │   └── mujoco_env.py                 # MuJoCo 物理环境基类封装
│   ├── trainer/
│   │   └── dqn_baseline.py               # ★ DQN 训练器（训练主逻辑）
│   ├── utils/
│   │   ├── controller_utils.py           # 控制器辅助（任务空间惯性矩阵等）
│   │   ├── mujoco_utils.py               # MuJoCo 工具（Jacobian、名称映射等）
│   │   └── transform_utils.py            # 四元数/旋转矩阵转换工具
│   ├── worlds/
│   │   ├── grasp.xml                     # MuJoCo 场景定义（UR5e + AG95 + 物体）
│   │   └── assets/meshes/                # 3D 模型网格（.stl/.obj）
│   ├── dataset/grasp_samples/            # 训练过程中的图像+标签数据存档
│   └── log/DQN/                          # TensorBoard 日志目录
│
└── trainer/
    └── dqn_baseline.py                   # 训练器备份/另一版本
```
---
## 四、核心实现详解
### 4.1 MuJoCo 仿真场景
**场景定义文件**：`grasprl/worlds/grasp.xml`

**机械臂**：UR5e 6 自由度串联机械臂
- 关节：`shoulder_pan` → `shoulder_lift` → `elbow` → `wrist_1` → `wrist_2` → `wrist_3`
- 每个关节配置 position actuator（gain=2000, damping=400），用于默认的关节位置控制
- 训练中使用 OSC 控制器绕过 actuator，通过 `qfrc_applied` 直接施加关节力矩

**夹爪**：AG95 二指平行夹爪
- 每侧 3 个关节：`inner_knuckle` → `outer_knuckle` → `finger`
- Equality 约束（`connect` + `joint`）确保两侧手指同步运动
- 两个 position actuator（kp=50, kv=5）控制开合：`left_finger_act` / `right_finger_act`
- 关节行程：`[0, 0.943]` rad（约 54°），0=全开，0.943=全闭

**桌面与物体**：
- 桌面：0.3×0.3×0.05m，中心高度 z=0.95m
- 6 个可抓取物体：3 个球体 + 3 个盒子（自由关节，可被推/抓）
- 物体命名：`ball_1` ~ `ball_3`，`box_1` ~ `box_3`

**相机**：
- `top_down` 固定相机：用于获取训练观测图像
- `eyeinhand` 手眼相机：安装在末端执行器旁

![MuJoCo仿真场景截图](images/1.png)

---
### 4.2 环境：GraspRobot（核心类）
**文件**：`grasprl/envs/grasp.py`

`GraspRobot` 继承自 `MujocoPhyEnv`，封装了完整的抓取任务环境。

#### 4.2.1 观测空间与动作空间
- **观测**：`defaultdict` 包含 RGB 图像 (64×64×3) 和深度图 (64×64)
- **动作**：`Box(low=-0.5, high=0.5, shape=[3])` → 世界坐标系下抓取目标位置 [x, y, z]

![Top-down相机观测视角](images/2.png)

#### 4.2.2 奖励设计（密集 + 稀疏混合）
| 奖励类型 | 系数 | 触发阶段 | 计算方式 |
|---------|------|---------|---------|
| 靠近奖励 | +3.0 | Phase 2 / Phase 3 | EE 到目标距离的减少量 |
| 闭合奖励 | +2.0 | Phase 4 | 夹爪闭合比例（qpos / 0.943） |
| 接触奖励 | +5.0 | Phase 4 | 手指与物体产生物理接触 |
| 抬起奖励 | +8.0 | Phase 6（成功时） | 物体实际抬起高度 / LIFT_HEIGHT |
| 成功奖励 | +100.0 | Phase 6（成功时） | 物体被稳定抓取并抬起 |
| 失败惩罚 | -3.0 | Phase 6（失败时） | 未检测到有效抓取 |

#### 4.2.3 多阶段抓取执行管线 (step 方法)
```
Phase 1: 打开夹爪 (frame_skip/2 步)
    ↓
Phase 2: 粗定位 → IK 移动到物体上方 20cm
    ↓
Phase 3: 测量手指-EE 偏移 → 分 3 段精准下降 (GRASP_DEPTH 控制)
    ↓
Phase 4: OSC 保持臂位姿 + 关闭夹爪 (渐进闭合 → 保压)
    ↓
Phase 5: IK 抬起物体 (LIFT_HEIGHT=0.25m)
    ↓
Phase 6: 检测结果 → 成功(移到放置区释放) / 失败(打开夹爪)
```

**核心控制逻辑**：
- **IK 求解** (`_ik_to_target`)：阻尼最小二乘 Jacobian 伪逆 IK，200 步迭代，含关节限位钳位和 nullspace 向心力（拉回固定 home pose）
- **OSC 微调** (`_move_eef_ik`)：IK 到位后，用操作空间控制器进行 20 步精细稳定
- **Actuator 同步** (`_sync_arm_ctrl`)：设置 arm actuator ctrl = qpos + 0.2·qvel，使 position actuator 产生零净力，避免与 OSC 冲突
- **夹爪闭合**：通过 position actuator（kp=50）渐进设置 ctrl 到 0.95，分加速段+保压段，保证充分闭合

#### 4.2.4 抓取成功检测（三重验证）
`check_grasp_success()` 不再仅靠物体 Z 位移判断，而是三重条件：
1. **物体抬升 > 3mm**：从抓取前的记录位置比较 Z 轴变化
2. **手指间距 < 12cm**：确认夹爪确实闭合（不是被推起来的）
3. **手指-物体接触**：通过 `get_finger_contacts()` 遍历 MuJoCo 接触对，确认至少一根手指与抬升物体的几何体有物理接触

---
### 4.3 Q 网络架构
**文件**：`grasprl/modules/qnet.py`

网络采用 **ResNet 风格编码器-解码器**结构，专为像素级抓取值函数设计：
```
输入: 4 通道图像 (RGB 归一化 + 深度反转)  [4, 64, 64]
    ↓
Perception_Module (编码器):
    Conv2d(4→64, 7×7, stride=2) → MaxPool
    ResBlock(64→128, stride=2)
    ResBlock(128→256, stride=2)
    ResBlock(256→512)              → [512, 8, 8]
    ↓
Grasping_Module_multidiscrete (解码器):
    ResBlock(512→256)
    ResBlock(256→128)
    Upsample (2×) → ResBlock(128→64)
    Upsample (2×) → Conv2d(64→1)   → [1, 64, 64]
    Flatten + Sigmoid                → 4096 维 Q 值
```

每个 `ResBlock` 包含：
- 两个 3×3 卷积 + BatchNorm + ReLU
- 跳跃连接（降采样时用 1×1 卷积对齐维度）

**关键模块**：
- `Perception_Module`：编码器，提取视觉特征
- `Grasping_Module`：解码器，输出抓取热力图
- `Grasping_Module_multidiscrete`：多通道解码器，用于多类别离散动作
- `MULTIDISCRETE_RESNET(n)`：工厂函数，输出 n×4096 维 Q 值（当前使用 n=1，即 4096 维）

---
### 4.4 训练器：DQN_Trainer
**文件**：`grasprl/trainer/dqn_baseline.py`（主训练版本）

#### 4.4.1 核心配置
| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 0.001 | Adam 优化器 |
| Weight Decay | 0.0001 | L2 正则化 |
| 经验回放容量 | 10000 | Replay Buffer 大小 |
| Batch Size | 32 | 每步训练样本数 |
| ε 起始值 | 1.0 | 初始探索率 |
| ε 终止值 | 0.01 | 最低探索率 |
| ε 衰减率 | 5000 | 指数衰减速率 |
| 目标网络更新频率 | 100 | 每 100 步同步 target net |
| Loss 函数 | SmoothL1Loss | Huber Loss（β=1） |

#### 4.4.2 状态预处理
`transform_state()` 将原始观测转换为网络输入：
1. RGB 图像：`ToTensor()` → `Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])`（ImageNet 统计量）
2. 深度图：`max() - depth`（反转，让近处更亮） → `ToTensor()`
3. 拼接 RGB + Depth → `[4, 64, 64]`
4. 通过 `VisualFeatureEnhancer`（4→16→4 卷积增强）后 clamp 到 [-1, 1]

#### 4.4.3 动作映射
`transform_action()` 将网络输出的像素索引 (0~4095) 转换为世界坐标：
1. 将像素索引还原为 (px, py) 坐标
2. **优先匹配实际物体**：遍历场景中所有可抓取物体，计算每个物体的世界坐标投影到图像上的像素位置
3. 选择距离输出像素最近的物体，返回其世界坐标 (wx, wy, wz)
4. 退路：若无可见物体，使用 `pixel2world()` 通过深度值反投影

#### 4.4.4 指令引导探索（Instruction-based Exploration）
这是项目的一个**关键创新**。传统 ε-greedy 探索时随机选择像素，但本项目在探索时直接使用**真实物体位置**作为探索目标：
```python
def select_action_by_instruction(self, state):
    if random.random() > epsilon:   # 贪心
        return q_net(state).argmax()
    else:                           # 引导探索
        available_objects = [可见物体]
        if available_objects:
            obj = random.choice(available_objects)
            return world2pixel(obj)  # 选中某个实际物体的像素
        else:
            return random pixel       # 退路随机
```

这样做的优势：
- 避免在空白区域无意义探索
- 初始阶段策略也能"碰到"物体，获得有效奖励信号
- 加速策略收敛

#### 4.4.5 DQN 学习更新
`learn()` 方法实现标准 DQN 更新：
```python
q_pred = q_net(state).gather(action)
q_target = reward + gamma * target_q_net(next_state).max()
loss = SmoothL1Loss(q_pred, q_target)
# + gradient clipping (max_norm=1.0)
```

---
### 4.5 操作空间控制器（OSC）
**文件**：`grasprl/controllers/operational_space_controller.py`

基于任务空间动力学的精确末端控制：
1. **计算末端 Jacobian**：`get_site_jac()` 获取 6×nv 的平移+旋转 Jacobian
2. **任务空间惯性矩阵**：`Mx = inv(J @ inv(M) @ J.T)`（伪逆处理奇异）
3. **位姿误差**：`pose_error()` = [位置差, 角度×旋转轴]
4. **速度限幅**：自适应缩放，距离越近速度越慢
5. **力矩转换**：`τ = J.T @ Mx @ u_task + 阻尼补偿 + 重力补偿`
6. **施加力矩**：通过 `qfrc_applied` 写入关节力

控制器参数：
- kp=80（位置增益）、ko=80（方向增益）、kv=50（阻尼）
- vmax_xyz=1（最大平移速度）、vmax_abg=2（最大旋转速度）
- min/max effort=±150（力矩安全限幅）

---
### 4.6 夹爪控制器
**文件**：`grasprl/controllers/joint_effort_controller.py`

`GripperEffortCtrl` 支持两种控制模式：
- **Actuator 模式**（有 `actuator_id`）：直接设置 `data.ctrl[act_id]`，利用 MuJoCo 内置 position actuator
- **Joint Effort 模式**：渐进 ramp-up 力矩（10 步），闭合方向施加正向力矩，打开方向施加 4 倍反向力矩

当前训练流程中，夹爪主要通过 position actuator 控制（`left_finger_act` / `right_finger_act`），`GripperEffortCtrl` 作为备用。

---
## 五、运行与测试
### 5.1 启动训练
```powershell
cd D:\nn\src\grasprl
# Windows PowerShell（注意设置 OpenMP 环境变量）
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m grasprl.run
```

### 5.2 训练过程输出
```
iter [1]/[100]  grasp_info=Failed  reward=-3.161  action=instruction
iter [2]/[100]  grasp_info=Failed  reward=-1.333  action=instruction
...
```
- `iter`：当前交互轮次
- `grasp_info`：抓取结果（Success/Failed）
- `reward`：本轮奖励（包含密集+稀疏）
- `action`：动作来源（greedy=网络输出 / instruction=引导探索）

![训练终端输出日志](images/3.png)

### 5.3 训练过程可视化（TensorBoard）
训练过程中所有核心指标通过TensorBoard实时监控，下图为训练全流程的指标变化曲线：

![TensorBoard训练指标曲线](images/4.png)

#### 核心指标解读：
1. **Reward 奖励曲线（橙色）**：每轮抓取的累计奖励，训练前期波动较大，后期稳步上升，说明策略从随机探索逐步收敛到有效抓取行为
2. **Loss 损失曲线（蓝色）**：DQN网络的SmoothL1损失，整体呈下降后平稳收敛趋势，网络拟合效果稳定
3. **Epsilon 探索率曲线（灰色）**：按照指数衰减策略从1.0逐步下降到0.01，符合ε-greedy探索-利用平衡设计
4. **平滑效果**：开启滑动平均（smoothing=0.6）后，曲线趋势更清晰，可直观观察训练整体走向

#### 本地交互式监控（可选）
如果需要本地查看可交互的原始训练日志，在项目根目录执行：
```powershell
tensorboard --logdir=grasprl/log/DQN/resnet_dqn_insne_v2
```
浏览器访问 `http://localhost:6006` 即可打开完整TensorBoard面板，支持缩放、导出、指标对比等操作。

### 5.4 数据集样本
训练过程中自动保存每次交互的视觉数据到 `grasprl/dataset/grasp_samples/`：
- `rgb_{iter}.png`：RGB 观测图像
- `depth_{iter}.npy`：深度图
- `label_{iter}.npy`：标签（action, grasp_success, reward）

使用 `count_dataset.py` 统计数据集中的抓取成功率。

### 5.5 硬编码抓取测试
`debug_step.py` 提供了一个**不使用 RL** 的端到端抓取测试脚本，用于验证机械臂和夹爪控制逻辑：
```powershell
cd D:\nn\src\grasprl
python debug_step.py
```

![硬编码抓取仿真运行界面](images/5.png)

---
## 六、项目改进历程
### 6.1 机械臂控制稳定性
| 问题 | 解决方案 |
|------|---------|
| QACC NaN / 手臂坍塌 | IK 后清零臂关节速度 + `_sync_arm_ctrl()` 消除 actuator 与 OSC 冲突 |
| Arm 位置 actuator 与 OSC 冲突 | `_sync_arm_ctrl()` 将 ctrl 设为与 qpos 同步，产生零净力 |
| 夹爪 actuator 被 sync 干扰 | `_sync_arm_ctrl(include_gripper=False)` 默认不同步夹爪 |
| IK 导致手臂扭曲 | 增加关节限位钳位 + nullspace 向心力权重从 0.05→0.3 + 固定 home pose |
| 夹爪闭合不充分 | 两阶段闭合：渐进关闭 → 保压段（3×frame_skip 步保持 ctrl=0.95） |
| Jacobian 伪逆 IK 在奇异位形附近求解不稳定 | `solve_ik_numerical()`：新增基于 scipy Nelder-Mead 无梯度优化的数值 IK 求解器，直接最小化 EE 位置误差（目标函数 `‖EE_pos − target‖`），不依赖 Jacobian 求逆，对奇异位形更鲁棒。收敛容差 1e-4，最大 1000 次迭代，误差 > 1cm 时打印警告 |
| 直接设置 qpos 导致机械臂瞬跳和物理震荡 | `move_joints_smooth()`：新增五次多项式（smoothstep）关节空间平滑插值。插值公式 `α³(6α²−15α+10)` 保证起点和终点的速度/加速度均为零。逐帧强制设置 qpos + 清零臂关节速度 + 臂/夹爪速度钳制（max_vel=1.0），最后 `mj_forward` 确保物理状态一致 |
| Phase 2 粗定位后 EE 在水平方向漂移，下降时对不准物体顶部 | XY 归零防漂移：Phase 2 粗定位阶段将 EE 的 XY 直接对准 DQN 输出的目标物体 XY 坐标（`approach_eef = [target_x, target_y, target_z+0.20]`），下降到距桌面 15cm 的安全高度。Phase 3 精准下降前再通过动态 finger-to-EE 偏移测量补偿末端到手指的真实几何差，确保下降轨迹垂直对准物体，消除水平漂移 |
| 机械臂基座过高导致抓取范围不足、夹爪执行器刚度太大导致震荡 | 调整 `grasp.xml` 场景参数：机械臂基座高度从 Z=1.30 降至 Z=0.80，使桌面进入臂可达范围中心；夹爪 position actuator 的 kp 从 2000 降至 50、kv=5，大幅降低闭合刚度，消除夹爪高频抖动；`<option>` 增加 `tolerance="1e-10"` 约束数值误差，抑制仿真不稳定 |

### 6.2 抓取算法改进
| 改进 | 效果 |
|------|------|
| 动态 finger-to-EE 偏移测量 | 替代硬编码偏移，消除手指末端与实际抓取点的误差 |
| 三段式精准下降 | 减少物体被推走的概率 |
| 密集奖励设计 | 为 DQN 提供中间信号，加速收敛 |
| 接触力检测验证抓取 | 减少"物体被推起来但未真正抓住"的假阳性 |
| 指令引导探索 | 让初始阶段也能碰到物体，避免完全随机空白探索 |

### 6.3 代码质量
| 改进 | 效果 |
|------|------|
| 移除直接 qpos 操作 | 消除绕过物理引擎导致的 NaN 和不稳定 |
| 缓存 actuator ID | 避免每帧 `mj_name2id` 查询 |
| 严格命名空间隔离 | 防止 `eval()` 执行任意代码 |
| 清理 dead code | 移除重复赋值、未使用方法 |

---
## 七、当前训练表现与已知问题
### 7.1 训练现状
- **EPOCH**：100 轮（每轮最多 30 次抓取尝试）
- **当前成功率**：硬编码抓取测试成功率100%，DQN训练复杂场景下抓取成功率持续优化中
- **奖励范围**：成功 +100~120，失败 -3~+5（取决于密集奖励积累）

### 7.2 已知挑战
1. **视觉特征学习困难**：64×64 分辨率下物体较小，网络难以精确区分物体像素
2. **奖励信用分配**：抓取成功受多阶段影响（IK 精度、夹爪闭合时机、物体位姿），难以追溯单一因素
3. **MuJoCo 仿真稳定性**：position actuator 与直接力矩控制混用时需精细同步，否则出现 QACC NaN
4. **夹爪闭合力不足**：AG95 position actuator 的 kp=50 对大质量物体闭合力偏弱

### 7.3 后续改进方向
- **算法层**：集成 Prioritized Experience Replay、NoisyNet、n-step return
- **网络层**：增大输入分辨率、加入 ViT/Feature Pyramid 结构
- **仿真层**：引入域随机化（物体质量/摩擦/位置随机）、障碍物场景
- **控制层**：尝试力控夹爪代替位置控制夹爪、加入触觉反馈
- **工程层**：参数配置化（argparse/yaml）、模型 checkpoint 管理与评估

---
## 八、核心术语
| 术语 | 解释（本项目语境） |
|------|-------------------|
| DQN | 深度 Q 网络，输入 RGB-D 图像，输出 64×64=4096 个像素的抓取 Q 值 |
| 目标网络 (Target Net) | 参数每 100 步从在线网络拷贝，用于计算稳定 TD target |
| 经验回放 (Replay Buffer) | 存储 (state, action, reward, next_state)，随机采样打破时序相关 |
| 操作空间控制器 (OSC) | 基于质量矩阵和 Jacobian 的 6D 末端位姿跟踪控制器 |
| IK（逆运动学） | Jacobian 伪逆法 + 阻尼 + nullspace 向心，将末端目标位姿转为关节角度 |
| GRASP_DEPTH | 抓取时的额外下压深度（0.08m），确保夹爪在物体两侧 |
| LIFT_HEIGHT | 抓取成功后的抬起高度（0.25m） |
| TABLE_HEIGHT | 桌面中心高度（0.95m），用于 IK 目标的安全钳位 |
| QACC NaN | MuJoCo 加速度 NaN 警告，通常由 actuator 与直接力控制的冲突引起 |
| 指令引导探索 | ε-greedy 探索时不随机选像素，而是选实际物体位置的像素，加速学习 |
