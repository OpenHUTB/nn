# 快速开始

## 环境要求

| 依赖 | 版本要求 |
|------|----------|
| Python | 3.8+ |
| MuJoCo | 3.4.0+ |
| NumPy | 最新稳定版 |

## 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或单独安装：

```bash
pip install mujoco numpy
```

## 基本运行

进入模块目录，直接运行主程序：

```bash
cd src/mechanical_arm_grasping
python main.py
```

默认配置：抓取 **5** 次，**中速**，**中力度**。

## 命令行参数

```bash
python main.py [抓取次数] [速度模式] [最低成功率] [夹爪力度]
```

| 参数位置 | 参数名 | 说明 | 可选值 | 默认值 |
|----------|--------|------|--------|--------|
| 第1个 | 抓取次数 | 机械臂抓取的总轮数 | 任意正整数 | 5 |
| 第2个 | 速度模式 | 关节运动速度 | `slow` / `medium` / `fast` | `medium` |
| 第3个 | 最低成功率 | 成功率达到此阈值才返回成功 | 0~100 浮点数 | 0（不检查） |
| 第4个 | 夹爪力度 | 夹爪闭合力度 | `light` / `medium` / `heavy` | `medium` |

## 使用示例

```bash
# 默认配置：抓取5次，中速，中力度
python main.py

# 快速抓取10次，要求80%成功率，重力度
python main.py 10 fast 80 heavy

# 慢速抓取3次，轻力度
python main.py 3 slow 0 light

# 抓取20次，中速，要求90%成功率
python main.py 20 medium 90
```

## 运行效果

程序运行后会：

1. 打开 **MuJoCo 可视化窗口**，展示 3 自由度机械臂仿真场景
2. 控制台实时显示**进度条**、**力矩信息**、**抓取状态**
3. 每次抓取完成后输出**计时统计**
4. 所有抓取完成后输出**总成功率**、**总耗时**和**平均耗时**
5. 自动将日志保存到 `grasp_log.txt` 文件

## 文件结构

```
src/mechanical_arm_grasping/
├── README.md              # 模块说明
├── main.py                # 主程序入口
├── robot_arm.xml          # MuJoCo 机械臂模型定义
├── robot_arm_control.py   # 控制接口（预留扩展）
└── grasp_log.txt          # 运行日志（自动生成）
```
