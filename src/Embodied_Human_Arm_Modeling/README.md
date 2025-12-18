# 仿人手臂 MuJoCo 仿真项目

## 项目结构
所有文件均位于 `src/Embodied_Human_Arm_Modeling/` 目录内。

## 可用文件

### 模型文件
1. `arm_model.xml` - 基础手臂模型（肩、肘、腕关节）
2. simulate_full_hand.xml` - 增强版模型（含手掌、手指、传感器）

### 仿真脚本
1. `simulate_arm.py` - 基础仿真脚本
2. simulate_full_hand.py - 增强版脚本（多模式控制、数据记录）


## 运行方式

### 方式1：使用启动器（推荐）
```bash
cd src/Embodied_Human_Arm_Modeling
python simulate_full_hand.py



