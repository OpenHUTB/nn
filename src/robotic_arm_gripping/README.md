# MuJoCo 机械臂抓取仿真

这是一个基于 MuJoCo 引擎实现的三指机械臂抓取小球的仿真项目，包含完整的机械臂模型定义和动作控制逻辑，能够实现机械臂前伸、检测接触、闭合手指抓取、举起目标物体的完整流程。

## 项目介绍
- 核心功能：两指机械臂对红色小球的自动抓取、举升、放下、复位全流程
- 控制模式：默认自动模式，支持键盘手动切换控制（肩部 / 肘部 / 手指开合）
- 物理特性：自定义接触摩擦、关节阻尼、碰撞检测，保证抓取稳定性
- 交互体验：实时状态反馈、可视化视角优化、非阻塞键盘监听

# 环境要求
## 软件依赖
- 核心依赖
```bash
pip install mujoco>=3.1.0 numpy
pip install mujoco numpy keyboard
```

# 系统兼容

- Windows/Linux/macOS
- Python 3.8+
- mujoco >= 2.3.0
- numpy >= 1.20.0
- keyboard >= 0.13.5

注意：keyboard 库可能需要管理员权限运行（Windows）或 root 权限（Linux/Mac）