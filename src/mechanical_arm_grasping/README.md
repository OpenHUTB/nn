# mechanical_arm_grasping 机械臂抓取仿真

## 一、项目理解与功能介绍
本模块基于 MuJoCo 物理引擎实现机械臂抓取仿真，是机器人学、运动控制、物理仿真的入门实践项目。
通过该项目可以理解机械臂模型构建、物理环境交互、关节控制逻辑以及抓取任务的基本实现流程，适合用于课程实验、算法验证与学习演示。

主要功能：
- 加载机械臂与抓取场景模型
- 实现物理引擎驱动的动态仿真
- 提供可视化界面观察机械臂运动
- 支持关节控制与抓取动作调试
  
## 二、运行环境
- Python >= 3.8
- mujoco >= 3.0
- numpy
- matplotlib
  
## 三、安装依赖
```bash
pip install mujoco numpy matplotlib
```
## 四、运行方式
python main.py

## 五、文件说明
main.py：仿真主程序，负责模型加载、物理步进、画面渲染
scene.xml：MuJoCo 场景与机械臂模型配置文件
README.md：项目说明文档

## 六、核心运行流程
加载 XML 模型文件
初始化物理引擎与渲染器
进入仿真循环更新状态
实时渲染机械臂与物体
支持关节控制与抓取交互

## 七、常见问题与解决方案
1. ModuleNotFoundError: No module named 'mujoco'
解决：执行 pip install mujoco 安装依赖库。
2. 报错找不到 scene.xml
解决：确保 main.py 和 scene.xml 在同一目录下。
3. 仿真窗口无法弹出 / 黑屏
解决：更新显卡驱动，确保支持 OpenGL。
4. 机械臂不运动
解决：添加 data.ctrl 关节控制代码。
