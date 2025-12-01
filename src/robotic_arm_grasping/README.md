# Robotic Arm Grasping

## 项目简介
本项目使用 MuJoCo 物理引擎实现机械臂抓取系统的仿真与优化。核心目标是利用 MuJoCo 强大的动力学建模能力，精准模拟机械臂、目标物体的物理交互（如夹持力、摩擦力、碰撞响应），完成从目标感知、抓取规划到动作执行的全流程闭环，并通过算法迭代提升抓取稳定性与成功率。

## 环境配置
### 第一步：安装 MuJoCo 引擎
参考 [MuJoCo 官方文档](https://mujoco.org/book/installation.html) 完成安装：
1. 下载 MuJoCo 2.3.7+ 压缩包，解压至对应路径：
   - Linux/Mac：`~/.mujoco/mujoco-2.3.7`
   - Windows：`C:\Users\YourName\.mujoco\mujoco-2.3.7`
2. 配置环境变量（可选，用于命令行直接调用 MuJoCo 引擎）

### 第二步：安装项目依赖
```bash
# 克隆仓库
git clone https://github.com/your-username/mujoco-robotic-grasping.git
cd mujoco-robotic-grasping

# 安装 Python 依赖
pip install -r requirements.txt

参考资料
MuJoCo 官方文档：核心 API 说明、模型导入规范、物理参数配置指南
MuJoCo Python 接口教程：Python 调用示例、仿真控制细节****