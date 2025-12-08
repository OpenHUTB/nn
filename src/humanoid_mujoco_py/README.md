# 人形机器人
# MuJoCo Python 模拟示例

这是一个使用MuJoCo物理引擎的Python示例项目，展示了如何加载模型、运行物理模拟并实时可视化结果。示例模拟了一个简单场景：一个盒子在重力作用下自由下落到地面。

## 项目结构
humanoid_mujoco_py/
├─ hello.xml          # MJCF模型文件，定义物理场景
├─ main.py            # Python主程序，运行模拟
└─ README.md          # 项目说明文档
## 环境要求

- Python 3.8 或更高版本
- MuJoCo 2.3.0 或更高版本
- 操作系统：Windows、macOS 或 Linux

## 安装步骤

1. **克隆或下载项目**

   ```bash
   git clone <仓库地址>
   cd mujoco_py_demo
   ```

2. **创建并激活虚拟环境（推荐）**

   ```bash
   # 创建虚拟环境
   python -m venv mujoco_env
   
   # 激活虚拟环境
   # Windows:
   mujoco_env\Scripts\activate
   # macOS/Linux:
   source mujoco_env/bin/activate
   ```

3. **安装依赖包**

   ```bash
   # 安装MuJoCo Python绑定
   pip install mujoco
   
   # 如需增强可视化功能（可选）
   pip install mujoco-viewer
   ```

## 使用方法

1. 确保`hello.xml`和`demo.py`在同一目录下

2. 运行模拟程序

   ```bash
   python demo.py
   ```

3. 程序将：
   - 加载`hello.xml`中定义的物理模型
   - 启动可视化窗口显示模拟过程
   - 在终端输出盒子的实时位置信息
   - 模拟持续10秒后自动结束

## 交互操作

在可视化窗口中，可以使用以下鼠标操作：
- 左键拖动：旋转视角
- 右键拖动：平移视角
- 滚轮：缩放视图
- Shift+左键：拖动场景中的物体

## 代码说明

- `hello.xml`：使用MJCF（MuJoCo XML格式）定义物理场景，包含：
  - 一个地面平面
  - 一个光源
  - 一个带自由关节的盒子（可在6个自由度上运动）

- `demo.py`：Python程序实现以下功能：
  - 加载MJCF模型
  - 初始化物理模拟
  - 启动可视化界面
  - 运行模拟循环并输出关键数据

## 扩展方向

1. **修改模型**：编辑`hello.xml`更改场景元素，如：
   - 添加更多物体
   - 改变物体形状、质量或颜色
   - 添加关节约束

2. **增强模拟**：修改`demo.py`实现：
   - 施加外力或扭矩
   - 改变重力方向或大小
   - 实现简单控制算法

3. **数据记录**：添加代码记录模拟数据，如：
   - 物体轨迹
   - 接触力
   - 速度和加速度变化

## 参考资料

- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)
- [MuJoCo GitHub 仓库](https://github.com/google-deepmind/mujoco)
- [MJCF 模型格式参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
