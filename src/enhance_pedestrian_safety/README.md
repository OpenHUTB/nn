# CVIPS - Cooperative Vehicle-Infrastructure Perception System

基于CARLA仿真平台的多传感器协同感知数据采集系统，支持V2X通信、多车辆协同、数据增强与验证。


## 项目简介
CVIPS是一个用于自动驾驶研究的多传感器数据采集系统，在CARLA仿真环境中生成高质量的协同感知数据集。
系统支持多车辆、多传感器协同工作，包含完整的V2X通信模拟、数据增强、验证和分析工具链。


## 核心功能
1. **多传感器数据采集**
   - 摄像头系统
   - 激光雷达
   - 传感器融合

2. **协同感知与V2X通信**
   - 多车辆协同
   - V2X消息系统
   - 共享感知

3. **智能交通场景生成**
   - 多样化场景
   - 交通流控制
   - 特殊事件

4. **数据处理与增强**
   - 图像处理
   - LiDAR处理
   - 数据增强

5. **质量保证与验证**
   - 数据验证器
   - 数据分析器
   - 性能监控

6. **多格式输出支持**
   - 标准格式
   - V2XFormer格式
   - KITTI格式


## 环境要求
- Python 3.7
- CARLA 0.9.14
- 内存：≥ 8GB（建议16GB）
- 存储：≥ 50GB可用空间


## 安装步骤
1. **克隆仓库**
   ```bash
   git clone https://github.com/Z-w-7799/nn.git

## 安装步骤
1. **克隆仓库**
   ```bash
   git clone https://github.com/Z-w-7799/nn.git
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置CARLA路径**
   - 方式1：设置环境变量
     ```bash
     export CARLA_PYTHON_PATH=/path/to/carla/PythonAPI/carla/dist
     ```
   - 方式2：使用命令行参数
     ```bash
     python main.py --carla-path /path/to/carla
     ```


## ROS 1 Kinetic封装运行步骤
该项目已封装为ROS 1 Kinetic包，运行流程：
1. 启动ROS核心
   ```bash
   roscore
   ```
2. 运行CVIPS节点
   ```bash
   rosrun carla_sensor main.py
   ```




