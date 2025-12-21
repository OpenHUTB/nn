# 多模式机器人导航系统
## 项目概述
你要开发的多模式机器人导航系统，核心是融合IMU、摄像头、激光雷达等多传感器数据，通过感知、跨域注意力、决策三大模块协作，最终输出机器人的行动策略，该系统基于CARLA 0.9.11模拟器开发，并采用深度学习技术完成训练与测试。

## 环境设置
### 1. 基础环境要求
- Python版本：3.7及以上
- PyTorch版本：1.7及以上（建议匹配CUDA版本以提升训练效率）
- CARLA模拟器版本：0.9.11（需保证客户端与服务端版本一致）

### 2. 快速配置步骤
```bash
# 1. 创建并激活虚拟环境
conda create -n robot_nav python=3.7 -y
conda activate robot_nav

# 2. 安装PyTorch（以CUDA 11.0为例，可按需调整）
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# 3. 安装CARLA 0.9.11
pip install carla==0.9.11

# 4. 补充常用依赖
pip install numpy opencv-python scipy matplotlib
```

### 3. 环境验证
```bash
# 验证Python版本
python --version

# 验证PyTorch
python -c "import torch; print(f'PyTorch版本：{torch.__version__}，CUDA可用：{torch.cuda.is_available()}')"

# 验证CARLA（需先启动CARLA 0.9.11服务端）
python -c "import carla; client=carla.Client('localhost',2000); client.set_timeout(5.0); print('CARLA连接成功' if client.get_world() else 'CARLA连接失败')"
```

## 总结
1. 环境核心要求：Python 3.7+、PyTorch 1.7+、CARLA 0.9.11，版本匹配是关键。
2. 配置流程：先创建虚拟环境，再依次安装PyTorch、CARLA及辅助依赖，最后验证环境可用性。
3. 系统核心逻辑：通过三大模块融合多传感器数据，基于CARLA模拟器实现机器人导航策略输出。