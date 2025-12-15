# Generate skeleton arm video

仓库根目录包含一个小脚本用于渲染并导出一个简单的 3D 骨骼手臂动画为 MP4 文件。

- 脚本位置：`generate_skeleton_video.py`（仓库根目录 `d:\nn`）
- 默认输出：`skeleton_arm.mp4`（保存到仓库根目录）

快速运行（Windows PowerShell，从仓库根目录运行）：

```powershell
& D:/nn/.venv/Scripts/Activate.ps1
python d:\nn\generate_skeleton_video.py
```

可选参数：

- `--out <filename>` : 输出文件名（默认 `skeleton_arm.mp4`）
- `--frames <n>` : 帧数（默认 240）
- `--fps <n>` : 帧率（默认 30）

注意：

- 该脚本使用无窗口 Matplotlib 后端，运行前请确保已安装：`numpy matplotlib imageio imageio-ffmpeg`。
- 如果播放器未自动打开，可在 PowerShell 中运行 `start skeleton_arm.mp4` 来查看生成的视频。
user-in-the-box-simulator
基于Gymnasium和MuJoCo的仿真器，集成生物力学模型、感知模块与强化学习任务，支持分层强化学习与可视化渲染

一、环境准备
1.虚拟环境创建与激活
# 切换到项目根目录
cd C:\Users\86186\user-in-the-box

# 创建Python 3.9虚拟环境（兼容性最优）
python -m venv venv --python=3.9

# 激活虚拟环境（Windows PowerShell）
\venv\Scripts\Activate.ps1
2.依赖安装 使用国内镜像源加速安装：
# 核心依赖
pip install gymnasium==1.2.1 mujoco==2.3.5 stable-baselines3==2.2.1 pygame==2.5.2 opencv-python==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 辅助依赖
pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.4 ruamel.yaml==0.18.6 certifi -i https://pypi.tuna.tsinghua.edu.cn/simple

二、核心文件说明

1.simulator.py（仿真器核心） 
功能：继承 gym.Env，实现仿真环境的初始化、步骤推进（step）、环境重置（reset）和可视化渲染（render），集成生物力学模型、感知模块和任务逻辑。 运行方式：需通过调用脚本（如 test_simulator.py）运行，示例见 “三、运行步骤”。
2.main.py（辅助脚本） 功能：基于 certifi 查询 CA 证书信息（路径或内容），用于验证网络请求的安全性。 
<<<<<<< HEAD

3.MoblArmsIndex.py:生物力学模型

=======
<<<<<<< HEAD
3.MoblArmsIndex.py:生物力学模型
=======
>>>>>>> f5c965a634bc42a4261d8907d2ed5530a8647006
>>>>>>> eeafa3d55c53ad4469c54fd3097a4ee7343b419c
运行方式：
# 查看证书路径
python main.py

# 查看证书内容
python main.py -c
<<<<<<< HEAD
=======
<<<<<<< HEAD
三、运行步骤  
=======
三、运行步骤
>>>>>>> f5c965a634bc42a4261d8907d2ed5530a8647006
>>>>>>> eeafa3d55c53ad4469c54fd3097a4ee7343b419c

三、运行步骤  
仿真器运行（simulator.py）
步骤 1：运行脚本 test_simulator.py

步骤 2：执行脚本

python test_simulator.py
此时会弹出 Pygame 窗口，展示仿真过程（如机械臂运动、感知模块渲染）。 2. 辅助脚本运行（main.py） 在终端执行以下命令，查看证书信息：

# 查看证书路径
python main.py

# 查看证书内容
python main.py -c

四、依赖清单
|库名称	|版本	|用途|
|------|-------|----|
|gymnasium|	1.2.1	|强化学习环境接口|
|mujoco|2.3.5|物理仿真引擎|
|stable-baselines3|2.2.1|强化学习算法库|
|pygame	|2.5.2|可视化渲染|
|opencv-python|4.9.0.80|图像感知处理|
|numpy|	1.26.4|	数值计算|
|scipy|	1.11.4|科学计算|
|matplotlib|3.8.4|数据可视化|
|ruamel.yaml|0.18.6	|配置文件解析|
|certifi	|2025.10.10	|CA 证书管理|