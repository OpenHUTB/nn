# 多场景仿真与控制优化项目 
# 一、项目概述
本项目围绕机器人仿真、车辆仿真、工程化规范、测试体系搭建四大方向，完成 14 项关键 PR 修改。覆盖了 CARLA 多模态导航、MuJoCo 机器人仿真、机械臂控制、车辆竞速、AirSim 仿真测试等多场景，解决了硬编码、参数不适配、运动抖动、环境不兼容、测试体系缺失、行走轨迹不自然等问题，提升了代码鲁棒性、可维护性、可测试性与仿真效果的真实性。
# 二、 修改明细
## 2.1 机器人仿真优化（MuJoCo / 机械臂 ）
###  2.1.1 添加人形机器人行走轨迹特性，增强模拟逻辑：
新增generate_walking_trajectory(t, freq, amplitude)轨迹生成器函数，基于正弦函数生成关节运动轨迹，实现左腿与右腿 180 度相位差交替运动，膝盖跟随臀部协调摆动，产生自然迈步效果。同时优化相机视角，设置距离 3.0、仰角 - 20 度、方位角 45 度，提供清晰的行走状态观测视角，提升人形机器人行走仿真的真实性与可视化效果。

### 运动轨迹生成器
    def generate_walking_trajectory(t, freq=1.0, amplitude=15.0):
    """
    生成行走运动轨迹
    :param t: 当前时间
    :param freq: 行走频率
    :param amplitude: 关节运动幅度（度）
    :return: 四个关节的目标角度 [l_hip, r_hip, l_knee, r_knee]
    """
    # 将角度转换为弧度
    amp_rad = np.deg2rad(amplitude)
    
    # 左腿和右腿交替运动
    l_hip_angle = amp_rad * np.sin(2 * np.pi * freq * t)
    r_hip_angle = -amp_rad * np.sin(2 * np.pi * freq * t)
    
    # 膝盖跟随臀部运动，产生自然的迈步效果
    l_knee_angle = amp_rad * 0.8 * np.cos(2 * np.pi * freq * t)
    r_knee_angle = -amp_rad * 0.8 * np.cos(2 * np.pi * freq * t)
    
    return np.array([l_hip_angle, r_hip_angle, l_knee_angle, r_knee_angle])

 ### 设置相机视角
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
            viewer.cam.lookat = [0, 0, 0.8]
 ### 使用预定义的运动轨迹
                trajectory = generate_walking_trajectory(data.time, freq=1.5, amplitude=20.0)
                data.ctrl[:] = trajectory
                
  <img width="1727" height="1121" alt="9924763740b2e1d8cf69ffac65ff0432" src="https://github.com/user-attachments/assets/965450fd-87f2-4a6c-946f-c5e39cb833f1" />



### 2.1.2修改模型 XML 关节旋转轴参数：
调整 MuJoCo 模型 XML 文件中 4 个关节的旋转轴配置，将原本的 axis="0 1 0" 统一改为 axis="1 0 0"，解决了关节旋转方向不统一导致的运动轨迹偏差、物理仿真不一致问题。
 <joint name="l_hip" type="hinge" axis="1 0 0" range="-30 30"/>

<img width="1728" height="1122" alt="2ad6d5024ea460e6e10ea035faaa6bf7" src="https://github.com/user-attachments/assets/ca0a953f-c26a-470c-83ab-4002b07c8e15" />



### 2.1.3 优化观测维度补零逻辑：
修改 TransformObservation.observation() 方法中观测维度补零逻辑，将硬编码的 np.zeros(332) 改为动态计算补零长度，兼容不同版本 MuJoCo 环境。
 ### 显式定义新的观察空间维度 
     def __init__(self, env):
        super(TransformObservation, self).__init__(env)
        # 显式定义新的观察空间维度 (44 + 332 = 376)
        self.target_obs_dim = 376  
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.target_obs_dim,), 
            dtype=np.float32
        )


### 2.1.4 提升机械臂 PD 控制稳定性：
给机械臂的电机力矩加上限制：肩关节力矩限制在±500、肘关节力矩限制在±800。避免动作太大导致疯狂抖动，控制更平稳、更精准。
 ### 肩关节控制
            shoulder_error = shoulder_target - data.qpos[shoulder_joint_id]
            shoulder_vel = data.qvel[shoulder_joint_id]
            data.ctrl[shoulder_act_id] = kp * shoulder_error - kd * shoulder_vel

            data.ctrl[shoulder_act_id] = np.clip(data.ctrl[shoulder_act_id], -500, 500) 
 ### 肘关节控制
            elbow_error = elbow_target - data.qpos[elbow_joint_id]
            elbow_vel = data.qvel[elbow_joint_id]
            data.ctrl[elbow_act_id] = kp * elbow_error - kd * elbow_vel
            data.ctrl[elbow_act_id] = np.clip(data.ctrl[elbow_act_id], -800, 800)


<img width="1727" height="1121" alt="dc1edb16f9a672397cdf42a7d1755cac" src="https://github.com/user-attachments/assets/fd62d840-7cb4-421d-8746-243a4ad5d872" />


## 2.2 车辆与仿真环境优化（CARLA/AirSim）
### 2.2.1 改进车速计算方式：
将车速直接赋值逻辑替换为一阶低通滤波公式（speed = 0.8×旧速度 + 0.2×新速度），解决车速数据跳变、面板显示抖动、车辆控制不稳定问题，实现车速值平滑无跳变，油门或刹车控制响应更线性。
   ### 专用速度传感器
                velocity = data.velocity
                self.vehicle_speed = 0.8 * self.vehicle_speed + 0.2 * (math.sqrt(velocity.x ** 2 + velocity.y ** 2) * 3.6)

### 2.2.2 增强 CARLA 强雨天仿真效果：
调整 CARLA 仿真环境的 WeatherParameters 配置，新增 rain_intensity=100.0 参数，解决雨天效果不明显问题，让仿真环境更真实，增强雨天视觉效果，满足恶劣天气导航测试需求。
### 设置强雨天天气
    weather = carla.WeatherParameters（
    cloudiness=90.0,
    precipitation=90.0,
    precipitation_deposits=90.0,
    wind_intensity=20.0,
    wetness=90.0,
    rain_intensity=100.0 
    ）

### 2.2.3 修复车辆竞速代码路径问题：
修复代码中 Python 路径硬编码问题，改为自动识别路径，让代码在 Windows、Linux 都能跑，不会报找不到文件的错。将原固定绝对路径/home/rosindustrial/catkin_ws/src/car_racing_ros/your_python_code改为基于当前脚本位置自动拼接的动态相对路径。新增os模块导入，配合路径动态解析。
###
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../your_python_code"))

### 2.2.4 修复车辆控制服务响应逻辑：
删除 src/button_control_vehicle/main.py 中对 response 对象不存在的 success 字段的赋值语句（response.success = False），解决服务调用时的 AttributeError 报错、响应逻辑中断问题，保障服务响应流程无报错，逻辑执行完整。

### 2.2.5 优化 AirSim 测试脚本：
对 AirSim 测试脚本进行结构优化与增加异常处理，将原始代码拆分为多个函数;增加异常捕获机制：在创建 AirSim 客户端时增加 try-except;增加 main 入口函数;原始脚本在 AirSim 未运行时可能抛出异常，导致程序直接终止,所以通过增加异常处理和模块化结构。
##
    def print_environment_info():
        """
        Print Python and AirSim version information.
        """
        print("Python 版本:", sys.version)

        try:
            print("AirSim 版本:", airsim.__version__)
            print("AirSim 导入成功！")
        except AttributeError:
            print("无法获取 AirSim 版本信息")


    def create_client_safe():
        """
        Safely create an AirSim client.

        Returns:
            client or None
        """
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            print("客户端创建成功")
            return client
        except Exception as e:
            print("客户端创建失败:", e)
            return None


    def main():
        """
        Main test entry.
        """
        print_environment_info()

        client = create_client_safe()

        if client is None:
            print("AirSim 未运行，跳过后续测试")
        else:
            print("AirSim 连接正常")


    if __name__ == "__main__":
        main() 

## 2.3 工程化规范与测试体系
### 2.3.1完善模块依赖配置：
为 src/humanoid_simulation_environment_establish/ 模块梳理依赖并生成标准化 requirements.txt，解决环境部署时依赖缺失、版本不兼容问题，支持一键安装依赖，快速复现模块运行环境。
#
    #基础数据处理
    pandas==2.1.0
    numpy==1.25.2
    #Web框架
    flask==2.3.3
    #数据库
    sqlalchemy==2.0.20
    pymysql==1.1.0
    #爬虫/请求
    requests==2.31.0
    beautifulsoup4==4.12.2
    #可视化
    matplotlib==3.7.2
    seaborn==0.12.2
    #其他
    python-dotenv==1.0.0
    openpyxl==3.1.2  # 处理Excel

### 2.3.2 适配 CARLA 多模态导航依赖：
为 carla_multimodal_navigator 项目新增适配多模态导航场景的 requirements.txt，包含carla、numpy、gym 等依赖包及版本，解决多模态导航功能部署时的依赖冲突、版本适配问题。
#
    # CARLA仿真环境交互
    carla>=0.9.15
    # 基础数据处理
    numpy>=1.24.0
    pandas>=2.0.0
    # 多模态感知/可视化
    opencv-python>=4.8.0
    matplotlib>=3.7.0
    pillow>=10.0.0
    # 路径规划/导航
    networkx>=3.2.0
    scipy>=1.11.0
    # 网络通信（wxapp交互/数据传输）
    requests>=2.31.0
    websockets>=12.0
    # 配置管理
    python-dotenv>=1.0.0
    # 日志/进度
    tqdm>=4.65.0
    logging>=0.4.9.6
    # 开发测试
    pytest>=7.4.0
    flake8>=6.0.0

### 2.3.3 更新 MuJoCo 下载命令：
更新 MuJoCo 的下载命令，改用 Invoke-WebReques。原WebClient在新版 Windows 或 网络环境下易因 TLS 协议不兼容导致下载失败，替换为Invoke-WebRequest并强制 TLS1.2，大幅提升下载成功率
#
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MUJOCO_REPO%' -OutFile '%MUJOCO_ZIP%' -UseBasicParsing"

### 2.3.4 扩展.gitignore 规则：
在 .gitignore 文件末尾新增仿真日志、临时解压文件、IDE 配置文件、虚拟环境目录等常见开发工具忽略规则，有助于解决无关文件提交至代码仓库导致的仓库体积膨胀、代码冲突问题，保持仓库文件整洁，仅保留核心代码与配置，减少无效提交。
#
    .claude/
    # Pytest
    .pytest_cache/

    # Coverage
    .coverage
    htmlcov/

    # VSCode
    .vscode/

    # JetBrains IDE
    .idea/

    # Temporary files
    *.tmp
    *.temp

    # Environment files
    .env.local
    .env.*.local

    # Jupyter
    .ipynb_checkpoints/

### 2.3.5搭建 pytest 测试基础：
添加 pytest 配置及基础导入测试，自动将项目根目录加入路径以修复找不到模块的报错，新增 src/hooks 模块导入测试用例并配置 pytest 基础运行规则，解决 pytest 运行时模块导入失败、无基础测试验证代码完整性的问题。
### tests/test_imports.py
    """
    Basic import tests for nn project.
    """

    import importlib
    import os
    import sys

    # Add project root to Python path
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


    def test_import_hooks():
        """
        Test importing hooks module.
        """

        module = importlib.import_module("hooks")

        assert module is not None


    def test_project_root_exists():
        """
        Ensure project root exists.
        """

        assert os.path.exists(PROJECT_ROOT)

### pytest.ini
    # Ignore heavy integration tests
    norecursedirs =
        src/airsim_control
        src/automatic_drive_deep_learning
        src/autonomous_vehicle_navigation_using_dl
        src/object_tracking_planning
        src/yolo12_object_detection
        src/mechanical_arm
        src/humanoid_balance
        src/lane_path_detection
        src/uav_navigation

    # Only run lightweight tests
    testpaths =
        tests


# 三、核心技术
## 3.1 鲁棒性提升
减少硬编码，改为动态计算，让代码适配不同版本、不同环境的参数变化；通过一阶低通滤波（车速）、力矩限幅（机械臂）等手段优化控制逻辑，抑制高频抖动，提升控制平滑性；AirSim 测试脚本新增多场景异常捕获机制，避免测试流程直接崩溃。
## 3.2 工程化规范
为核心模块与项目标准化管理依赖，新增 requirements.txt 统一版本，降低环境部署成本；扩展.gitignore 规则过滤无关文件，实现代码仓库轻量化管理；适配 Windows/Linux 双系统路径规则、不同版本差异，提升跨环境兼容性。
## 3.3 建成测试体系
完成 pytest 基础配置与导入测试，解决模块导入核心问题，为单元测试、集成测试拓展提供基础；完成 AirSim 测试脚本模块化重构，提升脚本可维护性与扩展能力，支撑后续功能迭代验证。

### Pytest 架构设计

<img width="2130" height="1160" alt="image" src="https://github.com/user-attachments/assets/8a4fe5e2-9867-4981-a646-492228848b80" />

## 3.4 仿真效果优化升级
新增人形机器人行走轨迹生成器，基于正弦函数实现交替迈步动作，模拟自然行走步态；统一 MuJoCo 机器人关节旋转轴方向，提升运动轨迹一致性；优化相机视角配置，提供清晰的行走状态观测视角；增强 CARLA 强雨天视觉效果，提升多模态导航场景的仿真真实性。

### 核心技术总览：

<img width="2136" height="177" alt="image" src="https://github.com/user-attachments/assets/20a15834-ddef-48d8-aa9b-218ff17d8013" />

# 四、后续规划
## 控制算法迭代：
基于机械臂 PD 控制优化基础，引入模型预测控制算法，进一步提升复杂轨迹跟踪精度；针对人形机器人行走控制，引入强化学习算法优化步态稳定性，实现长距离稳定行走。
## 仿真场景扩展：
为 CARLA 新增雾天、雪天等极端天气参数，为 AirSim 新增多无人机协同测试场景，为人形机器人仿真添加地面摩擦等真实物理特性，丰富仿真测试。
## 自动化测试拓展：
基于现有 pytest 基础，新增机械臂控制、CARLA 导航、人形机器人行走控制等核心功能的单元测试、集成测试用例，实现核心功能自动化验证。
## 仿真性能优化：
针对 MuJoCo、CARLA、AirSim 仿真帧率，优化计算逻辑，提升大规模仿真场景的运行效率。


# 五、总结
本次项目覆盖机器人、车辆、AirSim 仿真测试等场景，让机器人能更平稳行走、让车辆仿真更真实、让代码更规范、让环境更容易部署、让测试体系更丰富。所有 PR 均已通过审核并合并，核心功能验证通过，项目可以稳定运行，为后续更复杂的研究打下了基础。











