# 1. 导入模块（新增 cv2 和 os 用于图像保存）
import torch
import time
import numpy as np
import cv2  # 用于图像保存
import os   # 用于创建目录
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from models.dqn_agent import DQNAgent
from envs.carla_environment import CarlaEnvironment
import carla

# 2. 定义 IntegratedSystem 类
class IntegratedSystem:
    def __init__(self, device='cpu'):
        self.device = device
        self.perception = PerceptionModule().to(self.device)
        # 补充 input_dims 参数（与感知模块输出维度匹配）
        self.attention = CrossDomainAttention(
            num_blocks=6,
            input_dims=[256, 256, 6, 256, 256]
        ).to(self.device)
        self.decision = DecisionModule().to(self.device)

    def forward(self, image, lidar_data, imu_data):
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        policy, value = self.decision(fused_features)
        return policy, value

# 3. 定义传感器数据适配函数（桥接CARLA和模型）
def adapt_sensor_data(env, system):
    """
    从CARLA环境获取真实图像，转换为模型输入格式
    （LiDAR/IMU暂用模拟数据，后续可扩展为真实传感器）
    """
    # 1. 获取CARLA真实相机图像 (128, 128, 3) → 适配模型输入
    raw_image = env.get_observation()  # 真实RGB图像
    # 转换格式：HWC(128,128,3) → CHW(3,128,128) → 缩放至256×256（匹配模型输入）
    image = torch.FloatTensor(raw_image).permute(2, 0, 1).unsqueeze(0) / 255.0  # 归一化到[0,1]
    image = torch.nn.functional.interpolate(image, size=(256, 256), mode='bilinear')  # 缩放至256×256
    image = image.to(system.device)
    
    # 2. 模拟LiDAR数据（后续需添加CARLA LiDAR传感器）
    lidar_data = torch.randn(1, 256, 256).unsqueeze(0).to(system.device)
    
    # 3. 模拟IMU数据（后续需添加CARLA IMU传感器）
    imu_data = torch.randn(1, 6).to(system.device)
    
    return image, lidar_data, imu_data, raw_image  # 新增返回原始图像

# 4. 定义图像保存函数
def save_camera_image(raw_image, step, save_dir="carla_camera_images"):
    """
    保存CARLA相机原始图像到本地
    :param raw_image: 原始RGB图像 (128, 128, 3)
    :param step: 仿真步数
    :param save_dir: 保存目录
    """
    # 创建保存目录（不存在则创建）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 转换RGB格式为OpenCV的BGR格式（OpenCV默认BGR）
    image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
    
    # 定义保存路径
    save_path = os.path.join(save_dir, f"camera_step_{step:03d}.png")  # 03d 补零，如 001, 002
    
    # 保存图像
    cv2.imwrite(save_path, image_bgr)
    
    # 打印保存日志
    print(f"📸 第 {step} 步相机图像已保存：{save_path}")

# 5. 定义 run_simulation 函数
def run_simulation():
    # 初始化CARLA环境
    env = CarlaEnvironment()
    # 关键：调用reset()生成车辆和相机，初始化self.vehicle
    env.reset()
    
    # 校验车辆是否生成成功
    if env.vehicle is None:
        raise RuntimeError("❌ 车辆生成失败！请检查：\n1. CARLA模拟器是否启动\n2. 端口是否为2000\n3. 地图是否加载完成")
    
    # 初始化集成系统
    system = IntegratedSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("✅ 仿真开始，运行100步...")
    # 控制保存频率：比如每5步保存一张，或只保存前10张，避免文件过多
    save_frequency = 5  # 每5步保存一次
    max_save_images = 10  # 最多保存10张图像
    
    saved_count = 0
    for step in range(100):
        try:
            # 获取适配后的传感器数据 + 原始图像
            image, lidar_data, imu_data, raw_image = adapt_sensor_data(env, system)
            
            # 保存相机图像（按频率保存，且不超过最大数量）
            if (step + 1) % save_frequency == 0 and saved_count < max_save_images:
                save_camera_image(raw_image, step + 1)
                saved_count += 1
            
            # 前向传播得到策略
            policy, value = system.forward(image, lidar_data, imu_data)
            
            # 转换为CARLA控制信号（限制范围避免异常）
            throttle = float(torch.clamp(policy[0][0], 0, 1))  # 油门范围[0,1]
            steer = float(torch.clamp(policy[0][1], -1, 1))    # 转向范围[-1,1]
            control = carla.VehicleControl(throttle=throttle, steer=steer)
            
            # 应用控制指令到车辆
            env.vehicle.apply_control(control)
            
            # 打印运行日志（方便调试）
            print(f"第 {step+1:3d} 步 | 油门：{throttle:.2f} | 转向：{steer:.2f} | 相机图像像素范围：{raw_image.min()}~{raw_image.max()}")
            
            time.sleep(0.1)  # 模拟时间间隔
            
        except Exception as e:
            print(f"❌ 第 {step+1} 步出错：{str(e)}")
            break
    
    # 仿真结束，清理环境
    env.close()
    print("✅ 仿真结束，环境已清理")
    print(f"📂 共保存 {saved_count} 张相机图像到：carla_camera_images/ 目录")

# 6. 程序入口（放在最后）
if __name__ == "__main__":
    # 检查OpenCV是否安装
    try:
        import cv2
    except ImportError:
        print("❌ 未安装OpenCV，无法保存图像！请执行：pip install opencv-python")
        exit(1)
    
    run_simulation()