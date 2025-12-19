import torch
import time
import logging
import carla
from typing import Tuple, Optional
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from models.dqn_agent import DQNAgent
from envs.carla_environment import CarlaEnvironment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 系统配置
CONFIG = {
    "simulation_steps": 100,
    "step_delay": 0.1,
    "carla_host": "localhost",
    "carla_port": 2000,
    "image_size": (3, 256, 256),
    "lidar_size": (1, 256, 256),
    "imu_size": (1, 6),
    "max_throttle": 1.0,
    "max_steer": 1.0
}


class IntegratedSystem:
    """
    自动驾驶集成系统，整合感知、注意力和决策模块
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.logger = logging.getLogger(self.__class__.__name__)

        # 初始化各个模块
        try:
            self.perception = PerceptionModule().to(self.device)
            self.attention = CrossDomainAttention(num_blocks=6).to(self.device)
            self.decision = DecisionModule().to(self.device)
            self.logger.info(f"成功初始化所有模块，使用设备: {self.device}")
        except Exception as e:
            self.logger.error(f"模块初始化失败: {e}")
            raise

    def forward(self, image: torch.Tensor, lidar_data: torch.Tensor, imu_data: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        前向传播，处理输入数据并输出策略和价值

        Args:
            image: 图像数据 [B, 3, H, W]
            lidar_data: LiDAR数据 [B, 1, H, W]
            imu_data: IMU数据 [B, 6]

        Returns:
            policy: 决策策略 [B, num_actions]
            value: 状态价值 [B, 1]
        """
        try:
            # 感知模块处理
            scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)

            # 跨域注意力融合特征
            fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)

            # 决策模块输出
            policy, value = self.decision(fused_features)

            return policy, value
        except Exception as e:
            self.logger.error(f"前向传播过程出错: {e}")
            raise


def init_carla_environment(host: str = "localhost", port: int = 2000) -> Optional[CarlaEnvironment]:
    """
    初始化CARLA环境

    Args:
        host: CARLA服务器地址
        port: CARLA服务器端口

    Returns:
        初始化后的CarlaEnvironment实例
    """
    try:
        env = CarlaEnvironment(host=host, port=port)
        # 等待环境加载完成
        time.sleep(2.0)
        logger.info(f"成功连接到CARLA服务器 {host}:{port}")

        # 检查车辆是否正确初始化
        if hasattr(env, 'vehicle') and env.vehicle is not None:
            logger.info("CARLA车辆已成功初始化")
        else:
            logger.error("CARLA车辆初始化失败")
            return None

        return env
    except Exception as e:
        logger.error(f"CARLA环境初始化失败: {e}")
        return None


def clamp_control_values(throttle: float, steer: float) -> Tuple[float, float]:
    """
    限制控制信号在合法范围内

    Args:
        throttle: 油门值
        steer: 转向值

    Returns:
        限制后的油门和转向值
    """
    # 限制油门在 0-1 之间
    throttle_clamped = max(0.0, min(CONFIG["max_throttle"], throttle))
    # 限制转向在 -1 到 1 之间
    steer_clamped = max(-CONFIG["max_steer"], min(CONFIG["max_steer"], steer))

    return throttle_clamped, steer_clamped


def run_simulation():
    """运行CARLA仿真主函数"""
    # 1. 初始化CARLA环境
    env = init_carla_environment(
        host=CONFIG["carla_host"],
        port=CONFIG["carla_port"]
    )
    if env is None:
        logger.error("无法初始化CARLA环境，退出仿真")
        return

    # 2. 初始化集成系统
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        system = IntegratedSystem(device=device)
        logger.info(f"集成系统初始化完成，使用设备: {device}")
    except Exception as e:
        logger.error(f"集成系统初始化失败: {e}")
        return

    # 3. 运行仿真步骤
    logger.info(f"开始运行{CONFIG['simulation_steps']}步仿真")
    for step in range(CONFIG["simulation_steps"]):
        try:
            # 生成模拟输入数据（实际应用中应从CARLA环境获取真实数据）
            image = torch.randn(*CONFIG["image_size"]).unsqueeze(0).to(system.device)
            lidar_data = torch.randn(*CONFIG["lidar_size"]).unsqueeze(0).to(system.device)
            imu_data = torch.randn(*CONFIG["imu_size"]).to(system.device)

            # 前向传播获取决策
            policy, value = system.forward(image, lidar_data, imu_data)

            # 转换并限制控制信号
            throttle, steer = clamp_control_values(
                float(policy[0][0]),
                float(policy[0][1])
            )

            # 应用控制信号到车辆
            control = carla.VehicleControl(throttle=throttle, steer=steer)
            env.vehicle.apply_control(control)

            # 日志输出
            if step % 10 == 0:  # 每10步输出一次日志
                logger.info(
                    f"仿真步骤 {step:3d} | 油门: {throttle:.3f} | "
                    f"转向: {steer:.3f} | 状态价值: {value[0][0]:.3f}"
                )

            # 模拟步骤间延迟
            time.sleep(CONFIG["step_delay"])

        except Exception as e:
            logger.error(f"仿真步骤 {step} 出错: {e}")
            continue

    # 4. 清理资源
    try:
        # 停止车辆
        env.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1.0)

        # 清理CARLA环境
        if hasattr(env, 'cleanup'):
            env.cleanup()

        logger.info("仿真完成，已清理所有资源")
    except Exception as e:
        logger.error(f"清理资源时出错: {e}")


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)

    # 运行仿真
    try:
        run_simulation()
    except KeyboardInterrupt:
        logger.info("用户中断了仿真")
    except Exception as e:
        logger.error(f"仿真运行出错: {e}")