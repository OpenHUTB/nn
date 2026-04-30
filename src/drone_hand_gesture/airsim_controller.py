# -*- coding: utf-8 -*-
import time
import numpy as np
import cv2
from typing import Optional, Dict, Any

from core import BaseDroneController, ConfigManager, Logger


class AirSimController(BaseDroneController):
    def __init__(self, config: Optional[ConfigManager] = None, 
                 ip_address: Optional[str] = None, port: Optional[int] = None):
        super().__init__(config)
        self.ip_address: str = ip_address or self.config.get("airsim.ip_address", "127.0.0.1")
        self.port: int = port or self.config.get("airsim.port", 41451)
        self.vehicle_name: str = self.config.get("airsim.vehicle_name", "")
        self.client = None

    def connect(self) -> bool:
        try:
            import airsim

            self.logger.info(f"正在连接到 AirSim ({self.ip_address}:{self.port})...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()

            self.client.enableApiControl(True)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)

            self.connected = True
            self.state['armed'] = True

            self.logger.info(f"成功连接到 AirSim 模拟器！飞行器: {self.vehicle_name if self.vehicle_name else '默认'}")

            return True

        except ImportError:
            self.logger.error("未找到 airsim 模块，请运行: pip install airsim")
            return False

        except Exception as e:
            self.logger.error(f"连接失败 - {e}，请确保 AirSim 模拟器正在运行")
            return False

    def disconnect(self):
        if self.client and self.connected:
            try:
                self.client.armDisarm(False, vehicle_name=self.vehicle_name)
                self.client.enableApiControl(False)
                self.connected = False
                self.logger.info("已断开 AirSim 连接")
            except Exception as e:
                self.logger.warning(f"断开连接时出错 - {e}")

    def takeoff(self, altitude: Optional[float] = None) -> bool:
        if altitude is None:
            altitude = self.config.get("drone.takeoff_altitude", 2.0)

        if not self.connected:
            self.logger.error("未连接到 AirSim")
            return False

        try:
            self.logger.info(f"正在起飞到 {altitude} 米高度...")
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            import airsim
            self.client.moveToZAsync(-altitude, 1.0, vehicle_name=self.vehicle_name).join()
            self.state['flying'] = True
            self.state['mode'] = 'FLYING'
            self.state['armed'] = True
            self.logger.info("起飞完成！")
            return True
        except Exception as e:
            self.logger.error(f"起飞失败: {e}")
            return False

    def land(self) -> bool:
        if not self.connected:
            self.logger.error("未连接到 AirSim")
            return False

        try:
            self.logger.info("正在降落...")
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            self.state['flying'] = False
            self.state['mode'] = 'LANDED'
            self.logger.info("降落完成！")
            return True
        except Exception as e:
            self.logger.error(f"降落失败: {e}")
            return False

    def hover(self):
        if not self.connected:
            return
        try:
            self.client.moveToPositionAsync(
                self.state['position'][0],
                self.state['position'][1],
                self.state['position'][2],
                1.0,
                vehicle_name=self.vehicle_name
            )
            self.state['mode'] = 'HOVERING'
        except Exception as e:
            self.logger.warning(f"悬停控制失败 - {e}")

    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float = 0.5):
        """
        按速度控制无人机
        
        AirSim 使用 NED (North-East-Down) 坐标系:
        - X 轴: 前进方向 (正=前进)
        - Y 轴: 右移方向 (正=右移)  
        - Z 轴: 下降方向 (正=下降, 负=上升)
        
        Args:
            vx: 前进速度 (m/s), 正=前进, 负=后退
            vy: 右移速度 (m/s), 正=右移, 负=左移
            vz: 垂直速度 (m/s), 正=下降, 负=上升
            duration: 持续时间 (秒)
        """
        if not self.connected:
            return

        try:
            import airsim
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(),
                vehicle_name=self.vehicle_name
            )
            self._record_trajectory()
        except Exception as e:
            self.logger.warning(f"速度控制失败 - {e}")

    def send_command(self, command: str, intensity: float = 1.0):
        """
        发送命令到无人机（适配 AirSim 坐标系）
        
        Args:
            command: 命令名称
            intensity: 强度 (0-1)
        """
        self.logger.info(f"收到命令: {command}, 强度: {intensity}")

        if command == "takeoff":
            self.takeoff()
        elif command == "land":
            self.land()
        elif command == "hover":
            self.hover()
        elif command == "forward":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(speed, 0, 0)
        elif command == "backward":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(-speed, 0, 0)
        elif command == "left":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, -speed, 0)
        elif command == "right":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, speed, 0)
        elif command == "up":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, 0, -speed)
        elif command == "down":
            speed = self.config.get("drone.max_speed", 2.0) * intensity
            self.move_by_velocity(0, 0, speed)
        elif command == "stop":
            self.move_by_velocity(0, 0, 0)
            self.hover()

    def get_state(self) -> Dict[str, Any]:
        if not self.connected:
            return self.state

        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            self.state['position'] = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                -state.kinematics_estimated.position.z_val
            ])
            self.state['velocity'] = np.array([
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                -state.kinematics_estimated.linear_velocity.z_val
            ])
            return self.state
        except Exception as e:
            self.logger.warning(f"获取状态失败 - {e}")
            return self.state

    def get_camera_image(self, camera_name: str = "front_center", 
                        image_type: int = 0) -> Optional[np.ndarray]:
        """
        获取摄像头画面
        
        Args:
            camera_name: 摄像头名称 (front_center, front_right, front_left, bottom_center, back_center)
            image_type: 图像类型 (0=Scene, 1=DepthPlanar, 2=DepthPerspective, 3=DepthVis, 5=Segmentation)
        
        Returns:
            OpenCV 格式的图像 (BGR)，失败返回 None
        """
        if not self.connected:
            self.logger.warning("未连接到 AirSim，无法获取摄像头画面")
            return None

        try:
            import airsim

            # 请求图像
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, image_type, False, False)
            ])

            if not responses:
                self.logger.warning("未获取到图像响应")
                return None

            response = responses[0]

            # 将图像数据转换为 NumPy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

            # 重塑为 H x W x 3 (RGB)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # 原始图像是垂直翻转的，需要翻转回来
            img_rgb = np.flipud(img_rgb)

            # 转换为 BGR 格式 (OpenCV 使用 BGR)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            return img_bgr

        except ImportError:
            self.logger.error("未找到 cv2 或 numpy 模块")
            return None
        except Exception as e:
            self.logger.warning(f"获取摄像头画面失败 - {e}")
            return None


def test_airsim_connection():
    logger = Logger()
    logger.info("=" * 60)
    logger.info("AirSim 连接测试")
    logger.info("=" * 60)

    config = ConfigManager()
    controller = AirSimController(config)

    if controller.connect():
        logger.info("\nAirSim 连接成功！")
        state = controller.get_state()
        logger.info(f"位置: {state['position']}")
        controller.disconnect()
        return True
    else:
        logger.error("\nAirSim 连接失败")
        return False


if __name__ == "__main__":
    test_airsim_connection()
