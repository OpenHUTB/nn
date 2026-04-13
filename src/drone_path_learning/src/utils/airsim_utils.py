"""AirSim 客户端工具与封装"""

import numpy as np
import time
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AirSimConnector:
    """AirSim 多旋翼客户端的安全包装器"""

    def __init__(
        self, ip_address: str = "127.0.0.1", port: int = 41451, timeout: float = 5.0
    ):
        """
        初始化 AirSim 连接器。

        Args:
            ip_address: AirSim 服务器的 IP 地址
            port: 端口号
            timeout: 连接超时时间（秒）
        """
        self.ip_address = ip_address
        self.port = port
        self.timeout = timeout
        self.client = None
        self._is_connected = False

    def connect(self) -> bool:
        """
        建立到 AirSim 的连接。

        返回值:
            成功返回 True，否则返回 False
        """
        try:
            import airsim

            self.client = airsim.MultirotorClient(ip=self.ip_address)
            self.client.confirmConnection()
            self._is_connected = True
            logger.info(f"成功连接到 AirSim：{self.ip_address}")
            return True
        except Exception as e:
            logger.error(f"连接 AirSim 失败：{e}")
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        """检查是否已连接到 AirSim"""
        return self._is_connected and self.client is not None

    def disconnect(self):
        """安全地从 AirSim 断开连接"""
        try:
            if self.client is not None:
                self.client.reset()
                self.client = None
                self._is_connected = False
                logger.info("从 AirSim 断开连接")
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")

    def enable_api_control(self, enable: bool = True) -> bool:
        """启用/禁用 API 控制"""
        if not self.is_connected():
            return False
        try:
            self.client.enableApiControl(enable)
            return True
        except Exception as e:
            logger.error(f"设置 API 控制失败: {e}")
            return False

    def arm(self) -> bool:
        """无人机上锁"""
        if not self.is_connected():
            return False
        try:
            self.client.armDisarm(True)
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"无人机上锁失败: {e}")
            return False

    def disarm(self) -> bool:
        """无人机解锁"""
        if not self.is_connected():
            return False
        try:
            self.client.armDisarm(False)
            return True
        except Exception as e:
            logger.error(f"无人机解锁失败: {e}")
            return False

    def takeoff(self, timeout: float = 10.0) -> bool:
        """无人机起飞"""
        if not self.is_connected():
            return False
        try:
            future = self.client.takeoffAsync()
            future.join()
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"起飞失败: {e}")
            return False

    def land(self, timeout: float = 10.0) -> bool:
        """平稳着陆无人机"""
        if not self.is_connected():
            return False
        try:
            future = self.client.landAsync()
            future.join()
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"着陆失败: {e}")
            return False

    def reset(self) -> bool:
        """重置无人机到初始状态"""
        if not self.is_connected():
            return False
        try:
            self.client.reset()
            time.sleep(1.0)
            return True
        except Exception as e:
            logger.error(f"重置失败: {e}")
            return False

    def get_state(self) -> Optional[dict]:
        """获取无人机的当前状态"""
        if not self.is_connected():
            return None
        try:
            state = self.client.getMultirotorState()
            collision_info = self.client.simGetCollisionInfo()

            # 提取运动学数据
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity

            # AirSim 通常使用 Landed=0, Flying=1。
            # 当枚举值不可用时，采用 landed_state==0 的保守判定。
            landed_raw = getattr(state, "landed_state", 1)
            landed_value = getattr(landed_raw, "value", landed_raw)
            landed_state = int(landed_value)
            is_landed = landed_state == 0

            return {
                "position": np.array([pos.x_val, pos.y_val, pos.z_val]),
                "velocity": np.array([vel.x_val, vel.y_val, vel.z_val]),
                "collision": bool(getattr(collision_info, "has_collided", False)),
                "landed": is_landed,
            }
        except Exception as e:
            logger.error(f"获取无人机状态失败: {e}")
            return None

    def move_by_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float = 1.0,
        timeout: float = 10.0,
    ) -> bool:
        """
        以速度控制无人机运动。

        Args:
            vx, vy, vz: 速度分量 (m/s)
            duration: 运动持续时间（秒）
            timeout: 异步操作超时

        返回值:
            成功返回 True
        """
        if not self.is_connected():
            return False
        try:
            future = self.client.moveByVelocityAsync(vx, vy, vz, duration)
            future.join()
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"运动命令失败: {e}")
            return False

    def get_images(self, image_type: str = "depth") -> Optional[np.ndarray]:
        """
        从 AirSim 获取摄像机图像。

        Args:
            image_type: "depth"、"rgb" 或 "segmentation"

        返回值:
            图像数组，失败时返回 None
        """
        if not self.is_connected():
            return None

        try:
            import airsim

            # 映射图像类型
            if image_type == "depth":
                req_type = airsim.ImageType.DepthPerspective
            elif image_type == "rgb":
                req_type = airsim.ImageType.Scene
            else:
                req_type = airsim.ImageType.Segmentation

            # 从默认相机请求图像
            # 参数: camera_name, image_type, pixels_as_float, compress
            request = airsim.ImageRequest(
                "0", req_type, pixels_as_float=False, compress=False
            )
            responses = self.client.simGetImages([request])

            if len(responses) == 0 or responses[0] is None:
                logger.warning("从 AirSim 没有获取到图像响应")
                return None

            response = responses[0]

            # 检查是否有数据
            if response.image_data_uint8 is None or len(response.image_data_uint8) == 0:
                logger.warning(
                    f"接收到空图像数据. Height: {response.height}, Width: {response.width}"
                )
                return None

            # 转换为 numpy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            expected_size_rgb = response.height * response.width * 3
            expected_size_mono = response.height * response.width

            if img1d.size == expected_size_rgb:
                # RGB 图像
                img = img1d.reshape((response.height, response.width, 3))
            elif img1d.size == expected_size_mono:
                # 单通道（深度图）
                img = img1d.reshape((response.height, response.width))
            else:
                logger.warning(
                    f"图像大小异常: {img1d.size}, expected {expected_size_mono} or {expected_size_rgb}"
                )
                return None

            return img

        except Exception as e:
            logger.error(f"图像捕获失败: {e}")
            return None


def process_depth_image(
    raw_image: np.ndarray, target_size: Tuple[int, int] = (84, 84), invert: bool = True
) -> np.ndarray:
    """
    处理 AirSim 的原始深度图像。

    Args:
        raw_image: 来自 AirSim 的原始图像数组
        target_size: 目标输出大小 (H, W)
        invert: 是否反转深度（远处 = 亮）

    返回值:
        处理后的灰度图像 (H, W, 1)
    """
    try:
        from PIL import Image

        # 安全检查 - 确保有有效的图像数据
        if raw_image is None or raw_image.size == 0:
            logger.warning(f"无效的原始图像: {raw_image}")
            return np.zeros((*target_size, 1), dtype=np.uint8)

        # 处理单通道或多通道图像
        if len(raw_image.shape) == 3 and raw_image.shape[2] > 1:
            # 多通道图像，转换为灰度
            raw_image = np.mean(raw_image, axis=2)
        elif len(raw_image.shape) == 3:
            # 单通道且有额外维度
            raw_image = raw_image[:, :, 0]

        # 如需要，将浮点深度转换为 8 位
        if raw_image.dtype == np.float32 or raw_image.dtype == np.float64:
            # 反转深度：远处 = 亮，近处 = 暗
            if invert:
                processed = 255.0 / np.maximum(np.ones_like(raw_image), raw_image)
            else:
                processed = raw_image

            # 归一化到 0-255
            processed = np.clip(processed, 0, 255).astype(np.uint8)
        else:
            processed = raw_image.astype(np.uint8)

        # 使用 PIL 调整大小
        img = Image.fromarray(processed)
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

        # 转换为灰度并添加通道维度
        img_gray = np.array(img_resized.convert("L"), dtype=np.uint8)
        img_final = img_gray.reshape((*target_size, 1))

        return img_final

    except Exception as e:
        logger.error(f"图像处理失败: {e}")
        return np.zeros((*target_size, 1), dtype=np.uint8)
