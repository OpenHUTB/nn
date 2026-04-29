# drone_controller.py
"""无人机核心控制类模块

本模块封装了无人机飞行控制的核心功能，包括：
- 起飞和降落
- 飞向指定位置
- 碰撞检测和紧急处理
- 资源清理
"""

# 导入 AirSim 库，用于与仿真器通信
import airsim

# 导入 time 模块，用于延时和超时控制
import time

# 导入 os 模块，用于创建目录和路径操作
import os

# 导入 cv2 (OpenCV) 模块，用于图像处理
import cv2

# 导入 numpy 模块，用于数值计算
import numpy as np

# 从 config 模块导入飞行配置参数、相机配置和键盘控制参数
from config import (
    FlightConfig,
    RGB_CAMERA_NAME,
    DEFAULT_IMAGE_COUNT,
    KEYBOARD_VELOCITY,
    KEYBOARD_YAW_RATE,
    KEYBOARD_STEP,
)

# 从 collision_handler 模块导入碰撞处理器
from collision_handler import CollisionHandler


class DroneController:
    """无人机控制器类

    负责与 AirSim 仿真器交互，实现无人机的各项飞行控制功能。
    提供起飞、飞行、降落、紧急停止等方法。
    """

    def __init__(self):
        """初始化无人机控制器

        连接 AirSim 仿真器，获取控制权，解锁无人机。
        初始化碰撞检测处理器。
        """
        # 打印连接提示信息
        print("🔌 正在连接 AirSim...")
        # 创建 AirSim 多旋翼无人机客户端对象
        self.client = airsim.MultirotorClient()
        # 确认与仿真器的连接
        self.client.confirmConnection()
        # 打印连接成功信息
        print("✅ 连接成功！")

        # 打印获取控制权提示
        print("🎮 获取控制权...")
        # 启用 API 控制，使程序可以控制无人机
        self.client.enableApiControl(True)
        # 打印解锁提示
        print("🔓 解锁电机...")
        # 解锁无人机螺旋桨
        self.client.armDisarm(True)
        # 打印初始化完成信息
        print("✅ 初始化完成")

        # 创建碰撞检测处理器实例
        self.collision_handler = CollisionHandler(self.client)

        # 相机相关初始化
        self.image_count = DEFAULT_IMAGE_COUNT  # 图片保存计数器
        # 使用脚本所在目录的绝对路径，确保无论从哪里运行都能找到
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "drone_images")
        # 创建保存目录（如果不存在）
        self._ensure_output_dir()

        # 键盘控制速度属性（供 keyboard_control.py 使用）
        self.velocity = KEYBOARD_VELOCITY

    def takeoff(self):
        """执行起飞操作

        首先执行 takeoff 命令，然后上升到指定高度。
        包含超时检测，防止起飞失败时程序卡死。

        返回:
            bool: 起飞成功返回 True，失败返回 False
        """
        # 打印起飞提示
        print("🚀 起飞中...")

        # 记录起飞开始时间，用于检测超时
        start_time = time.time()
        # 执行起飞命令并等待完成
        self.client.takeoffAsync().join()

        # 检查起飞是否超时
        if time.time() - start_time > FlightConfig.TAKEOFF_TIMEOUT:
            # 起飞超时，打印错误信息并返回失败
            print("❌ 起飞超时！")
            return False

        # 打印上升高度信息
        print(f"📈 上升到 {abs(FlightConfig.TAKEOFF_HEIGHT)} 米...")
        # 异步上升到指定高度，速度参数 1 m/s
        self.client.moveToZAsync(FlightConfig.TAKEOFF_HEIGHT, 1).join()

        # 获取当前位置并打印
        pos = self.get_position()
        print(
            f"✅ 起飞完成，当前位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})"
        )
        return True

    def fly_to_position(self, x, y, z, velocity=None, show_progress=True):
        """飞向目标位置

        控制无人机从当前位置飞向指定的目标点。
        在飞行过程中持续检测碰撞，到达目标后停止。

        参数:
            x (float): 目标点 X 坐标
            y (float): 目标点 Y 坐标
            z (float): 目标点 Z 坐标
            velocity (float): 飞行速度，默认为 None（使用配置中的速度）
            show_progress (bool): 是否显示飞行进度，默认为 True

        返回:
            bool: 成功到达返回 True，超时或碰撞返回 False
        """
        # 如果未指定速度，使用配置文件中的默认速度
        if velocity is None:
            velocity = FlightConfig.FLIGHT_VELOCITY

        # 打印目标位置信息
        print(f"✈️  正在飞往: ({x}, {y}, {z})")
        # 获取起始位置
        start_pos = self.get_position()
        # 计算总距离
        total_distance = (
            (start_pos.x_val - x) ** 2 +
            (start_pos.y_val - y) ** 2 +
            (start_pos.z_val - z) ** 2
        ) ** 0.5
        # 打印起始位置和总距离
        print(f"   起始位置: ({start_pos.x_val:.1f}, {start_pos.y_val:.1f}, {start_pos.z_val:.1f})")
        print(f"   目标距离: {total_distance:.1f}m")

        # 发送异步飞往目标位置的命令（非阻塞）
        self.client.moveToPositionAsync(x, y, z, velocity)

        # 记录飞行开始时间
        start_time = time.time()
        # 上一进度打印时间
        last_print_time = 0
        # 进入飞行监控循环
        while time.time() - start_time < FlightConfig.MAX_FLIGHT_TIME:
            # 检查是否发生严重碰撞
            is_serious, _ = self.collision_handler.check_collision()
            if is_serious:
                # 发生严重碰撞，取消当前任务
                self.client.cancelLastTask()
                # 悬停等待
                self.client.hoverAsync().join()
                return False

            # 检查是否已到达目标位置
            if self.is_at_position(x, y, z):
                # 成功到达目标点
                print(f"📍 成功到达目标点 ({x}, {y}, {z})")
                return True

            # 显示飞行进度（每0.5秒更新一次）
            current_time = time.time()
            if show_progress and current_time - last_print_time >= 0.5:
                pos = self.get_position()
                current_distance = (
                    (pos.x_val - x) ** 2 +
                    (pos.y_val - y) ** 2 +
                    (pos.z_val - z) ** 2
                ) ** 0.5
                progress = max(0, min(100, (1 - current_distance / total_distance) * 100)) if total_distance > 0 else 100
                speed = self.get_speed()
                print(f"   进度: {progress:5.1f}% | 剩余: {current_distance:5.1f}m | 速度: {speed:.1f}m/s    ", end="\r")
                last_print_time = current_time

            # 短暂休眠，减少 CPU 占用
            time.sleep(0.1)

        # 清除进度行
        print(" " * 80, end="\r")
        # 飞行超时
        print("❌ 飞行超时！")
        return False

    def safe_land(self):
        """安全降落

        执行多步降落流程，包括悬停稳定、下降、确认落地。
        如果常规降落失败，会尝试紧急复位。

        返回:
            bool: 降落成功返回 True，失败返回 False
        """
        # 打印降落开始提示
        print("\n🛬 开始安全降落流程...")

        # 尝试多次降落（配置文件中的最大次数）
        for attempt in range(FlightConfig.LANDING_MAX_ATTEMPTS):
            try:
                # 打印当前尝试次数
                print(
                    f"   尝试 {attempt + 1}/{FlightConfig.LANDING_MAX_ATTEMPTS}: 稳定无人机..."
                )
                # 进入悬停状态
                self.client.hoverAsync().join()
                # 等待 1 秒稳定
                time.sleep(1)

                # 获取当前位置
                pos = self.get_position()
                # 计算当前高度
                current_height = -pos.z_val
                # 打印位置和高度信息
                print(f"   当前位置: ({pos.x_val:.1f}, {pos.y_val:.1f})")
                print(f"   当前高度: {current_height:.2f}m")

                # 如果高度很低，认为已经在地面
                if current_height < 0.3:
                    print("✅ 无人机已经在地面")
                    return True

                # 打印降落命令提示
                print("   执行降落命令...")
                # 执行降落命令并等待完成
                self.client.landAsync().join()

                # 等待降落完成，轮询检查高度
                for _ in range(
                    int(
                        FlightConfig.LANDING_MAX_WAIT
                        / FlightConfig.LANDING_CHECK_INTERVAL
                    )
                ):
                    # 等待检查间隔时间
                    time.sleep(FlightConfig.LANDING_CHECK_INTERVAL)
                    # 获取当前 Z 轴坐标
                    current_z = self.get_position().z_val
                    # Z >= 0 表示已经触地
                    if current_z >= 0:
                        print("✅ 降落成功！")
                        return True

            except Exception as e:
                # 当前尝试失败，打印错误信息
                print(f"❌ 降落尝试 {attempt + 1} 失败: {e}")
                # 等待 1 秒后重试
                time.sleep(1)

        # 所有尝试都失败，尝试紧急复位
        print("⚠️  常规降落失败，尝试复位...")
        try:
            # 调用 AirSim 的复位功能
            self.client.reset()
            # 等待复位完成
            time.sleep(2)
            print("✅ 已复位")
            return True
        except:
            # 复位也失败
            return False

    def emergency_stop(self):
        """紧急停止

        在发生严重错误或碰撞时执行紧急停止，
        取消当前任务、悬停、解锁控制权。
        """
        # 打印紧急停止提示
        print("🚨 执行紧急停止！")
        try:
            # 取消当前正在执行的任务
            self.client.cancelLastTask()
            # 进入悬停状态
            self.client.hoverAsync().join()
            # 锁定电机
            self.client.armDisarm(False)
            # 释放 API 控制权
            self.client.enableApiControl(False)
            print("✅ 已紧急停止")
        except Exception as e:
            # 紧急停止过程中发生异常
            print(f"⚠️  紧急停止异常: {e}")

    def get_position(self):
        """获取无人机当前位置

        从 AirSim 获取无人机的实时位置信息。

        返回:
            Position: 包含 x_val, y_val, z_val 属性的位置对象
        """
        # 获取无人机当前状态中的位置信息
        return self.client.getMultirotorState().kinematics_estimated.position

    def is_at_position(self, x, y, z):
        """判断无人机是否到达目标位置

        计算当前位置与目标位置的欧几里得距离，
        如果距离小于容差值则认为已到达。

        参数:
            x (float): 目标点 X 坐标
            y (float): 目标点 Y 坐标
            z (float): 目标点 Z 坐标

        返回:
            bool: 在容差范围内返回 True，否则返回 False
        """
        # 获取当前位置
        pos = self.get_position()
        # 计算到目标点的欧几里得距离
        distance = (
            (pos.x_val - x) ** 2 + (pos.y_val - y) ** 2 + (pos.z_val - z) ** 2
        ) ** 0.5
        # 比较距离与容差值
        return distance < FlightConfig.ARRIVAL_TOLERANCE

    def cleanup(self):
        """清理资源

        在程序结束时调用，释放无人机控制权。
        确保无人机处于安全状态。
        """
        try:
            # 锁定电机
            self.client.armDisarm(False)
            # 释放 API 控制权
            self.client.enableApiControl(False)
            print("✅ 资源清理完成")
        except:
            # 清理过程中忽略异常
            pass

    # ==================== 相机控制方法 ====================

    def _ensure_output_dir(self):
        """确保输出目录存在

        如果保存图片的目录不存在，则创建该目录。
        """
        # 如果目录不存在
        if not os.path.exists(self.output_dir):
            # 创建目录（包括父目录）
            os.makedirs(self.output_dir)
            print(f"📁 创建图片保存目录: {self.output_dir}")

    def set_output_dir(self, directory: str):
        """设置图片保存目录

        参数:
            directory (str): 新的图片保存目录路径
        """
        self.output_dir = directory
        self._ensure_output_dir()
        print(f"📁 图片保存目录已设置为: {directory}")

    def capture_image(self, filename: str = None, show_preview: bool = False) -> str:
        """拍摄并保存 RGB 图像

        从无人机的 RGB 相机捕获当前视角图像并保存到文件。

        参数:
            filename (str): 保存的文件名，默认为 None（自动生成）
            show_preview (bool): 是否显示图像预览，默认为 False

        返回:
            str: 保存的文件路径，失败返回 None
        """
        try:
            # 如果未指定文件名，自动生成带时间戳的文件名
            if filename is None:
                self.image_count += 1
                # 获取当前时间戳
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # 获取当前位置用于文件名
                pos = self.get_position()
                # 生成文件名：rgb_时间戳_X_Y_序号.png
                filename = f"rgb_{timestamp}_{pos.x_val:.1f}_{pos.y_val:.1f}_n{self.image_count}.png"

            # 构建完整文件路径
            filepath = os.path.join(self.output_dir, filename)

            # 调用 AirSim API 捕获图像
            # CameraType: 0 = 视角相机（RGB），1 = 深度，2 = 分割
            # ImageType: 0 = 场景（RGB），1 = 深度，2 = 分割，3 = 表面法线，4 = 红外
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        RGB_CAMERA_NAME, airsim.ImageType.Scene, False, False
                    )
                ]
            )

            # 检查响应是否有效
            if responses is None or len(responses) == 0:
                print("❌ 未能获取图像数据")
                return None

            # 获取图像数据
            response = responses[0]
            # 将原始图像数据转换为 numpy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            # 根据图像宽度重新整形为 3 通道 RGB 图像
            img = img1d.reshape(response.height, response.width, 3)

            # 保存图像为 PNG 格式
            cv2.imwrite(filepath, img)

            # 获取当前高度信息
            pos = self.get_position()
            height = -pos.z_val

            # 打印保存成功信息
            print(f"📸 拍照成功！保存至: {filepath}")
            print(f"   位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {height:.1f}m)")

            # 如果需要显示预览
            if show_preview:
                # 在窗口中显示图像
                cv2.imshow("RGB Camera Preview", img)
                # 等待按键，0 表示无限等待
                cv2.waitKey(0)
                # 关闭所有窗口
                cv2.destroyAllWindows()

            return filepath

        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"❌ 拍照失败: {e}")
            return None

    def capture_depth_image(self, filename: str = None) -> str:
        """拍摄并保存深度图像

        从无人机的深度相机捕获当前深度图像并保存到文件。
        深度图像以伪彩色方式保存，蓝色表示近，红色表示远。

        参数:
            filename (str): 保存的文件名，默认为 None（自动生成）

        返回:
            str: 保存的文件路径，失败返回 None
        """
        try:
            # 如果未指定文件名，自动生成带时间戳的文件名
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                pos = self.get_position()
                filename = f"depth_{timestamp}_{pos.x_val:.1f}_{pos.y_val:.1f}.png"

            filepath = os.path.join(self.output_dir, filename)

            # 获取深度图像
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        RGB_CAMERA_NAME, airsim.ImageType.DepthPerspective, True, False
                    )
                ]
            )

            if responses is None or len(responses) == 0:
                print("❌ 未能获取深度图像数据")
                return None

            response = responses[0]
            # 深度图像数据是浮点数数组
            img1d = np.array(response.image_data_float, dtype=np.float32)
            img = img1d.reshape(response.height, response.width)

            # 将深度值归一化到 0-255 范围并转换为伪彩色
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_uint8 = img_normalized.astype(np.uint8)
            # 应用伪彩色映射（JET 色彩表：蓝->绿->红）
            img_colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)

            # 保存深度图像
            cv2.imwrite(filepath, img_colored)

            pos = self.get_position()
            print(f"📸 深度拍照成功！保存至: {filepath}")
            print(f"   位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}m)")

            return filepath

        except Exception as e:
            print(f"❌ 深度拍照失败: {e}")
            return None

    def capture_segmentation_image(self, filename: str = None) -> str:
        """拍摄并保存分割图像

        从无人机的分割相机捕获当前分割图像并保存到文件。
        分割图像将场景中的不同物体用不同颜色标记。

        参数:
            filename (str): 保存的文件名，默认为 None（自动生成）

        返回:
            str: 保存的文件路径，失败返回 None
        """
        try:
            # 如果未指定文件名，自动生成带时间戳的文件名
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                pos = self.get_position()
                filename = f"seg_{timestamp}_{pos.x_val:.1f}_{pos.y_val:.1f}.png"

            filepath = os.path.join(self.output_dir, filename)

            # 获取分割图像
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        RGB_CAMERA_NAME, airsim.ImageType.Segmentation, False, False
                    )
                ]
            )

            if responses is None or len(responses) == 0:
                print("❌ 未能获取分割图像数据")
                return None

            response = responses[0]
            # 将原始图像数据转换为 numpy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(response.height, response.width, 3)

            # 保存分割图像
            cv2.imwrite(filepath, img)

            pos = self.get_position()
            print(f"📸 分割拍照成功！保存至: {filepath}")
            print(f"   位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}m)")

            return filepath

        except Exception as e:
            print(f"❌ 分割拍照失败: {e}")
            return None

    def capture_all_cameras(self, prefix: str = None) -> dict:
        """同时拍摄 RGB、深度和分割图像

        一次性从无人机获取所有类型的图像，适合需要完整数据的场景。

        参数:
            prefix (str): 文件名前缀，默认为 None（使用时间戳）

        返回:
            dict: 包含三种图像保存路径的字典，失败返回 None
        """
        try:
            # 生成文件名前缀
            if prefix is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                pos = self.get_position()
                prefix = f"all_{timestamp}_{pos.x_val:.1f}_{pos.y_val:.1f}"

            # 同时请求三种图像
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        RGB_CAMERA_NAME, airsim.ImageType.Scene, False, False
                    ),
                    airsim.ImageRequest(
                        RGB_CAMERA_NAME, airsim.ImageType.DepthPerspective, True, False
                    ),
                    airsim.ImageRequest(
                        RGB_CAMERA_NAME, airsim.ImageType.Segmentation, False, False
                    ),
                ]
            )

            if responses is None or len(responses) < 3:
                print("❌ 未能获取完整的图像数据")
                return None

            result = {}

            # 处理 RGB 图像
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            rgb_path = os.path.join(self.output_dir, f"{prefix}_rgb.png")
            cv2.imwrite(rgb_path, img_rgb)
            result["rgb"] = rgb_path

            # 处理深度图像
            img1d = np.array(responses[1].image_data_float, dtype=np.float32)
            img_depth = img1d.reshape(responses[1].height, responses[1].width)
            img_normalized = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX)
            img_depth_colored = cv2.applyColorMap(
                img_normalized.astype(np.uint8), cv2.COLORMAP_JET
            )
            depth_path = os.path.join(self.output_dir, f"{prefix}_depth.png")
            cv2.imwrite(depth_path, img_depth_colored)
            result["depth"] = depth_path

            # 处理分割图像
            img1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            img_seg = img1d.reshape(responses[2].height, responses[2].width, 3)
            seg_path = os.path.join(self.output_dir, f"{prefix}_seg.png")
            cv2.imwrite(seg_path, img_seg)
            result["segmentation"] = seg_path

            pos = self.get_position()
            print(f"📸 全景拍照成功！保存 3 张图像到: {self.output_dir}/")
            print(f"   位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}m)")

            return result

        except Exception as e:
            print(f"❌ 全景拍照失败: {e}")
            return None

    # ==================== 键盘控制方法 ====================

    def move_forward(self):
        """向前移动"""
        pos = self.get_position()
        self.client.moveToPositionAsync(
            pos.x_val + KEYBOARD_STEP, pos.y_val, pos.z_val, KEYBOARD_VELOCITY
        )
        print(f"⬆️  前进 +X: {KEYBOARD_STEP}m")

    def move_backward(self):
        """向后移动"""
        pos = self.get_position()
        self.client.moveToPositionAsync(
            pos.x_val - KEYBOARD_STEP, pos.y_val, pos.z_val, KEYBOARD_VELOCITY
        )
        print(f"⬇️  后退 -X: {KEYBOARD_STEP}m")

    def move_left(self):
        """向左移动"""
        pos = self.get_position()
        self.client.moveToPositionAsync(
            pos.x_val, pos.y_val + KEYBOARD_STEP, pos.z_val, KEYBOARD_VELOCITY
        )
        print(f"⬅️  左移 +Y: {KEYBOARD_STEP}m")

    def move_right(self):
        """向右移动"""
        pos = self.get_position()
        self.client.moveToPositionAsync(
            pos.x_val, pos.y_val - KEYBOARD_STEP, pos.z_val, KEYBOARD_VELOCITY
        )
        print(f"➡️  右移 -Y: {KEYBOARD_STEP}m")

    def move_up(self):
        """上升"""
        pos = self.get_position()
        new_z = pos.z_val - KEYBOARD_STEP  # 负值表示上升
        self.client.moveToZAsync(new_z, KEYBOARD_VELOCITY)
        print(f"🔼  上升 -Z: {KEYBOARD_STEP}m (高度: {abs(new_z):.1f}m)")

    def move_down(self):
        """下降"""
        pos = self.get_position()
        new_z = pos.z_val + KEYBOARD_STEP  # 正值表示下降
        # 防止降到地面以下
        if new_z > -0.5:
            new_z = -0.5
        self.client.moveToZAsync(new_z, KEYBOARD_VELOCITY)
        print(f"🔽  下降 +Z: {KEYBOARD_STEP}m (高度: {abs(new_z):.1f}m)")

    def hover(self):
        """悬停

        让无人机停止移动并保持在当前位置悬停。
        """
        self.client.hoverAsync()

    def go_forward_continuous(self):
        """持续向前飞行"""
        self.client.moveByVelocityAsync(self.velocity, 0, 0, 0.1)

    def go_backward_continuous(self):
        """持续向后飞行"""
        self.client.moveByVelocityAsync(-self.velocity, 0, 0, 0.1)

    def go_left_continuous(self):
        """持续向左飞行"""
        self.client.moveByVelocityAsync(0, -self.velocity, 0, 0.1)

    def go_right_continuous(self):
        """持续向右飞行"""
        self.client.moveByVelocityAsync(0, self.velocity, 0, 0.1)

    def rotate_left_continuous(self):
        """持续向左旋转（偏航）"""
        self.client.rotateByYawRateAsync(-KEYBOARD_YAW_RATE, 0.1)

    def rotate_right_continuous(self):
        """持续向右旋转（偏航）"""
        self.client.rotateByYawRateAsync(KEYBOARD_YAW_RATE, 0.1)

    def go_up_continuous(self):
        """持续上升

        使用速度控制，让无人机以设定速度上升。
        AirSim 中 Z 轴向下为正，所以上升需要负的 Z 速度。
        """
        self.client.moveByVelocityAsync(0, 0, -self.velocity, 0.1)

    def go_down_continuous(self):
        """持续下降

        使用速度控制，让无人机以设定速度下降。
        AirSim 中 Z 轴向下为正，所以下降需要正的 Z 速度。
        防止降到地面以下（z > -0.5）。
        """
        pos = self.get_position()
        # 只有高于地面安全高度时才下降
        if pos.z_val < -0.5:
            self.client.moveByVelocityAsync(0, 0, self.velocity, 0.1)

    def get_telemetry(self):
        """获取并打印无人机状态信息"""
        pos = self.get_position()
        state = self.client.getMultirotorState()
        linear_vel = state.kinematics_estimated.linear_velocity
        collision_info = self.client.simGetCollisionInfo()

        height = abs(pos.z_val)
        speed = (linear_vel.x_val**2 + linear_vel.y_val**2 + linear_vel.z_val**2) ** 0.5

        # 获取飞行姿态
        orientation = state.kinematics_estimated.orientation
        pitch, roll, yaw = airsim.to_euler_angles(orientation)
        yaw_deg = round(yaw * 180 / 3.14159, 1)

        print(f"\n{'─' * 40}")
        print(f"📊 无人机状态:")
        print(f"   位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
        print(f"   高度: {height:.2f}m")
        print(f"   速度: {speed:.2f} m/s")
        print(f"   朝向: {yaw_deg}°")
        print(f"   碰撞: {'⚠️ 是' if collision_info.has_collided else '✅ 否'}")
        print(f"{'─' * 40}\n")

    def get_velocity(self):
        """获取无人机当前速度

        返回:
            tuple: (vx, vy, vz) 速度分量
        """
        state = self.client.getMultirotorState()
        vel = state.kinematics_estimated.linear_velocity
        return vel.x_val, vel.y_val, vel.z_val

    def get_speed(self):
        """获取无人机当前速率

        返回:
            float: 速率（米/秒）
        """
        vx, vy, vz = self.get_velocity()
        return (vx**2 + vy**2 + vz**2) ** 0.5
