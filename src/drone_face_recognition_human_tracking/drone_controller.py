# 无人机控制模块
import cv2
import numpy as np
import threading


class DroneController:
    def __init__(self, use_simulator=True):
        """
        初始化无人机控制器

        Args:
            use_simulator: 是否使用模拟器（True=AirSim, False=真实/备用模式）
        """
        self.client = None
        self.connected = False
        self.use_simulator = use_simulator

        # 控制参数
        self.velocity = 2.0  # 米/秒
        self.duration = 0.1  # 控制持续时间

        # 虚拟位置（用于测试）
        self.virtual_position = {'x': 0, 'y': 0, 'z': 10}

        # 用于测试的虚拟摄像头
        self.test_video_source = 0  # 0=默认摄像头，或视频文件路径
        self.cap = None
        print(f"🚁 初始化{'模拟' if use_simulator else '真实'}无人机控制器")

    def connect(self):
        """连接到无人机或模拟器"""
        try:
            print("🤖 正在连接无人机...")

            if self.use_simulator:
                # 尝试连接AirSim
                try:
                    import airsim
                    self.client = airsim.MultirotorClient("127.0.0.1", 41451)
                    self.client.confirmConnection()
                    self.client.enableApiControl(True)
                    self.client.armDisarm(True)

                    # 起飞
                    self.client.takeoffAsync().join()

                    self.connected = True
                    print("✅ AirSim无人机连接成功")
                    return True

                except ImportError:
                    print("⚠️  AirSim未安装，切换到备用模式")
                    return self.connect_backup_mode()

                except Exception as e:
                    print(f"⚠️  AirSim连接失败: {e}")
                    print("切换到备用模式...")
                    return self.connect_backup_mode()

            else:
                # 真实无人机模式（需要具体SDK）
                print("🛸 真实无人机模式（需要具体SDK）")
                return self.connect_backup_mode()

        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return self.connect_backup_mode()

    def connect_backup_mode(self):
        """备用连接模式（使用本地摄像头）"""
        try:
            print("📷 使用备用模式：本地摄像头")

            # 打开本地摄像头
            self.cap = cv2.VideoCapture(self.test_video_source)
            if not self.cap.isOpened():
                print("❌ 无法打开摄像头")
                return False

            self.connected = True
            print("✅ 备用模式连接成功（使用本地摄像头）")
            return True

        except Exception as e:
            print(f"❌ 备用模式连接失败: {e}")
            return False

    def get_frame(self):
        """获取无人机摄像头图像"""
        if not self.connected:
            return None

        try:
            if self.use_simulator and hasattr(self, 'client') and self.client is not None:
                # AirSim模式
                import airsim
                responses = self.client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                ])

                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                return img_bgr

            else:
                # 备用模式：本地摄像头
                if self.cap is None:
                    return None

                ret, frame = self.cap.read()
                if not ret:
                    # 如果视频结束，重新打开
                    if isinstance(self.test_video_source, str):
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.test_video_source)
                        ret, frame = self.cap.read()

                    if not ret:
                        return None

                return frame

        except Exception as e:
            print(f"❌ 获取图像失败: {e}")
            return None

    def move_to_target(self, control_command):
        """根据控制指令移动无人机"""
        if not self.connected:
            return

        try:
            # 解析控制指令
            vx = control_command.get('forward', 0.0) * self.velocity
            vy = control_command.get('right', 0.0) * self.velocity
            vz = control_command.get('up', 0.0) * self.velocity
            yaw_rate = control_command.get('yaw', 0.0) * 30.0  # 度/秒

            if self.use_simulator and hasattr(self, 'client') and self.client is not None:
                # AirSim控制
                import airsim
                self.client.moveByVelocityAsync(
                    vx, vy, vz, self.duration,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    airsim.YawMode(True, yaw_rate)
                )
            else:
                # 备用模式：模拟控制
                print(f"🎮 模拟控制指令: 前进={vx:.2f}, 右移={vy:.2f}, 上升={vz:.2f}, 偏航={yaw_rate:.1f}")

                # 更新虚拟位置（简化模型）
                self.virtual_position['x'] += vx * self.duration
                self.virtual_position['y'] += vy * self.duration
                self.virtual_position['z'] += vz * self.duration

        except Exception as e:
            print(f"❌ 控制无人机失败: {e}")

    def hover(self):
        """悬停无人机"""
        if self.connected:
            if self.use_simulator and hasattr(self, 'client') and self.client is not None:
                self.client.hoverAsync().join()
            else:
                print("🛸 模拟悬停")

    def return_to_home(self):
        """返回起始点"""
        if self.connected:
            if self.use_simulator and hasattr(self, 'client') and self.client is not None:
                self.client.goHomeAsync().join()
            else:
                print("🏠 模拟返回起点")
                # 重置虚拟位置
                self.virtual_position = {'x': 0, 'y': 0, 'z': 10}

    def get_position(self):
        """获取无人机当前位置"""
        if self.connected:
            if self.use_simulator and hasattr(self, 'client') and self.client is not None:
                import airsim
                state = self.client.getMultirotorState()
                return {
                    'x': state.kinematics_estimated.position.x_val,
                    'y': state.kinematics_estimated.position.y_val,
                    'z': state.kinematics_estimated.position.z_val
                }
            else:
                # 返回虚拟位置
                return self.virtual_position
        return None

    def disconnect(self):
        """断开连接"""
        if self.connected:
            try:
                if self.use_simulator and hasattr(self, 'client') and self.client is not None:
                    print("🛬 正在降落AirSim无人机...")
                    self.client.landAsync().join()
                    self.client.armDisarm(False)
                    self.client.enableApiControl(False)
                    print("✅ AirSim无人机已安全降落")
                else:
                    print("🛬 模拟降落无人机")

                # 关闭摄像头
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

            except Exception as e:
                print(f"❌ 断开连接时出错: {e}")
            finally:
                self.connected = False
                print("✅ 无人机已断开连接")


# 测试函数
def test_drone_controller():
    """测试无人机控制器"""
    print("🧪 测试无人机控制器...")

    # 创建控制器（不使用AirSim）
    drone = DroneController(use_simulator=False)

    if drone.connect():
        print("✅ 连接成功")

        # 测试获取图像
        for i in range(5):
            frame = drone.get_frame()
            if frame is not None:
                print(f"📸 获取到图像: {frame.shape}")
                cv2.imshow('Test Frame', frame)
                cv2.waitKey(100)
            else:
                print("❌ 无法获取图像")

        # 测试控制
        drone.move_to_target({'forward': 0.5, 'right': 0.0, 'up': 0.0, 'yaw': 0.0})

        # 获取位置
        pos = drone.get_position()
        print(f"📍 当前位置: {pos}")

        drone.hover()
        drone.disconnect()

        cv2.destroyAllWindows()
        print("✅ 测试完成")
    else:
        print("❌ 连接失败")


if __name__ == "__main__":
    test_drone_controller()