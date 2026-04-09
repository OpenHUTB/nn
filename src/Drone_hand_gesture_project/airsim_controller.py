# -*- coding: utf-8 -*-
"""
AirSim 无人机控制器
连接到 Microsoft AirSim 模拟器，实现真实的无人机控制
"""

import time
import numpy as np
from typing import Optional, Dict, Any


class AirSimController:
    """AirSim 无人机控制器"""
    
    def __init__(self, ip_address: str = "127.0.0.1", port: int = 41451):
        """
        初始化 AirSim 控制器
        
        Args:
            ip_address: AirSim 服务器地址
            port: AirSim RPC 端口
        """
        self.ip_address = ip_address
        self.port = port
        self.client = None
        self.connected = False
        self.drone_name = "Drone1"
        
        # 无人机状态
        self.state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),  # roll, pitch, yaw
            'angular_velocity': np.array([0.0, 0.0, 0.0]),
            'battery': 100.0,
            'armed': False,
            'flying': False
        }
        
        # 控制参数
        self.control_params = {
            'max_speed': 5.0,  # m/s
            'max_acceleration': 2.0,  # m/s²
            'takeoff_altitude': 2.0,  # m
        }
        
        # 传感器数据
        self.sensor_data = {
            'camera_image': None,
            'depth_image': None,
            'lidar_data': None,
            'imu_data': None
        }
        
    def connect(self) -> bool:
        """连接到 AirSim 模拟器"""
        try:
            import airsim
            
            print(f"正在连接到 AirSim ({self.ip_address}:{self.port})...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            # 启用无人机控制
            self.client.enableApiControl(True)
            self.client.armDisarm(True, vehicle_name=self.drone_name)
            
            self.connected = True
            self.state['armed'] = True
            
            print("✅ 成功连接到 AirSim 模拟器！")
            print(f"   无人机：{self.drone_name}")
            print(f"   状态：已解锁")
            
            return True
            
        except ImportError:
            print("❌ 未找到 airsim 模块")
            print("   请运行：pip install airsim")
            return False
            
        except Exception as e:
            print(f"❌ 连接失败：{e}")
            print("   请确保 AirSim 模拟器正在运行")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.client and self.connected:
            try:
                self.client.armDisarm(False, vehicle_name=self.drone_name)
                self.client.enableApiControl(False)
                self.connected = False
                print("✅ 已断开 AirSim 连接")
            except Exception as e:
                print(f"⚠️  断开连接时出错：{e}")
    
    def takeoff(self, altitude: Optional[float] = None) -> bool:
        """起飞"""
        if not self.connected:
            print("❌ 未连接到 AirSim")
            return False
        
        try:
            alt = altitude or self.control_params['takeoff_altitude']
            print(f"🚁 正在起飞到 {alt} 米高度...")
            
            # 异步起飞
            self.client.takeoffAsync(altitude=alt, vehicle_name=self.drone_name).join()
            
            self.state['flying'] = True
            print("✅ 起飞完成！")
            return True
            
        except Exception as e:
            print(f"❌ 起飞失败：{e}")
            return False
    
    def land(self) -> bool:
        """降落"""
        if not self.connected:
            print("❌ 未连接到 AirSim")
            return False
        
        try:
            print("🛬 正在降落...")
            self.client.landAsync(vehicle_name=self.drone_name).join()
            self.state['flying'] = False
            print("✅ 降落完成！")
            return True
            
        except Exception as e:
            print(f"❌ 降落失败：{e}")
            return False
    
    def hover(self):
        """悬停"""
        if not self.connected:
            return
        
        try:
            self.client.moveToPositionAsync(
                self.state['position'][0],
                self.state['position'][1],
                self.state['position'][2],
                1.0,
                vehicle_name=self.drone_name
            )
        except:
            pass
    
    def move_by_velocity(self, vx: float, vy: float, vz: float, duration: float = 1.0):
        """
        按速度控制无人机
        
        Args:
            vx: X 轴速度 (m/s), 前进为正
            vy: Y 轴速度 (m/s), 右移为正
            vz: Z 轴速度 (m/s), 下降为正
            duration: 持续时间 (秒)
        """
        if not self.connected:
            return
        
        try:
            # AirSim 坐标系：Z 轴向上，所以 vz 需要取反
            self.client.moveByVelocityZAsync(
                vx, vy, -vz,
                duration,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(),
                vehicle_name=self.drone_name
            )
        except Exception as e:
            print(f"⚠️  速度控制失败：{e}")
    
    def move_to_position(self, x: float, y: float, z: float, speed: float = 2.0):
        """
        移动到指定位置
        
        Args:
            x: X 坐标 (米)
            y: Y 坐标 (米)
            z: Z 坐标 (米，高度)
            speed: 移动速度 (m/s)
        """
        if not self.connected:
            return False
        
        try:
            print(f"📍 移动到位置 ({x:.1f}, {y:.1f}, {z:.1f})")
            self.client.moveToPositionAsync(
                x, y, -z, speed,
                vehicle_name=self.drone_name
            ).join()
            return True
        except Exception as e:
            print(f"❌ 移动失败：{e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """获取无人机状态"""
        if not self.connected:
            return self.state
        
        try:
            # 获取位置
            position = self.client.getMultirotorState(vehicle_name=self.drone_name).position
            self.state['position'] = np.array([
                position.x_val,
                position.y_val,
                -position.z_val  # 转换为 Z 轴向上
            ])
            
            # 获取速度
            velocity = self.client.getMultirotorState(vehicle_name=self.drone_name).velocity
            self.state['velocity'] = np.array([
                velocity.x_val,
                velocity.y_val,
                -velocity.z_val
            ])
            
            # 获取姿态（四元数转欧拉角）
            orientation = self.client.getMultirotorState(vehicle_name=self.drone_name).orientation
            self.state['orientation'] = self._quaternion_to_euler(
                orientation.w_val, orientation.x_val,
                orientation.y_val, orientation.z_val
            )
            
            return self.state
            
        except Exception as e:
            print(f"⚠️  获取状态失败：{e}")
            return self.state
    
    def get_camera_image(self, camera_name: int = 0, image_type: str = "scene") -> Optional[np.ndarray]:
        """
        获取相机图像
        
        Args:
            camera_name: 相机 ID (0=前置，1=下置等)
            image_type: 图像类型 ("scene"=场景，"depth"=深度，"segmentation"=分割)
            
        Returns:
            图像数组 (H, W, 3) 或 None
        """
        if not self.connected:
            return None
        
        try:
            import airsim
            
            # 设置图像类型
            if image_type == "depth":
                image_request = airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis)
            elif image_type == "segmentation":
                image_request = airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation)
            else:
                image_request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene)
            
            responses = self.client.simGetImages([image_request], vehicle_name=self.drone_name)
            
            if responses:
                response = responses[0]
                
                # 转换为 numpy 数组
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_height = response.height
                img_width = response.width
                
                img = img1d.reshape(img_height, img_width, 3)
                self.sensor_data['camera_image'] = img
                
                return img
                
        except Exception as e:
            print(f"⚠️  获取图像失败：{e}")
            return None
    
    def _quaternion_to_euler(self, w, x, y, z) -> np.ndarray:
        """四元数转欧拉角"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def record_flight_data(self, duration: float = 10.0, interval: float = 0.1) -> list:
        """
        记录飞行数据
        
        Args:
            duration: 记录时长 (秒)
            interval: 采样间隔 (秒)
            
        Returns:
            飞行数据列表
        """
        if not self.connected:
            print("❌ 未连接到 AirSim")
            return []
        
        print(f"📊 开始记录飞行数据，时长 {duration} 秒...")
        
        flight_data = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            state = self.get_state()
            
            data_point = {
                'timestamp': time.time() - start_time,
                'position': state['position'].tolist(),
                'velocity': state['velocity'].tolist(),
                'orientation': state['orientation'].tolist()
            }
            
            flight_data.append(data_point)
            time.sleep(interval)
        
        print(f"✅ 记录完成，共 {len(flight_data)} 个数据点")
        return flight_data
    
    def save_flight_data(self, data: list, filename: str = "flight_data.npy"):
        """保存飞行数据到文件"""
        try:
            np.save(filename, data)
            print(f"✅ 飞行数据已保存到 {filename}")
        except Exception as e:
            print(f"❌ 保存失败：{e}")


def test_airsim_connection():
    """测试 AirSim 连接"""
    print("=" * 60)
    print("AirSim 连接测试")
    print("=" * 60)
    
    controller = AirSimController()
    
    if controller.connect():
        print("\n✅ AirSim 连接成功！")
        
        # 获取状态
        state = controller.get_state()
        print(f"\n无人机状态:")
        print(f"  位置：{state['position']}")
        print(f"  速度：{state['velocity']}")
        print(f"  姿态：{np.degrees(state['orientation'])}°")
        
        # 获取图像
        img = controller.get_camera_image()
        if img is not None:
            print(f"\n✅ 成功获取相机图像：{img.shape}")
        
        controller.disconnect()
        return True
    else:
        print("\n❌ AirSim 连接失败")
        print("请确保:")
        print("  1. AirSim 模拟器已启动")
        print("  2. settings.json 已正确配置")
        return False


if __name__ == "__main__":
    test_airsim_connection()
