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
        self.vehicle_name = ""  # 空字符串表示默认飞行器
        
        # 无人机状态
        self.state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0]),
            'battery': 100.0,
            'armed': False,
            'flying': False
        }
        
    def connect(self) -> bool:
        """连接到 AirSim 模拟器"""
        try:
            import airsim
            
            print(f"正在连接到 AirSim ({self.ip_address}:{self.port})...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            self.client.enableApiControl(True)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            
            self.connected = True
            self.state['armed'] = True
            
            print("[OK] 成功连接到 AirSim 模拟器！")
            print(f"   飞行器：{self.vehicle_name if self.vehicle_name else '默认'}")
            print(f"   状态：已解锁")
            
            return True
            
        except ImportError:
            print("[ERROR] 未找到 airsim 模块")
            print("   请运行：pip install airsim")
            return False
            
        except Exception as e:
            print(f"[ERROR] 连接失败 - {e}")
            print("   请确保 AirSim 模拟器正在运行")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.client and self.connected:
            try:
                self.client.armDisarm(False, vehicle_name=self.vehicle_name)
                self.client.enableApiControl(False)
                self.connected = False
                print("[OK] 已断开 AirSim 连接")
            except Exception as e:
                print(f"Warning: 断开连接时出错 - {e}")
    
    def takeoff(self, altitude: float = 2.0) -> bool:
        """起飞"""
        if not self.connected:
            print("[ERROR] 未连接到 AirSim")
            return False
        
        try:
            print(f"[INFO] 正在起飞到 {altitude} 米高度...")
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            # 起飞后上升到指定高度
            import airsim
            self.client.moveToZAsync(-altitude, 1.0, vehicle_name=self.vehicle_name).join()
            self.state['flying'] = True
            print("[OK] 起飞完成！")
            return True
        except Exception as e:
            print(f"[ERROR] 起飞失败：{e}")
            return False
    
    def land(self) -> bool:
        """降落"""
        if not self.connected:
            print("[ERROR] 未连接到 AirSim")
            return False
        
        try:
            print("[INFO] 正在降落...")
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            self.state['flying'] = False
            print("[OK] 降落完成！")
            return True
        except Exception as e:
            print(f"[ERROR] 降落失败：{e}")
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
                vehicle_name=self.vehicle_name
            )
        except:
            pass
    
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
        except Exception as e:
            print(f"Warning: 速度控制失败 - {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """获取无人机状态"""
        if not self.connected:
            return self.state
        
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            # 修复状态获取 - 使用正确的属性名称
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
            print(f"Warning: 获取状态失败 - {e}")
            return self.state


def test_airsim_connection():
    """测试 AirSim 连接"""
    print("=" * 60)
    print("AirSim 连接测试")
    print("=" * 60)
    
    controller = AirSimController()
    
    if controller.connect():
        print("\n[OK] AirSim 连接成功！")
        state = controller.get_state()
        print(f"位置：{state['position']}")
        controller.disconnect()
        return True
    else:
        print("\n[ERROR] AirSim 连接失败")
        return False


if __name__ == "__main__":
    test_airsim_connection()
