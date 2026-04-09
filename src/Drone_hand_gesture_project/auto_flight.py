# -*- coding: utf-8 -*-
"""
自动飞行模式
实现预设轨迹的自动飞行
"""

import numpy as np
import time
from typing import Optional, List, Tuple


class AutoFlight:
    """自动飞行控制器"""
    
    def __init__(self, controller=None):
        """
        初始化自动飞行
        
        Args:
            controller: 无人机控制器实例
        """
        self.controller = controller
        self.is_flying = False
        self.current_trajectory = None
    
    def connect(self) -> bool:
        """连接控制器"""
        if self.controller:
            return self.controller.connected
        
        try:
            from airsim_controller import AirSimController
            self.controller = AirSimController()
            return self.controller.connect()
        except Exception as e:
            print(f"❌ 连接失败：{e}")
            return False
    
    def takeoff(self, altitude: float = 2.0) -> bool:
        """起飞"""
        if not self.controller:
            return False
        return self.controller.takeoff(altitude)
    
    def land(self) -> bool:
        """降落"""
        if not self.controller:
            return False
        return self.controller.land()
    
    def fly_circle(self, center: Tuple[float, float, float], 
                   radius: float, altitude: float,
                   speed: float = 1.0, laps: int = 1) -> bool:
        """
        圆形轨迹飞行
        
        Args:
            center: 圆心 (x, y)
            radius: 半径 (米)
            altitude: 高度 (米)
            speed: 速度 (m/s)
            laps: 圈数
        """
        if not self.controller:
            print("❌ 控制器未连接")
            return False
        
        print(f"🔵 开始圆形飞行：半径={radius}m, 高度={altitude}m, 圈数={laps}")
        
        self.is_flying = True
        cx, cy = center
        
        for lap in range(laps):
            if not self.is_flying:
                break
            
            print(f"  第 {lap+1}/{laps} 圈")
            
            # 分成 36 个点
            for angle in np.linspace(0, 2*np.pi, 36):
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                
                self.controller.move_to_position(x, y, altitude, speed)
                time.sleep(0.1)
        
        self.is_flying = False
        print("✅ 圆形飞行完成")
        return True
    
    def fly_figure8(self, center: Tuple[float, float, float],
                    width: float, height: float, altitude: float,
                    speed: float = 1.0) -> bool:
        """
        8 字形轨迹飞行
        
        Args:
            center: 中心点 (x, y)
            width: 宽度
            height: 高度
            altitude: 飞行高度
            speed: 速度
        """
        if not self.controller:
            return False
        
        print(f"♾️  开始 8 字形飞行：宽={width}m, 高={height}m")
        
        self.is_flying = True
        cx, cy = center
        
        # 8 字形参数方程
        for t in np.linspace(0, 2*np.pi, 72):
            if not self.is_flying:
                break
            
            x = cx + width * np.sin(t)
            y = cy + height * np.sin(2*t) / 2
            
            self.controller.move_to_position(x, y, altitude, speed)
            time.sleep(0.1)
        
        self.is_flying = False
        print("✅ 8 字形飞行完成")
        return True
    
    def fly_square(self, center: Tuple[float, float, float],
                   size: float, altitude: float,
                   speed: float = 1.0) -> bool:
        """
        方形轨迹飞行
        
        Args:
            center: 中心点
            size: 边长
            altitude: 高度
            speed: 速度
        """
        if not self.controller:
            return False
        
        print(f"⬜ 开始方形飞行：边长={size}m")
        
        self.is_flying = True
        cx, cy = center
        half = size / 2
        
        # 四个角点
        waypoints = [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
        
        for i, (x, y) in enumerate(waypoints):
            if not self.is_flying:
                break
            print(f"  航点 {i+1}/4")
            self.controller.move_to_position(x, y, altitude, speed)
            time.sleep(0.5)
        
        # 回到起点
        self.controller.move_to_position(waypoints[0][0], waypoints[0][1], altitude, speed)
        
        self.is_flying = False
        print("✅ 方形飞行完成")
        return True
    
    def emergency_stop(self):
        """紧急停止"""
        self.is_flying = False
        print("🛑 紧急停止！")
        
        if self.controller:
            self.controller.hover()
    
    def stop(self):
        """停止自动飞行"""
        self.is_flying = False
        print("⏹ 停止自动飞行")


def demo_auto_flight():
    """演示自动飞行"""
    print("=" * 60)
    print("自动飞行演示")
    print("=" * 60)
    
    auto = AutoFlight()
    
    if not auto.connect():
        print("\n⚠️ 无法连接，使用模拟模式")
        # 模拟飞行
        print("\n模拟圆形飞行...")
        time.sleep(2)
        print("✅ 模拟完成")
        return
    
    # 起飞
    auto.takeoff(altitude=2.0)
    time.sleep(1)
    
    # 执行飞行
    auto.fly_circle((0, 0), radius=3.0, altitude=2.0, laps=1)
    
    # 降落
    auto.land()
    
    print("\n✅ 演示完成！")


if __name__ == "__main__":
    demo_auto_flight()
