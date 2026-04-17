# flight_path.py
"""航点规划模块"""

from typing import List, Tuple

class FlightPath:
    """飞行路径管理"""
    
    @staticmethod
    def square_path(size: float = 10, height: float = -3) -> List[Tuple[float, float, float]]:
        """正方形路径"""
        return [
            (size, 0, height),
            (size, size, height),
            (0, size, height),
            (0, 0, height)
        ]
    
    @staticmethod
    def rectangle_path(width: float = 20, height: float = 10, altitude: float = -3) -> List[Tuple[float, float, float]]:
        """矩形路径"""
        return [
            (width, 0, altitude),
            (width, height, altitude),
            (0, height, altitude),
            (0, 0, altitude)
        ]
    
    @staticmethod
    def custom_path(waypoints: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """自定义路径"""
        return waypoints
    
    @staticmethod
    def print_path(waypoints: List[Tuple[float, float, float]]):
        """打印路径信息"""
        print("\n🗺️  飞行路径规划:")
        for i, (x, y, z) in enumerate(waypoints, 1):
            print(f"   航点{i}: ({x}, {y}, {z})")