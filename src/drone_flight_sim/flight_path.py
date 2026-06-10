# flight_path.py
"""航点规划模块

本模块负责生成各种飞行路径，包括：
- 正方形路径
- 矩形路径
- 自定义路径
- 路径信息打印
"""

# 从 typing 模块导入 List 和 Tuple 类型，用于函数参数和返回值的类型注解
from typing import List, Tuple


class FlightPath:
    """飞行路径管理类

    提供静态方法生成不同形状的飞行路径。
    所有方法都是静态方法，可以通过类名直接调用。
    """

    @staticmethod
    def square_path(size: float = 10, height: float = -3) -> List[Tuple[float, float, float]]:
        """生成正方形飞行路径

        正方形路径以原点 (0, 0) 为左下角点，沿顺时针方向依次飞行四个角点。

        参数:
            size (float): 正方形边长（米），默认为 10
            height (float): 飞行高度（米，负值向上），默认为 -3

        返回:
            List[Tuple[float, float, float]]: 飞行点坐标列表，每个元素是 (x, y, z) 元组
        """
        # 返回四个角点的坐标列表，按顺时针顺序排列
        # 起点/左上角 (0, 0, height)
        # 右上角 (size, 0, height)
        # 右下角 (size, size, height)
        # 左下角 (0, size, height)
        return [
            # 起点/左上角点
            (0, 0, height),
            # 右上角点
            (size, 0, height),
            # 右下角点
            (size, size, height),
            # 左下角点
            (0, size, height)
        ]

    @staticmethod
    def rectangle_path(width: float = 20, length: float = 10, altitude: float = -3) -> List[Tuple[float, float, float]]:
        """生成矩形飞行路径

        矩形路径以原点 (0, 0) 为左下角点，沿顺时针方向依次飞行四个角点。

        参数:
            width (float): 矩形宽度，即 X 方向的距离（米），默认为 20
            length (float): 矩形长度，即 Y 方向的距离（米），默认为 10
            altitude (float): 飞行高度（米，负值向上），默认为 -3

        返回:
            List[Tuple[float, float, float]]: 飞行点坐标列表，每个元素是 (x, y, z) 元组
        """
        # 返回矩形四个角点的坐标列表
        # 右下角 (width, 0, altitude)
        # 左下角 (width, length, altitude)
        # 左上角 (0, length, altitude)
        # 右上角 (0, 0, altitude)
        return [
            # 右下角点
            (width, 0, altitude),
            # 左下角点
            (width, length, altitude),
            # 左上角点
            (0, length, altitude),
            # 右上角点
            (0, 0, altitude)
        ]
    
    @staticmethod
    def triangle_path(size: float = 10, height: float = -3) -> List[Tuple[float, float, float]]:
        """生成三角形飞行路径"""

        return [
            (0, 0, height),          # 起点
            (size, 0, height),       # 右下角
            (size / 2, size, height) # 顶点
        ]
    
    @staticmethod
    def custom_path(waypoints: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """自定义飞行路径

        直接使用传入的航点列表作为飞行路径，
        适用于用户需要指定特定飞行轨迹的场景。

        参数:
            waypoints (List[Tuple[float, float, float]]): 用户自定义的航点坐标列表

        返回:
            List[Tuple[float, float, float]]: 传入的航点坐标列表（直接返回原列表）
        """
        # 直接返回用户传入的航点列表
        return waypoints
    


    @staticmethod
    def print_path(waypoints: List[Tuple[float, float, float]]):
        """打印飞行路径信息

        在控制台以格式化的方式显示飞行路径的详细信息，
        包括路径点的数量和每个航点的坐标。

        参数:
            waypoints (List[Tuple[float, float, float]]): 需要打印的航点坐标列表
        """
        # 打印路径标题
        print("\n🗺️  飞行路径规划:")
        # 遍历所有航点，使用 enumerate 获取索引（从 1 开始）
        for i, (x, y, z) in enumerate(waypoints, 1):
            # 格式化输出每个航点的编号和坐标
            print(f"   航点{i}: ({x}, {y}, {z})")
