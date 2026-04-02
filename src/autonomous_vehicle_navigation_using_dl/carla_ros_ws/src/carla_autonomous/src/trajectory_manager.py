"""
轨迹管理器 - 管理规划路线和轨迹点
"""

import carla
import math

class TrajectoryManager:
    """轨迹管理器"""
    
    def __init__(self, env):
        self.env = env
        self.route_points = []
        
    def get_route_points(self):
        """获取规划路线点"""
        if not self.route_points:
            self.route_points = self._extract_route_points()
        
        return self.route_points
    
    def _extract_route_points(self):
        """从环境中提取规划路线的点"""
        route_points = []
        try:
            # 获取规划轨迹
            trajectory = self.env.trajectory(draw=False)
            
            for waypoint, road_option in trajectory:
                location = waypoint.transform.location
                route_points.append((location.x, location.y, location.z))
            
            print(f"📊 提取到 {len(route_points)} 个路径点")
            
            return route_points
            
        except Exception as e:
            print(f"❌ 提取路线点失败: {e}")
            return []
    
    def calculate_route_length(self):
        """计算规划路线总长度"""
        if len(self.route_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self.route_points) - 1):
            x1, y1, _ = self.route_points[i]
            x2, y2, _ = self.route_points[i+1]
            total_length += math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        return total_length
    
    def find_closest_point(self, x, y):
        """找到距离给定位置最近的路径点"""
        if not self.route_points:
            return -1, float('inf')
        
        min_distance = float('inf')
        closest_index = -1
        
        for i, point in enumerate(self.route_points):
            px, py, _ = point
            distance = math.sqrt((px - x)**2 + (py - y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index, min_distance
    
    def get_remaining_route(self, current_x, current_y):
        """获取剩余路线"""
        if not self.route_points:
            return []
        
        closest_idx, _ = self.find_closest_point(current_x, current_y)
        
        if closest_idx >= 0:
            return self.route_points[closest_idx:]
        else:
            return self.route_points
    
    def reset(self):
        """重置轨迹管理器"""
        self.route_points = []
        print("🔄 轨迹管理器已重置")
    
    def get_route_info(self):
        """获取路线信息"""
        if not self.route_points:
            return {
                'point_count': 0,
                'total_length': 0.0,
                'has_route': False
            }
        
        total_length = self.calculate_route_length()
        
        return {
            'point_count': len(self.route_points),
            'total_length': total_length,
            'has_route': True,
            'start_point': self.route_points[0] if self.route_points else None,
            'end_point': self.route_points[-1] if self.route_points else None
        }