"""
路线可视化模块 - 在CARLA世界中绘制路线、车辆和路径
"""

import carla
import math
import time
import config as cfg

class RouteVisualizer:
    """路线可视化器"""
    
    def __init__(self, world):
        self.world = world
        self.vehicle_history = []  # 存储车辆历史位置
        
        # 使用配置文件中的颜色定义
        self.route_color = carla.Color(
            int(cfg.PLANNED_ROUTE_COLOR[0] * cfg.PLANNED_ROUTE_BRIGHTNESS),
            int(cfg.PLANNED_ROUTE_COLOR[1] * cfg.PLANNED_ROUTE_BRIGHTNESS),
            int(cfg.PLANNED_ROUTE_COLOR[2] * cfg.PLANNED_ROUTE_BRIGHTNESS)
        )
        self.path_color = carla.Color(
            int(cfg.HISTORY_PATH_COLOR[0] * cfg.HISTORY_PATH_BRIGHTNESS),
            int(cfg.HISTORY_PATH_COLOR[1] * cfg.HISTORY_PATH_BRIGHTNESS),
            int(cfg.HISTORY_PATH_COLOR[2] * cfg.HISTORY_PATH_BRIGHTNESS)
        )
        self.vehicle_color = carla.Color(
            int(cfg.VEHICLE_MARKER_COLOR[0] * cfg.VEHICLE_MARKER_BRIGHTNESS),
            int(cfg.VEHICLE_MARKER_COLOR[1] * cfg.VEHICLE_MARKER_BRIGHTNESS),
            int(cfg.VEHICLE_MARKER_COLOR[2] * cfg.VEHICLE_MARKER_BRIGHTNESS)
        )
        self.arrow_color = carla.Color(
            int(cfg.ARROW_COLOR[0] * cfg.ARROW_BRIGHTNESS),
            int(cfg.ARROW_COLOR[1] * cfg.ARROW_BRIGHTNESS),
            int(cfg.ARROW_COLOR[2] * cfg.ARROW_BRIGHTNESS)
        )
        
        # 显示高度
        self.route_height = cfg.ROUTE_HEIGHT
        self.path_height = cfg.PATH_HEIGHT
        self.vehicle_height = cfg.VEHICLE_HEIGHT
        
        # 存储绘制对象
        self.route_lines = []
        self.start_marker = None
        self.end_marker = None
        
    def draw_planned_route(self, route_points):
        """绘制规划路线（常亮显示）"""
        if not cfg.SHOW_PLANNED_ROUTE:
            return False
            
        # 清除之前的路线
        self.clear_route()
        
        if len(route_points) < 2:
            return False
        
        print(f"📏 绘制规划路线，共 {len(route_points)} 个点")
        
        # 绘制整条路线
        for i in range(len(route_points) - 1):
            start = self._create_location(route_points[i], self.route_height)
            end = self._create_location(route_points[i+1], self.route_height)
            
            # 绘制线段，使用长生命时间保证常亮
            line = self.world.debug.draw_line(
                start, end,
                thickness=cfg.PLANNED_ROUTE_THICKNESS,
                color=self.route_color,
                life_time=1000.0,
                persistent_lines=True
            )
            self.route_lines.append(line)
        
        # 绘制起点和终点标记
        self._draw_start_end_points(route_points)
        
        return True
    
    def _draw_start_end_points(self, route_points):
        """绘制起点和终点标记"""
        if len(route_points) == 0:
            return
        
        # 起点标记
        start_point = route_points[0]
        start_loc = self._create_location(start_point, self.route_height + 0.1)
        
        self.start_marker = self.world.debug.draw_point(
            start_loc,
            size=0.5,
            color=carla.Color(255, 165, 0),  # 橙色
            life_time=1000.0,
            persistent_lines=True
        )
        
        # 起点文字
        self.world.debug.draw_string(
            self._create_location(start_point, self.route_height + 1.0),
            'START',
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=1000.0
        )
        
        # 终点标记
        end_point = route_points[-1]
        end_loc = self._create_location(end_point, self.route_height + 0.1)
        
        self.end_marker = self.world.debug.draw_point(
            end_loc,
            size=0.5,
            color=carla.Color(255, 0, 255),  # 洋红色
            life_time=1000.0,
            persistent_lines=True
        )
        
        # 终点文字
        self.world.debug.draw_string(
            self._create_location(end_point, self.route_height + 1.0),
            'GOAL',
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=1000.0
        )
    
    def update_vehicle_display(self, x, y, heading):
        """更新车辆显示（位置和朝向）"""
        # 保存历史位置
        self.vehicle_history.append((x, y, heading, time.time()))
        
        # 保持历史点数量
        if len(self.vehicle_history) > cfg.HISTORY_PATH_MAX_POINTS:
            self.vehicle_history = self.vehicle_history[-cfg.HISTORY_PATH_MAX_POINTS:]
        
        # 绘制车辆当前位置和朝向
        if cfg.SHOW_VEHICLE_MARKER:
            self._draw_vehicle_current(x, y, heading)
        
        # 绘制历史路径
        if cfg.SHOW_HISTORY_PATH:
            self._draw_vehicle_history()
        
        # 更新信息显示
        self._update_info_display(x, y, heading)
    
    def _draw_vehicle_current(self, x, y, heading):
        """绘制车辆当前位置和朝向"""
        # 车辆位置点
        vehicle_loc = self._create_location((x, y, 0), self.vehicle_height)
        
        self.world.debug.draw_point(
            vehicle_loc,
            size=cfg.VEHICLE_MARKER_SIZE,
            color=self.vehicle_color,
            life_time=0.5,  # 稍微延长显示时间
            persistent_lines=False
        )
        
        # 车辆朝向箭头
        if cfg.SHOW_ARROW:
            arrow_length = cfg.ARROW_LENGTH
            angle_rad = math.radians(heading)
            end_x = x + arrow_length * math.cos(angle_rad)
            end_y = y + arrow_length * math.sin(angle_rad)
            
            self.world.debug.draw_arrow(
                vehicle_loc,
                self._create_location((end_x, end_y, 0), self.vehicle_height),
                thickness=cfg.ARROW_THICKNESS,
                arrow_size=0.6,
                color=self.arrow_color,
                life_time=0.5,
                persistent_lines=False
            )
        
        # 车辆轮廓（三角形）
        self._draw_vehicle_outline(x, y, heading)
    
    def _draw_vehicle_outline(self, x, y, heading):
        """绘制车辆轮廓三角形"""
        size = 1.2
        angle_rad = math.radians(heading)
        
        # 前顶点
        front_x = x + size * math.cos(angle_rad)
        front_y = y + size * math.sin(angle_rad)
        
        # 左后顶点
        left_x = x + size * 0.7 * math.cos(angle_rad + math.radians(140))
        left_y = y + size * 0.7 * math.sin(angle_rad + math.radians(140))
        
        # 右后顶点
        right_x = x + size * 0.7 * math.cos(angle_rad - math.radians(140))
        right_y = y + size * 0.7 * math.sin(angle_rad - math.radians(140))
        
        # 连接成三角形
        points = [
            self._create_location((front_x, front_y, 0), self.vehicle_height),
            self._create_location((left_x, left_y, 0), self.vehicle_height),
            self._create_location((right_x, right_y, 0), self.vehicle_height),
            self._create_location((front_x, front_y, 0), self.vehicle_height)  # 闭合
        ]
        
        for i in range(len(points) - 1):
            self.world.debug.draw_line(
                points[i], points[i+1],
                thickness=0.01,
                color=carla.Color(50, 255, 50),  # 亮绿色轮廓
                life_time=0.5,
                persistent_lines=False
            )
    
    def _draw_vehicle_history(self):
        """绘制车辆历史路径"""
        if len(self.vehicle_history) < 2:
            return
        
        # 确定要绘制的点范围
        if cfg.HISTORY_PATH_MAX_POINTS > 0:
            start_idx = max(0, len(self.vehicle_history) - cfg.HISTORY_PATH_MAX_POINTS)
        else:
            start_idx = 0
        
        for i in range(start_idx, len(self.vehicle_history) - 1):
            x1, y1, _, t1 = self.vehicle_history[i]
            x2, y2, _, t2 = self.vehicle_history[i+1]
            
            start = self._create_location((x1, y1, 0), self.path_height)
            end = self._create_location((x2, y2, 0), self.path_height)
            
            # 根据时间远近调整透明度
            if cfg.HISTORY_PATH_FADE_OUT:
                time_diff = t2 - t1
                if time_diff > 0:
                    # 计算衰减因子：越新的点越亮
                    age = len(self.vehicle_history) - i - 1
                    alpha = max(0.1, 1.0 / (1.0 + age * 0.05))
                else:
                    alpha = 0.5
            else:
                alpha = 1.0
                
            # 应用亮度配置
            brightness_factor = cfg.HISTORY_PATH_BRIGHTNESS * alpha
            
            color = carla.Color(
                int(self.path_color.r * brightness_factor),
                int(self.path_color.g * brightness_factor),
                int(self.path_color.b * brightness_factor)
            )
            
            self.world.debug.draw_line(
                start, end,
                thickness=cfg.HISTORY_PATH_THICKNESS,
                color=color,
                life_time=0.5,
                persistent_lines=False
            )
    
    def _update_info_display(self, x, y, heading):
        """更新信息显示"""
        info_height = 10.0
        
        # 计算路径长度
        path_length = self.calculate_path_length()
        
        # 显示车辆信息
        info_text = f"Vehicle: ({x:.1f}, {y:.1f}) | Heading: {heading:.1f}°"
        self.world.debug.draw_string(
            carla.Location(-30, 3, info_height),
            info_text,
            draw_shadow=True,
            color=carla.Color(255, 255, 255),
            life_time=0.3
        )
        
        # 显示路径长度
        path_text = f"Path Length: {path_length:.1f}m"
        self.world.debug.draw_string(
            carla.Location(-30, 2, info_height),
            path_text,
            draw_shadow=True,
            color=carla.Color(200, 200, 255),
            life_time=0.3
        )
        
        # 显示历史点数
        history_text = f"History Points: {len(self.vehicle_history)}/{cfg.HISTORY_PATH_MAX_POINTS}"
        self.world.debug.draw_string(
            carla.Location(-30, 1, info_height),
            history_text,
            draw_shadow=True,
            color=carla.Color(255, 200, 200),
            life_time=0.3
        )
        
        # 显示可视化状态
        viz_status = f"Viz: Arrow={cfg.SHOW_ARROW}, History={cfg.SHOW_HISTORY_PATH}, Route={cfg.SHOW_PLANNED_ROUTE}"
        self.world.debug.draw_string(
            carla.Location(-30, 0, info_height),
            viz_status,
            draw_shadow=True,
            color=carla.Color(200, 255, 200),
            life_time=0.3
        )
    
    def calculate_path_length(self):
        """计算已行驶路径长度"""
        if len(self.vehicle_history) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.vehicle_history)):
            x1, y1, _, _ = self.vehicle_history[i-1]
            x2, y2, _, _ = self.vehicle_history[i]
            dx = x2 - x1
            dy = y2 - y1
            total_length += math.sqrt(dx*dx + dy*dy)
        
        return total_length
    
    def reset_history(self):
        """重置历史记录"""
        self.vehicle_history = []
        print("🔄 车辆历史记录已重置")
    
    def clear_route(self):
        """清除路线绘制"""
        # CARLA会自动清理过期的debug绘制
        self.route_lines = []
        self.start_marker = None
        self.end_marker = None
    
    def _create_location(self, point, z_offset=0):
        """创建Location对象"""
        if len(point) >= 3:
            return carla.Location(point[0], point[1], point[2] + z_offset)
        else:
            return carla.Location(point[0], point[1], z_offset)
    
    def get_vehicle_history(self):
        """获取车辆历史记录"""
        return self.vehicle_history