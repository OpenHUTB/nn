"""
3D环境建设模块
实现地形生成、环境建模、可视化等功能
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import random
import time
from scipy import ndimage
from scipy.interpolate import CubicSpline

class TerrainConfig:
    """地形配置参数"""
    
    def __init__(self, 
                 width: float = 100.0,
                 length: float = 100.0,
                 height_range: float = 20.0,
                 resolution: float = 2.0,
                 seed: int = 42,
                 has_hills: bool = True,
                 has_valleys: bool = True,
                 has_river: bool = False,
                 vegetation_density: float = 0.1,
                 tree_density: float = 0.02,
                 water_level: float = 5.0):
        self.width = width
        self.length = length
        self.height_range = height_range
        self.resolution = resolution
        self.seed = seed
        self.has_hills = has_hills
        self.has_valleys = has_valleys
        self.has_river = has_river
        self.vegetation_density = vegetation_density
        self.tree_density = tree_density
        self.water_level = water_level

class TerrainGenerator:
    """地形生成器"""
    
    def __init__(self, config: TerrainConfig):
        """初始化地形生成器"""
        self.config = config
        self.heightmap = None
        self.terrain_mesh = None
        self.obstacles_mesh = []
        self.water_mesh = None
        
        # 设置随机种子
        np.random.seed(config.seed)
        random.seed(config.seed)
        
    def generate_terrain(self) -> np.ndarray:
        """生成地形高度图"""
        # 计算网格尺寸
        nx = int(self.config.width / self.config.resolution)
        ny = int(self.config.length / self.config.resolution)
        
        # 初始化高度图
        heightmap = np.zeros((nx, ny))
        
        # 添加基本噪声
        for i in range(nx):
            for j in range(ny):
                noise = np.random.randn() * 0.5
                heightmap[i, j] = noise
        
        # 使用高斯滤波平滑
        heightmap = ndimage.gaussian_filter(heightmap, sigma=1.0)
        
        # 归一化到 [0, 1]
        heightmap_min = heightmap.min()
        heightmap_max = heightmap.max()
        if heightmap_max - heightmap_min > 0:
            heightmap = (heightmap - heightmap_min) / (heightmap_max - heightmap_min)
        
        # 应用高度范围
        heightmap = heightmap * self.config.height_range
        
        # 添加丘陵
        if self.config.has_hills:
            num_hills = random.randint(3, 8)
            for _ in range(num_hills):
                center_x = random.uniform(0.2, 0.8) * nx
                center_y = random.uniform(0.2, 0.8) * ny
                radius = random.uniform(5, 15)
                height = random.uniform(3, 8)
                
                for i in range(nx):
                    for j in range(ny):
                        distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                        if distance < radius:
                            # 高斯形状的山丘
                            hill_height = height * np.exp(-(distance**2) / (2 * (radius/3)**2))
                            heightmap[i, j] += hill_height
        
        # 添加山谷/河流
        if self.config.has_valleys or self.config.has_river:
            if self.config.has_river:
                # 创建弯曲的河流
                river_path = self._generate_river_path(nx, ny)
                for point in river_path:
                    x, y = point
                    radius = 3
                    for i in range(max(0, int(x)-radius), min(nx, int(x)+radius+1)):
                        for j in range(max(0, int(y)-radius), min(ny, int(y)+radius+1)):
                            distance = np.sqrt((i - x)**2 + (j - y)**2)
                            if distance < radius:
                                heightmap[i, j] -= (1 - distance/radius) * 3
            else:
                # 添加随机山谷
                num_valleys = random.randint(2, 5)
                for _ in range(num_valleys):
                    start_x = random.uniform(0.1, 0.9) * nx
                    start_y = random.uniform(0.1, 0.9) * ny
                    length = random.uniform(20, 40)
                    angle = random.uniform(0, 2*np.pi)
                    width = random.uniform(3, 8)
                    
                    for i in range(nx):
                        for j in range(ny):
                            # 计算点到线段的距离
                            distance = self._distance_to_line(i, j, start_x, start_y, length, angle)
                            if distance < width:
                                valley_depth = 2 * (1 - distance/width)
                                heightmap[i, j] -= valley_depth
        
        # 确保最小高度为0
        heightmap = heightmap - heightmap.min()
        
        self.heightmap = heightmap
        self._create_mesh()
        
        return heightmap
    
    def _generate_river_path(self, nx: int, ny: int) -> list:
        """生成河流路径"""
        path = []
        
        # 随机起点和终点
        start_x = random.uniform(0.1 * nx, 0.4 * nx)
        start_y = 0 if random.random() < 0.5 else ny-1
        
        end_x = random.uniform(0.6 * nx, 0.9 * nx)
        end_y = ny-1 if start_y == 0 else 0
        
        # 生成控制点进行插值
        num_points = 10
        t = np.linspace(0, 1, num_points)
        
        # 添加一些随机偏移创建弯曲效果
        control_x = np.linspace(start_x, end_x, num_points)
        control_y = np.linspace(start_y, end_y, num_points)
        
        for i in range(1, num_points-1):
            control_x[i] += random.uniform(-0.1 * nx, 0.1 * nx)
            control_y[i] += random.uniform(-0.1 * ny, 0.1 * ny)
        
        # 三次样条插值
        cs_x = CubicSpline(t, control_x)
        cs_y = CubicSpline(t, control_y)
        
        # 生成更密集的点
        t_dense = np.linspace(0, 1, 100)
        for t_val in t_dense:
            x = cs_x(t_val)
            y = cs_y(t_val)
            if 0 <= x < nx and 0 <= y < ny:
                path.append((x, y))
        
        return path
    
    def _distance_to_line(self, x: float, y: float, start_x: float, start_y: float, 
                         length: float, angle: float) -> float:
        """计算点到线段的距离"""
        end_x = start_x + length * np.cos(angle)
        end_y = start_y + length * np.sin(angle)
        
        # 计算点到直线的最短距离
        numerator = abs((end_y - start_y)*x - (end_x - start_x)*y + end_x*start_y - end_y*start_x)
        denominator = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2)
        
        return numerator / denominator if denominator > 0 else float('inf')
    
    def _create_mesh(self):
        """从高度图创建网格"""
        if self.heightmap is None:
            return
        
        nx, ny = self.heightmap.shape
        
        # 创建顶点
        vertices = []
        for i in range(nx):
            for j in range(ny):
                x = i * self.config.resolution
                y = j * self.config.resolution
                z = self.heightmap[i, j]
                vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # 创建三角形面片
        faces = []
        for i in range(nx-1):
            for j in range(ny-1):
                idx1 = i * ny + j
                idx2 = i * ny + (j+1)
                idx3 = (i+1) * ny + j
                idx4 = (i+1) * ny + (j+1)
                
                # 两个三角形组成一个网格单元
                faces.append([idx1, idx2, idx3])
                faces.append([idx2, idx4, idx3])
        
        self.terrain_mesh = {
            'vertices': vertices,
            'faces': np.array(faces)
        }
        
        # 创建水面
        if self.config.water_level > 0:
            self._create_water_mesh()
    
    def _create_water_mesh(self):
        """创建水面网格"""
        nx, ny = self.heightmap.shape
        
        vertices = []
        faces = []
        
        # 创建平面水面
        water_height = self.config.water_level
        
        # 四个顶点
        vertices.append([0, 0, water_height])
        vertices.append([0, ny * self.config.resolution, water_height])
        vertices.append([nx * self.config.resolution, 0, water_height])
        vertices.append([nx * self.config.resolution, ny * self.config.resolution, water_height])
        
        # 两个三角形
        faces.append([0, 1, 2])
        faces.append([1, 3, 2])
        
        self.water_mesh = {
            'vertices': np.array(vertices),
            'faces': np.array(faces)
        }
    
    def add_building(self, position: list, dimensions: list):
        """添加建筑物障碍物"""
        x, y, z_base = position
        width, depth, height = dimensions
        
        # 建筑物底部顶点
        vertices = [
            [x, y, z_base],
            [x + width, y, z_base],
            [x + width, y + depth, z_base],
            [x, y + depth, z_base],
            [x, y, z_base + height],
            [x + width, y, z_base + height],
            [x + width, y + depth, z_base + height],
            [x, y + depth, z_base + height]
        ]
        
        # 立方体的6个面（12个三角形）
        faces = [
            [0, 1, 2], [0, 2, 3],  # 底面
            [4, 5, 6], [4, 6, 7],  # 顶面
            [0, 1, 5], [0, 5, 4],  # 前面
            [2, 3, 7], [2, 7, 6],  # 后面
            [1, 2, 6], [1, 6, 5],  # 右面
            [0, 3, 7], [0, 7, 4]   # 左面
        ]
        
        building_mesh = {
            'vertices': np.array(vertices),
            'faces': np.array(faces),
            'centroid': [x + width/2, y + depth/2, z_base + height/2],
            'bounds': [[x, y, z_base], [x+width, y+depth, z_base+height]]
        }
        
        self.obstacles_mesh.append(building_mesh)
        
        # 调整地形高度（建筑物区域需要平整）
        if self.heightmap is not None:
            nx, ny = self.heightmap.shape
            building_grid_x = int(x / self.config.resolution)
            building_grid_y = int(y / self.config.resolution)
            building_grid_w = int(width / self.config.resolution)
            building_grid_d = int(depth / self.config.resolution)
            
            for i in range(building_grid_x, min(building_grid_x + building_grid_w, nx)):
                for j in range(building_grid_y, min(building_grid_y + building_grid_d, ny)):
                    if 0 <= i < nx and 0 <= j < ny:
                        # 将建筑物区域的地形高度设置为建筑物基础高度
                        self.heightmap[i, j] = z_base
    
    def add_tree_obstacle(self, position: tuple, height: float = 10.0):
        """添加树木障碍物"""
        x, y = position
        
        # 获取地形高度
        if self.heightmap is not None:
            nx, ny = self.heightmap.shape
            grid_x = min(int(x / self.config.resolution), nx-1)
            grid_y = min(int(y / self.config.resolution), ny-1)
            z_base = self.heightmap[grid_x, grid_y]
        else:
            z_base = 0
        
        # 树冠（圆锥体）
        crown_radius = 2.0
        crown_height = height * 0.7
        crown_base = z_base + height * 0.3
        
        # 创建树冠网格（圆锥体）
        crown_vertices = []
        crown_faces = []
        
        # 圆锥顶点
        crown_vertices.append([x, y, crown_base + crown_height])
        
        # 圆锥底部圆周上的点
        num_segments = 8
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            vx = x + crown_radius * np.cos(angle)
            vy = y + crown_radius * np.sin(angle)
            crown_vertices.append([vx, vy, crown_base])
        
        # 圆锥侧面三角形
        for i in range(num_segments):
            crown_faces.append([0, i+1, (i+1)%num_segments + 1])
        
        # 圆锥底面三角形
        center_idx = num_segments + 1
        crown_vertices.append([x, y, crown_base])  # 底面中心点
        for i in range(num_segments):
            crown_faces.append([center_idx, i+1, (i+1)%num_segments + 1])
        
        tree_mesh = {
            'vertices': np.array(crown_vertices),
            'faces': np.array(crown_faces),
            'centroid': [x, y, crown_base + crown_height/2],
            'bounds': [[x-crown_radius, y-crown_radius, z_base], 
                      [x+crown_radius, y+crown_radius, z_base+height]],
            'type': 'tree'
        }
        
        self.obstacles_mesh.append(tree_mesh)
    
    def add_vegetation(self, density: float = None):
        """添加植被"""
        if density is None:
            density = self.config.vegetation_density
        
        if self.heightmap is None:
            return
        
        nx, ny = self.heightmap.shape
        num_trees = int(nx * ny * density)
        
        for _ in range(num_trees):
            x = random.uniform(0, nx * self.config.resolution)
            y = random.uniform(0, ny * self.config.resolution)
            
            # 获取地形高度
            grid_x = min(int(x / self.config.resolution), nx-1)
            grid_y = min(int(y / self.config.resolution), ny-1)
            z_base = self.heightmap[grid_x, grid_y]
            
            # 避免在水面或建筑物上种植
            if (self.config.water_level > 0 and z_base < self.config.water_level + 1):
                continue
            
            # 随机树木高度
            tree_height = random.uniform(5, 15)
            self.add_tree_obstacle((x, y), tree_height)
    
    def export_environment(self, filepath: str):
        """导出环境数据"""
        export_data = {
            'config': {
                'width': self.config.width,
                'length': self.config.length,
                'height_range': self.config.height_range,
                'resolution': self.config.resolution,
                'water_level': self.config.water_level
            },
            'terrain': {
                'heightmap': self.heightmap.tolist() if self.heightmap is not None else [],
                'mesh': {
                    'vertices': self.terrain_mesh['vertices'].tolist() if self.terrain_mesh else [],
                    'faces': self.terrain_mesh['faces'].tolist() if self.terrain_mesh else []
                }
            },
            'obstacles': [
                {
                    'type': obs.get('type', 'building'),
                    'vertices': obs['vertices'].tolist(),
                    'faces': obs['faces'].tolist(),
                    'centroid': obs.get('centroid', []),
                    'bounds': obs.get('bounds', [])
                } for obs in self.obstacles_mesh
            ],
            'water': {
                'vertices': self.water_mesh['vertices'].tolist() if self.water_mesh else [],
                'faces': self.water_mesh['faces'].tolist() if self.water_mesh else []
            } if self.water_mesh else None
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"环境数据已导出到: {filepath}")

class Drone3DModel:
    """无人机3D模型"""
    
    def __init__(self, model_type: str = "quadcopter", scale: float = 1.0):
        """
        初始化无人机模型
        
        Args:
            model_type: 模型类型 ('quadcopter', 'fixed_wing')
            scale: 模型缩放比例
        """
        self.model_type = model_type
        self.scale = scale
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])  # 欧拉角 (roll, pitch, yaw)
        
        # 创建模型网格
        self.mesh = self._create_model_mesh()
    
    def _create_model_mesh(self):
        """创建无人机模型网格"""
        if self.model_type == "quadcopter":
            return self._create_quadcopter_mesh()
        elif self.model_type == "fixed_wing":
            return self._create_fixed_wing_mesh()
        else:
            return self._create_basic_drone_mesh()
    
    def _create_quadcopter_mesh(self):
        """创建四旋翼无人机模型"""
        vertices = []
        faces = []
        
        # 机身（中心立方体）
        body_size = 0.3 * self.scale
        half_size = body_size / 2
        
        # 机身顶点
        body_vertices = [
            [-half_size, -half_size, -half_size],
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, half_size],
            [-half_size, half_size, half_size]
        ]
        
        # 机身面
        body_faces = [
            [0, 1, 2], [0, 2, 3],  # 底面
            [4, 5, 6], [4, 6, 7],  # 顶面
            [0, 1, 5], [0, 5, 4],  # 前面
            [2, 3, 7], [2, 7, 6],  # 后面
            [1, 2, 6], [1, 6, 5],  # 右面
            [0, 3, 7], [0, 7, 4]   # 左面
        ]
        
        vertices.extend(body_vertices)
        face_offset = len(vertices) - 8
        faces.extend([[v + face_offset for v in face] for face in body_faces])
        
        # 四个旋翼臂
        arm_length = 0.8 * self.scale
        arm_radius = 0.05 * self.scale
        
        arm_positions = [
            [arm_length, 0, 0],      # 右
            [-arm_length, 0, 0],     # 左
            [0, arm_length, 0],      # 前
            [0, -arm_length, 0]      # 后
        ]
        
        for i, (dx, dy, dz) in enumerate(arm_positions):
            # 旋翼臂（圆柱体简化）
            arm_vertices = [
                [dx - arm_radius, dy - arm_radius, dz],
                [dx + arm_radius, dy - arm_radius, dz],
                [dx + arm_radius, dy + arm_radius, dz],
                [dx - arm_radius, dy + arm_radius, dz],
                [0 - arm_radius, 0 - arm_radius, 0],
                [0 + arm_radius, 0 - arm_radius, 0],
                [0 + arm_radius, 0 + arm_radius, 0],
                [0 - arm_radius, 0 + arm_radius, 0]
            ]
            
            arm_faces = [
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7]
            ]
            
            vertices.extend(arm_vertices)
            face_offset = len(vertices) - 8
            faces.extend([[v + face_offset for v in face] for face in arm_faces])
            
            # 旋翼（圆形）
            rotor_radius = 0.2 * self.scale
            num_segments = 8
            
            rotor_center = [dx, dy, dz + 0.1 * self.scale]
            rotor_vertices = [rotor_center]
            
            for j in range(num_segments):
                angle = 2 * np.pi * j / num_segments
                vx = dx + rotor_radius * np.cos(angle)
                vy = dy + rotor_radius * np.sin(angle)
                rotor_vertices.append([vx, vy, dz + 0.1 * self.scale])
            
            vertices.extend(rotor_vertices)
            
            # 旋翼面
            center_idx = len(vertices) - num_segments - 1
            for j in range(num_segments):
                vertex_idx = center_idx + j + 1
                next_vertex_idx = center_idx + ((j + 1) % num_segments) + 1
                faces.append([center_idx, vertex_idx, next_vertex_idx])
        
        return {
            'vertices': np.array(vertices),
            'faces': np.array(faces),
            'color': '#3498db'  # 蓝色
        }
    
    def _create_fixed_wing_mesh(self):
        """创建固定翼无人机模型"""
        vertices = []
        faces = []
        
        # 机身（流线型）
        body_length = 1.5 * self.scale
        body_radius = 0.1 * self.scale
        
        # 创建机身顶点（简化）
        num_sections = 8
        for i in range(num_sections):
            t = i / (num_sections - 1)
            z = (t - 0.5) * body_length
            radius = body_radius * (1 - abs(t-0.5)*1.5)  # 中间粗，两头细
            
            for j in range(8):
                angle = 2 * np.pi * j / 8
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                vertices.append([x, y, z])
        
        # 创建机翼
        wing_span = 2.0 * self.scale
        wing_chord = 0.4 * self.scale
        
        wing_vertices = [
            [-wing_chord/2, -wing_span/2, 0],
            [wing_chord/2, -wing_span/2, 0],
            [wing_chord/2, wing_span/2, 0],
            [-wing_chord/2, wing_span/2, 0]
        ]
        
        wing_faces = [[0, 1, 2], [0, 2, 3]]
        
        vertices.extend(wing_vertices)
        face_offset = len(vertices) - 4
        faces.extend([[v + face_offset for v in face] for face in wing_faces])
        
        # 尾翼
        tail_span = 0.6 * self.scale
        tail_chord = 0.2 * self.scale
        
        tail_vertices = [
            [-tail_chord/2, -tail_span/2, -body_length/2],
            [tail_chord/2, -tail_span/2, -body_length/2],
            [tail_chord/2, tail_span/2, -body_length/2],
            [-tail_chord/2, tail_span/2, -body_length/2]
        ]
        
        tail_faces = [[0, 1, 2], [0, 2, 3]]
        
        vertices.extend(tail_vertices)
        face_offset = len(vertices) - 4
        faces.extend([[v + face_offset for v in face] for face in tail_faces])
        
        return {
            'vertices': np.array(vertices),
            'faces': np.array(faces),
            'color': '#e74c3c'  # 红色
        }
    
    def _create_basic_drone_mesh(self):
        """创建基本无人机模型"""
        vertices = [
            [0, 0, 0],
            [0.5 * self.scale, 0, 0],
            [0, 0.5 * self.scale, 0],
            [-0.5 * self.scale, 0, 0],
            [0, -0.5 * self.scale, 0],
            [0, 0, 0.5 * self.scale]
        ]
        
        faces = [
            [0, 1, 5],
            [0, 2, 5],
            [0, 3, 5],
            [0, 4, 5],
            [1, 2, 0],
            [2, 3, 0],
            [3, 4, 0],
            [4, 1, 0]
        ]
        
        return {
            'vertices': np.array(vertices),
            'faces': np.array(faces),
            'color': '#2ecc71'  # 绿色
        }
    
    def update_pose(self, position: np.ndarray, orientation: np.ndarray = None):
        """
        更新无人机位姿
        
        Args:
            position: 位置 [x, y, z]
            orientation: 欧拉角 [roll, pitch, yaw]（弧度）
        """
        self.position = position
        
        if orientation is not None:
            self.orientation = orientation
    
    def get_transformed_mesh(self):
        """获取变换后的模型网格"""
        if self.mesh is None:
            return None
        
        vertices = self.mesh['vertices'].copy()
        
        # 应用旋转
        roll, pitch, yaw = self.orientation
        
        # 绕Z轴旋转（偏航）
        if yaw != 0:
            rotation_z = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            vertices = vertices @ rotation_z.T
        
        # 绕Y轴旋转（俯仰）
        if pitch != 0:
            rotation_y = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            vertices = vertices @ rotation_y.T
        
        # 绕X轴旋转（滚转）
        if roll != 0:
            rotation_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
            vertices = vertices @ rotation_x.T
        
        # 应用平移
        vertices += self.position
        
        return {
            'vertices': vertices,
            'faces': self.mesh['faces'].copy(),
            'color': self.mesh['color']
        }

class Environment3DVisualizer:
    """3D环境可视化器"""
    
    def __init__(self, render_engine: str = "matplotlib"):
        """
        初始化可视化器
        
        Args:
            render_engine: 渲染引擎 ('matplotlib' 或 'plotly')
        """
        self.render_engine = render_engine
        
    def visualize_terrain(self, terrain_gen: TerrainGenerator, 
                         drones: list = None,
                         paths: list = None):
        """
        可视化地形环境
        
        Args:
            terrain_gen: 地形生成器
            drones: 无人机列表
            paths: 路径列表
        """
        if self.render_engine == "matplotlib":
            self._visualize_with_matplotlib(terrain_gen, drones, paths)
        elif self.render_engine == "plotly":
            self._visualize_with_plotly(terrain_gen, drones, paths)
        else:
            print(f"未知的渲染引擎: {self.render_engine}，使用matplotlib")
            self._visualize_with_matplotlib(terrain_gen, drones, paths)
    
    def _visualize_with_matplotlib(self, terrain_gen: TerrainGenerator,
                                  drones: list,
                                  paths: list):
        """使用matplotlib进行可视化"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制地形
        if terrain_gen.terrain_mesh:
            vertices = terrain_gen.terrain_mesh['vertices']
            faces = terrain_gen.terrain_mesh['faces']
            
            # 创建Poly3DCollection
            poly3d = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] 
                     for face in faces[::10]]  # 每10个面绘制一个以加速
            
            terrain_collection = Poly3DCollection(poly3d, 
                                                 facecolors='#8B7355', 
                                                 edgecolors='#654321',
                                                 alpha=0.8,
                                                 linewidths=0.3)
            ax.add_collection3d(terrain_collection)
        
        # 绘制水面
        if terrain_gen.water_mesh:
            vertices = terrain_gen.water_mesh['vertices']
            faces = terrain_gen.water_mesh['faces']
            
            poly3d = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] 
                     for face in faces]
            
            water_collection = Poly3DCollection(poly3d, 
                                               facecolors='#1E90FF', 
                                               edgecolors='#1C86EE',
                                               alpha=0.3,
                                               linewidths=0.5)
            ax.add_collection3d(water_collection)
        
        # 绘制障碍物
        for obstacle in terrain_gen.obstacles_mesh:
            vertices = obstacle['vertices']
            faces = obstacle['faces']
            
            if obstacle.get('type') == 'tree':
                color = '#228B22'  # 森林绿
                alpha = 0.7
            else:
                color = '#A0522D'  # 棕色
                alpha = 0.8
            
            poly3d = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] 
                     for face in faces]
            
            obstacle_collection = Poly3DCollection(poly3d, 
                                                  facecolors=color, 
                                                  edgecolors='#654321',
                                                  alpha=alpha,
                                                  linewidths=0.5)
            ax.add_collection3d(obstacle_collection)
        
        # 绘制无人机
        if drones:
            for i, drone in enumerate(drones):
                drone_mesh = drone.get_transformed_mesh()
                if drone_mesh:
                    vertices = drone_mesh['vertices']
                    faces = drone_mesh['faces']
                    color = drone_mesh['color']
                    
                    # 绘制无人机
                    for face in faces:
                        poly = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
                        ax.add_collection3d(Poly3DCollection([poly], 
                                                           facecolors=color, 
                                                           edgecolors='black',
                                                           alpha=1.0,
                                                           linewidths=1))
                    
                    # 标记无人机位置
                    ax.scatter(drone.position[0], drone.position[1], drone.position[2], 
                             color=color, s=100, marker='^', label=f'无人机 {i+1}')
        
        # 绘制路径
        if paths:
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF']
            for i, path in enumerate(paths):
                if len(path) > 1:
                    color = colors[i % len(colors)]
                    xs = path[:, 0]
                    ys = path[:, 1]
                    zs = path[:, 2]
                    
                    ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=0.8, 
                           label=f'路径 {i+1}')
                    ax.scatter(xs, ys, zs, color=color, s=20, alpha=0.6)
        
        # 设置坐标轴
        config = terrain_gen.config
        ax.set_xlim(0, config.width)
        ax.set_ylim(0, config.length)
        ax.set_zlim(0, config.height_range + 5)
        
        ax.set_xlabel('X (米)', fontsize=12)
        ax.set_ylabel('Y (米)', fontsize=12)
        ax.set_zlabel('Z (米)', fontsize=12)
        
        ax.set_title('3D环境可视化', fontsize=14, fontweight='bold')
        
        # 添加图例
        if drones or paths:
            ax.legend()
        
        # 设置视角
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_with_plotly(self, terrain_gen: TerrainGenerator,
                              drones: list,
                              paths: list):
        """使用plotly进行交互式可视化"""
        fig = make_subplots(rows=1, cols=1, 
                           specs=[[{'type': 'scene'}]])
        
        # 绘制地形
        if terrain_gen.terrain_mesh:
            vertices = terrain_gen.terrain_mesh['vertices']
            faces = terrain_gen.terrain_mesh['faces']
            
            # 创建三角网格
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            
            i = faces[:, 0]
            j = faces[:, 1]
            k = faces[:, 2]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='#8B7355',
                opacity=0.8,
                name='地形',
                showlegend=True
            ))
        
        # 绘制水面
        if terrain_gen.water_mesh:
            vertices = terrain_gen.water_mesh['vertices']
            faces = terrain_gen.water_mesh['faces']
            
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            
            i = faces[:, 0]
            j = faces[:, 1]
            k = faces[:, 2]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='#1E90FF',
                opacity=0.3,
                name='水面',
                showlegend=True
            ))
        
        # 绘制障碍物
        for idx, obstacle in enumerate(terrain_gen.obstacles_mesh):
            vertices = obstacle['vertices']
            faces = obstacle['faces']
            
            if obstacle.get('type') == 'tree':
                color = '#228B22'
                name = f'树木 {idx+1}'
            else:
                color = '#A0522D'
                name = f'建筑物 {idx+1}'
            
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            
            i = faces[:, 0]
            j = faces[:, 1]
            k = faces[:, 2]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=color,
                opacity=0.7,
                name=name,
                showlegend=True
            ))
        
        # 绘制无人机
        if drones:
            for i, drone in enumerate(drones):
                drone_mesh = drone.get_transformed_mesh()
                if drone_mesh:
                    vertices = drone_mesh['vertices']
                    faces = drone_mesh['faces']
                    color = drone_mesh['color']
                    
                    x = vertices[:, 0]
                    y = vertices[:, 1]
                    z = vertices[:, 2]
                    
                    i_idx = faces[:, 0]
                    j_idx = faces[:, 1]
                    k_idx = faces[:, 2]
                    
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i_idx, j=j_idx, k=k_idx,
                        color=color,
                        opacity=1.0,
                        name=f'无人机 {i+1}',
                        showlegend=True
                    ))
        
        # 绘制路径
        if paths:
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF']
            for i, path in enumerate(paths):
                if len(path) > 1:
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Scatter3d(
                        x=path[:, 0],
                        y=path[:, 1],
                        z=path[:, 2],
                        mode='lines+markers',
                        line=dict(color=color, width=4),
                        marker=dict(color=color, size=3),
                        name=f'路径 {i+1}',
                        showlegend=True
                    ))
        
        # 更新布局
        config = terrain_gen.config
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (米)', range=[0, config.width]),
                yaxis=dict(title='Y (米)', range=[0, config.length]),
                zaxis=dict(title='Z (米)', range=[0, config.height_range + 5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=config.length/config.width, z=0.5),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            title=dict(
                text='3D环境交互式可视化',
                font=dict(size=20, family='Arial', color='black')
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1000,
            height=800
        )
        
        fig.show()
    
    def save_visualization(self, filepath: str):
        """保存可视化结果"""
        print(f"可视化结果可保存为: {filepath}")
        # 在实际使用中，可以根据render_engine保存为不同格式