"""
基于MuJoCo的自动驾驶仿真数据生成核心代码

本模块实现了完整的自动驾驶车辆仿真系统，包含以下主要功能：
1. 车辆动力学仿真 - 利用MuJoCo物理引擎模拟真实车辆运动
2. LiDAR点云生成 - 模拟激光雷达传感器数据采集
3. 物体检测与标注 - 自动识别环境中的障碍物并生成标注
4. 温度监控系统 - 模拟车内温度变化及空调控制
5. 故障监测系统 - 实时监控传感器和执行器健康状态
6. 数据可视化 - 生成各类图表和分析报告
7. 键盘控制 - 支持手动控制车辆运动方向
"""
import os
import json
import numpy as np
import mujoco
from mujoco import viewer
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict, deque

# 用于键盘控制
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("注意: 未安装pygame，键盘控制功能不可用。可以通过 'pip install pygame' 安装。")

# -------------------------- 配置参数 --------------------------
# 场景文件路径
XML_PATH = "models/simple_car.xml"
# 输出目录
OUTPUT_DIR = "output/simulation_results"
# LiDAR参数
LIDAR_PARAMS = {
    "pos": [0, 0, 0.8],  # LiDAR在车辆上的安装位置
    "range": 30.0,  # 探测范围（m）
    "azimuth_res": 1.0,  # 方位角分辨率（°）
    "elevation_res": 2.0,  # 俯仰角分辨率（°）
    "elevation_min": -15,  # 最小俯仰角（°）
    "elevation_max": 15,  # 最大俯仰角（°）
    "lines": 16,  # 线束数
}
# 仿真帧数
SIMULATION_FRAMES = 1000

# 温度监测参数
TEMPERATURE_PARAMS = {
    "ambient_temp": 25.0,  # 环境基础温度 (摄氏度)
    "temp_variation": 5.0,  # 温度变化幅度
    "heat_sources": ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5"],  # 热源物体
    "comfort_range": [18.0, 28.0],  # 舒适温度范围 (摄氏度)
    "ac_power": 1.0  # 空调功率系数
}

# 添加故障监测参数
FAULT_MONITORING_PARAMS = {
    "sensor_fault_threshold": 0.1,  # 传感器故障阈值
    "actuator_fault_threshold": 0.15,  # 执行器故障阈值
    "health_check_interval": 20,  # 健康检查间隔（帧）
}

# -------------------------------------------------------------

class MojocoDataSim:
    """
    MuJoCo自动驾驶仿真主类
    
    该类负责管理整个仿真过程，包括：
    - 车辆模型加载与初始化
    - 传感器数据生成
    - 物体检测与识别
    - 温度监控与空调控制
    - 故障监测
    - 数据保存与可视化
    - 键盘控制交互
    """
    
    def __init__(self, xml_path, output_dir):
        """
        初始化仿真系统
        
        :param xml_path: MuJoCo模型文件路径
        :param output_dir: 输出数据存储目录
        """
        # 初始化输出目录结构
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/lidar", exist_ok=True)          # LiDAR点云数据目录
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)     # 物体检测标注目录
        os.makedirs(f"{output_dir}/visualization", exist_ok=True)   # 可视化图表目录
        os.makedirs(f"{output_dir}/distance_analysis", exist_ok=True)  # 距离分析图表目录
        os.makedirs(f"{output_dir}/fault_reports", exist_ok=True)   # 故障监测报告目录

        # 加载MuJoCo模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)  # 加载物理模型
        self.data = mujoco.MjData(self.model)                # 创建仿真数据实例
        
        # 创建可视化窗口
        self.viewer = viewer.launch_passive(self.model, self.data)

        print("可视化窗口已启动")
        print("仿真将在3秒后开始...")
        time.sleep(3)

        # 初始化空调系统状态
        self.ac_status = False           # 空调开关状态 (False=关闭, True=开启)
        self.ac_target_temp = 23.0       # 空调目标温度 (摄氏度)
        self.comfort_min_temp = TEMPERATURE_PARAMS["comfort_range"][0]  # 舒适温度下限
        self.comfort_max_temp = TEMPERATURE_PARAMS["comfort_range"][1]  # 舒适温度上限

        # 初始化故障监测系统
        self.fault_monitor = FaultMonitor(self.model, self.data)  # 创建故障监测器实例
        self.fault_history = []          # 存储故障历史记录
        self.health_scores = []          # 存储系统健康评分历史

        # 键盘控制相关初始化
        self.keyboard_control = False     # 键盘控制功能是否可用
        if PYGAME_AVAILABLE:
            try:
                pygame.init()
                pygame.display.set_mode((400, 200))
                pygame.display.set_caption('小车控制')
                self.font = pygame.font.Font(None, 36)       # 标题字体
                self.small_font = pygame.font.Font(None, 24)  # 正文字体
                self.keyboard_control = True
                print("已启用键盘控制功能")
                print("使用方向键控制小车: 上键-前进, 下键-后退, 左键-左转, 右键-右转")
            except Exception as e:
                print(f"无法初始化键盘控制界面: {e}")
        else:
            print("键盘控制不可用: 未安装pygame")

        # 车辆控制参数
        self.max_speed = 10.0     # 最大行驶速度
        self.turn_rate = 0.5      # 转向速率

    def get_world_pose(self, body_name):
        """
        获取指定物体的世界位姿（位置和姿态）
        
        :param body_name: 物体名称（如'vehicle', 'obstacle1'等）
        :return: tuple(位置向量, 四元数)
            - 位置向量: [x, y, z] 世界坐标系下的位置
            - 四元数: [w, x, y, z] 表示物体的姿态
        """
        # 根据物体名称查找物体ID
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的物体")
        
        # 获取物体在世界坐标系中的位置
        pos = self.data.xpos[body_id].copy()
        
        # 获取物体的姿态（旋转矩阵）并转换为四元数
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[body_id])
        return pos, quat

    def generate_realistic_lidar_data(self):
        """
        基于MuJoCo光线追踪生成真实的LiDAR点云数据
        
        该方法模拟真实的LiDAR传感器工作原理：
        1. 从LiDAR传感器位置发射多个激光束
        2. 检测每个激光束与环境中物体的碰撞点
        3. 记录有效碰撞点构成点云数据
        
        :return: numpy数组，形状为(N, 3)，N为检测到的点数，每行包含[x, y, z]坐标
        """
        try:
            # 获取车辆位置和朝向
            vehicle_pos, vehicle_quat = self.get_world_pose("vehicle")

            # 获取LiDAR传感器的位置和朝向
            lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
            if lidar_site_id >= 0:
                # 获取LiDAR传感器在世界坐标系中的位置
                lidar_pos = self.data.site_xpos[lidar_site_id].copy()
                # 获取LiDAR传感器的旋转矩阵（姿态）
                lidar_mat = self.data.site_xmat[lidar_site_id].reshape(3, 3)
            else:
                # 如果找不到LiDAR站点，使用默认位置（车辆位置+相对偏移）
                lidar_offset = np.array(LIDAR_PARAMS["pos"])
                lidar_pos = vehicle_pos + lidar_offset
                lidar_mat = np.eye(3)  # 单位矩阵表示无旋转
        except ValueError:
            # 如果无法获取车辆位姿，使用默认值
            vehicle_pos = np.array([0, 0, 0.5])
            lidar_pos = vehicle_pos + np.array(LIDAR_PARAMS["pos"])
            lidar_mat = np.eye(3)

        # 生成扫描角度范围
        # 方位角：水平方向的角度，从0°到360°
        azimuth_angles = np.arange(0, 360, LIDAR_PARAMS["azimuth_res"])
        # 俯仰角：垂直方向的角度，从最小值到最大值
        elevation_angles = np.arange(
            LIDAR_PARAMS["elevation_min"],
            LIDAR_PARAMS["elevation_max"] + LIDAR_PARAMS["elevation_res"],
            LIDAR_PARAMS["elevation_res"]
        )

        # 存储点云数据的列表
        point_cloud = []

        # 遍历所有角度组合，生成激光束
        for az in azimuth_angles:
            for el in elevation_angles:
                # 将角度转换为弧度（numpy三角函数使用弧度单位）
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)

                # 计算激光束的方向向量（在LiDAR局部坐标系中）
                # 使用球面坐标转换为直角坐标
                dir_local = np.array([
                    np.cos(el_rad) * np.cos(az_rad),  # X分量
                    np.cos(el_rad) * np.sin(az_rad),  # Y分量
                    np.sin(el_rad)                    # Z分量
                ])

                # 归一化方向向量，确保长度为1
                dir_local = dir_local / np.linalg.norm(dir_local)

                # 将方向向量从LiDAR局部坐标系转换到世界坐标系
                # 通过旋转矩阵实现坐标变换
                dir_world = lidar_mat @ dir_local

                # 创建射线检测参数
                geom_group = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)  # 检测所有几何体组
                geom_id = np.zeros(1, dtype=np.int32)  # 用于返回碰撞的几何体ID

                # 调用MuJoCo的射线检测函数
                distance = mujoco.mj_ray(
                    self.model, self.data,
                    lidar_pos,    # 射线起点（LiDAR传感器位置）
                    dir_world,    # 射线方向（世界坐标系）
                    geom_group,   # 几何体组（检测哪些类型的物体）
                    1,            # flg_static: 检测静态几何体
                    -1,           # bodyexclude: 不排除任何body
                    geom_id       # 返回碰撞的几何体ID
                )

                # 记录有效的点云数据
                if distance >= 0 and distance <= LIDAR_PARAMS["range"]:
                    # 计算碰撞点在世界坐标系中的位置
                    hit_pos = lidar_pos + dir_world * distance
                    point_cloud.append(hit_pos)

        # 转换为numpy数组并返回
        if len(point_cloud) > 0:
            point_cloud = np.array(point_cloud)
        else:
            # 如果没有检测到点，返回空数组
            point_cloud = np.empty((0, 3))

        return point_cloud

    def detect_objects_with_direction(self):
        """检测环境中的物体并计算相对于小车的方向"""
        detected_objects = []

        # 获取车辆位置和朝向
        try:
            vehicle_pos, vehicle_quat = self.get_world_pose("vehicle")
        except ValueError:
            vehicle_pos = np.array([0, 0, 0.5])
            vehicle_quat = np.array([1, 0, 0, 0])  # 默认朝向

        # 遍历所有物体
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith("obstacle"):
                # 获取物体位置
                pos = self.data.xpos[i].copy()

                # 计算与车辆的距离
                distance = np.linalg.norm(pos - vehicle_pos)

                # 只有在检测范围内才记录
                if distance <= 20.0:  # 扩大检测范围
                    # 计算相对于车辆的方向（方位角和俯仰角）
                    relative_pos = pos - vehicle_pos

                    # 计算方位角（水平角度）
                    azimuth = np.arctan2(relative_pos[1], relative_pos[0])

                    # 计算俯仰角（垂直角度）
                    elevation = np.arctan2(relative_pos[2], np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2))

                    # 获取物体类型（根据名称）
                    obj_type = "box"

                    # 获取物体的几何信息用于更好的可视化
                    geom_id = self.model.body_geomadr[i]
                    if geom_id >= 0:
                        size = self.model.geom_size[geom_id][:3].copy()
                    else:
                        size = [0.5, 0.5, 0.5]  # 默认大小

                    detected_objects.append({
                        "id": i,
                        "name": body_name,
                        "type": obj_type,
                        "position": pos.tolist(),
                        "distance": float(distance),
                        "azimuth": float(azimuth),  # 方位角（弧度）
                        "elevation": float(elevation),  # 俯仰角（弧度）
                        "azimuth_deg": float(np.degrees(azimuth)),  # 方位角（度）
                        "elevation_deg": float(np.degrees(elevation)),  # 俯仰角（度）
                        "size": size.tolist()
                    })

        return detected_objects

    def calculate_avoidance_control(self, lidar_data, detected_objects):
        """基于传感器数据计算避障控制指令"""
        # 初始化控制指令
        left_speed = 5.0
        right_speed = 5.0
        steering_angle = 0.0

        if len(detected_objects) > 0:
            # 找到最近的障碍物
            closest_obj = min(detected_objects, key=lambda x: x['distance'])

            if closest_obj['distance'] < 5.0:  # 如果障碍物很近
                obj_pos = np.array(closest_obj['position'])
                try:
                    vehicle_pos, _ = self.get_world_pose("vehicle")
                    # 计算障碍物相对于车辆的方向
                    direction = obj_pos[:2] - vehicle_pos[:2]  # 只考虑XY平面
                    angle_to_obstacle = np.arctan2(direction[1], direction[0])

                    # 简单避障策略：向相反方向转弯
                    if angle_to_obstacle > 0:  # 障碍物在左侧
                        steering_angle = -0.5  # 向右转
                    else:  # 障碍物在右侧
                        steering_angle = 0.5  # 向左转

                    # 如果非常接近，减速
                    if closest_obj['distance'] < 3.0:
                        left_speed = 2.0
                        right_speed = 2.0
                except ValueError:
                    pass

        return left_speed, right_speed, steering_angle

    def generate_annotations(self):
        """生成物体检测标注数据"""
        # 检测到的物体
        detected_objects = self.detect_objects_with_direction()

        annotations = {
            "frame": self.frame_count,
            "timestamp": time.time(),
            "objects": detected_objects
        }
        return annotations

    def save_data(self, lidar_data, annotations):
        """保存数据"""
        # 保存LiDAR点云（NPY格式）
        np.save(f"{self.output_dir}/lidar/frame_{self.frame_count:04d}.npy", lidar_data)
        print(f"已保存点云数据: frame_{self.frame_count:04d}.npy (共{len(lidar_data)}个点)")

        # 保存标注数据（JSON格式）
        with open(f"{self.output_dir}/annotations/frame_{self.frame_count:04d}.json", "w") as f:
            json.dump(annotations, f, indent=4)

        self.frame_count += 1

    def visualize_detection(self, lidar_data, annotations):
        """生成物体识别效果图"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制LiDAR点云数据
        if len(lidar_data) > 0:
            ax.scatter(lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2],
                       c='blue', s=0.5, alpha=0.6, label='LiDAR点云')

        # 绘制检测到的物体
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, obj in enumerate(annotations['objects']):
            pos = np.array(obj['position'])
            size = np.array(obj['size'])

            # 绘制物体中心点
            ax.scatter(pos[0], pos[1], pos[2],
                       c=colors[i % len(colors)], s=100, marker='o',
                       label=f"{obj['name']}")

            # 绘制物体边界框
            corners = self._generate_bounding_box_corners(pos, size)
            self._plot_bounding_box(ax, corners, colors[i % len(colors)])

        # 尝试绘制小车
        try:
            vehicle_pos, _ = self.get_world_pose("vehicle")
            ax.scatter(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2],
                       c='cyan', s=200, marker='s', label='小车')
        except ValueError:
            # 如果无法获取小车位置，则不绘制
            pass

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'物体识别效果图 - 帧 {self.frame_count:04d}')
        ax.legend()

        # 保存可视化图像
        plt.savefig(f"{self.output_dir}/visualization/frame_{self.frame_count:04d}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已生成识别效果图: frame_{self.frame_count:04d}.png")

    def visualize_distance_analysis(self, annotations):
        """生成距离和方位分析图"""
        if not annotations['objects']:
            return

        # 创建一个新的图形用于距离和方位分析
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # 提取物体信息
        object_names = [obj['name'] for obj in annotations['objects']]
        distances = [obj['distance'] for obj in annotations['objects']]
        azimuths = [obj['azimuth_deg'] for obj in annotations['objects']]
        elevations = [obj['elevation_deg'] for obj in annotations['objects']]

        # 绘制距离柱状图
        bars = ax1.bar(range(len(object_names)), distances, color=['red', 'green', 'orange', 'purple', 'brown'])
        ax1.set_xlabel('物体')
        ax1.set_ylabel('距离 (m)')
        ax1.set_title(f'物体距离分析 - 帧 {self.frame_count:04d}')
        ax1.set_xticks(range(len(object_names)))
        ax1.set_xticklabels(object_names, rotation=45)

        # 在柱状图上添加数值标签
        for i, (bar, dist) in enumerate(zip(bars, distances)):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{dist:.1f}m', ha='center', va='bottom')

        # 绘制极坐标图显示方位
        ax2 = plt.subplot(122, projection='polar')
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (azimuth, distance, name) in enumerate(zip(azimuths, distances, object_names)):
            # 转换为极坐标（需要弧度）
            theta = np.radians(azimuth)
            ax2.plot([0, theta], [0, distance], 'o-', color=colors[i % len(colors)],
                     label=f'{name} ({distance:.1f}m)', markersize=8)

        ax2.set_title(f'物体方位分析 - 帧 {self.frame_count:04d}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)

        # 保存分析图像
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/distance_analysis/frame_{self.frame_count:04d}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已生成距离和方位分析图: frame_{self.frame_count:04d}.png")

    def _generate_bounding_box_corners(self, position, size):
        """生成包围盒的8个顶点"""
        x, y, z = position
        sx, sy, sz = size

        corners = np.array([
            [x - sx, y - sy, z - sz], [x + sx, y - sy, z - sz], [x + sx, y + sy, z - sz], [x - sx, y + sy, z - sz],
            # 底面
            [x - sx, y - sy, z + sz], [x + sx, y - sy, z + sz], [x + sx, y + sy, z + sz], [x - sx, y + sy, z + sz]  # 顶面
        ])
        return corners

    def _plot_bounding_box(self, ax, corners, color):
        """绘制包围盒"""
        # 底面和顶面
        for i in range(2):
            # 四条边
            ax.plot(corners[i * 4:(i + 1) * 4, 0], corners[i * 4:(i + 1) * 4, 1], corners[i * 4:(i + 1) * 4, 2],
                    c=color, alpha=0.7)
            # 连接首尾
            ax.plot([corners[i * 4 + 3, 0], corners[i * 4, 0]],
                    [corners[i * 4 + 3, 1], corners[i * 4, 1]],
                    [corners[i * 4 + 3, 2], corners[i * 4, 2]],
                    c=color, alpha=0.7)

        # 连接顶面和底面
        for i in range(4):
            ax.plot([corners[i, 0], corners[i + 4, 0]],
                    [corners[i, 1], corners[i + 4, 1]],
                    [corners[i, 2], corners[i + 4, 2]],
                    c=color, alpha=0.7)

    def simulate_temperature_data(self):
        """
        模拟温度数据采集
        基于车辆位置和热源位置计算温度
        """
        try:
            # 获取车辆位置
            vehicle_pos, _ = self.get_world_pose("vehicle")
            lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
            if lidar_site_id >= 0:
                sensor_pos = self.data.site_xpos[lidar_site_id].copy()
            else:
                sensor_pos = vehicle_pos + np.array(LIDAR_PARAMS["pos"])
        except ValueError:
            sensor_pos = np.array([0, 0, 0.8])

        # 基础环境温度
        temperature = TEMPERATURE_PARAMS["ambient_temp"]

        # 遍历热源物体，计算对温度的影响
        for heat_source in TEMPERATURE_PARAMS["heat_sources"]:
            try:
                # 获取热源位置
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, heat_source)
                heat_pos = self.data.xpos[body_id].copy()

                # 计算传感器与热源的距离
                distance = np.linalg.norm(sensor_pos - heat_pos)

                # 根据距离计算温度影响（假设热源散发热量遵循平方反比定律）
                # 距离越近，温度越高
                temp_increase = TEMPERATURE_PARAMS["temp_variation"] / (distance + 1)  # 避免除零
                temperature += temp_increase

            except Exception:
                # 如果找不到热源，跳过
                continue

        # 添加随机噪声模拟真实传感器
        noise = np.random.normal(0, 0.5)  # 均值为0，标准差为0.5的高斯噪声
        temperature += noise

        # 如果空调开启，调整温度
        if self.ac_status:
            # 空调效果：逐渐向目标温度靠近
            temp_diff = self.ac_target_temp - temperature
            ac_effect = temp_diff * TEMPERATURE_PARAMS["ac_power"] * 0.05  # 空调效果系数
            temperature += ac_effect

        return temperature

    def visualize_temperature_data(self, temperature, detected_objects):
        """
        生成温度分布可视化图
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # 左侧图：温度随时间变化趋势
        # 由于我们是单帧数据，这里展示当前温度信息
        ax1.set_xlim(0, 10)
        ax1.set_ylim(TEMPERATURE_PARAMS["ambient_temp"] - 5,
                     TEMPERATURE_PARAMS["ambient_temp"] + TEMPERATURE_PARAMS["temp_variation"] + 5)
        ax1.axhline(y=TEMPERATURE_PARAMS["ambient_temp"], color='b', linestyle='--',
                    label=f'环境温度: {TEMPERATURE_PARAMS["ambient_temp"]:.1f}°C')
        ax1.bar([5], [temperature], width=2, color='r', alpha=0.7,
                label=f'测量温度: {temperature:.1f}°C')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('温度 (°C)')
        ax1.set_title(f'温度监测 - 帧 {self.frame_count:04d}')
        ax1.legend()
        ax1.grid(True)

        # 右侧图：温度与物体距离关系
        if detected_objects:
            distances = [obj['distance'] for obj in detected_objects]
            object_names = [obj['name'] for obj in detected_objects]

            # 计算每个物体附近的预期温度
            expected_temps = []
            try:
                sensor_pos = self.data.site_xpos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
                ].copy()

                for obj in detected_objects:
                    obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj['name'])
                    obj_pos = self.data.xpos[obj_id].copy()
                    distance = np.linalg.norm(sensor_pos - obj_pos)
                    expected_temp = TEMPERATURE_PARAMS["ambient_temp"] + \
                                   TEMPERATURE_PARAMS["temp_variation"] / (distance + 1)
                    expected_temps.append(expected_temp)
            except:
                # 如果出错，使用简化计算
                expected_temps = [TEMPERATURE_PARAMS["ambient_temp"] +
                                 TEMPERATURE_PARAMS["temp_variation"] / (d + 1) for d in distances]

            x_pos = range(len(distances))
            ax2.bar(x_pos, expected_temps, alpha=0.7, color='orange', label='预期温度')
            ax2.axhline(y=temperature, color='r', linestyle='-', label=f'实测温度: {temperature:.1f}°C')

            ax2.set_xlabel('物体')
            ax2.set_ylabel('温度 (°C)')
            ax2.set_title('物体距离与温度关系')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([name[-1] for name in object_names])  # 只显示编号
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, '无检测到物体', horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('物体距离与温度关系')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualization/temp_frame_{self.frame_count:04d}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已生成温度可视化图: temp_frame_{self.frame_count:04d}.png")

    def generate_thermal_map(self, temperature, detected_objects):
        """
        生成热力图（二维温度分布图）
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # 创建网格点用于绘制热力图
        grid_size = 50
        x_range = np.linspace(-10, 15, grid_size)
        y_range = np.linspace(-8, 8, grid_size)
        X, Y = np.meshgrid(x_range, y_range)

        # 计算每个网格点的温度值
        Z = np.zeros_like(X)
        sensor_height = 0.8  # 传感器高度

        for i in range(grid_size):
            for j in range(grid_size):
                # 当前网格点位置
                point_pos = np.array([X[j, i], Y[j, i], sensor_height])

                # 基础环境温度
                temp = TEMPERATURE_PARAMS["ambient_temp"]

                # 遍历热源计算温度贡献
                for heat_source in TEMPERATURE_PARAMS["heat_sources"]:
                    try:
                        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, heat_source)
                        heat_pos = self.data.xpos[body_id].copy()
                        distance = np.linalg.norm(point_pos - heat_pos)
                        temp_increase = TEMPERATURE_PARAMS["temp_variation"] / (distance + 1)
                        temp += temp_increase
                    except:
                        continue

                Z[j, i] = temp

        # 绘制热力图
        im = ax.contourf(X, Y, Z, levels=50, cmap='hot')
        plt.colorbar(im, ax=ax, label='温度 (°C)')

        # 绘制车辆位置
        try:
            vehicle_pos, _ = self.get_world_pose("vehicle")
            ax.plot(vehicle_pos[0], vehicle_pos[1], 'bo', markersize=10, label='车辆')
        except:
            pass

        # 绘制障碍物位置
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        for i, obstacle in enumerate(TEMPERATURE_PARAMS["heat_sources"]):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obstacle)
                pos = self.data.xpos[body_id].copy()
                ax.plot(pos[0], pos[1], 's', color=colors[i % len(colors)],
                       markersize=8, label=obstacle)
            except:
                continue

        # 绘制温度传感器测量值
        try:
            lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
            if lidar_site_id >= 0:
                sensor_pos = self.data.site_xpos[lidar_site_id].copy()
                ax.plot(sensor_pos[0], sensor_pos[1], 'wo', markersize=6,
                       markeredgecolor='black', label=f'传感器({temperature:.1f}°C)')
        except:
            pass

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'环境温度分布图 - 帧 {self.frame_count:04d}')
        ax.legend()
        ax.grid(True)

        plt.savefig(f"{self.output_dir}/visualization/thermal_map_{self.frame_count:04d}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已生成温度分布热力图: thermal_map_{self.frame_count:04d}.png")

    def check_and_control_ac(self, temperature):
        """
        检查温度并控制空调开关
        :param temperature: 当前温度
        :return: 是否开启了空调
        """
        # 检查温度是否超出舒适范围
        if temperature < self.comfort_min_temp or temperature > self.comfort_max_temp:
            # 如果温度不在舒适范围内，开启空调
            if not self.ac_status:
                self.ac_status = True
                print(f"🌡️ 空调已开启，当前温度: {temperature:.1f}°C，目标温度: {self.ac_target_temp:.1f}°C")
        else:
            # 如果温度在舒适范围内，关闭空调
            if self.ac_status:
                self.ac_status = False
                print(f"❄️ 空调已关闭，当前温度: {temperature:.1f}°C，处于舒适范围内")

        return self.ac_status

    def visualize_ac_control(self, temperature, ac_status):
        """
        生成空调控制状态可视化图
        :param temperature: 当前温度
        :param ac_status: 空调状态
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制温度范围
        comfort_min = self.comfort_min_temp
        comfort_max = self.comfort_max_temp
        ambient_temp = TEMPERATURE_PARAMS["ambient_temp"]

        # 绘制舒适温度区域
        ax.axvspan(comfort_min, comfort_max, alpha=0.3, color='green', label='舒适温度区间')

        # 绘制环境温度线
        ax.axvline(ambient_temp, color='blue', linestyle='--', linewidth=1, label=f'环境温度 ({ambient_temp}°C)')

        # 绘制当前温度
        color = 'red' if ac_status else 'orange'
        status_label = '空调运行中' if ac_status else '空调关闭'
        ax.scatter(temperature, 1, s=100, color=color, label=f'当前温度 {temperature:.1f}°C ({status_label})')

        # 绘制目标温度（如果空调开启）
        if ac_status:
            ax.axvline(self.ac_target_temp, color='purple', linestyle='-.', linewidth=1,
                      label=f'目标温度 ({self.ac_target_temp}°C)')

        ax.set_xlim(ambient_temp - 10, ambient_temp + 10)
        ax.set_ylim(0, 2)
        ax.set_xlabel('温度 (°C)')
        ax.set_title(f'空调控制系统状态 - 帧 {self.frame_count:04d}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 移除y轴刻度
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualization/ac_control_{self.frame_count:04d}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已生成空调控制状态图: ac_control_{self.frame_count:04d}.png")

    def generate_temperature_trend(self):
        """
        生成温度变化趋势图
        """
        # 读取之前保存的温度数据
        temp_files = []
        for file in os.listdir(f"{self.output_dir}/annotations"):
            if file.startswith("temp_frame_") and file.endswith(".json"):
                temp_files.append(file)

        if not temp_files:
            return

        # 按帧排序
        temp_files.sort()

        frames = []
        temperatures = []
        ac_statuses = []

        # 读取温度数据
        for file in temp_files:
            with open(f"{self.output_dir}/annotations/{file}", "r") as f:
                temp_data = json.load(f)
                frames.append(temp_data["frame"])
                temperatures.append(temp_data["temperature"])
                ac_statuses.append(temp_data["ac_status"])

        # 绘制温度变化趋势图
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制温度曲线
        ax.plot(frames, temperatures, 'o-', color='red', linewidth=2, markersize=4, label='实测温度')

        # 绘制舒适温度区间
        ax.axhspan(self.comfort_min_temp, self.comfort_max_temp, alpha=0.2, color='green',
                  label='舒适温度区间')

        # 绘制环境温度线
        ax.axhline(TEMPERATURE_PARAMS["ambient_temp"], color='blue', linestyle='--',
                  label=f'环境温度 ({TEMPERATURE_PARAMS["ambient_temp"]}°C)')

        # 标记空调开启的时间点
        ac_on_frames = [frames[i] for i in range(len(frames)) if ac_statuses[i]]
        ac_on_temps = [temperatures[i] for i in range(len(temperatures)) if ac_statuses[i]]
        if ac_on_frames:
            ax.scatter(ac_on_frames, ac_on_temps, color='purple', s=50, marker='^',
                      label='空调运行中', zorder=5)

        ax.set_xlabel('帧序号')
        ax.set_ylabel('温度 (°C)')
        ax.set_title('温度变化趋势与空调控制状态')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualization/temperature_trend.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("已生成温度变化趋势图: temperature_trend.png")

    def generate_temperature_summary_chart(self):
        """
        生成温度调节汇总图，综合显示温度变化、空调状态和调节效果
        """
        # 读取之前保存的温度数据
        temp_files = []
        for file in os.listdir(f"{self.output_dir}/annotations"):
            if file.startswith("temp_frame_") and file.endswith(".json"):
                temp_files.append(file)

        if not temp_files:
            return

        # 按帧排序
        temp_files.sort()

        frames = []
        temperatures = []
        ac_statuses = []
        target_temps = []

        # 读取温度数据
        for file in temp_files:
            with open(f"{self.output_dir}/annotations/{file}", "r") as f:
                temp_data = json.load(f)
                frames.append(temp_data["frame"])
                temperatures.append(temp_data["temperature"])
                ac_statuses.append(temp_data["ac_status"])
                target_temps.append(temp_data.get("ac_target_temp", None))

        # 创建汇总图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 第一个子图：温度变化和空调状态
        ax1.plot(frames, temperatures, 'o-', color='red', linewidth=2, markersize=4, label='实测温度')
        ax1.axhspan(self.comfort_min_temp, self.comfort_max_temp, alpha=0.2, color='green',
                   label='舒适温度区间')
        ax1.axhline(TEMPERATURE_PARAMS["ambient_temp"], color='blue', linestyle='--',
                   label=f'环境温度 ({TEMPERATURE_PARAMS["ambient_temp"]}°C)')

        # 标记空调开启的时间点
        ac_on_frames = [frames[i] for i in range(len(frames)) if ac_statuses[i]]
        ac_on_temps = [temperatures[i] for i in range(len(temperatures)) if ac_statuses[i]]
        if ac_on_frames:
            ax1.scatter(ac_on_frames, ac_on_temps, color='purple', s=50, marker='^',
                       label='空调运行中', zorder=5)

        ax1.set_ylabel('温度 (°C)')
        ax1.set_title('温度变化与空调控制状态')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 第二个子图：空调工作时的目标温度和调节效果
        # 只显示空调开启时的数据点
        ac_frames = []
        ac_temps = []
        ac_targets = []
        temp_differences = []

        for i in range(len(frames)):
            if ac_statuses[i] and target_temps[i] is not None:
                ac_frames.append(frames[i])
                ac_temps.append(temperatures[i])
                ac_targets.append(target_temps[i])
                temp_differences.append(abs(temperatures[i] - target_temps[i]))

        if ac_frames:
            ax2.plot(ac_frames, ac_temps, 'o-', color='red', linewidth=2, markersize=4, label='实测温度')
            ax2.plot(ac_frames, ac_targets, 's-', color='purple', linewidth=2, markersize=4, label='目标温度')

            # 添加温度差值的柱状图
            ax3 = ax2.twinx()
            bars = ax3.bar(ac_frames, temp_differences, alpha=0.3, color='orange', width=1.0, label='温度差值')
            ax3.set_ylabel('温度差值 (°C)', color='orange')
            ax3.tick_params(axis='y', labelcolor='orange')

            # 添加数值标签
            for bar, diff in zip(bars, temp_differences):
                height = bar.get_height()
                ax3.annotate(f'{diff:.1f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='orange')

        ax2.set_xlabel('帧序号')
        ax2.set_ylabel('温度 (°C)')
        ax2.set_title('空调调节效果分析')
        ax2.grid(True, alpha=0.3)

        # 合并图例
        if ac_frames:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax2.legend(lines2 + lines3, labels2 + labels3, loc='upper left')
        else:
            ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualization/temperature_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("已生成温度调节汇总图: temperature_summary.png")

    def visualize_real_time_avoidance(self, lidar_data, detected_objects, left_speed, right_speed, steering_angle):
        """
        生成实时避障图
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制LiDAR点云数据
        if len(lidar_data) > 0:
            ax.scatter(lidar_data[:, 0], lidar_data[:, 1],
                      c='blue', s=1, alpha=0.6, label='LiDAR点云')

        # 绘制检测到的物体
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, obj in enumerate(detected_objects):
            pos = np.array(obj['position'])
            # 绘制物体中心点
            ax.scatter(pos[0], pos[1],
                      c=colors[i % len(colors)], s=100, marker='o',
                      label=f"{obj['name']} (距离: {obj['distance']:.1f}m)")

            # 绘制安全距离圆
            circle = plt.Circle((pos[0], pos[1]), 3.0, fill=False,
                              color=colors[i % len(colors)], linestyle='--', alpha=0.7)
            ax.add_patch(circle)

        # 尝试绘制小车
        try:
            vehicle_pos, _ = self.get_world_pose("vehicle")
            # 绘制小车位置
            ax.scatter(vehicle_pos[0], vehicle_pos[1],
                      c='cyan', s=200, marker='s', label='小车')

            # 绘制小车方向 (简化表示)
            direction_length = 2.0
            ax.arrow(vehicle_pos[0], vehicle_pos[1],
                    direction_length * np.cos(steering_angle*2),
                    direction_length * np.sin(steering_angle*2),
                    head_width=0.3, head_length=0.3, fc='cyan', ec='cyan')

        except ValueError:
            # 如果无法获取小车位置，则不绘制
            pass

        # 绘制避障决策信息
        ax.text(0.02, 0.98, f'左轮速度: {left_speed:.1f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.93, f'右轮速度: {right_speed:.1f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.88, f'转向角度: {steering_angle:.1f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 根据转向角度确定转向方向文字
        if steering_angle > 0:
            turn_direction = "左转"
        elif steering_angle < 0:
            turn_direction = "右转"
        else:
            turn_direction = "直行"

        ax.text(0.02, 0.83, f'转向方向: {turn_direction}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'实时避障图 - 帧 {self.frame_count:04d}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # 确保目录存在
        os.makedirs(f"{self.output_dir}/visualization", exist_ok=True)

        # 保存可视化图像
        plt.savefig(f"{self.output_dir}/visualization/avoidance_{self.frame_count:04d}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已生成实时避障图: avoidance_{self.frame_count:04d}.png")

    def run_simulation(self):
        """
        运行MuJoCo仿真主循环并生成数据
        
        仿真主循环流程：
        1. 每20帧采集传感器数据并保存
        2. 实时更新车辆控制指令
        3. 执行物理仿真步骤
        4. 监控系统健康状态
        """
        print("开始仿真...")
        self.frame_count = 0

        # 查找车辆的驱动关节和转向关节索引
        # 后轮驱动电机
        rear_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_wheel_motor")
        rear_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_wheel_motor")
        # 前轮转向伺服电机
        front_left_steer_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_left_steering")
        front_right_steer_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_right_steering")

        if rear_left_idx >= 0 and rear_right_idx >= 0:
            print("找到了车辆驱动关节")

        # 上一帧检测到的物体数量（用于比较和输出变化）
        prev_detected_count = 0

        # 初始化控制变量（默认自动巡航状态）
        left_speed, right_speed, steering_angle = 5.0, 5.0, 0.0

        # 仿真主循环
        for i in range(SIMULATION_FRAMES):
            # 每20帧生成和保存一次数据（降低数据生成频率以提高性能）
            if i % 20 == 0:
                # 生成传感器数据和标注
                lidar_data = self.generate_realistic_lidar_data()
                annotations = self.generate_annotations()

                # 新增：模拟温度数据
                temperature = self.simulate_temperature_data()
                

                # 检查并控制空调
                ac_status = self.check_and_control_ac(temperature)

                # 基于传感器数据计算控制指令（自动避障）
                left_speed, right_speed, steering_angle = self.calculate_avoidance_control(
                    lidar_data, annotations["objects"]
                )

                # 显示检测到的物体数量
                detected_count = len(annotations["objects"])
                if detected_count != prev_detected_count:
                    if detected_count > 0:
                        print(f"检测到 {detected_count} 个物体:")
                        for obj in annotations["objects"]:
                            print(f"  - {obj['name']} 距离: {obj['distance']:.2f}m, "
                                  f"方位角: {obj['azimuth_deg']:.1f}°, "
                                  f"俯仰角: {obj['elevation_deg']:.1f}°")
                    else:
                        print("未检测到附近物体")
                    prev_detected_count = detected_count

                # 保存数据
                self.save_data(lidar_data, annotations)

                # 生成识别效果图
                self.visualize_detection(lidar_data, annotations)

                # 新增：生成实时避障图
                self.visualize_real_time_avoidance(lidar_data, annotations["objects"],
                                                 left_speed, right_speed, steering_angle)

                # 新增：生成温度可视化图
                self.visualize_temperature_data(temperature, annotations["objects"])

                # 新增：生成温度分布热力图
                self.generate_thermal_map(temperature, annotations["objects"])

                # 新增：生成空调控制状态图
                self.visualize_ac_control(temperature, ac_status)

                # 在保存数据时也保存温度信息
                temp_data = {
                    "frame": self.frame_count,
                    "temperature": temperature,
                    "unit": "celsius",
                    "ac_status": ac_status,
                    "ac_target_temp": self.ac_target_temp if ac_status else None
                }
                with open(f"{self.output_dir}/annotations/temp_frame_{self.frame_count:04d}.json", "w") as f:
                    json.dump(temp_data, f, indent=4)

                print(f"已仿真 {i}/{SIMULATION_FRAMES} 帧")

            # 设置控制输入
            if rear_left_idx >= 0:
                self.data.ctrl[rear_left_idx] = left_speed  # 左后轮速度
            if rear_right_idx >= 0:
                self.data.ctrl[rear_right_idx] = right_speed  # 右后轮速度
            # 设置前轮转向
            if front_left_steer_idx >= 0:
                self.data.ctrl[front_left_steer_idx] = steering_angle
            if front_right_steer_idx >= 0:
                self.data.ctrl[front_right_steer_idx] = steering_angle

            # 执行仿真步长
            mujoco.mj_step(self.model, self.data)

            # 更新可视化
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.sync()

            # 控制仿真速度以便观察
            time.sleep(0.01)

            # 新增：故障监测
            if i % FAULT_MONITORING_PARAMS["health_check_interval"] == 0:
                health_report = self.fault_monitor.check_system_health()
                self.health_scores.append((self.frame_count, health_report["overall_health"]))

                # 记录故障
                if health_report["faults"]:
                    self.fault_history.append({
                        "frame": self.frame_count,
                        "timestamp": time.time(),
                        "report": health_report
                    })

                    # 生成故障报告和图表
                    self.generate_fault_report(health_report)
                    self.visualize_fault_status(health_report)
                    self.generate_health_trend()

                    # 打印故障信息
                    self.print_fault_info(health_report)

        # 生成最终的温度趋势图
        self.generate_temperature_trend()

        # 生成温度调节汇总图
        self.generate_temperature_summary_chart()

        # 生成最终的健康趋势图
        self.generate_health_trend()

        print(f"仿真完成！数据已保存到：{self.output_dir}")

    def print_fault_info(self, health_report):
        """打印故障信息"""
        faults = health_report["faults"]
        if faults:
            print(f"🚨 检测到 {len(faults)} 个故障:")
            for fault in faults:
                print(f"  - {fault['component']}: {fault['description']} (严重程度: {fault['severity']})")
        else:
            print("✅ 系统健康状态良好")

    def generate_fault_report(self, health_report):
        """生成故障报告JSON文件"""
        fault_data = {
            "frame": self.frame_count,
            "timestamp": time.time(),
            "health_report": health_report
        }

        with open(f"{self.output_dir}/fault_reports/frame_{self.frame_count:04d}.json", "w") as f:
            json.dump(fault_data, f, indent=4, ensure_ascii=False)

        print(f"已生成故障报告: frame_{self.frame_count:04d}.json")

    def visualize_fault_status(self, health_report):
        """生成故障状态可视化图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 左侧图：各组件健康状态
        components = list(health_report["component_health"].keys())
        health_scores = list(health_report["component_health"].values())

        # 使用不同颜色表示健康状态
        colors = []
        for score in health_scores:
            if score > 0.8:
                colors.append('green')  # 健康
            elif score > 0.5:
                colors.append('orange')  # 警告
            else:
                colors.append('red')  # 严重

        bars = ax1.bar(range(len(components)), health_scores, color=colors)
        ax1.set_xlabel('组件')
        ax1.set_ylabel('健康评分 (0-1)')
        ax1.set_title(f'组件健康状态 - 帧 {self.frame_count:04d}')
        ax1.set_xticks(range(len(components)))
        ax1.set_xticklabels(components, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, health_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')

        # 右侧图：故障详情
        faults = health_report["faults"]
        if faults:
            fault_names = [f"{fault['component']}" for fault in faults]
            severities = [fault['severity'] for fault in faults]

            # 严重程度颜色映射
            severity_colors = []
            for severity in severities:
                if severity > 0.7:
                    severity_colors.append('red')
                elif severity > 0.4:
                    severity_colors.append('orange')
                else:
                    severity_colors.append('yellow')

            bars = ax2.bar(range(len(faults)), severities, color=severity_colors)
            ax2.set_xlabel('故障组件')
            ax2.set_ylabel('严重程度 (0-1)')
            ax2.set_title(f'检测到的故障 - 帧 {self.frame_count:04d}')
            ax2.set_xticks(range(len(faults)))
            ax2.set_xticklabels(fault_names, rotation=45, ha='right')
            ax2.set_ylim(0, 1.1)

            # 添加数值标签和描述
            for i, (bar, fault) in enumerate(zip(bars, faults)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{fault['description']}", ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, '当前无故障', horizontalalignment='center',
                    verticalalignment='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title(f'故障状态 - 帧 {self.frame_count:04d}')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fault_reports/fault_status_{self.frame_count:04d}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已生成故障状态图: fault_status_{self.frame_count:04d}.png")

    def generate_health_trend(self):
        """生成系统健康趋势图"""
        if not self.health_scores:
            return

        frames, scores = zip(*self.health_scores)

        plt.figure(figsize=(12, 6))
        plt.plot(frames, scores, 'o-', linewidth=2, markersize=6, color='blue')
        plt.xlabel('帧序号')
        plt.ylabel('整体健康评分 (0-1)')
        plt.title('系统健康状态趋势')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)

        # 添加健康状态区域
        plt.axhspan(0.8, 1.0, alpha=0.2, color='green', label='健康 (>0.8)')
        plt.axhspan(0.5, 0.8, alpha=0.2, color='yellow', label='警告 (0.5-0.8)')
        plt.axhspan(0.0, 0.5, alpha=0.2, color='red', label='严重 (<0.5)')

        plt.legend()

        # 标注故障点
        fault_frames = [record["frame"] for record in self.fault_history]
        fault_scores = [record["report"]["overall_health"] for record in self.fault_history]
        if fault_frames:
            plt.scatter(fault_frames, fault_scores, color='red', s=50, zorder=5, label='检测到故障')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fault_reports/health_trend.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("已生成健康趋势图: health_trend.png")


class FaultMonitor:
    """故障监测系统"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.sensor_names = []
        self.actuator_names = []

        # 获取传感器列表
        for i in range(model.nsensor):
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name:
                self.sensor_names.append(sensor_name)

        # 获取执行器列表
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                self.actuator_names.append(actuator_name)

        # 存储历史数据用于异常检测
        self.sensor_history = defaultdict(lambda: deque(maxlen=50))
        self.actuator_history = defaultdict(lambda: deque(maxlen=50))

    def check_system_health(self):
        """检查系统整体健康状态"""
        faults = []
        component_health = {}

        # 检查传感器健康状态
        sensor_health = self._check_sensor_health(faults)
        component_health.update(sensor_health)

        # 检查执行器健康状态
        actuator_health = self._check_actuator_health(faults)
        component_health.update(actuator_health)

        # 计算整体健康评分
        if component_health:
            overall_health = sum(component_health.values()) / len(component_health)
        else:
            overall_health = 1.0

        return {
            "overall_health": overall_health,
            "component_health": component_health,
            "faults": faults
        }

    def _check_sensor_health(self, faults):
        """检查传感器健康状态"""
        sensor_health = {}

        for sensor_name in self.sensor_names:
            try:
                # 获取传感器索引
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                if sensor_id < 0:
                    continue

                # 获取传感器数据
                sensor_data = self.data.sensordata[sensor_id]

                # 存储历史数据
                self.sensor_history[sensor_name].append(sensor_data)

                # 计算健康评分（基于数据变化）
                if len(self.sensor_history[sensor_name]) > 5:
                    recent_data = list(self.sensor_history[sensor_name])[-5:]
                    # 计算数据变化率
                    data_changes = [abs(recent_data[i] - recent_data[i-1])
                                  for i in range(1, len(recent_data))]
                    avg_change = sum(data_changes) / len(data_changes)

                    # 健康评分（变化率越小越健康，但不能为0）
                    health_score = max(0.1, 1.0 - avg_change)

                    # 检查是否有故障
                    if avg_change < FAULT_MONITORING_PARAMS["sensor_fault_threshold"]:
                        faults.append({
                            "component": f"传感器:{sensor_name}",
                            "description": "传感器数据无变化，可能存在故障",
                            "severity": 1.0 - health_score
                        })
                else:
                    health_score = 1.0

                sensor_health[f"传感器:{sensor_name}"] = health_score

            except Exception as e:
                # 传感器读取失败
                sensor_health[f"传感器:{sensor_name}"] = 0.0
                faults.append({
                    "component": f"传感器:{sensor_name}",
                    "description": f"传感器读取失败: {str(e)}",
                    "severity": 1.0
                })

        return sensor_health

    def _check_actuator_health(self, faults):
        """检查执行器健康状态"""
        actuator_health = {}

        for actuator_name in self.actuator_names:
            try:
                # 获取执行器索引
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id < 0:
                    continue

                # 获取执行器数据
                actuator_data = self.data.ctrl[actuator_id]

                # 存储历史数据
                self.actuator_history[actuator_name].append(actuator_data)

                # 计算健康评分（基于数据有效性）
                if len(self.actuator_history[actuator_name]) > 5:
                    recent_data = list(self.actuator_history[actuator_name])[-5:]
                    # 检查是否有NaN或无穷大值
                    invalid_count = sum(1 for x in recent_data if np.isnan(x) or np.isinf(x))
                    health_score = 1.0 - (invalid_count / len(recent_data))

                    # 检查是否有故障
                    if invalid_count > 0:
                        faults.append({
                            "component": f"执行器:{actuator_name}",
                            "description": f"执行器数据异常 ({invalid_count}/{len(recent_data)} 帧)",
                            "severity": invalid_count / len(recent_data)
                        })
                else:
                    health_score = 1.0

                actuator_health[f"执行器:{actuator_name}"] = health_score

            except Exception as e:
                # 执行器读取失败
                actuator_health[f"执行器:{actuator_name}"] = 0.0
                faults.append({
                    "component": f"执行器:{actuator_name}",
                    "description": f"执行器读取失败: {str(e)}",
                    "severity": 1.0
                })

        return actuator_health

if __name__ == "__main__":
    print("正在初始化仿真器...")
    try:
        sim = MojocoDataSim(XML_PATH, OUTPUT_DIR)
        sim.run_simulation()
    except FileNotFoundError as e:
        print(f"找不到模型文件: {e}")
        print("请确认XML文件路径是否正确")
    except Exception as e:
        print(f"仿真过程中出现错误: {e}")
        import traceback

        traceback.print_exc()