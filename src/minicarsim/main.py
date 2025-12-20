"""
基于MuJoCo的自动驾驶仿真数据生成核心代码
功能：车辆动力学仿真、LiDAR点云生成、摄像头图像生成、自动标注、数据保存
"""
import os
import json
import numpy as np
import mujoco
from mujoco import viewer
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    "heat_sources": ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5"]  # 热源物体
}


# -------------------------------------------------------------

class MojocoDataSim:
    def __init__(self, xml_path, output_dir):
        # 初始化输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/lidar", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        os.makedirs(f"{output_dir}/visualization", exist_ok=True)  # 新增可视化目录
        os.makedirs(f"{output_dir}/distance_analysis", exist_ok=True)  # 新增距离分析目录

        # 加载MuJoCo模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # 创建可视化窗口
        self.viewer = viewer.launch_passive(self.model, self.data)

        print("可视化窗口已启动")
        print("仿真将在3秒后开始...")
        time.sleep(3)

    def get_world_pose(self, body_name):
        """
        获取指定物体的世界位姿
        :param body_name: 物体名称
        :return: 位置和四元数
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的物体")
        pos = self.data.xpos[body_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[body_id])
        return pos, quat

    def generate_realistic_lidar_data(self):
        """基于MuJoCo光线追踪生成真实的LiDAR点云数据"""
        try:
            # 获取车辆位置和朝向
            vehicle_pos, vehicle_quat = self.get_world_pose("vehicle")

            # 获取LiDAR传感器的位置和朝向
            lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
            if lidar_site_id >= 0:
                lidar_pos = self.data.site_xpos[lidar_site_id].copy()
                # 获取LiDAR的旋转矩阵
                lidar_mat = self.data.site_xmat[lidar_site_id].reshape(3, 3)
            else:
                # 如果找不到LiDAR站点，使用默认位置
                lidar_offset = np.array(LIDAR_PARAMS["pos"])
                lidar_pos = vehicle_pos + lidar_offset
                lidar_mat = np.eye(3)  # 单位矩阵表示无旋转
        except ValueError:
            vehicle_pos = np.array([0, 0, 0.5])
            lidar_pos = vehicle_pos + np.array(LIDAR_PARAMS["pos"])
            lidar_mat = np.eye(3)

        # 生成角度范围
        azimuth_angles = np.arange(0, 360, LIDAR_PARAMS["azimuth_res"])  # 方位角：0~360°
        elevation_angles = np.arange(
            LIDAR_PARAMS["elevation_min"],
            LIDAR_PARAMS["elevation_max"] + LIDAR_PARAMS["elevation_res"],
            LIDAR_PARAMS["elevation_res"]
        )  # 俯仰角

        point_cloud = []

        # 遍历所有角度，生成激光束
        for az in azimuth_angles:
            for el in elevation_angles:
                # 转换为弧度
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)

                # 计算激光束的方向向量（局部坐标系）
                dir_local = np.array([
                    np.cos(el_rad) * np.cos(az_rad),
                    np.cos(el_rad) * np.sin(az_rad),
                    np.sin(el_rad)
                ])

                # 归一化方向向量
                dir_local = dir_local / np.linalg.norm(dir_local)

                # 将方向向量从LiDAR坐标系转换到世界坐标系
                dir_world = lidar_mat @ dir_local

                # 创建参数
                geom_group = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
                geom_id = np.zeros(1, dtype=np.int32)

                # 调用射线检测
                distance = mujoco.mj_ray(
                    self.model, self.data,
                    lidar_pos,  # 射线起点
                    dir_world,  # 射线方向（世界坐标系）
                    geom_group,  # 几何体组
                    1,  # flg_static: 检测静态几何体
                    -1,  # bodyexclude: 不排除任何body
                    geom_id  # 返回碰撞的几何体ID
                )

                # 记录点云数据
                if distance >= 0 and distance <= LIDAR_PARAMS["range"]:  # 如果检测到碰撞且在范围内
                    # 计算交点位置
                    hit_pos = lidar_pos + dir_world * distance
                    point_cloud.append(hit_pos)

        # 转换为numpy数组
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
                        steering_angle = -5.0  # 向右转
                    else:  # 障碍物在右侧
                        steering_angle = 5.0  # 向左转

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

    def run_simulation(self):
        """运行MuJoCo仿真并生成数据"""
        print("开始仿真...")
        self.frame_count = 0

        # 设置简单的控制输入
        # 查找车辆的驱动关节
        rear_left_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_wheel_motor")
        rear_right_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_wheel_motor")
        front_left_steer_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_left_steering")
        front_right_steer_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_right_steering")

        if rear_left_idx >= 0 and rear_right_idx >= 0:
            print("找到了车辆驱动关节")

        prev_detected_count = 0

        for i in range(SIMULATION_FRAMES):
            # 每20帧生成和保存一次数据
            if i % 20 == 0:
                # 生成传感器数据和标注
                lidar_data = self.generate_realistic_lidar_data()
                annotations = self.generate_annotations()
                
                # 新增：模拟温度数据
                temperature = self.simulate_temperature_data()

                # 基于传感器数据计算控制指令
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
                
                # 新增：生成温度可视化图
                self.visualize_temperature_data(temperature, annotations["objects"])
                
                # 新增：生成温度分布热力图
                self.generate_thermal_map(temperature, annotations["objects"])

                # 在保存数据时也保存温度信息
                temp_data = {
                    "frame": self.frame_count,
                    "temperature": temperature,
                    "unit": "celsius"
                }
                with open(f"{self.output_dir}/annotations/temp_frame_{self.frame_count:04d}.json", "w") as f:
                    json.dump(temp_data, f, indent=4)

                print(f"已仿真 {i}/{SIMULATION_FRAMES} 帧")
            else:
                # 使用上一帧的控制指令
                try:
                    left_speed, right_speed, steering_angle
                except NameError:
                    left_speed, right_speed, steering_angle = 5.0, 5.0, 0.0

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

        print(f"仿真完成！数据已保存到：{self.output_dir}")


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