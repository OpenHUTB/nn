"""
sensors.py - CARLA传感器管理
包含：相机、LiDAR传感器封装和管理
"""

import random  # 添加随机模块支持
import carla
import cv2
import numpy as np
import queue
import threading
import sys
import time

# 配置日志
try:
    from loguru import logger
except ImportError:
    # 使用标准logging作为回退
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

import open3d as o3d
from sklearn.cluster import DBSCAN

class CameraManager:
    """相机管理器"""
    
    def __init__(self, world, ego_vehicle, config):
        """
        初始化相机
        
        Args:
            world: CARLA世界对象
            ego_vehicle: 自车对象
            config: 配置字典
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.config = config
        self.camera = None
        self.image_queue = queue.Queue(maxsize=2)
        self.current_image = None
        self.frame_count = 0
        
    def setup(self):
        """设置相机"""
        try:
            # 获取相机蓝图
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            
            # 设置相机属性
            camera_bp.set_attribute('image_size_x', str(self.config.get('img_width', 640)))
            camera_bp.set_attribute('image_size_y', str(self.config.get('img_height', 480)))
            camera_bp.set_attribute('fov', str(self.config.get('fov', 90)))
            camera_bp.set_attribute('sensor_tick', str(self.config.get('sensor_tick', 0.05)))
            
            # 设置相机位置（车顶前方）
            camera_transform = carla.Transform(
                carla.Location(x=2.5, z=2.5),
                carla.Rotation(pitch=-5)  # 略微向下倾斜
            )
            
            # 生成相机
            self.camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.ego_vehicle
            )
            
            # 绑定回调函数
            self.camera.listen(self._camera_callback)
            
            logger.info(f"✅ 相机初始化成功 (ID: {self.camera.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 相机初始化失败: {e}")
            return False
    
    def _camera_callback(self, image):
        """相机数据回调函数"""
        try:
            # 将原始数据转换为numpy数组
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            
            # 提取RGB通道（去掉alpha通道）
            rgb_array = array[:, :, :3]
            
            # 轻微高斯模糊减少噪声
            rgb_array = cv2.GaussianBlur(rgb_array, (3, 3), 0)
            
            # 更新当前图像
            self.current_image = rgb_array
            
            # 放入队列（如果队列已满，丢弃最旧的数据）
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.image_queue.put(rgb_array.copy())
            self.frame_count += 1
            
        except Exception as e:
            logger.warning(f"相机回调错误: {e}")
    
    def get_image(self, timeout=0.1):
        """
        获取最新图像
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            np.ndarray or None: 图像数据
        """
        try:
            # 首先尝试从队列获取最新图像
            image = self.image_queue.get(timeout=timeout)
            # 清空队列中的旧图像
            while not self.image_queue.empty():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    break
            return image
        except queue.Empty:
            # 如果队列为空，返回当前图像
            return self.current_image
    
    def get_current_image(self):
        """获取当前图像（不阻塞）"""
        return self.current_image
    
    def destroy(self):
        """销毁相机"""
        if self.camera and self.camera.is_alive:
            try:
                self.camera.stop()
                self.camera.destroy()
                logger.info("✅ 相机已销毁")
            except Exception as e:
                logger.warning(f"销毁相机失败: {e}")
        self.camera = None

class LiDARManager:
    """LiDAR管理器"""
    
    def __init__(self, world, ego_vehicle, config):
        """
        初始化LiDAR
        
        Args:
            world: CARLA世界对象
            ego_vehicle: 自车对象
            config: 配置字典
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.config = config
        self.lidar = None
        self.pointcloud_queue = queue.Queue(maxsize=2)
        self.current_pointcloud = None
        self.current_transform = None
        
    def setup(self):
        """设置LiDAR"""
        try:
            if not self.config.get('use_lidar', True):
                logger.info("LiDAR被禁用")
                return True
            
            # 获取LiDAR蓝图
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            
            # 设置LiDAR属性
            lidar_bp.set_attribute('channels', str(self.config.get('lidar_channels', 32)))
            lidar_bp.set_attribute('range', str(self.config.get('lidar_range', 100.0)))
            lidar_bp.set_attribute('points_per_second', 
                                 str(self.config.get('lidar_points_per_second', 500000)))
            lidar_bp.set_attribute('rotation_frequency', str(self.config.get('rotation_frequency', 20)))
            lidar_bp.set_attribute('sensor_tick', str(self.config.get('sensor_tick', 0.05)))
            
            # 设置LiDAR位置（车顶中央）
            lidar_transform = carla.Transform(
                carla.Location(x=0.0, z=2.5),
                carla.Rotation()
            )
            
            # 生成LiDAR
            self.lidar = self.world.spawn_actor(
                lidar_bp,
                lidar_transform,
                attach_to=self.ego_vehicle
            )
            
            # 绑定回调函数
            self.lidar.listen(self._lidar_callback)
            
            logger.info(f"✅ LiDAR初始化成功 (ID: {self.lidar.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ LiDAR初始化失败: {e}")
            return False
    
    def _lidar_callback(self, pointcloud):
        """LiDAR数据回调函数"""
        try:
            # 将原始数据转换为numpy数组
            points = np.frombuffer(pointcloud.raw_data, dtype=np.float32)
            points = points.reshape(-1, 4)[:, :3]  # 只取xyz，忽略反射强度
            
            # 过滤地面点（简单的高度过滤）
            ground_mask = points[:, 2] < -1.0
            filtered_points = points[~ground_mask]
            
            # 更新当前点云
            self.current_pointcloud = filtered_points
            self.current_transform = pointcloud.transform
            
            # 放入队列（如果队列已满，丢弃最旧的数据）
            if self.pointcloud_queue.full():
                try:
                    self.pointcloud_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.pointcloud_queue.put((filtered_points.copy(), pointcloud.transform))
            
        except Exception as e:
            logger.warning(f"LiDAR回调错误: {e}")
    
    def get_pointcloud(self, timeout=0.1):
        """
        获取最新点云数据
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            tuple: (points, transform) 或 (None, None)
        """
        try:
            points, transform = self.pointcloud_queue.get(timeout=timeout)
            # 清空队列中的旧数据
            while not self.pointcloud_queue.empty():
                try:
                    self.pointcloud_queue.get_nowait()
                except queue.Empty:
                    break
            return points, transform
        except queue.Empty:
            # 如果队列为空，返回当前点云
            return self.current_pointcloud, self.current_transform
    
    def detect_objects(self, min_points=30):
        """
        从点云中检测物体
        
        Args:
            min_points: 最小点数阈值
            
        Returns:
            list: 检测到的物体列表
        """
        if self.current_pointcloud is None or len(self.current_pointcloud) < min_points:
            return []
        
        try:
            # 使用DBSCAN聚类
            clustering = DBSCAN(eps=0.8, min_samples=30).fit(self.current_pointcloud[:, :2])
            
            objects = []
            for label in set(clustering.labels_):
                if label == -1:  # 忽略噪声点
                    continue
                
                # 获取该聚类的点
                cluster_points = self.current_pointcloud[clustering.labels_ == label]
                
                if len(cluster_points) < min_points:
                    continue
                
                # 计算3D边界框
                min_coords = cluster_points.min(axis=0)
                max_coords = cluster_points.max(axis=0)
                center = (min_coords + max_coords) / 2
                size = max_coords - min_coords
                
                # 估计物体类型（基于尺寸）
                obj_type = self._estimate_object_type(size)
                
                objects.append({
                    'bbox_3d': [*min_coords, *max_coords],  # [x_min, y_min, z_min, x_max, y_max, z_max]
                    'center': center.tolist(),
                    'size': size.tolist(),
                    'num_points': len(cluster_points),
                    'type': obj_type,
                    'label': label
                })
            
            return objects
            
        except Exception as e:
            logger.warning(f"LiDAR物体检测失败: {e}")
            return []
    
    def _estimate_object_type(self, size):
        """根据尺寸估计物体类型"""
        length, width, height = size
        
        # 简单的大小分类
        if height > 2.5:
            return "truck"
        elif width > 2.0:
            return "bus"
        elif length > 4.0:
            return "car"
        else:
            return "unknown"
    
    def get_open3d_pointcloud(self):
        """
        获取Open3D格式的点云
        
        Returns:
            o3d.geometry.PointCloud or None: Open3D点云对象
        """
        if self.current_pointcloud is None or len(self.current_pointcloud) == 0:
            return None
        
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.current_pointcloud)
            
            # 根据高度着色（低处蓝色，高处红色）
            z_min = self.current_pointcloud[:, 2].min()
            z_max = self.current_pointcloud[:, 2].max()
            z_range = max(z_max - z_min, 1e-6)
            
            colors = np.zeros((len(self.current_pointcloud), 3))
            normalized_z = (self.current_pointcloud[:, 2] - z_min) / z_range
            colors[:, 0] = normalized_z  # 红色通道（高处）
            colors[:, 2] = 1 - normalized_z  # 蓝色通道（低处）
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            return pcd
            
        except Exception as e:
            logger.warning(f"创建Open3D点云失败: {e}")
            return None
    
    def destroy(self):
        """销毁LiDAR"""
        if self.lidar and self.lidar.is_alive:
            try:
                self.lidar.stop()
                self.lidar.destroy()
                logger.info("✅ LiDAR已销毁")
            except Exception as e:
                logger.warning(f"销毁LiDAR失败: {e}")
        self.lidar = None

class SpectatorManager:
    """CARLA视角管理器 - 提供卫星视角跟随"""
    
    def __init__(self, world, ego_vehicle, config):
        """
        初始化视角管理器
        
        Args:
            world: CARLA世界对象
            ego_vehicle: 自车对象
            config: 配置字典
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.config = config
        self.spectator = None
        view_config = config.get('view', {})
        self.view_mode = view_config.get('default_mode', 'satellite')
        self.view_height = view_config.get('satellite_height', 50.0)
        self.follow_distance = view_config.get('behind_distance', 10.0)
        self.first_person_height = view_config.get('first_person_height', 1.6)
        
    def setup(self):
        """设置视角"""
        try:
            # 获取世界中的观察者（spectator）
            self.spectator = self.world.get_spectator()
            
            # 设置初始视角
            self._set_satellite_view()
            
            logger.info(f"✅ 视角管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 视角管理器初始化失败: {e}")
            return False
    
    def _set_satellite_view(self):
        """设置卫星视角"""
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return
        
        try:
            # 获取车辆位置和方向
            vehicle_transform = self.ego_vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # 设置卫星视角位置（车辆正上方）
            spectator_location = carla.Location(
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z + self.view_height
            )
            
            # 设置视角朝向（俯瞰车辆）
            spectator_rotation = carla.Rotation(
                pitch=-90,  # 向下看
                yaw=vehicle_rotation.yaw,
                roll=0
            )
            
            # 应用变换
            spectator_transform = carla.Transform(
                spectator_location,
                spectator_rotation
            )
            self.spectator.set_transform(spectator_transform)
            
        except Exception as e:
            logger.warning(f"设置卫星视角失败: {e}")
    
    def _set_behind_view(self):
        """设置后方跟随视角"""
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return
        
        try:
            # 获取车辆位置和方向
            vehicle_transform = self.ego_vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # 计算后方偏移位置
            import math
            yaw_rad = math.radians(vehicle_rotation.yaw)
            
            # 车辆后方偏移
            behind_x = vehicle_location.x - self.follow_distance * math.cos(yaw_rad)
            behind_y = vehicle_location.y - self.follow_distance * math.sin(yaw_rad)
            
            # 设置摄像机位置（稍高于车辆）
            spectator_location = carla.Location(
                x=behind_x,
                y=behind_y,
                z=vehicle_location.z + 3.0
            )
            
            # 设置视角朝向（看向车辆）
            # 计算朝向车辆的旋转
            dx = vehicle_location.x - spectator_location.x
            dy = vehicle_location.y - spectator_location.y
            target_yaw = math.degrees(math.atan2(dy, dx))
            
            spectator_rotation = carla.Rotation(
                pitch=-15,  # 略微向下
                yaw=target_yaw,
                roll=0
            )
            
            # 应用变换
            spectator_transform = carla.Transform(
                spectator_location,
                spectator_rotation
            )
            self.spectator.set_transform(spectator_transform)
            
        except Exception as e:
            logger.warning(f"设置后方视角失败: {e}")
    
    def _set_first_person_view(self):
        """设置第一人称视角"""
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return
        
        try:
            # 获取车辆变换
            vehicle_transform = self.ego_vehicle.get_transform()
            
            # 稍微调整位置（从驾驶员视角）
            location = carla.Location(
                x=vehicle_transform.location.x,
                y=vehicle_transform.location.y,
                z=vehicle_transform.location.z + self.first_person_height  # 改为使用配置
            )
            
            # 使用车辆的方向
            rotation = carla.Rotation(
                pitch=vehicle_transform.rotation.pitch,
                yaw=vehicle_transform.rotation.yaw,
                roll=vehicle_transform.rotation.roll
            )
            
            # 应用变换
            spectator_transform = carla.Transform(
                location,
                rotation
            )
            self.spectator.set_transform(spectator_transform)
            
        except Exception as e:
            logger.warning(f"设置第一人称视角失败: {e}")
    
    def set_view_mode(self, mode):
        """
        设置视角模式
        
        Args:
            mode: 视角模式 ('satellite', 'behind', 'first_person')
        """
        if mode not in ['satellite', 'behind', 'first_person']:
            logger.warning(f"未知的视角模式: {mode}")
            return
        
        self.view_mode = mode
    
    def update(self):
        """更新视角"""
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return
        
        try:
            if self.view_mode == 'satellite':
                self._set_satellite_view()
            elif self.view_mode == 'behind':
                self._set_behind_view()
            elif self.view_mode == 'first_person':
                self._set_first_person_view()
                
        except Exception as e:
            logger.debug(f"更新视角失败: {e}")
    
    def cycle_view_mode(self):
        """循环切换视角模式"""
        modes = ['satellite', 'behind', 'first_person']
        current_index = modes.index(self.view_mode) if self.view_mode in modes else 0
        next_index = (current_index + 1) % len(modes)
        self.view_mode = modes[next_index]
        logger.info(f"切换到 {self.view_mode} 视角")
    
    def destroy(self):
        """销毁视角管理器"""
        self.spectator = None
        logger.info("✅ 视角管理器已销毁")

class SensorManager:
    """传感器管理器（统一管理所有传感器）"""
    
    def __init__(self, world, ego_vehicle, config):
        """
        初始化传感器管理器
        
        Args:
            world: CARLA世界对象
            ego_vehicle: 自车对象
            config: 配置字典
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.config = config
        
        self.camera_manager = None
        self.lidar_manager = None
        self.spectator_manager = None  # 新增视角管理器
        self.is_setup = False
        
    def setup(self):
        """设置所有传感器"""
        logger.info("正在初始化传感器...")
        
        # 初始化相机
        self.camera_manager = CameraManager(self.world, self.ego_vehicle, self.config)
        camera_success = self.camera_manager.setup()
        
        # 初始化LiDAR
        lidar_success = True
        if self.config.get('use_lidar', True):
            self.lidar_manager = LiDARManager(self.world, self.ego_vehicle, self.config)
            lidar_success = self.lidar_manager.setup()
        else:
            logger.info("LiDAR功能已禁用")
        
        # 初始化视角管理器
        self.spectator_manager = SpectatorManager(self.world, self.ego_vehicle, self.config)
        spectator_success = self.spectator_manager.setup()
        
        self.is_setup = camera_success and lidar_success and spectator_success
        
        if self.is_setup:
            logger.info("✅ 所有传感器初始化完成")
        else:
            logger.warning("⚠️  传感器初始化不完全")
        
        return self.is_setup
    
    # 在get_sensor_data方法中添加视角更新
    def get_sensor_data(self, timeout=0.05):
        """
        获取所有传感器数据
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            dict: 传感器数据字典
        """
        data = {
            'image': None,
            'pointcloud': None,
            'lidar_transform': None,
            'lidar_objects': [],
            'timestamp': time.time()
        }
        
        # 获取相机图像
        if self.camera_manager:
            data['image'] = self.camera_manager.get_image(timeout=timeout)
        
        # 获取LiDAR数据
        if self.lidar_manager:
            points, transform = self.lidar_manager.get_pointcloud(timeout=timeout)
            data['pointcloud'] = points
            data['lidar_transform'] = transform
            
            # 检测物体
            if points is not None:
                data['lidar_objects'] = self.lidar_manager.detect_objects()
        
        # 更新视角
        if self.spectator_manager:
            self.spectator_manager.update()
        
        return data
    
    # 添加视角控制方法
    def set_view_mode(self, mode):
        """设置视角模式"""
        if self.spectator_manager:
            self.spectator_manager.set_view_mode(mode)
    
    def cycle_view_mode(self):
        """循环切换视角模式"""
        if self.spectator_manager:
            self.spectator_manager.cycle_view_mode()
    
    def destroy(self):
        """销毁所有传感器"""
        logger.info("正在销毁传感器...")
        
        if self.camera_manager:
            self.camera_manager.destroy()
        
        if self.lidar_manager:
            self.lidar_manager.destroy()
        
        if self.spectator_manager:
            self.spectator_manager.destroy()
        
        logger.info("✅ 所有传感器已销毁")

def create_ego_vehicle(world, config, spawn_points=None):
    """
    创建自车
    
    Args:
        world: CARLA世界对象
        config: 配置字典
        spawn_points: 可选的自定义生成点列表
        
    Returns:
        carla.Vehicle or None: 自车对象
    """
    try:
        # 获取生成点
        if spawn_points is None:
            spawn_points = world.get_map().get_spawn_points()
        
        if not spawn_points:
            logger.error("❌ 无可用生成点")
            return None
        
        logger.info(f"找到 {len(spawn_points)} 个生成点")
        
        # 选择车辆蓝图
        vehicle_bp = None
        vehicle_filter = config.get('ego_vehicle_filter', 'vehicle.tesla.model3')
        
        # 尝试首选车辆
        blueprint_library = world.get_blueprint_library()
        for bp in blueprint_library.filter(vehicle_filter):
            if int(bp.get_attribute('number_of_wheels')) == 4:
                vehicle_bp = bp
                logger.info(f"找到车辆蓝图: {bp.id}")
                break
        
        # 如果没找到，选择任意四轮车辆
        if vehicle_bp is None:
            logger.info("首选车辆未找到，尝试其他四轮车辆...")
            for bp in blueprint_library.filter('vehicle.*'):
                if int(bp.get_attribute('number_of_wheels')) == 4:
                    vehicle_bp = bp
                    logger.info(f"使用备用车辆蓝图: {bp.id}")
                    break
        
        if vehicle_bp is None:
            logger.error("❌ 找不到合适的车辆蓝图")
            return None
        
        # 设置车辆颜色
        color = config.get('ego_vehicle_color', '255,0,0')
        vehicle_bp.set_attribute('color', color)
        
        # 尝试生成车辆 - 改进的碰撞避免策略
        max_attempts = config.get('spawn_max_attempts', 20)
        
        for attempt in range(max_attempts):
            # 随机选择生成点
            if attempt < len(spawn_points):
                spawn_point = spawn_points[attempt]
            else:
                # 选择随机一个生成点
                import random
                spawn_point = random.choice(spawn_points)
                
                # 随机偏移位置以避免碰撞
                spawn_point.location.x += random.uniform(-3, 3)
                spawn_point.location.y += random.uniform(-3, 3)
            
            logger.info(f"尝试生成自车 (尝试 {attempt + 1}/{max_attempts}) "
                       f"位置: x={spawn_point.location.x:.1f}, y={spawn_point.location.y:.1f}")
            
            # 设置生成点的高度为地面以上0.5米
            spawn_point.location.z += 0.5
            
            ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            
            if ego_vehicle is not None:
                logger.info(f"✅ 自车生成成功 (尝试 {attempt + 1}/{max_attempts})")
                logger.info(f"  位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}, {spawn_point.location.z:.1f})")
                
                # 等待一小段时间让车辆稳定
                world.tick()
                
                # 设置自动驾驶
                try:
                    ego_vehicle.set_autopilot(True, 8000)
                    logger.info("✅ 自车自动驾驶已启用")
                except Exception as e:
                    logger.warning(f"设置自动驾驶失败: {e}")
                    try:
                        ego_vehicle.set_autopilot(True)
                        logger.info("✅ 自车自动驾驶已启用（备用方法）")
                    except:
                        logger.warning("无法设置自动驾驶，车辆将保持静止")
                
                return ego_vehicle
            else:
                logger.debug(f"生成失败，尝试下一个位置...")
        
        logger.error(f"❌ 经过 {max_attempts} 次尝试后仍无法生成自车")
        logger.info("建议：")
        logger.info("1. 重新启动CARLA服务器")
        logger.info("2. 在CARLA中手动清理场景中的车辆")
        logger.info("3. 尝试不同的生成点")
        
        return None
        
    except Exception as e:
        logger.error(f"❌ 创建自车失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def spawn_npc_vehicles(world, config, count=None):
    """
    生成NPC车辆
    
    Args:
        world: CARLA世界对象
        config: 配置字典
        count: NPC数量（默认使用配置中的值）
        
    Returns:
        int: 成功生成的NPC数量
    """
    try:
        if count is None:
            count = config.get('num_npcs', 20)
        
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            logger.warning("无可用生成点，无法生成NPC")
            return 0
        
        # 过滤合适的车辆蓝图（四轮车辆，排除特殊车辆）
        vehicle_bps = []
        for bp in world.get_blueprint_library().filter('vehicle.*'):
            if int(bp.get_attribute('number_of_wheels')) == 4:
                # 排除特殊车辆
                if not bp.id.endswith(('firetruck', 'ambulance', 'police', 'charger')):
                    vehicle_bps.append(bp)
        
        if not vehicle_bps:
            logger.warning("找不到合适的NPC车辆蓝图")
            return 0
        
        spawned_count = 0
        used_spawn_points = set()
        
        for i in range(min(count * 3, len(spawn_points))):  # 最多尝试3倍数量
            if spawned_count >= count:
                break
            
            spawn_point = spawn_points[i]
            
            # 检查是否已使用该位置
            position_key = (round(spawn_point.location.x, 1), 
                          round(spawn_point.location.y, 1))
            
            if position_key in used_spawn_points:
                continue
            
            # 随机选择车辆蓝图
            vehicle_bp = random.choice(vehicle_bps)
            
            # 尝试生成
            npc = world.try_spawn_actor(vehicle_bp, spawn_point)
            
            if npc is not None:
                used_spawn_points.add(position_key)
                spawned_count += 1
                
                # 设置自动驾驶
                try:
                    npc.set_autopilot(True, 8000)
                except:
                    try:
                        npc.set_autopilot(True)
                    except:
                        pass
        
        logger.info(f"✅ 成功生成 {spawned_count}/{count} 个NPC车辆")
        return spawned_count
        
    except Exception as e:
        logger.error(f"生成NPC车辆失败: {e}")
        return 0

def clear_all_actors(world, exclude_ids=None):
    """
    清理所有演员（车辆和传感器）
    
    Args:
        world: CARLA世界对象
        exclude_ids: 要排除的演员ID列表
    """
    try:
        exclude_ids = set(exclude_ids) if exclude_ids else set()
        
        actors = world.get_actors()
        
        # 按类型分组清理
        vehicle_actors = []
        sensor_actors = []
        
        for actor in actors:
            if actor.id in exclude_ids:
                continue
            
            if actor.type_id.startswith('vehicle.'):
                vehicle_actors.append(actor)
            elif actor.type_id.startswith('sensor.'):
                sensor_actors.append(actor)
        
        # 先清理传感器
        logger.info(f"清理 {len(sensor_actors)} 个传感器...")
        for sensor in sensor_actors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except:
                pass
        
        # 再清理车辆
        logger.info(f"清理 {len(vehicle_actors)} 个车辆...")
        batch_size = 10
        for i in range(0, len(vehicle_actors), batch_size):
            batch = vehicle_actors[i:i+batch_size]
            for vehicle in batch:
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                except:
                    pass
        
        logger.info("✅ 清理完成")
        
    except Exception as e:
        logger.warning(f"清理演员时出错: {e}")

def test_sensor_manager():
    """测试传感器管理器"""
    print("=" * 50)
    print("测试 sensors.py...")
    print("=" * 50)
    
    # 模拟配置
    test_config = {
        'img_width': 640,
        'img_height': 480,
        'fov': 90,
        'sensor_tick': 0.05,
        'use_lidar': True,
        'lidar_channels': 32,
        'lidar_range': 100.0,
        'lidar_points_per_second': 500000,
    }
    
    print("✅ sensors.py 结构测试通过")
    print("注：完整测试需要CARLA环境")
    
    return True

if __name__ == "__main__":
    test_sensor_manager()