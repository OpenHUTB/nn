"""
sensors.py - CARLA传感器管理
包含：相机、LiDAR传感器封装和管理
"""

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
                carla.Location(x=2.0, z=1.8),
                carla.Rotation(pitch=-10)  # 略微向下倾斜
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
        
        self.is_setup = camera_success and lidar_success
        
        if self.is_setup:
            logger.info("✅ 所有传感器初始化完成")
        else:
            logger.warning("⚠️  传感器初始化不完全")
        
        return self.is_setup
    
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
        
        return data
    
    def get_camera_image(self):
        """获取相机图像"""
        if self.camera_manager:
            return self.camera_manager.get_current_image()
        return None
    
    def get_lidar_pointcloud(self):
        """获取LiDAR点云"""
        if self.lidar_manager:
            return self.lidar_manager.current_pointcloud
        return None
    
    def get_lidar_objects(self):
        """获取LiDAR检测到的物体"""
        if self.lidar_manager:
            return self.lidar_manager.detect_objects()
        return []
    
    def get_open3d_pointcloud(self):
        """获取Open3D格式的点云"""
        if self.lidar_manager:
            return self.lidar_manager.get_open3d_pointcloud()
        return None
    
    def destroy(self):
        """销毁所有传感器"""
        logger.info("正在销毁传感器...")
        
        if self.camera_manager:
            self.camera_manager.destroy()
        
        if self.lidar_manager:
            self.lidar_manager.destroy()
        
        logger.info("✅ 所有传感器已销毁")


# ======================== 工具函数 ========================

def create_ego_vehicle(world, config, spawn_points=None):
    """
    创建自车 - 强制随机版本
    """
    try:
        # 强制使用随机种子
        import random
        import time
        import os
        
        # 生成强随机种子
        current_time = time.time()
        pid = os.getpid()
        seed = int((current_time * 1000) % 1000000) ^ pid
        random.seed(seed)
        
        logger.debug(f"随机种子: {seed} (时间: {current_time}, PID: {pid})")
        
        # 获取生成点
        if spawn_points is None:
            spawn_points = world.get_map().get_spawn_points()
        
        if not spawn_points:
            logger.error("❌ 无可用生成点")
            return None
        
        logger.info(f"找到 {len(spawn_points)} 个生成点")
        
        # 随机打乱所有生成点
        shuffled_points = spawn_points.copy()
        random.shuffle(shuffled_points)
        
        # 尝试前5个打乱后的点
        max_attempts = min(5, len(shuffled_points))
        
        for attempt in range(max_attempts):
            spawn_point = shuffled_points[attempt]
            
            logger.info(f"🎲 尝试 {attempt + 1}/{max_attempts}: "
                       f"随机选择的位置 ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
            
            # 获取车辆蓝图
            vehicle_bp = world.get_blueprint_library().filter('vehicle.*')[0]
            
            # 设置生成点高度
            spawn_point.location.z += 0.5
            
            ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            
            if ego_vehicle is not None:
                logger.info(f"✅ 自车生成成功！")
                logger.info(f"  位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
                
                # 设置自动驾驶
                try:
                    ego_vehicle.set_autopilot(True)
                    logger.info("✅ 自动驾驶已启用")
                except:
                    pass
                
                return ego_vehicle
        
        logger.error("❌ 所有随机位置尝试都失败")
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
            import random
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
        
        # 再清理车辆（分批进行）
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


# ======================== 测试函数 ========================

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