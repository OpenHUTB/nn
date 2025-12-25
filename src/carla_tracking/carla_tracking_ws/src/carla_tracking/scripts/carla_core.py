# carla_core.py - CARLA核心逻辑（无ROS依赖）
import carla
import numpy as np
import torch
import queue
import time
import json
import logging
from collections import deque
import cv2
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# 【复制原代码中所有类和工具函数】
# 包括：ConfigManager、OptimizedKalmanFilter、OptimizedTrack、OptimizedSORT、
# VehicleController、NPCManager、所有工具函数（preprocess_depth_image等）、
# CARLA相关函数（setup_carla_client、spawn_ego_vehicle等）、
# 回调函数（camera_callback、depth_camera_callback）、
# load_detection_model函数

# 移除原main函数，新增CARLA核心类
class CarlaTrackingCore:
    def __init__(self, config_path=None):
        # 初始化配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # 初始化变量
        self.world = None
        self.client = None
        self.vehicle = None
        self.camera = None
        self.depth_camera = None
        self.image_queue = queue.Queue(maxsize=3)
        self.depth_queue = queue.Queue(maxsize=3)
        self.npc_manager = None
        self.model = None
        self.class_names = None
        self.tracker = None
        self.controller = None
        
        # 状态变量
        self.running = False
        self.frame_count = 0
        self.detection_results = {
            "boxes": [],
            "tracks": [],
            "tracks_info": [],
            "fps": 0.0
        }
        
        # 日志
        self.logger = logging.getLogger("carla_tracking_ros")

    def initialize(self):
        """初始化CARLA和模型"""
        # 1. 连接CARLA
        self.world, self.client = setup_carla_client(self.config_manager)
        
        # 2. 生成主车辆
        self.vehicle = spawn_ego_vehicle(self.world, self.config_manager)
        self.controller = VehicleController(self.vehicle, self.config_manager)
        
        # 3. 生成传感器
        bp_lib = self.world.get_blueprint_library()
        # RGB相机
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.config['camera']['width']))
        camera_bp.set_attribute('image_size_y', str(self.config['camera']['height']))
        camera_bp.set_attribute('fov', str(self.config['camera']['fov']))
        camera_bp.set_attribute('sensor_tick', '0.05')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5, yaw=0))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: camera_callback(image, self.image_queue))
        
        # 深度相机
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.config['camera']['width']))
        depth_bp.set_attribute('image_size_y', str(self.config['camera']['height']))
        depth_bp.set_attribute('fov', str(self.config['camera']['fov']))
        depth_bp.set_attribute('sensor_tick', '0.05')
        self.depth_camera = self.world.spawn_actor(depth_bp, camera_transform, attach_to=self.vehicle)
        self.depth_camera.listen(lambda image: depth_camera_callback(image, self.depth_queue))
        
        # 4. 生成NPC
        self.npc_manager = NPCManager(self.client, self.config_manager)
        self.npc_manager.spawn_npcs(self.world, count=self.config['npc']['count'], ego_vehicle=self.vehicle)
        
        # 5. 加载模型和跟踪器
        self.model, self.class_names = load_detection_model(self.config['detection']['model_type'])
        self.tracker = OptimizedSORT(self.config['tracking'])
        
        # 等待初始化完成
        for _ in range(10):
            self.world.tick()
        
        self.running = True
        self.logger.info("CARLA核心初始化完成")

    def step(self):
        """单步运行（一帧）"""
        if not self.running:
            return
        
        frame_start = time.time()
        self.frame_count += 1
        
        # 1. 同步CARLA
        self.world.tick()
        self.npc_manager.update_npc_behavior()
        
        # 2. 获取图像
        if self.image_queue.empty():
            return
        origin_image = self.image_queue.get()
        image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        
        # 3. 获取深度图像
        depth_image = None
        if not self.depth_queue.empty():
            depth_image = self.depth_queue.get()
        
        # 4. 目标检测
        boxes, labels, probs, depths = [], [], [], []
        conf_thres = self.config['detection']['conf_thres']
        iou_thres = self.config['detection']['iou_thres']
        device = self.config['detection']['device']
        
        results = self.model(image, conf=conf_thres, iou=iou_thres, device=device, imgsz=640, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if cls in [2, 3, 5, 7]:  # 只保留车辆类
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
                    probs.append(conf)
                    if depth_image is not None:
                        dist = get_target_distance(depth_image, [x1, y1, x2, y2])
                        depths.append(dist)
        
        # 5. 目标跟踪
        detections = []
        for i in range(len(boxes)):
            det = boxes[i] + [probs[i]] + [labels[i]]
            detections.append(det)
        track_results = self.tracker.update(detections, depths)
        tracks_info = self.tracker.get_tracks_info()
        
        # 6. 更新控制（自动模式）
        obstacle_distances = [t['distance'] for t in tracks_info if t['distance'] is not None]
        control = self.controller.update_control(obstacle_distances)
        self.vehicle.apply_control(control)
        
        # 7. 计算FPS
        total_time = time.time() - frame_start
        fps = 1.0 / total_time if total_time > 0 else 0.0
        
        # 8. 更新结果
        self.detection_results = {
            "image": origin_image,  # BGR格式
            "depth_image": depth_image,
            "boxes": boxes,
            "tracks": track_results,
            "tracks_info": tracks_info,
            "fps": fps,
            "frame_count": self.frame_count
        }

    def cleanup(self):
        """清理资源"""
        self.running = False
        # 停止传感器
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.depth_camera:
            self.depth_camera.stop()
            self.depth_camera.destroy()
        # 销毁NPC和主车辆
        if self.npc_manager:
            self.npc_manager.destroy_all_npcs()
        if self.vehicle:
            self.vehicle.destroy()
        # 恢复CARLA设置
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("资源清理完成")
