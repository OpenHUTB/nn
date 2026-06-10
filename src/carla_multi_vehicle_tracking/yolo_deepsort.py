#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO + DeepSort 多车跟踪系统
兼容 Python 3.8.10 + CARLA 0.9.13 + Windows 10

【新增：碰撞预警模块】
"""

from __future__ import print_function, absolute_import
import sys
import os
import argparse
import traceback

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch 未安装，将使用 CPU")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[WARNING] Ultralytics 未安装")

try:
    from deep_sort.deep_sort import DeepSort
    from deep_sort.utils.parser import get_config
    DEEPSORT_AVAILABLE = True
except ImportError as e:
    DEEPSORT_AVAILABLE = False
    print(f"[WARNING] DeepSort 导入失败: {e}")

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("[WARNING] CARLA 未安装")

import random

# ==================== 配置常量 ====================
class_id = [2, 3, 5, 7]
class_name = {2: 'car', 3: 'motobike', 5: 'bus', 7: 'truck'}

img_w = 256 * 4
img_h = 256 * 3
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
output_path = "output.mp4"

# 【新增：碰撞预警模块】配置变量
COLLISION_DISTANCE_THRESHOLD_RATIO = 0.10
COLLISION_DISTANCE_THRESHOLD = int(img_w * COLLISION_DISTANCE_THRESHOLD_RATIO)
COLLISION_WARNING_COLOR = (0, 0, 255)
COLLISION_WARNING_TEXT_COLOR = (0, 0, 255)
COLLISION_WARNING_BOX_THICKNESS = 3
COLLISION_WARNING_TEXT_SCALE = 0.8
COLLISION_WARNING_TEXT_THICKNESS = 2
COLLISION_WARNING_TEXT_POSITION = (10, 30)
COLLISION_WARNING_MESSAGE = "COLLISION WARNING!"

# ==================== 碰撞预警模块 ====================
def check_vehicle_collision(tracked_vehicles, frame_width):
    """
    【新增：碰撞预警模块】检测车辆之间的碰撞风险
    
    参数:
        tracked_vehicles: 跟踪到的车辆列表
        frame_width: 画面宽度
    
    返回:
        collision_pairs: 有碰撞风险的车辆对列表
    """
    collision_pairs = []
    
    if len(tracked_vehicles) < 2:
        return collision_pairs
    
    vehicles = []
    for output in tracked_vehicles:
        if len(output) >= 5:
            try:
                x1, y1, x2, y2 = map(int, output[0:4])
                track_id = int(output[4])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                vehicles.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'track_id': track_id
                })
            except (ValueError, TypeError, IndexError):
                continue
    
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            vehicle1 = vehicles[i]
            vehicle2 = vehicles[j]
            
            dx = vehicle1['center'][0] - vehicle2['center'][0]
            dy = vehicle1['center'][1] - vehicle2['center'][1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            
            if distance < COLLISION_DISTANCE_THRESHOLD:
                collision_pairs.append((
                    vehicle1['bbox'], 
                    vehicle2['bbox'], 
                    vehicle1['track_id'], 
                    vehicle2['track_id']
                ))
                print(f"[警告] 车辆ID:{vehicle1['track_id']} 和 车辆ID:{vehicle2['track_id']} 距离过近，存在碰撞风险")
    
    return collision_pairs


def draw_collision_warning(frame, collision_pairs):
    """
    【新增：碰撞预警模块】在视频帧上绘制预警
    """
    if len(collision_pairs) > 0:
        # 绘制警告文字
        cv2.putText(
            frame,
            COLLISION_WARNING_MESSAGE,
            COLLISION_WARNING_TEXT_POSITION,
            cv2.FONT_HERSHEY_SIMPLEX,
            COLLISION_WARNING_TEXT_SCALE,
            COLLISION_WARNING_TEXT_COLOR,
            COLLISION_WARNING_TEXT_THICKNESS,
            cv2.LINE_AA
        )
        
        # 绘制警告框
        for bbox1, bbox2, id1, id2 in collision_pairs:
            x1, y1, x2, y2 = bbox1
            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                         COLLISION_WARNING_COLOR, COLLISION_WARNING_BOX_THICKNESS)
            
            x1, y1, x2, y2 = bbox2
            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                         COLLISION_WARNING_COLOR, COLLISION_WARNING_BOX_THICKNESS)
    
    return frame


# ==================== 主跟踪类 ====================
class VehicleTracker:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        print(f"[INFO] 使用设备: {self.device}")
        print(f"[INFO] Python 版本: {sys.version}")
        
        # 检查依赖
        self._check_dependencies()
        
        # 加载模型
        self.model = None
        self.deepsort = None
        
        if ULTRALYTICS_AVAILABLE:
            self._load_yolo_model()
        
        if DEEPSORT_AVAILABLE:
            self._load_deepsort()
        
        print("[INFO] 初始化完成，准备开始跟踪")
    
    def _check_dependencies(self):
        """检查依赖是否完整"""
        print("\n[INFO] 检查依赖...")
        
        deps_status = {
            'PyTorch': TORCH_AVAILABLE,
            'Ultralytics': ULTRALYTICS_AVAILABLE,
            'DeepSort': DEEPSORT_AVAILABLE,
            'CARLA': CARLA_AVAILABLE
        }
        
        for name, available in deps_status.items():
            status = "✓" if available else "✗"
            print(f"  [{status}] {name}")
        
        if not ULTRALYTICS_AVAILABLE:
            print("[ERROR] 必须安装 ultralytics: pip install ultralytics==8.0.150")
        
        if not DEEPSORT_AVAILABLE:
            print("[ERROR] 必须安装 deep_sort 模块")
    
    def _load_yolo_model(self):
        """加载 YOLO 模型"""
        model_paths = [
            'weights/yolov8n.pt',
            'weights/best.pt',
            'yolov8n.pt'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print(f"[WARNING] 未找到 YOLO 模型文件，使用默认路径: {model_paths[0]}")
            model_path = model_paths[0]
        
        try:
            self.model = YOLO(model_path)
            print(f"[INFO] YOLO 模型加载成功: {model_path}")
        except Exception as e:
            print(f"[ERROR] 加载 YOLO 模型失败: {str(e)}")
            traceback.print_exc()
    
    def _load_deepsort(self):
        """加载 DeepSort 跟踪器"""
        weight_paths = [
            "deep_sort/deep/checkpoint/ckpt.t7",
            "deep_sort/deepSORT/ckpt.t7"
        ]
        
        weight_path = None
        for path in weight_paths:
            if os.path.exists(path):
                weight_path = path
                break
        
        if not weight_path:
            print(f"[ERROR] 未找到 DeepSort 权重文件: {weight_paths[0]}")
            print("[INFO] 跳过 DeepSort，跟踪功能可能受限")
            return
        
        try:
            self.cfg = get_config()
            cfg_path = 'deep_sort/configs/deep_sort.yaml'
            if os.path.exists(cfg_path):
                self.cfg.merge_from_file(cfg_path)
            
            self.deepsort = DeepSort(weight_path, max_age=70)
            print(f"[INFO] DeepSort 加载成功: {weight_path}")
        except Exception as e:
            print(f"[ERROR] 加载 DeepSort 失败: {str(e)}")
            traceback.print_exc()
    
    def yolo_details(self, frame):
        """YOLO 检测"""
        if not self.model:
            return frame, [], [], []
        
        try:
            results = self.model(frame)
            bbox_xyxy = []
            conf_score = []
            cls_id = []
            
            for box in results:
                if hasattr(box, 'boxes') and box.boxes is not None:
                    data_list = box.boxes.data.tolist()
                    for row in data_list:
                        if len(row) >= 6:
                            class_id_val = int(row[5])
                            if class_id_val in class_id:
                                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                                conf = row[4]
                                bbox_xyxy.append([x1, y1, x2, y2])
                                conf_score.append(conf)
                                cls_id.append(class_id_val)
            
            return frame, bbox_xyxy, conf_score, cls_id
        except Exception as e:
            print(f"[ERROR] YOLO 检测失败: {str(e)}")
            return frame, [], [], []
    
    def colour_label(self, label):
        """生成颜色标签"""
        label_colour = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(label_colour)
    
    def draw_bbox(self, frame, output, conf, cls_id):
        """绘制边界框"""
        try:
            x1, y1, x2, y2 = map(int, output[0:4])
            track_id = int(output[4])
            label = class_name.get(cls_id, str(cls_id))
            
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            colour = self.colour_label(track_id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c_id = f'{label} {track_id}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), colour, -1)
            cv2.putText(frame, c_id, (x1, y1 + t_size[1] + 4), 
                       cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
        except Exception as e:
            print(f"[ERROR] 绘制边界框失败: {str(e)}")
        
        return frame
    
    def process_frame(self, frame):
        """处理单帧图像"""
        frame, bbox_xyxy, conf_score, cls_id = self.yolo_details(frame)
        
        if len(bbox_xyxy) > 0 and self.deepsort is not None:
            try:
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)
                
                if len(outputs) > 0:
                    min_len = min(len(outputs), len(conf_score), len(cls_id))
                    for i in range(min_len):
                        frame = self.draw_bbox(frame, outputs[i], conf_score[i], cls_id[i])
                    
                    # 碰撞预警
                    collision_pairs = check_vehicle_collision(outputs, img_w)
                    frame = draw_collision_warning(frame, collision_pairs)
            except Exception as e:
                print(f"[ERROR] DeepSort 更新失败: {str(e)}")
        
        return frame
    
    def run_video_mode(self, video_path):
        """视频文件模式"""
        if not os.path.exists(video_path):
            print(f"[ERROR] 视频文件不存在: {video_path}")
            print("[INFO] 可用视频:")
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                videos = glob.glob(ext)
                for v in videos:
                    print(f"  - {v}")
            return
        
        print(f"[INFO] 视频模式: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频文件: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] 视频信息: {frame_width}x{frame_height} @ {fps}fps")
        
        video_writer = None
        if self.args.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = self.process_frame(frame)
                
                if not self.args.no_display:
                    cv2.imshow('Vehicle Tracking', frame)
                
                if video_writer:
                    video_writer.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[INFO] 已处理 {frame_count} 帧")
                
                if cv2.waitKey(1) == ord('q'):
                    break
        except KeyboardInterrupt:
            print("[INFO] 用户中断")
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            print(f"[INFO] 视频模式完成，共处理 {frame_count} 帧")
    
    def run_carla_mode(self):
        """CARLA 模拟器模式"""
        if not CARLA_AVAILABLE:
            print("[ERROR] CARLA 未安装，无法使用 CARLA 模式")
            print("[INFO] 请安装: pip install carla==0.9.13")
            return
        
        if not ULTRALYTICS_AVAILABLE:
            print("[ERROR] Ultralytics 未安装，无法进行目标检测")
            return
        
        print("[INFO] CARLA 模式: 尝试连接到 localhost:2000")
        
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)
            
            try:
                world = client.get_world()
                print("[INFO] ✓ 成功连接到 CARLA 模拟器")
            except Exception as e:
                print(f"[ERROR] 连接 CARLA 失败: {str(e)}")
                print("[INFO] 请确保 CARLA 模拟器已启动")
                print("[INFO] 或使用 --video 参数运行视频模式")
                return
            
            # 获取地图和生成点
            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                print("[ERROR] 无法获取生成点")
                return
            
            print(f"[INFO] 找到 {len(spawn_points)} 个生成点")
            
            # 生成主车辆
            vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
            vehicle_bp.set_attribute('role_name', 'ego')
            ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
            if not ego_vehicle:
                print("[ERROR] 无法生成主车辆")
                return
            
            print("[INFO] ✓ 主车辆生成成功")
            
            # 设置相机
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(img_w))
            camera_bp.set_attribute('image_size_y', str(img_h))
            camera_bp.set_attribute('fov', '110')
            
            camera_location = carla.Location(2, 0, 1)
            camera_rotation = carla.Rotation(0, 180, 0)
            camera_init_trans = carla.Transform(camera_location, camera_rotation)
            
            # CARLA 0.9.13 附件类型：Rigid, SpringArm
            camera = world.spawn_actor(
                camera_bp, 
                camera_init_trans, 
                attach_to=ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid
            )
            
            print("[INFO] ✓ 相机生成成功")
            
            # 生成 NPC 车辆
            npc_count = 0
            for i in range(20):
                vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
                npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
                if npc:
                    npc.set_autopilot(True)
                    npc_count += 1
            
            print(f"[INFO] ✓ 生成了 {npc_count} 个 NPC 车辆")
            
            # 图像数据
            camera_data = {'image': np.zeros((img_h, img_w, 3), dtype=np.uint8)}
            
            def capture_image(image):
                try:
                    image_data_array = np.array(image.raw_data)
                    image_rgb = image_data_array.reshape((image.height, image.width, 4))[:, :, :3]
                    camera_data['image'] = image_rgb
                except Exception as e:
                    print(f"[ERROR] 图像捕获失败: {str(e)}")
            
            camera.listen(capture_image)
            ego_vehicle.set_autopilot(True)
            
            # 视频写入器
            video_writer = None
            if self.args.save_output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, 14.0, (img_w, img_h))
            
            print("[INFO] ✓ 开始跟踪... 按 'q' 退出")
            
            frame_count = 0
            try:
                while True:
                    frame = camera_data['image'].copy()
                    frame = self.process_frame(frame)
                    
                    if not self.args.no_display:
                        cv2.imshow('CARLA Tracking', frame)
                    
                    if video_writer:
                        video_writer.write(frame)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[INFO] 已处理 {frame_count} 帧")
                    
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
            except KeyboardInterrupt:
                print("[INFO] 用户中断")
            finally:
                print("[INFO] 清理资源...")
                camera.stop()
                camera.destroy()
                ego_vehicle.destroy()
                
                for npc in world.get_actors().filter('vehicle*'):
                    try:
                        npc.destroy()
                    except:
                        pass
                
                if video_writer:
                    video_writer.release()
                
                cv2.destroyAllWindows()
                print("[INFO] ✓ CARLA 模式结束")
        
        except Exception as e:
            print(f"[ERROR] CARLA 运行失败: {str(e)}")
            traceback.print_exc()
    
    def run(self):
        """运行跟踪"""
        if self.args.video:
            self.run_video_mode(self.args.video)
        else:
            self.run_carla_mode()


# ==================== 主函数 ====================
def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO + DeepSort 多车跟踪系统\n'
                   '兼容 Python 3.8.10 + CARLA 0.9.13',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, default=None,
                       help='视频文件路径（跳过 CARLA）')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示画面')
    parser.add_argument('--save-output', action='store_true',
                       help='保存输出视频')
    
    return parser.parse_args()


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("YOLO + DeepSort 多车跟踪系统")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"平台: {sys.platform}")
    print(f"工作目录: {os.getcwd()}")
    print("=" * 60)


if __name__ == '__main__':
    check_environment()
    
    args = parse_args()
    
    try:
        tracker = VehicleTracker(args)
        tracker.run()
    except KeyboardInterrupt:
        print("\n[INFO] 程序被用户中断")
    except Exception as e:
        print(f"\n[FATAL ERROR] 程序异常终止: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
