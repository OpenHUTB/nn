# vision_module.py
import carla
import cv2
import numpy as np
import queue
import time  
from ultralytics import YOLO

class VisionSystem:
    def __init__(self, ego_vehicle, world, fov='90', res_x='640', res_y='480'):
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.camera_sensor = None
        self.image_queue = queue.Queue()
        
        print(f"⏳ [视觉模块] 正在加载 YOLOv8 模型...")
        self.yolo_model = YOLO("yolov8n.pt") 
        print(f"✅ [视觉模块] YOLOv8 模型加载完毕。")
        
        self.last_seen_classes = set()
        self.last_alert_time = {
            "person": 0.0,
            "car": 0.0
        }
        
        self.focal_length = 320.0 
        
        self._setup_camera(fov, res_x, res_y)

    def _setup_camera(self, fov, res_x, res_y):
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', res_x) 
        camera_bp.set_attribute('image_size_y', res_y)
        camera_bp.set_attribute('fov', fov)
        
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.try_spawn_actor(camera_bp, cam_transform, attach_to=self.ego_vehicle)
        
        if self.camera_sensor:
            self.camera_sensor.listen(self._camera_callback)
            print("✅ [视觉模块] RGB 摄像头已挂载。")

    def _camera_callback(self, image):
        self.image_queue.put(image)

    def process_and_render(self):
        """处理图像，并返回：(图像帧, 当前正前方的最短障碍物距离)"""
        if not self.image_queue.empty():
            image = self.image_queue.get()
            
            img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (image.height, image.width, 4))
            img_bgr = img_array[:, :, :3]
            
            results = self.yolo_model(img_bgr, verbose=False)
            current_seen_classes = set()
            min_distance = float('inf') 
            
            roi_left = 200
            roi_right = 440
            
            # 🌟 新增：设置雷达的“最远有效预警距离” (单位：米)
            # 你可以随时调整这个值，40米对于城市道路巡航是一个很舒服的预警距离
            radar_max_range = 40.0
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_width = x2 - x1
                box_height = y2 - y1
                box_center_x = (x1 + x2) / 2
                
                if roi_left < box_center_x < roi_right:
                    if cls_name in ["car", "person"]:
                        real_height = 1.7 if cls_name == "person" else 1.5
                        distance = (self.focal_length * real_height) / max(1.0, box_height)
                        
                        # (AEB 会用到这个最小距离，哪怕在40米外也要持续算)
                        if distance < min_distance:
                            min_distance = distance
                            
                        # 🌟 核心拦截逻辑：只有距离小于 40 米时，才将其计入“雷达监控名单”
                        if distance <= radar_max_range:
                            current_seen_classes.add(cls_name)
            
            newly_appeared = current_seen_classes - self.last_seen_classes
            current_time = time.time()
            
            # 配合距离过滤，更新了控制台的文案
            if "person" in newly_appeared and current_time - self.last_alert_time.get("person", 0) > 3.0:
                print(f"\033[93m[视觉雷达] ⚠️ 正前方 {int(radar_max_range)} 米内发现行人。\033[0m")
                self.last_alert_time["person"] = current_time 
                    
            if "car" in newly_appeared and current_time - self.last_alert_time.get("car", 0) > 3.0:
                print(f"\033[96m[视觉雷达] ⚠️ 正前方 {int(radar_max_range)} 米内发现车辆。\033[0m")
                self.last_alert_time["car"] = current_time
            
            self.last_seen_classes = current_seen_classes
            
            annotated_frame = results[0].plot()
            cv2.line(annotated_frame, (roi_left, 0), (roi_left, 480), (0, 255, 0), 2)
            cv2.line(annotated_frame, (roi_right, 0), (roi_right, 480), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Ego Lane ROI", (roi_left + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("CARLA YOLOv8 Vision", annotated_frame)
            cv2.waitKey(1)
            
            return annotated_frame, min_distance
            
        return None, float('inf')

    def destroy(self):
        if self.camera_sensor:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
        cv2.destroyAllWindows() 
        cv2.waitKey(1)
        print("🧹 [视觉模块] 摄像头已卸载，窗口已关闭。")