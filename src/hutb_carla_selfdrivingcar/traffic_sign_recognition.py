import carla
import math
import cv2
import numpy as np

class TrafficSignRecognition:
    """交通标志识别系统（TSR）"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.blueprint_lib = world.get_blueprint_library()
        
        self.detected_sign = None
        self.detected_speed = None
        self.sign_distance = float('inf')
        
        self.setup_camera()
    
    def setup_camera(self):
        """设置前置摄像头"""
        camera_bp = self.blueprint_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        
        self.camera.listen(self.process_image)
    
    def process_image(self, image):
        """处理摄像头图像"""
        # 将CARLA图像转换为OpenCV格式
        img = np.array(image.raw_data).reshape((480, 640, 4))[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 模拟交通标志识别（简化版）
        self.detect_signs()
    
    def detect_signs(self):
        """检测交通标志（使用CARLA API）"""
        self.detected_sign = None
        self.detected_speed = None
        self.sign_distance = float('inf')
        
        # 获取车辆位置
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        
        # 检测交通标志
        for actor in self.world.get_actors().filter('traffic.sign.*'):
            sign_location = actor.get_location()
            dx = sign_location.x - vehicle_location.x
            dy = sign_location.y - vehicle_location.y
            
            distance = math.sqrt(dx**2 + dy**2)
            dot_product = dx * forward_vector.x + dy * forward_vector.y
            
            # 只检测前方50米内的标志
            if dot_product > 0 and distance < 50:
                if distance < self.sign_distance:
                    self.sign_distance = distance
                    sign_type = self.identify_sign(actor)
                    if sign_type:
                        self.detected_sign = sign_type['name']
                        self.detected_speed = sign_type.get('speed', None)
    
    def identify_sign(self, actor):
        """识别交通标志类型"""
        sign_id = actor.type_id
        
        # 限速标志
        if 'speed_limit.30' in sign_id:
            return {'name': '限速30', 'speed': 30}
        elif 'speed_limit.40' in sign_id:
            return {'name': '限速40', 'speed': 40}
        elif 'speed_limit.50' in sign_id:
            return {'name': '限速50', 'speed': 50}
        elif 'speed_limit.60' in sign_id:
            return {'name': '限速60', 'speed': 60}
        elif 'speed_limit.80' in sign_id:
            return {'name': '限速80', 'speed': 80}
        elif 'stop' in sign_id.lower():
            return {'name': '停车标志', 'speed': 0}
        elif 'yield' in sign_id.lower():
            return {'name': '让行标志', 'speed': 15}
        
        return None
    
    def get_target_speed(self, default_speed=50):
        """获取目标速度"""
        if self.detected_speed is not None:
            return self.detected_speed
        return default_speed
    
    def get_sign_info(self):
        """获取识别到的标志信息"""
        return {
            'sign': self.detected_sign,
            'speed': self.detected_speed,
            'distance': self.sign_distance
        }
    
    def destroy(self):
        """销毁传感器"""
        if self.camera:
            self.camera.destroy()
