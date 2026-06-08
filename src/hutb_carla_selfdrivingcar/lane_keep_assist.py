import carla
import numpy as np
import cv2
import queue
import math

class LaneKeepAssist:
    def __init__(self, vehicle, world, bp_lib):
        self.vehicle = vehicle
        self.world = world
        self.bp_lib = bp_lib
        self.queue = queue.Queue()
        self.steering_pid = PIDController(kp=0.8, ki=0.0, kd=0.3)
        self.lane_width = 3.5  # 标准车道宽度（米）
        self.enabled = True
        
    def setup_camera(self):
        """设置前置摄像头"""
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        
        transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.process_image(image))
        print("[LKA] 摄像头已设置")
        
    def process_image(self, image):
        """处理摄像头图像，检测车道线"""
        try:
            # 转换为numpy数组
            img = np.array(image.raw_data)
            img = img.reshape((image.height, image.width, 4))
            img = img[:, :, :3]  # 只保留RGB
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 感兴趣区域（只检测前方道路）
            height, width = edges.shape
            mask = np.zeros_like(edges)
            polygon = np.array([[
                [0, height],
                [width, height],
                [width, height//2],
                [0, height//2]
            ]], dtype=np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # 霍夫变换检测直线
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, 
                                   minLineLength=30, maxLineGap=100)
            
            if lines is not None:
                left_lines, right_lines = self.separate_lines(lines)
                
                # 计算车道线
                if left_lines and right_lines:
                    left_info = self.average_lines(left_lines)
                    right_info = self.average_lines(right_lines)
                    lane_offset, angle = self.calculate_lane_position(left_info, right_info)
                    self.queue.put({'offset': lane_offset, 'angle': angle, 'detected': True})
                else:
                    self.queue.put({'detected': False})
            else:
                self.queue.put({'detected': False})
                
        except Exception as e:
            self.queue.put({'detected': False})
            
    def separate_lines(self, lines):
        """分离左右车道线"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            if slope < -0.3:  # 左车道线（斜率为负）
                left_lines.append(line[0])
            elif slope > 0.3:  # 右车道线（斜率为正）
                right_lines.append(line[0])
                
        return left_lines, right_lines
    
    def average_lines(self, lines):
        """计算平均车道线"""
        if not lines:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        return np.polyfit(x_coords, y_coords, 1)
    
    def calculate_lane_position(self, left_info, right_info):
        """计算车道位置偏移"""
        if left_info is None or right_info is None:
            return 0.0, 0.0
            
        # 计算左右车道线的x坐标（底部）
        width = 640
        height = 480
        left_x_bottom = (height - left_info[1]) / left_info[0]
        right_x_bottom = (height - right_info[1]) / right_info[0]
        
        # 车道中心
        lane_center = (left_x_bottom + right_x_bottom) / 2
        
        # 计算车辆中心（图像中心）
        vehicle_center = width / 2
        
        # 计算偏移
        lane_offset = (lane_center - vehicle_center) / (right_x_bottom - left_x_bottom + 1e-6)
        
        # 计算角度
        left_slope = left_info[0]
        right_slope = right_info[0]
        avg_slope = (left_slope + right_slope) / 2
        angle = math.atan(avg_slope) * 180 / math.pi
        
        return lane_offset * self.lane_width, angle
    
    def get_steering(self):
        """获取转向控制"""
        if not self.enabled:
            return 0.0
            
        try:
            data = self.queue.get_nowait()
            if data.get('detected', False):
                offset = data['offset']
                angle = data['angle']
                
                # PID控制
                steer = self.steering_pid.compute(offset)
                
                # 限制转向角度
                steer = max(-1.0, min(1.0, steer))
                
                return steer
        except queue.Empty:
            pass
            
        return 0.0  # 无检测结果时保持当前方向
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'camera'):
            self.camera.destroy()
            print("[LKA] 摄像头已销毁")


class PIDController:
    """简单的PID控制器"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0.0
        self.integral = 0.0
        
    def compute(self, error):
        """计算PID输出"""
        self.integral += error
        derivative = error - self.last_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output