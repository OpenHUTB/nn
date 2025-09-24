import cv2
import numpy as np
import time

class ObstacleDetector:
    def __init__(self, camera_index=0, min_area=500, max_area=10000):
        """
        初始化障碍检测器
        :param camera_index: 摄像头索引，默认0为内置摄像头
        :param min_area: 最小障碍物面积，过滤小噪声
        :param max_area: 最大障碍物面积，过滤过大区域
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.min_area = min_area
        self.max_area = max_area
        
        # 定义颜色范围（这里使用HSV颜色空间，可根据实际场景调整）
        self.lower_color = np.array([0, 0, 0])
        self.upper_color = np.array([180, 255, 80])
        
        # 获取图像中心，用于判断障碍物位置
        self.frame_center_x = 640 // 2
        self.frame_center_y = 480 // 2

    def preprocess_image(self, frame):
        """预处理图像，为检测做准备"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建掩码，提取指定颜色范围
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # 去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 边缘检测
        edges = cv2.Canny(mask, 50, 150)
        
        return mask, edges

    def detect_obstacles(self, frame):
        """检测图像中的障碍物"""
        mask, edges = self.preprocess_image(frame)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤面积过小或过大的轮廓
            if self.min_area < area < self.max_area:
                # 计算最小外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算障碍物中心坐标
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 计算障碍物相对于图像中心的位置
                relative_pos = "center"
                if center_x < self.frame_center_x - 50:
                    relative_pos = "left"
                elif center_x > self.frame_center_x + 50:
                    relative_pos = "right"
                
                # 估算距离（简单的大小估算，实际应用需要校准）
                distance = self.estimate_distance(w, h)
                
                obstacles.append({
                    "position": (x, y, w, h),
                    "center": (center_x, center_y),
                    "area": area,
                    "relative_position": relative_pos,
                    "distance": distance
                })
                
                # 在图像上绘制边界框和信息
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"Dist: {distance:.1f}m", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, obstacles

    def estimate_distance(self, width, height):
        """简单估算障碍物距离（需要根据实际情况校准）"""
        # 这里使用简单的反比例关系，实际应用中应使用相机校准数据
        avg_size = (width + height) / 2
        distance = 2000 / avg_size  # 系数需要根据实际情况调整
        return min(max(distance, 0.5), 5.0)  # 限制距离范围在0.5-5米

    def run(self):
        """运行障碍检测主循环"""
        print("开始障碍检测... 按 'q' 退出")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取图像")
                    break
                
                # 检测障碍物
                result_frame, obstacles = self.detect_obstacles(frame)
                
                # 显示结果
                cv2.imshow("Obstacle Detection", result_frame)
                
                # 打印障碍物信息
                if obstacles:
                    print(f"检测到 {len(obstacles)} 个障碍物:")
                    for i, obs in enumerate(obstacles):
                        print(f"  障碍物 {i+1}: 位置={obs['relative_position']}, 距离={obs['distance']:.1f}m")
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)  # 稍微延迟，降低CPU占用
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("检测结束")

if __name__ == "__main__":
    try:
        # 创建并运行障碍检测器
        detector = ObstacleDetector(camera_index=0)
        detector.run()
    except Exception as e:
        print(f"发生错误: {str(e)}")
