import cv2
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class MonoCamera:
    """单目相机类"""
    def __init__(self, focal_length, principal_point, image_size, height, pitch):
        """
        初始化单目相机
        
        参数:
            focal_length: 焦距 [fx, fy]
            principal_point: 主点 [cx, cy]
            image_size: 图像尺寸 [高度, 宽度]
            height: 相机安装高度 (米)
            pitch: 俯仰角度 (度)
        """
        self.focal_length = np.array(focal_length)
        self.principal_point = np.array(principal_point)
        self.image_size = np.array(image_size)
        self.height = height
        self.pitch = np.deg2rad(pitch)  # 转换为弧度
        
        # 内参矩阵
        self.K = np.array([
            [focal_length[0], 0, principal_point[0]],
            [0, focal_length[1], principal_point[1]],
            [0, 0, 1]
        ])
        
        # 外参：旋转和平移矩阵
        # 相机相对于地面的旋转（仅俯仰）
        self.R = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch), -np.sin(self.pitch)],
            [0, np.sin(self.pitch), np.cos(self.pitch)]
        ])
        
        # 平移向量 (相机在地面坐标系中的位置)
        self.t = np.array([0, -height, 0])  # 相机在Y轴负方向（向下）
        
    def pixel_to_world(self, pixel_coords):
        """
        像素坐标转换到世界坐标（车辆坐标系）
        """
        # 像素坐标转为齐次坐标
        pixels_h = np.column_stack([pixel_coords, np.ones(len(pixel_coords))])
        
        # 反投影到相机坐标系
        camera_coords = np.linalg.inv(self.K) @ pixels_h.T
        
        # 旋转到世界坐标系
        world_coords = self.R @ camera_coords
        
        # 计算与地面的交点 (Y=0 平面)
        # 解方程: Y = Y0 + t * Vy = 0
        t = -self.t[1] / world_coords[1, :]
        
        # 计算3D坐标
        points_3d = self.t[:, np.newaxis] + world_coords * t
        
        return points_3d[:2, :].T  # 只返回X和Z坐标（车辆坐标系）

class BirdsEyeView:
    """鸟瞰图转换类"""
    def __init__(self, camera, out_view, output_size):
        """
        初始化鸟瞰图配置
        
        参数:
            camera: MonoCamera实例
            out_view: [x_min, x_max, y_min, y_max] 车辆坐标系中的区域
            output_size: [高度, 宽度] 输出图像大小
        """
        self.camera = camera
        self.out_view = out_view
        self.output_size = output_size
        
        # 计算透视变换矩阵
        self.compute_transform_matrix()
        
    def compute_transform_matrix(self):
        """计算透视变换矩阵"""
        # 定义地面上的四个点（车辆坐标系）
        x_min, x_max, y_min, y_max = self.out_view
        
        # 地面上的四个角点
        world_points = np.array([
            [x_min, y_min],  # 左下
            [x_max, y_min],  # 右下
            [x_max, y_max],  # 右上
            [x_min, y_max]   # 左上
        ], dtype=np.float32)
        
        # 将这些点投影到图像平面
        pixel_points = []
        for point in world_points:
            # 地面上的3D点 (Y=0)
            world_3d = np.array([point[0], 0, point[1]])
            
            # 转换到相机坐标系
            camera_coords = self.camera.R.T @ (world_3d - self.camera.t)
            
            # 投影到图像平面
            pixel_h = self.camera.K @ camera_coords
            pixel = pixel_h[:2] / pixel_h[2]
            pixel_points.append(pixel)
        
        pixel_points = np.array(pixel_points, dtype=np.float32)
        
        # 鸟瞰图的四个角点
        output_points = np.array([
            [0, self.output_size[0]],  # 左下
            [*self.output_size[::-1]],  # 右下 (w, h)
            [self.output_size[1], 0],  # 右上
            [0, 0]  # 左上
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        self.M = cv2.getPerspectiveTransform(pixel_points, output_points)
        self.M_inv = cv2.getPerspectiveTransform(output_points, pixel_points)
        
    def transform_image(self, image):
        """将图像转换为鸟瞰图"""
        return cv2.warpPerspective(image, self.M, 
                                  (self.output_size[1], self.output_size[0]))
    
    def image_to_vehicle(self, image_points):
        """将鸟瞰图像素坐标转换到车辆坐标系"""
        if len(image_points) == 0:
            return np.array([])
        
        # 转换到原始图像坐标
        image_points_h = np.column_stack([image_points, np.ones(len(image_points))])
        original_pixels = (self.M_inv @ image_points_h.T).T
        
        # 归一化
        original_pixels = original_pixels[:, :2] / original_pixels[:, 2:3]
        
        # 转换到车辆坐标系
        vehicle_points = self.camera.pixel_to_world(original_pixels)
        return vehicle_points

class VehicleDetector:
    """车辆检测器类"""
    def __init__(self):
        """初始化车辆检测器"""
        # 使用OpenCV的预训练车辆检测器
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 或者使用YOLO（如果需要更精确的检测）
        try:
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i[0] - 1] 
                                 for i in self.net.getUnconnectedOutLayers()]
            self.use_yolo = True
        except:
            print("YOLO模型未找到，使用HOG检测器")
            self.use_yolo = False
            
    def detect(self, image):
        """检测图像中的车辆"""
        if self.use_yolo:
            return self.detect_yolo(image)
        else:
            return self.detect_hog(image)
    
    def detect_hog(self, image):
        """使用HOG检测车辆"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects, weights = self.hog.detectMultiScale(gray, winStride=(4, 4),
                                                   padding=(8, 8), scale=1.05)

        # 使用列表推导式简化过滤和转换
        bboxes = [[x, y, x + w, y + h] for (x, y, w, h) in rects if w > 50 and h > 50]
        scores = [1.0] * len(bboxes)  # HOG不返回置信度分数
        return np.array(bboxes), np.array(scores)
    
    def detect_yolo(self, image):
        """使用YOLO检测车辆"""
        height, width = image.shape[:2]
        
        # 预处理图像
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), 
                                     (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        bboxes = []
        scores = []
        
        for output in outputs:
            for detection in output:
                scores_array = detection[5:]
                class_id = np.argmax(scores_array)
                confidence = scores_array[class_id]
                
                # 只检测车辆类（在COCO数据集中，汽车是2，卡车是7，公共汽车是5）
                if class_id in [2, 5, 7] and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    bboxes.append([x, y, x+w, y+h])
                    scores.append(float(confidence))
        
        return np.array(bboxes), np.array(scores)

class LaneDetector:
    """车道线检测器类"""
    def __init__(self, approx_lane_width=0.25, sensitivity=0.25):
        """
        初始化车道线检测器
        
        参数:
            approx_lane_width: 近似车道线宽度 (米)
            sensitivity: 检测灵敏度
        """
        self.approx_lane_width = approx_lane_width
        self.sensitivity = sensitivity
        
    def segment_lane_markers(self, birds_eye_image, birds_eye_config, vehicle_roi):
        """
        分割车道线标记
        
        参数:
            birds_eye_image: 鸟瞰图
            birds_eye_config: BirdsEyeView实例
            vehicle_roi: 感兴趣区域 [x_min, x_max, y_min, y_max]
        """
        # 转换为灰度图
        if len(birds_eye_image.shape) == 3:
            gray = cv2.cvtColor(birds_eye_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = birds_eye_image.copy()
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 创建ROI掩码
        mask = np.zeros_like(edges)
        x_min, x_max, y_min, y_max = vehicle_roi
        
        # 转换ROI坐标到图像像素坐标
        height, width = edges.shape
        
        # 注意：鸟瞰图中x对应图像宽度方向，y对应图像高度方向
        roi_pixels = [
            (int((y_min - birds_eye_config.out_view[2]) / 
                 (birds_eye_config.out_view[3] - birds_eye_config.out_view[2]) * width),
             int((x_max - birds_eye_config.out_view[0]) / 
                 (birds_eye_config.out_view[1] - birds_eye_config.out_view[0]) * height)),
            (int((y_max - birds_eye_config.out_view[2]) / 
                 (birds_eye_config.out_view[3] - birds_eye_config.out_view[2]) * width),
             int((x_max - birds_eye_config.out_view[0]) / 
                 (birds_eye_config.out_view[1] - birds_eye_config.out_view[0]) * height)),
            (int((y_max - birds_eye_config.out_view[2]) / 
                 (birds_eye_config.out_view[3] - birds_eye_config.out_view[2]) * width),
             int((x_min - birds_eye_config.out_view[0]) / 
                 (birds_eye_config.out_view[1] - birds_eye_config.out_view[0]) * height)),
            (int((y_min - birds_eye_config.out_view[2]) / 
                 (birds_eye_config.out_view[3] - birds_eye_config.out_view[2]) * width),
             int((x_min - birds_eye_config.out_view[0]) / 
                 (birds_eye_config.out_view[1] - birds_eye_config.out_view[0]) * height))
        ]
        
        # 创建多边形掩码
        cv2.fillPoly(mask, [np.array(roi_pixels, dtype=np.int32)], 255)
        
        # 应用ROI掩码
        masked_edges = cv2.bitwise_and(edges, mask)
        
        return masked_edges
    
    def find_lane_boundaries(self, binary_image, birds_eye_config, max_lanes=2):
        """
        寻找车道边界
        
        参数:
            binary_image: 二值化图像
            birds_eye_config: BirdsEyeView实例
            max_lanes: 最大车道线数量
        """
        # 寻找非零像素
        y_indices, x_indices = np.where(binary_image > 0)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            return []
        
        # 转换到车辆坐标系
        image_points = np.column_stack([x_indices, y_indices])
        vehicle_points = birds_eye_config.image_to_vehicle(image_points)
        
        if len(vehicle_points) == 0:
            return []
        
        # 使用DBSCAN聚类车道点
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(vehicle_points)
        labels = clustering.labels_
        
        boundaries = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
                
            cluster_points = vehicle_points[labels == label]
            
            if len(cluster_points) < 50:  # 过滤小的聚类
                continue
            
            # 拟合二次曲线: y = ax² + bx + c (在车辆坐标系中)
            try:
                # 按x坐标排序
                sorted_idx = np.argsort(cluster_points[:, 0])
                x_sorted = cluster_points[sorted_idx, 0]
                y_sorted = cluster_points[sorted_idx, 1]
                
                # 拟合多项式
                coeffs = np.polyfit(x_sorted, y_sorted, 2)
                
                # 创建边界对象
                boundary = {
                    'coeffs': coeffs,
                    'x_range': [np.min(x_sorted), np.max(x_sorted)],
                    'points': cluster_points
                }
                
                boundaries.append(boundary)
                
                if len(boundaries) >= max_lanes:
                    break
                    
            except:
                continue
        
        return boundaries
    
    def classify_lane_types(self, boundaries, boundary_points):
        """分类车道线类型"""
        if not boundaries:
            return boundaries
        
        # 简单的分类：根据y坐标的平均值判断是左车道线还是右车道线
        for boundary in boundaries:
            mean_y = np.mean(boundary['points'][:, 1])
            if mean_y > 0:
                boundary['type'] = 'right'
            else:
                boundary['type'] = 'left'
        
        return boundaries

def compute_vehicle_locations(bboxes, camera):
    """
    计算车辆位置
    
    参数:
        bboxes: 边界框 [[x1, y1, x2, y2], ...]
        camera: MonoCamera实例
    """
    if len(bboxes) == 0:
        return np.array([])

    if len(bboxes) == 0:
        return np.array([])

    # 批量计算边界框底边中心点
    centers = np.column_stack([(bboxes[:, 0] + bboxes[:, 2]) / 2, bboxes[:, 3]])
    vehicle_points = camera.pixel_to_world(centers)
    return vehicle_points

def visualize_results(frame, camera, sensor_out, int_out=None):
    """
    可视化结果
    
    参数:
        frame: 原始帧
        camera: MonoCamera实例
        sensor_out: 传感器输出
        int_out: 中间结果
    """
    # 创建显示图像
    display_image = frame.copy()
    
    # 绘制车道线
    if 'leftEgoBoundary' in sensor_out and sensor_out['leftEgoBoundary']:
        left_boundary = sensor_out['leftEgoBoundary']
        # 在图像上绘制左车道线
        for i in range(len(left_boundary['x_range']) - 1):
            x1, x2 = left_boundary['x_range'][i], left_boundary['x_range'][i+1]
            y1 = np.polyval(left_boundary['coeffs'], x1)
            y2 = np.polyval(left_boundary['coeffs'], x2)
            
            # 转换回像素坐标
            point1_3d = np.array([x1, 0, y1])
            camera_coords1 = camera.R.T @ (point1_3d - camera.t)
            pixel1_h = camera.K @ camera_coords1
            pixel1 = (pixel1_h[:2] / pixel1_h[2]).astype(int)
            
            point2_3d = np.array([x2, 0, y2])
            camera_coords2 = camera.R.T @ (point2_3d - camera.t)
            pixel2_h = camera.K @ camera_coords2
            pixel2 = (pixel2_h[:2] / pixel2_h[2]).astype(int)
            
            cv2.line(display_image, tuple(pixel1), tuple(pixel2), (0, 255, 0), 2)
    
    if 'rightEgoBoundary' in sensor_out and sensor_out['rightEgoBoundary']:
        right_boundary = sensor_out['rightEgoBoundary']
        # 在图像上绘制右车道线
        for i in range(len(right_boundary['x_range']) - 1):
            x1, x2 = right_boundary['x_range'][i], right_boundary['x_range'][i+1]
            y1 = np.polyval(right_boundary['coeffs'], x1)
            y2 = np.polyval(right_boundary['coeffs'], x2)
            
            # 转换回像素坐标
            point1_3d = np.array([x1, 0, y1])
            camera_coords1 = camera.R.T @ (point1_3d - camera.t)
            pixel1_h = camera.K @ camera_coords1
            pixel1 = (pixel1_h[:2] / pixel1_h[2]).astype(int)
            
            point2_3d = np.array([x2, 0, y2])
            camera_coords2 = camera.R.T @ (point2_3d - camera.t)
            pixel2_h = camera.K @ camera_coords2
            pixel2 = (pixel2_h[:2] / pixel2_h[2]).astype(int)
            
            cv2.line(display_image, tuple(pixel1), tuple(pixel2), (0, 0, 255), 2)
    
    # 绘制车辆检测框
    if 'vehicleBoxes' in sensor_out:
        for bbox in sensor_out['vehicleBoxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    # 显示车辆位置
    if 'vehicleLocations' in sensor_out and len(sensor_out['vehicleLocations']) > 0:
        for location in sensor_out['vehicleLocations']:
            cv2.putText(display_image, f"({location[0]:.1f}, {location[1]:.1f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display_image

def main():
    """主函数"""
    # 相机参数
    focal_length = [309.4362, 344.2161]
    principal_point = [318.9034, 257.5352]
    image_size = [360, 640]
    height = 2.1798
    pitch = 14
    
    # 创建相机
    camera = MonoCamera(focal_length, principal_point, image_size, height, pitch)
    
    # 创建鸟瞰图配置
    dist_ahead = 13
    space_to_side = 6
    bottom_offset = 3
    out_view = [bottom_offset, dist_ahead, -space_to_side, space_to_side]
    birds_eye_size = [250, 500]  # [高度, 宽度]
    
    birds_eye_config = BirdsEyeView(camera, out_view, birds_eye_size)
    
    # 创建检测器
    vehicle_detector = VehicleDetector()
    lane_detector = LaneDetector()
    
    # 选择视频文件
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi")]
    )
    
    if not file_path:
        print("用户取消选择")
        return
    
    print(f"用户选择: {file_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(file_path)
    
    # 设置起始时间
    start_time = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频")
        return
    
    # 主循环
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为BGR到RGB（如果需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 生成鸟瞰图
        birds_eye_image = birds_eye_config.transform_image(frame)
        
        # 检测车辆
        bboxes, scores = vehicle_detector.detect(frame)
        locations = compute_vehicle_locations(bboxes, camera)
        
        # 车道线检测
        vehicle_roi = [bottom_offset - 1, dist_ahead + 2, 
                      -space_to_side - 3, space_to_side + 3]
        birds_eye_bw = lane_detector.segment_lane_markers(
            birds_eye_image, birds_eye_config, vehicle_roi
        )
        
        # 寻找车道边界
        boundaries = lane_detector.find_lane_boundaries(
            birds_eye_bw, birds_eye_config, max_lanes=2
        )
        
        # 分类车道类型
        boundaries = lane_detector.classify_lane_types(boundaries, None)
        
        # 寻找自我车道
        left_ego = None
        right_ego = None
        
        if boundaries:
            # 计算在x=0处的y值
            distances = []
            for boundary in boundaries:
                y_at_zero = np.polyval(boundary['coeffs'], 0)
                distances.append(y_at_zero)
            
            # 寻找最近的左车道和右车道
            left_distances = [d for d in distances if d > 0]
            right_distances = [d for d in distances if d < 0]
            
            if left_distances:
                min_left = min(left_distances)
                left_ego = boundaries[distances.index(min_left)]
            
            if right_distances:
                max_right = max(right_distances)
                right_ego = boundaries[distances.index(max_right)]
        
        # 准备传感器输出
        sensor_out = {
            'leftEgoBoundary': left_ego,
            'rightEgoBoundary': right_ego,
            'vehicleLocations': locations,
            'xVehiclePoints': list(range(bottom_offset, dist_ahead)),
            'vehicleBoxes': bboxes
        }
        
        # 准备中间输出（用于调试）
        int_out = {
            'birdsEyeImage': birds_eye_image,
            'birdsEyeConfig': birds_eye_config,
            'vehicleScores': scores,
            'vehicleROI': vehicle_roi,
            'birdsEyeBW': birds_eye_bw
        }
        
        # 可视化结果
        result_frame = visualize_results(frame, camera, sensor_out, int_out)
        
        # 显示结果
        cv2.imshow('Lane and Vehicle Detection', result_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 显示中间结果（按's'键）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Original Frame')
            
            plt.subplot(132)
            plt.imshow(cv2.cvtColor(birds_eye_image, cv2.COLOR_BGR2RGB))
            plt.title('Birds Eye View')
            
            plt.subplot(133)
            plt.imshow(birds_eye_bw, cmap='gray')
            plt.title('Lane Detection')
            
            plt.tight_layout()
            plt.show()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
