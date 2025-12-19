import cv2
import numpy as np
import time
import math
import open3d as o3d  # LiDAR点云处理
from threading import Thread
import serial  # 串口读取轮速/IMU（适配Arduino/车载MCU）

# 适配YOLOv5（需提前下载YOLOv5权重文件）
from ultralytics import YOLO


class AutonomousCarPerception:
    def __init__(self):
        # 1. 初始化硬件接口
        self.lidar = None  # LiDAR对象（示例用Open3D模拟，实际替换为Velodyne/Ouster SDK）
        self.camera = cv2.VideoCapture(0)  # 车载摄像头（0为默认摄像头，可替换为视频流/CSI摄像头）
        self.serial_port = serial.Serial("/dev/ttyUSB0", 9600, timeout=0.1)  # 串口读取IMU/轮速
        self.yolo_model = YOLO("yolov5s.pt")  # 轻量化YOLOv5模型

        # 2. 感知数据缓存
        self.perception_data = {
            # 自身状态
            "speed": 0.0,  # 车速 (m/s)
            "heading": 0.0,  # 航向角 (°，正北为0，顺时针)
            "position": {"lat": 0.0, "lon": 0.0},  # GPS位置
            "imu": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},  # IMU姿态
            # 视觉感知
            "lane": {"detected": False, "center_offset": 0.0},  # 车道线偏移 (m)
            "visual_obstacle": [],  # 视觉检测障碍物 [{"class": "car", "distance": 5.0, "x": 1.2}]
            # LiDAR感知
            "lidar_obstacle": [],  # LiDAR检测障碍物 [{"distance": 4.8, "width": 1.5, "angle": 5.0}]
            # 融合结果
            "fused_obstacle": [],  # 融合后障碍物
            "timestamp": time.time()
        }

        # 3. 线程控制
        self.running = True
        self.vision_thread = Thread(target=self._vision_loop)
        self.lidar_thread = Thread(target=self._lidar_loop)
        self.state_thread = Thread(target=self._state_loop)
        self.fusion_thread = Thread(target=self._fusion_loop)

    # ------------------- 1. 自身状态感知 -------------------
    def _read_state_sensors(self):
        """读取轮速计、IMU、GPS数据（串口/ROS话题）"""
        try:
            # 示例：从串口读取MCU发送的状态数据（格式：speed,heading,roll,pitch,yaw,lat,lon）
            if self.serial_port.in_waiting > 0:
                data = self.serial_port.readline().decode().strip().split(",")
                if len(data) == 7:
                    self.perception_data["speed"] = float(data[0])
                    self.perception_data["heading"] = float(data[1])
                    self.perception_data["imu"]["roll"] = float(data[2])
                    self.perception_data["imu"]["pitch"] = float(data[3])
                    self.perception_data["imu"]["yaw"] = float(data[4])
                    self.perception_data["position"]["lat"] = float(data[5])
                    self.perception_data["position"]["lon"] = float(data[6])
        except Exception as e:
            print(f"状态传感器读取失败: {e}")

    def _state_loop(self):
        """状态感知循环（20Hz）"""
        while self.running:
            self._read_state_sensors()
            self.perception_data["timestamp"] = time.time()
            time.sleep(0.05)

    # ------------------- 2. 视觉感知 -------------------
    def _detect_lane(self, frame):
        """车道线检测（霍夫变换）"""
        # 预处理：灰度化 → 高斯模糊 → 边缘检测 → 感兴趣区域(ROI)裁剪
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 裁剪ROI（只检测画面下半部分的车道线）
        roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        # 霍夫直线检测
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
        lane_detected = False
        center_offset = 0.0  # 车道中心偏移（单位：米，正值偏右，负值偏左）

        if lines is not None:
            lane_detected = True
            # 分离左右车道线
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # 避免除零
                if slope < -0.5:  # 左车道线（负斜率）
                    left_lines.append(line)
                elif slope > 0.5:  # 右车道线（正斜率）
                    right_lines.append(line)

            # 拟合左右车道线并计算中心偏移
            if left_lines and right_lines:
                # 左车道线拟合
                left_x = [line[0][0] for line in left_lines] + [line[0][2] for line in left_lines]
                left_y = [line[0][1] for line in left_lines] + [line[0][3] for line in left_lines]
                left_fit = np.polyfit(left_y, left_x, 1)
                left_x_base = int(left_fit[0] * height + left_fit[1])

                # 右车道线拟合
                right_x = [line[0][0] for line in right_lines] + [line[0][2] for line in right_lines]
                right_y = [line[0][1] for line in right_lines] + [line[0][3] for line in right_lines]
                right_fit = np.polyfit(right_y, right_x, 1)
                right_x_base = int(right_fit[0] * height + right_fit[1])

                # 计算车道中心与画面中心的偏移（像素→米，需标定）
                lane_center = (left_x_base + right_x_base) / 2
                frame_center = width / 2
                pixel2meter = 0.01  # 1像素=0.01米（需实际标定）
                center_offset = (lane_center - frame_center) * pixel2meter

                # 绘制车道线
                cv2.line(frame, (left_x_base, height), (int(left_fit[0] * height / 2 + left_fit[1]), int(height / 2)),
                         (0, 255, 0), 2)
                cv2.line(frame, (right_x_base, height),
                         (int(right_fit[0] * height / 2 + right_fit[1]), int(height / 2)), (0, 255, 0), 2)
                cv2.line(frame, (int(lane_center), height), (int(frame_center), int(height / 2)), (0, 0, 255), 2)

        # 更新车道线数据
        self.perception_data["lane"]["detected"] = lane_detected
        self.perception_data["lane"]["center_offset"] = center_offset
        return frame

    def _detect_visual_obstacle(self, frame):
        """YOLOv5检测视觉障碍物（车辆/行人/障碍物）"""
        # 推理（过滤小目标，只保留car/person/bicycle）
        results = self.yolo_model(frame, conf=0.5, classes=[0, 1, 2])
        obstacles = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取检测框信息
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = self.yolo_model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                # 简易距离估算（基于检测框高度，需标定）
                box_height = y2 - y1
                distance = 500 / box_height  # 示例公式：高度越小，距离越远

                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{cls}: {distance:.1f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                obstacles.append({
                    "class": cls,
                    "distance": distance,
                    "x": (x1 + x2) / 2 - frame.shape[1] / 2,  # 横向偏移（像素）
                    "confidence": conf
                })

        self.perception_data["visual_obstacle"] = obstacles
        return frame

    def _vision_loop(self):
        """视觉感知循环（10Hz）"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                # 1. 车道线检测
                frame = self._detect_lane(frame)
                # 2. 视觉障碍物检测
                frame = self._detect_visual_obstacle(frame)
                # 显示画面
                cv2.putText(frame, f"Lane Offset: {self.perception_data['lane']['center_offset']:.2f}m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Car Perception (Vision)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
            time.sleep(0.1)

    # ------------------- 3. LiDAR感知 -------------------
    def _process_lidar_pointcloud(self):
        """LiDAR点云处理（聚类检测障碍物）"""
        try:
            # 示例：生成模拟点云（实际替换为LiDAR SDK读取）
            points = np.random.rand(1000, 3) * 20 - 10  # [-10,10]米范围
            points = points[points[:, 2] > -1]  # 过滤地面以下点

            # Open3D点云处理
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 下采样 + 去除离群点
            pcd = pcd.voxel_down_sample(voxel_size=0.2)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

            # 聚类检测障碍物（DBSCAN）
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10))

            obstacles = []
            max_label = labels.max()
            if max_label >= 0:
                for label in range(max_label + 1):
                    cluster_points = np.asarray(pcd.points)[labels == label]
                    # 计算障碍物中心、距离、尺寸
                    center = np.mean(cluster_points, axis=0)
                    distance = np.linalg.norm(center)  # 距车辆的距离
                    angle = math.degrees(math.atan2(center[1], center[0]))  # 方位角
                    width = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
                    length = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])

                    obstacles.append({
                        "distance": distance,
                        "angle": angle,
                        "width": width,
                        "length": length
                    })

            self.perception_data["lidar_obstacle"] = obstacles
        except Exception as e:
            print(f"LiDAR处理失败: {e}")

    def _lidar_loop(self):
        """LiDAR感知循环（5Hz）"""
        while self.running:
            self._process_lidar_pointcloud()
            time.sleep(0.2)

    # ------------------- 4. 多传感器融合 -------------------
    def _fusion_obstacle(self):
        """融合视觉+LiDAR障碍物数据（加权融合）"""
        visual_obstacles = self.perception_data["visual_obstacle"]
        lidar_obstacles = self.perception_data["lidar_obstacle"]
        fused_obstacles = []

        # 简易融合规则：LiDAR距离优先，视觉补充类别
        for lidar_obs in lidar_obstacles:
            # 匹配视觉障碍物（方位角±5°内）
            matched_visual = None
            for visual_obs in visual_obstacles:
                # 视觉方位角估算（基于横向偏移）
                visual_angle = math.degrees(math.atan2(visual_obs["x"] * 0.01, visual_obs["distance"]))
                if abs(visual_angle - lidar_obs["angle"]) < 5:
                    matched_visual = visual_obs
                    break

            fused_obstacles.append({
                "distance": lidar_obs["distance"],  # LiDAR距离更精准
                "angle": lidar_obs["angle"],
                "width": lidar_obs["width"],
                "class": matched_visual["class"] if matched_visual else "unknown",
                "confidence": matched_visual["confidence"] if matched_visual else 0.8
            })

        self.perception_data["fused_obstacle"] = fused_obstacles

    def _fusion_loop(self):
        """数据融合循环（5Hz）"""
        while self.running:
            self._fusion_obstacle()
            time.sleep(0.2)

    # ------------------- 5. 安全判断与接口 -------------------
    def check_safety(self):
        """安全状态判断"""
        # 1. 碰撞风险：融合障碍物距离<3米
        for obs in self.perception_data["fused_obstacle"]:
            if obs["distance"] < 3.0:
                return False, f"前方{obs['distance']:.1f}米检测到{obs['class']}，碰撞风险！"
        # 2. 车道偏离风险：偏移>0.5米
        if abs(self.perception_data["lane"]["center_offset"]) > 0.5 and self.perception_data["lane"]["detected"]:
            return False, f"车道偏移{self.perception_data['lane']['center_offset']:.2f}米，偏离风险！"
        # 3. 车速风险：超速（>10m/s=36km/h）
        if self.perception_data["speed"] > 10.0:
            return False, f"车速{self.perception_data['speed']:.1f}m/s，超速！"
        return True, "安全"

    def get_perception_result(self):
        """获取最新感知结果"""
        return self.perception_data

    def start(self):
        """启动感知系统"""
        print("启动无人车感知系统...")
        self.vision_thread.start()
        self.lidar_thread.start()
        self.state_thread.start()
        self.fusion_thread.start()

    def stop(self):
        """停止感知系统"""
        self.running = False
        # 等待线程结束
        self.vision_thread.join()
        self.lidar_thread.join()
        self.state_thread.join()
        self.fusion_thread.join()
        # 释放资源
        self.camera.release()
        self.serial_port.close()
        cv2.destroyAllWindows()
        print("无人车感知系统已停止")


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 初始化感知系统
    car_perception = AutonomousCarPerception()

    try:
        # 启动感知
        car_perception.start()

        # 主循环打印感知结果
        while True:
            data = car_perception.get_perception_result()
            is_safe, msg = car_perception.check_safety()

            print(f"\n=== 无人车感知数据 [{time.ctime()}] ===")
            print(f"车速: {data['speed']:.2f}m/s, 航向角: {data['heading']:.1f}°")
            print(
                f"车道线: {'检测到' if data['lane']['detected'] else '未检测到'}, 偏移: {data['lane']['center_offset']:.2f}m")
            print(f"视觉障碍物数量: {len(data['visual_obstacle'])}")
            print(f"LiDAR障碍物数量: {len(data['lidar_obstacle'])}")
            print(f"融合障碍物数量: {len(data['fused_obstacle'])}")
            print(f"安全状态: {'✅ 安全' if is_safe else '❌ 危险'} - {msg}")

            time.sleep(1)

    except KeyboardInterrupt:
        car_perception.stop()
        print("程序已手动终止")
