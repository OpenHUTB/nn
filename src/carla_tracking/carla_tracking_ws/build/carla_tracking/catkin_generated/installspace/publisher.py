# publisher.py - ROS发布器
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Int32, Header
from cv_bridge import CvBridge
from carla_tracking.msg import TrackedObject
import numpy as np
import json

class CarlaPublisher:
    def __init__(self):
        # 初始化ROS节点（由主节点管理，此处不独立初始化）
        self.bridge = CvBridge()
        
        # 创建发布者
        self.image_pub = rospy.Publisher('/carla/camera/rgb', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/carla/camera/depth', Image, queue_size=10)
        self.fps_pub = rospy.Publisher('/carla/performance/fps', Float32, queue_size=10)
        self.frame_pub = rospy.Publisher('/carla/performance/frame_count', Int32, queue_size=10)
        self.tracks_pub = rospy.Publisher('/carla/tracking/objects', TrackedObject, queue_size=10)
        self.tracks_info_pub = rospy.Publisher('/carla/tracking/info', Image, queue_size=10)  # 可视化结果
        
        self.logger = rospy.loginfo

    def publish_image(self, cv_image, topic='rgb'):
        """发布图像（RGB/Depth）"""
        try:
            if topic == 'rgb':
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.image_pub.publish(ros_image)
            elif topic == 'depth':
                # 深度图像转换为ROS消息（32位浮点）
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="32FC1")
                self.depth_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"发布图像失败: {e}")

    def publish_tracks(self, tracks, tracks_info):
        """发布跟踪结果"""
        # 发布单个跟踪目标
        for track in tracks:
            obj = TrackedObject()
            obj.x1 = track[0]
            obj.y1 = track[1]
            obj.x2 = track[2]
            obj.y2 = track[3]
            obj.track_id = int(track[4])
            obj.class_id = int(track[5])
            obj.confidence = float(track[6])
            
            # 补充距离和速度
            track_info = next((t for t in tracks_info if t['id'] == track[4]), None)
            if track_info:
                obj.distance = track_info.get('distance', 0.0)
                obj.velocity = track_info.get('velocity', 0.0)
            self.tracks_pub.publish(obj)
        
        # 发布FPS和帧数
        self.fps_pub.publish(self.last_fps)
        self.frame_pub.publish(self.last_frame)

    def publish_visualization(self, cv_image):
        """发布可视化结果"""
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.tracks_info_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"发布可视化图像失败: {e}")

    def update(self, detection_results):
        """更新并发布所有数据"""
        # 保存最新状态
        self.last_fps = detection_results['fps']
        self.last_frame = detection_results['frame_count']
        
        # 发布RGB图像
        if 'image' in detection_results and detection_results['image'] is not None:
            self.publish_image(detection_results['image'], 'rgb')
        
        # 发布深度图像
        if 'depth_image' in detection_results and detection_results['depth_image'] is not None:
            self.publish_image(detection_results['depth_image'], 'depth')
        
        # 发布跟踪结果
        if 'tracks' in detection_results and detection_results['tracks']:
            self.publish_tracks(detection_results['tracks'], detection_results['tracks_info'])
        
        # 生成并发布可视化图像
        if 'image' in detection_results and detection_results['image'] is not None:
            vis_image = self._draw_visualization(detection_results['image'], 
                                                detection_results['tracks'],
                                                detection_results['tracks_info'])
            self.publish_visualization(vis_image)

    def _draw_visualization(self, image, tracks, tracks_info):
        """绘制可视化结果（复用原代码的draw_bounding_boxes）"""
        # 【复制原代码中的draw_bounding_boxes和draw_trajectories函数】
        # 此处简化实现，绘制跟踪框和ID
        vis_image = image.copy()
        track_ids = [t[4] for t in tracks]
        boxes = [t[:4] for t in tracks]
        distances = [t['distance'] for t in tracks_info]
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制ID和距离
            text = f"ID:{int(track_ids[i])} D:{distances[i]:.1f}m" if i < len(distances) else f"ID:{int(track_ids[i])}"
            cv2.putText(vis_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis_image
