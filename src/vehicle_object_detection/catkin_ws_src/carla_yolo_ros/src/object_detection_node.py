#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

# 定义 ROS 的库路径
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

# 【第一步】如果有 ROS 路径，先移除，确保 import cv2 加载的是虚拟环境里的新版本
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2

# 【第二步】再把 ROS 路径加回到末尾，确保后续能找到 sensor_msgs
sys.path.append(ros_path)

import rospy
import numpy as np
import carla
from sensor_msgs.msg import Image  # 现在这一行就能正常运行了
from ultralytics import YOLO

class CarlaYoloNode:
    def __init__(self):
        rospy.init_node('carla_yolo_node', anonymous=True)
        
        # 参数配置
        self.host = rospy.get_param('~host', '192.168.133.1')
        self.port = rospy.get_param('~port', 2000)
        self.model_path = rospy.get_param('~model_path', '')
        self.conf_thres = rospy.get_param('~confidence', 0.5)
        
        self.pub_img = rospy.Publisher('/carla/camera/detection', Image, queue_size=1)
        
        # 连接 CARLA
        try:
            self.client = carla.Client(self.host, self.port, worker_threads=1)
            self.client.set_timeout(30.0)
            self.world = self.client.get_world()
        except Exception as e:
            rospy.logerr(f"Connect Fail: {e}")
            sys.exit(1)

        # 加载 YOLO 模型
        rospy.loginfo(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        self.setup_sensor()

    def setup_sensor(self):
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # 1. 生成带自动驾驶的车辆
        vehicle_bp = bp_lib.filter("model3")[0]
        self.vehicle = None
        for point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, point)
                break
            except RuntimeError:
                continue
        
        if not self.vehicle:
            rospy.logerr("Cannot spawn vehicle")
            return

        self.vehicle.set_autopilot(True, 8005)
        
        # 2. 挂载摄像头
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        self.camera = self.world.spawn_actor(
            camera_bp, 
            carla.Transform(carla.Location(x=1.5, z=2.4)), 
            attach_to=self.vehicle
        )
        self.camera.listen(self.callback)

    def callback(self, data):
        if rospy.is_shutdown(): return
        
        # 手动将 CARLA 原始数据转为 Numpy (避免使用 cv_bridge)
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((data.height, data.width, 4))
        rgb = array[:, :, :3]
        
        # YOLO 推理
        results = self.model(rgb, verbose=False, conf=self.conf_thres)
        
        # 绘制检测框 (转为 BGR 供 OpenCV 绘图)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if int(box.cls[0]) < len(self.model.names):
                    label = self.model.names[int(box.cls[0])]
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(bgr, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 手动封装 ROS Image 消息
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_link"
        msg.height = bgr.shape[0]
        msg.width = bgr.shape[1]
        msg.encoding = "bgr8"
        msg.step = bgr.shape[1] * 3
        msg.data = bgr.tobytes()
        self.pub_img.publish(msg)

    def cleanup(self):
        if self.camera: self.camera.destroy()
        if self.vehicle: self.vehicle.destroy()

if __name__ == '__main__':
    node = CarlaYoloNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.cleanup()
