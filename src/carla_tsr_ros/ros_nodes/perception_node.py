#!/usr/bin/env python3
import os
import json
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node', anonymous=False)
        model_path = rospy.get_param('~model_path', 'yolov8n.pt')
        self.conf = rospy.get_param('~conf_threshold', 0.5)
        self.target_class = rospy.get_param('~target_class', 11)
        self.infer_every_n = rospy.get_param('~infer_every_n', 2)
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(model_path):
            candidate = os.path.join(pkg_root, model_path)
            if os.path.exists(candidate):
                model_path = candidate
        rospy.loginfo("Loading YOLO: " + model_path)
        self.model = YOLO(model_path)
        rospy.loginfo("YOLO loaded")
        self.frame_count = 0
        self.latest_annotated = None
        self.det_pub = rospy.Publisher('/perception/traffic_signs', String, queue_size=10)
        self.img_pub = rospy.Publisher('/perception/annotated_image', Image, queue_size=1)
        rospy.Subscriber('/camera/image_raw', Image, self.on_image, queue_size=1)
        rospy.loginfo("Perception ready")

    @staticmethod
    def imgmsg_to_cv2(msg):
        n_channels = 3 if msg.encoding in ('bgr8', 'rgb8') else 1
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, n_channels)

    @staticmethod
    def cv2_to_imgmsg(frame):
        msg = Image()
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = 3 * frame.shape[1]
        msg.data = frame.tobytes()
        return msg

    def on_image(self, msg):
        try:
            frame = self.imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logerr("decode error: " + str(e))
            return
        self.frame_count += 1
        if self.frame_count % self.infer_every_n == 0:
            results = self.model(frame, classes=[self.target_class], conf=self.conf, verbose=False)
            detections = []
            for r in results:
                for row in r.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, c, cls_id = row
                    cls_id = int(cls_id)
                    detections.append({
                        'class_id': cls_id,
                        'class_name': self.model.names[cls_id],
                        'confidence': round(float(c), 3),
                        'bbox': [round(float(x1),1), round(float(y1),1), round(float(x2),1), round(float(y2),1)],
                    })
            payload = {'stamp': msg.header.stamp.to_sec(), 'detections': detections}
            self.det_pub.publish(String(data=json.dumps(payload)))
            if detections:
                rospy.loginfo("Detected: " + str([d['class_name'] for d in detections]))
            self.latest_annotated = results[0].plot()
        if self.latest_annotated is not None:
            out_msg = self.cv2_to_imgmsg(self.latest_annotated)
            out_msg.header = msg.header
            self.img_pub.publish(out_msg)


if __name__ == '__main__':
    try:
        PerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
