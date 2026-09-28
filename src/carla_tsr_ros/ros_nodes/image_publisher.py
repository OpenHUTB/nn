#!/usr/bin/env python3
import os
import glob
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image


class ImagePublisher:
    def __init__(self):
        rospy.init_node('image_publisher', anonymous=False)
        image_dir = rospy.get_param('~image_dir', 'data')
        rate_hz = rospy.get_param('~publish_rate', 5.0)
        self.loop = rospy.get_param('~loop', True)
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_dir = image_dir if os.path.isabs(image_dir) else os.path.join(pkg_root, image_dir)
        self.images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            self.images.extend(sorted(glob.glob(os.path.join(full_dir, ext))))
        if not self.images:
            rospy.logerr("No images found in " + full_dir)
            raise RuntimeError("No images found in " + full_dir)
        rospy.loginfo("Found %d images" % len(self.images))
        self.pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
        self.rate = rospy.Rate(rate_hz)
        self.idx = 0

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

    def run(self):
        while not rospy.is_shutdown():
            path = self.images[self.idx]
            frame = cv2.imread(path)
            if frame is None:
                rospy.logwarn("Failed to read " + path)
                self.idx = (self.idx + 1) % len(self.images)
                continue
            msg = self.cv2_to_imgmsg(frame)
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'camera'
            self.pub.publish(msg)
            rospy.loginfo_throttle(5.0, "Publishing " + os.path.basename(path))
            self.idx += 1
            if self.idx >= len(self.images):
                if self.loop:
                    self.idx = 0
                else:
                    break
            self.rate.sleep()


if __name__ == '__main__':
    try:
        ImagePublisher().run()
    except rospy.ROSInterruptException:
        pass
