#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


bridge = CvBridge()
def image_publisher():
    pub = rospy.Publisher('image_raw', Image, queue_size=10)
    rospy.init_node('image_publisher', anonymous=True)
    rate = rospy.Rate(0.5) # 10hz

    while not rospy.is_shutdown():
        # Create and populate your image message
        img_msg = Image()
        img = cv2.imread('./tests/garden.jpg')
        ros_img = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        # Publish the image message
        pub.publish(ros_img)
        rate.sleep()

if __name__ == '__main__':
    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass
