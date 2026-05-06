#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from os import listdir
from os.path import join, isfile

FOLDER = '/mnt/ysli-1/Lab/Dataset/CARLA/img'
HZ = 1
files = listdir(FOLDER)

def publisher():
    rospy.init_node('AVOD_to_ROS', anonymous=True)
    br = CvBridge()
    pub = rospy.Publisher('main_to_img', Image, queue_size=10)
    rate = rospy.Rate(HZ) # 10hz
    idx = 0
    print(len(files))

    while not rospy.is_shutdown() and idx < len(files):
        filepath = join(FOLDER, files[idx])
        #if isfile(filepath):
        cv2img = cv2.imread(filepath)
        pub.publish(br.cv2_to_imgmsg(cv2img))
        rospy.loginfo("img = " + str(idx))
        #else:
            #rospy.loginfo("ERROR : cannot open")
        idx = (idx + 1) # % len(files)
        rate.sleep()

if __name__ == '__main__':
    publisher()