#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.point_cloud2 import PointCloud2 as pc2
from cv_bridge import CvBridge
import cv2
import open3d as o3d
import numpy as np
import ctypes
import struct

br = CvBridge()
img_idx = 7419
plc_idx = 7419
FOLDER = "/mnt/ysli-1/Lab/Dataset/CARLA/"


def img_callback(data):
    global img_idx
    cv_img = br.imgmsg_to_cv2(data)
    file_name = FOLDER + "img/" + "{:d}.png".format(img_idx).zfill(6)
    img_idx += 1
    cv2.imwrite(file_name, cv_img, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    rospy.loginfo("save img = " + str(img_idx))


def img_listener():
    rospy.loginfo("Init img listener")
    rospy.init_node('img_listener', anonymous=True)
    rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, img_callback)
    rospy.spin()


if __name__ == '__main__':
    img_listener()