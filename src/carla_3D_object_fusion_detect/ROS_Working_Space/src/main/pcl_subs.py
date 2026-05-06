#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
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

def plc_callback(data):
    global plc_idx
    xyz = np.array([[0,0,0]])
    rgb = np.array([[0,0,0]])
    #self.lock.acquire()
    gen = pc2.read_points(data, skip_nans=True)
    int_data = list(gen)

    for x in int_data:
        test = x[3] 
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,test)
        i = struct.unpack('>l',s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)
        # prints r,g,b values in the 0-255 range
                    # x,y,z can be retrieved from the x[0],x[1],x[2]
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
        rgb = np.append(rgb,[[r,g,b]], axis = 0)

    out_pcd = o3d.geometry.PointCloud()    
    out_pcd.points = o3d.utility.Vector3dVector(xyz)
    out_pcd.colors = o3d.utility.Vector3dVector(rgb)

    plc_idx += 1
    file_name = FOLDER + "pcl/" + "{0:d}.ply".format(plc_idx).zfill(6)
    o3d.io.write_point_cloud(file_name, out_pcd)
    rospy.loginfo("save plc = " + str(plc_idx))


def plc_listener():
    rospy.loginfo("Init plc listener")
    rospy.init_node('plc_listener', anonymous=True)
    rospy.Subscriber("/carla/ego_vehicle/dvs_front/events", PointCloud2, plc_callback)
    rospy.spin()


if __name__ == '__main__':
    plc_listener()