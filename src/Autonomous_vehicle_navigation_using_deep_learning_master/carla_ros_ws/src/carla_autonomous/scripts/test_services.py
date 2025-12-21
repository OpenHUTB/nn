#!/usr/bin/env python3
"""
测试ROS服务
"""

import rospy
from carla_autonomous.srv import StartEpisode, Reset, Stop

def test_services():
    rospy.init_node('test_client')
    
    # 等待服务可用
    rospy.loginfo("等待服务...")
    rospy.wait_for_service('/carla/start_episode')
    rospy.wait_for_service('/carla/reset')
    rospy.wait_for_service('/carla/stop')
    
    try:
        # 测试重置服务
        reset_service = rospy.ServiceProxy('/carla/reset', Reset)
        response = reset_service()
        rospy.loginfo(f"重置服务响应: {response}")
        
        # 测试开始episode服务
        start_service = rospy.ServiceProxy('/carla/start_episode', StartEpisode)
        response = start_service()
        rospy.loginfo(f"开始episode响应: {response.success}, {response.message}")
        
        # 等待5秒
        rospy.sleep(5)
        
        # 测试停止服务
        stop_service = rospy.ServiceProxy('/carla/stop', Stop)
        response = stop_service()
        rospy.loginfo(f"停止服务响应: {response.success}, {response.message}")
        
    except rospy.ServiceException as e:
        rospy.logerr(f"服务调用失败: {e}")

if __name__ == '__main__':
    test_services()