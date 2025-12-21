#!/usr/bin/env python3
"""
CARLA自动驾驶ROS服务定义
"""

from rospy import Service, ServiceProxy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Float32MultiArray

# 自定义服务
from carla_autonomous.srv import *

class Reset(Service):
    """重置环境服务"""
    def __init__(self):
        super().__init__('reset', Reset, self.callback)
    
    def callback(self, req):
        # 返回状态数组
        return ResetResponse(Float32MultiArray())

class StartEpisode(Service):
    """开始episode服务"""
    def __init__(self):
        super().__init__('start_episode', StartEpisode, self.callback)
    
    def callback(self, req):
        return StartEpisodeResponse(True, "Episode started")

class Stop(Service):
    """停止服务"""
    def __init__(self):
        super().__init__('stop', Stop, self.callback)
    
    def callback(self, req):
        return StopResponse(True, "Stopped")