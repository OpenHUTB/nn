#!/usr/bin/env python3
import rospy
import sys
import os
sys.path.append(os.path.dirname(__file__))

import carla_env_multi_obs

_original_init = carla_env_multi_obs.CarlaEnvMultiObs.__init__

def _patched_init(self, *args, **kwargs):
    """通过构造参数注入 ROS 中的 CARLA 地址，避免先连接再被 __init__ 清掉 client/world。"""
    kwargs.setdefault('carla_host', rospy.get_param('~carla_host', 'localhost'))
    kwargs.setdefault('carla_port', int(rospy.get_param('~carla_port', 2000)))
    _original_init(self, *args, **kwargs)

carla_env_multi_obs.CarlaEnvMultiObs.__init__ = _patched_init

from train_agent import main as train_main

if __name__ == '__main__':
    rospy.init_node('carla_rl_train', anonymous=True)
    rospy.loginfo("🚀 Starting training via ROS...")
    try:
        train_main()
    except KeyboardInterrupt:
        rospy.loginfo("Training stopped by user.")
    except Exception as e:
        rospy.logerr(f"Training error: {e}")
        raise

