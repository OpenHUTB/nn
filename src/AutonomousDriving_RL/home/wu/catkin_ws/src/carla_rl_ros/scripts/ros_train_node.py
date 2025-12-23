#!/usr/bin/env python3
import rospy
import sys
import os
import time
sys.path.append(os.path.dirname(__file__))

import carla_env_multi_obs

_original_init = carla_env_multi_obs.CarlaEnvMultiObs.__init__

def _patched_init(self):
    carla_host = rospy.get_param('~carla_host', 'localhost')
    carla_port = rospy.get_param('~carla_port', 2000)
    for attempt in range(3):
        try:
            rospy.loginfo(f"🔄 Connecting to CARLA at {carla_host}:{carla_port}")
            self.client = carla.Client(carla_host, carla_port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            break
        except Exception as e:
            rospy.logwarn(f"⚠️ Retry {attempt+1}: {e}")
            time.sleep(2)
    else:
        raise RuntimeError("❌ CARLA connection failed")
    _original_init(self)

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

