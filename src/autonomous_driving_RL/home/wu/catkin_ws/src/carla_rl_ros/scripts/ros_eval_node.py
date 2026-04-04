#!/usr/bin/env python3
"""
ROS wrapper for eval_agent.py
Patches CarlaEnv to support remote CARLA server via ROS params.
"""

import rospy
import sys
import os
# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(__file__))

# Patch CarlaEnv to accept remote CARLA host
import carla_env_multi_obs

_original_init = carla_env_multi_obs.CarlaEnvMultiObs.__init__

def _patched_init(self, *args, **kwargs):
    kwargs.setdefault('carla_host', rospy.get_param('~carla_host', 'localhost'))
    kwargs.setdefault('carla_port', int(rospy.get_param('~carla_port', 2000)))
    _original_init(self, *args, **kwargs)

carla_env_multi_obs.CarlaEnvMultiObs.__init__ = _patched_init

if __name__ == '__main__':
    rospy.init_node('carla_rl_eval', anonymous=True)

    # 读取 ROS 参数
    model_path = rospy.get_param('~model_path', './checkpoints/best_model.zip')
    steps = rospy.get_param('~steps', 1000)
    targets_str = rospy.get_param('~targets', '')
    target_x = rospy.get_param('~target_x', None)
    target_y = rospy.get_param('~target_y', None)
    waypoint_dist = rospy.get_param('~waypoint_dist', 4.0)
    steer_gain = rospy.get_param('~steer_gain', 1.8)
    arrival_radius = rospy.get_param('~arrival_radius', 1.0)
    visualize_path = rospy.get_param('~visualize_path', False)

    # 构造命令行参数
    argv = ['ros_eval_node.py', '--model_path', model_path, '--steps', str(steps)]
    if targets_str:
        argv.extend(['--targets', targets_str])
    if target_x is not None and target_y is not None:
        argv.extend(['--target_x', str(target_x), '--target_y', str(target_y)])
    argv.extend([
        '--waypoint_dist', str(waypoint_dist),
        '--steer_gain', str(steer_gain),
        '--arrival_radius', str(arrival_radius)
    ])
    if visualize_path:
        argv.append('--visualize_path')

    # 临时替换 sys.argv 并运行 main
    original_argv = sys.argv
    sys.argv = argv

    try:
        from eval_agent import main as eval_main
        eval_main()
    except Exception as e:
        rospy.logerr(f"Error in evaluation: {e}")
        raise
    finally:
        sys.argv = original_argv
