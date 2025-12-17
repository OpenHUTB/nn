#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import rospy
from std_msgs.msg import Float32MultiArray, Float32, Bool

from uitb import Simulator


class RCCarNode(object):
    def __init__(self, simulator_folder, rate_hz=30.0):
        rospy.loginfo("RCCarNode init, simulator_folder: %s", simulator_folder)

        # 创建 simulator（RC Car via Joystick）
        self.env = Simulator.get(simulator_folder)

        self.rate = rospy.Rate(rate_hz)

        # 当前 action（从 ROS 话题拿）
        self.current_action = None

        # 发布者：观测 / 奖励 / 终止标志
        self.obs_pub = rospy.Publisher(
            "/rc_car/obs", Float32MultiArray, queue_size=1
        )
        self.reward_pub = rospy.Publisher(
            "/rc_car/reward", Float32, queue_size=1
        )
        self.done_pub = rospy.Publisher(
            "/rc_car/done", Bool, queue_size=1
        )

        # 订阅 action（你后面可以用别的节点发布这个话题，比如摇杆转指令）
        rospy.Subscriber(
            "/rc_car/action", Float32MultiArray, self.action_callback
        )

        # 先 reset 一次
        self.reset_env()

    def action_callback(self, msg):
        # 保存最新的动作
        self.current_action = list(msg.data)

    def reset_env(self):
        obs, info = self.env.reset()
        self.publish_obs(obs)
        rospy.loginfo("Environment reset")

    def publish_obs(self, obs):
        msg = Float32MultiArray()
        # obs 可能是 numpy 数组，统一转 list
        try:
            data = obs.flatten().tolist()
        except AttributeError:
            # 已经是 list/tuple
            data = list(obs)
        msg.data = data
        self.obs_pub.publish(msg)

    def step_once(self):
        # 如果还没收到 action，就用零动作（或随机动作，看你需求）
        if self.current_action is None:
            # 用零向量动作（假设 action_space 是 Box）
            import numpy as np

            action_dim = self.env.action_space.shape[0]
            action = np.zeros(action_dim, dtype=float)
        else:
            action = self.current_action

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # 发布结果
        self.publish_obs(obs)
        self.reward_pub.publish(Float32(data=float(reward)))
        self.done_pub.publish(Bool(data=done))

        if done:
            rospy.loginfo("Episode done, auto reset")
            self.reset_env()

    def spin(self):
        rospy.loginfo("RCCarNode spinning...")
        while not rospy.is_shutdown():
            self.step_once()
            self.rate.sleep()


def main():
    rospy.init_node("rc_car_sim_node")

    # 从参数服务器取 simulator_folder，如果没给就用默认路径
    default_sim_folder = str(
        Path(__file__).resolve().parents[2]
        / "simulators"
        / "mobl_arms_index_remote_driving"
    )
    simulator_folder = rospy.get_param("~simulator_folder", default_sim_folder)

    rate_hz = rospy.get_param("~rate", 30.0)

    node = RCCarNode(simulator_folder=simulator_folder, rate_hz=rate_hz)
    node.spin()


if __name__ == "__main__":
    main()
