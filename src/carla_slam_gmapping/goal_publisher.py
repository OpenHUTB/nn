#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导航目标点发布器
功能：向move_base发布导航目标点
使用：python3 goal_publisher.py [x] [y] [yaw]
"""

import rospy
import sys
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionGoal
import math


class GoalPublisher:
    def __init__(self):
        rospy.init_node('goal_publisher', anonymous=True)
        
        # 发布器
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        rospy.loginfo("Goal Publisher initialized")
        rospy.sleep(1.0)  # 等待发布器准备好
    
    def publish_goal(self, x, y, yaw=0.0):
        """
        发布导航目标点
        
        Args:
            x: 目标点X坐标（米）
            y: 目标点Y坐标（米）
            yaw: 目标朝向（弧度）
        """
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        
        # 位置
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # 朝向（欧拉角转四元数）
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        
        rospy.loginfo(f"Publishing goal: x={x}, y={y}, yaw={yaw:.2f} rad")
        self.goal_pub.publish(goal)
        rospy.loginfo("Goal published successfully!")
    
    def publish_goal_sequence(self, goals):
        """
        发布一系列导航目标点（按顺序）
        
        Args:
            goals: 目标点列表，每个元素为 (x, y, yaw)
        """
        rospy.loginfo(f"Publishing {len(goals)} goals in sequence...")
        
        for i, (x, y, yaw) in enumerate(goals):
            rospy.loginfo(f"Goal {i+1}/{len(goals)}")
            self.publish_goal(x, y, yaw)
            
            # 等待用户确认或超时
            input(f"Press Enter to send next goal (or Ctrl+C to stop)...")


def predefined_goals():
    """
    预定义的测试目标点集合（Town01场景）
    """
    return {
        "straight": [
            (10.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
        ],
        "square": [
            (20.0, 0.0, 0.0),
            (20.0, 20.0, 1.57),
            (0.0, 20.0, 3.14),
            (0.0, 0.0, -1.57),
        ],
        "explore": [
            (50.0, 0.0, 0.0),
            (50.0, 50.0, 1.57),
            (-50.0, 50.0, 3.14),
            (-50.0, -50.0, -1.57),
            (0.0, 0.0, 0.0),
        ],
    }


def main():
    if len(sys.argv) < 2:
        print("\n=== Goal Publisher Usage ===")
        print("1. Single goal:")
        print("   python3 goal_publisher.py <x> <y> [yaw]")
        print("   Example: python3 goal_publisher.py 10.0 5.0 1.57")
        print("\n2. Predefined goal sequences:")
        print("   python3 goal_publisher.py straight  # 直线前进")
        print("   python3 goal_publisher.py square    # 方形路径")
        print("   python3 goal_publisher.py explore   # 探索路径")
        print("\n3. Interactive mode:")
        print("   python3 goal_publisher.py interactive")
        return
    
    try:
        publisher = GoalPublisher()
        
        # 预定义路径
        if sys.argv[1] in predefined_goals():
            goals = predefined_goals()[sys.argv[1]]
            publisher.publish_goal_sequence(goals)
        
        # 交互模式
        elif sys.argv[1] == "interactive":
            print("\n=== Interactive Goal Publisher ===")
            print("Enter goal coordinates (or 'q' to quit)")
            while not rospy.is_shutdown():
                try:
                    x = input("X coordinate (m): ")
                    if x.lower() == 'q':
                        break
                    y = input("Y coordinate (m): ")
                    yaw = input("Yaw angle (rad, default=0.0): ")
                    
                    x = float(x)
                    y = float(y)
                    yaw = float(yaw) if yaw else 0.0
                    
                    publisher.publish_goal(x, y, yaw)
                except ValueError:
                    print("Invalid input! Please enter numbers.")
                except KeyboardInterrupt:
                    break
        
        # 单个目标点
        else:
            x = float(sys.argv[1])
            y = float(sys.argv[2])
            yaw = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
            
            publisher.publish_goal(x, y, yaw)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Goal Publisher shutdown")
    except Exception as e:
        rospy.logerr(f"Error: {e}")


if __name__ == '__main__':
    main()
