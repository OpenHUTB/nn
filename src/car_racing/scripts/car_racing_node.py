import rospy
import sys
import os
from std_msgs.msg import Float32, String

# 导入同功能包内的你的原代码（不用跨目录，ROS封装的核心）
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../your_python_code"))

from dqn_model.DQN_model import Agent as DQNAgent
from double_dqn_model.double_dqn import DoubleDQNAgent as DDQNAgent
from compare_models import evaluate_ddqn  # 评估函数

# ROS节点
class CarRacingROS:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('car_racing_ros_node', anonymous=True)
        self.avg_score_pub = rospy.Publisher('/car/avg_score', Float32, queue_size=10)
        self.status_pub = rospy.Publisher('/car/status', String, queue_size=10)

        #  加载原代码模型
        self.agent = DDQNAgent(
            state_space_shape=(4, 84, 84),
            action_n=5,
            load_state=True,
            load_model='training/saved_models/DoubleDQN.pt'  #原路径
        )
        rospy.loginfo("你的Python原代码模型加载完成")

    def run(self):
        # 3. 运行原代码评估逻辑
        avg_score = evaluate_ddqn(self.agent, num_episodes=5)  # 调用原函数

        # 4. 把结果发布到ROS
        self.avg_score_pub.publish(avg_score)
        self.status_pub.publish(f"原代码评估完成，平均得分：{avg_score:.1f}")
        rospy.loginfo(f"📌 ROS封装完成，评估得分：{avg_score:.1f}")
        rospy.spin()  # 保持节点运行

if __name__ == "__main__":
    try:
        ros_node = CarRacingROS()
        ros_node.run()
    except rospy.ROSInterruptException:
        pass
