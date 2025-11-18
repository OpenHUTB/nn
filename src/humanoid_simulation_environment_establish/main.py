import mujoco
from mujoco import viewer
import numpy as np
import rospy
import rospkg  # 用于获取ROS包路径

def main():
    # 初始化ROS节点（anonymous=True确保节点名唯一）
    rospy.init_node('humanoid_main_node', anonymous=True)
    rospy.loginfo("人形机器人核心控制节点启动")

    # 通过rospkg获取模型路径（避免相对路径问题）
    try:
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('humanoid_motion')  # 替换为你的功能包名
        model_path = f"{pkg_path}/xml/humanoid.xml"  # 模型相对路径
    except Exception as e:
        rospy.logerr(f"获取包路径失败: {e}")
        return

    # 加载MuJoCo模型
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        rospy.loginfo("模型加载成功")
    except Exception as e:
        rospy.logerr(f"模型加载失败: {e}")
        return

    # 设置控制频率（200Hz，根据仿真需求调整）
    control_rate = rospy.Rate(200)

    # 启动仿真可视化
    with mujoco.viewer.launch_passive(model, data) as viewer:
        rospy.loginfo("仿真可视化启动")
        try:
            # ROS循环（直到节点关闭）
            while not rospy.is_shutdown():
                # 核心控制逻辑（示例：简单关节控制）
                mujoco.mj_step(model, data)  # 单步仿真
                data.ctrl[:] = 0.1 * np.sin(data.time)  # 示例：正弦波控制关节

                # 同步可视化
                viewer.sync()
                # 控制循环频率
                control_rate.sleep()

        except KeyboardInterrupt:
            rospy.loginfo("用户中断，停止仿真")
        finally:
            rospy.loginfo("核心控制节点关闭")
