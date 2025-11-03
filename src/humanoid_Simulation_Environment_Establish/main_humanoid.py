# 标准库
import rospy
import rospkg  # 用于获取ROS包路径
# 第三方库
import mujoco
from mujoco import viewer  

def main():

    rospy.init_node('humanoid_standup_node', anonymous=True)
    rospy.loginfo("人形机器人起身控制节点启动")

    kp_gain = rospy.get_param('~kp_gain', 5.0)  
    model_rel_path = rospy.get_param('~model_path', 'xml/humanoid.xml')  

    try:
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('humanoid_motion') 
        model_path = f"{pkg_path}/{model_rel_path}"
        rospy.loginfo(f"模型路径: {model_path}")
    except Exception as e:
        rospy.logerr(f"获取包路径失败: {e}")
        return

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        rospy.loginfo("模型加载成功")
    except Exception as e:
        rospy.logerr(f"模型加载失败: {e}")
        return
    
    target_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, target_data, 1)  
    target_qpos = target_data.qpos.copy()
    mujoco.mj_resetDataKeyframe(model, data, 0) 

    control_rate = rospy.Rate(200)

    with viewer.launch_passive(model, data) as v:
        rospy.loginfo("开始起身控制（按Ctrl+C停止）")
        try:
       
            while not rospy.is_shutdown():
                mujoco.mj_step(model, data)  

                qpos_error = target_qpos[7:] - data.qpos[7:]
                data.ctrl[:] = kp_gain * qpos_error

                rospy.loginfo_throttle(1, 
                                      f"时间: {data.time:.2f}, 躯干高度: {data.qpos[2]:.2f}")

                v.sync()  
                control_rate.sleep()  

        except KeyboardInterrupt:
            rospy.loginfo("用户中断，停止控制")
        finally:
            rospy.loginfo("起身控制节点关闭")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass