# 标准库
import rospy
import rospkg  # 用于获取ROS包路径
# 第三方库
import mujoco
from mujoco import viewer  # 注意：MuJoCo 2.x版本支持，3.x需用mujoco_viewer

def main():
    # 初始化ROS节点
    rospy.init_node('humanoid_standup_node', anonymous=True)
    rospy.loginfo("人形机器人起身控制节点启动")

    # 从ROS参数服务器获取配置（可通过launch文件修改）
    kp_gain = rospy.get_param('~kp_gain', 5.0)  # 比例增益
    model_rel_path = rospy.get_param('~model_path', 'xml/humanoid.xml')  # 模型相对路径

    # 获取模型绝对路径（解决相对路径问题）
    try:
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('humanoid_motion')  # 功能包路径
        model_path = f"{pkg_path}/{model_rel_path}"  # 拼接绝对路径
        rospy.loginfo(f"模型路径: {model_path}")
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

    # 初始化目标姿势（站立，关键帧1）和初始姿势（深蹲，关键帧0）
    target_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, target_data, 1)  # 站立关键帧
    target_qpos = target_data.qpos.copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)  # 初始深蹲姿势

    # 设置控制频率（200Hz，与原time.sleep(0.005)对应）
    control_rate = rospy.Rate(200)

    # 启动可视化
    with viewer.launch_passive(model, data) as v:
        rospy.loginfo("开始起身控制（按Ctrl+C停止）")
        try:
            # ROS循环（直到节点关闭）
            while not rospy.is_shutdown():
                mujoco.mj_step(model, data)  # 单步仿真

                # 比例控制（只控制电机关节，跳过前7个根关节）
                qpos_error = target_qpos[7:] - data.qpos[7:]
                data.ctrl[:] = kp_gain * qpos_error

                # 输出躯干高度（ROS日志）
                rospy.loginfo_throttle(1,  # 每1秒输出一次，避免刷屏
                                      f"时间: {data.time:.2f}, 躯干高度: {data.qpos[2]:.2f}")

                v.sync()  # 同步可视化
                control_rate.sleep()  # 控制循环频率

        except KeyboardInterrupt:
            rospy.loginfo("用户中断，停止控制")
        finally:
            rospy.loginfo("起身控制节点关闭")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass