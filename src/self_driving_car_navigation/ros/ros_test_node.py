#!/home/pan-j-l/my_ros_project/venv_py37/bin/python3
import rospy
import torch
from models.perception_module import PerceptionModule
from models.decision_module import DecisionModule

def main():
    rospy.init_node('ros_test_node', anonymous=True)
    rospy.loginfo("测试节点启动成功！")
    
    try:
        # 初始化感知/决策模块
        perception_model = PerceptionModule()
        decision_model = DecisionModule()
        rospy.loginfo("感知模块和决策模块初始化成功！")

        # 测试模块功能（模拟传感器数据）
        mock_imu = torch.randn(1, 6)  # 模拟1个batch的IMU数据
        mock_image = torch.randn(1, 3, 128, 128)  # 模拟图像数据
        mock_lidar = torch.randn(1, 360)  # 模拟LiDAR数据

        # 感知模块处理
        fused_feature = perception_model.process(mock_imu, mock_image, mock_lidar)
        rospy.loginfo(f"感知特征生成成功，shape: {fused_feature.shape}")

        # 决策模块输出控制指令
        control_cmd = decision_model.get_control_cmd(fused_feature)
        rospy.loginfo(f"决策控制指令：线速度x={control_cmd['linear_x']:.2f}m/s，角速度z={control_cmd['angular_z']:.2f}rad/s")

    except Exception as e:
        rospy.logerr(f"模块测试失败：{str(e)}")
        return
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点已被中断退出。")