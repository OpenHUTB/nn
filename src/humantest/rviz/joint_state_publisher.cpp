#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <vector>
#include <string>

int main(int argc, char**argv) {
    // 初始化节点
    ros::init(argc, argv, "joint_state_publisher");
    ros::NodeHandle nh;

    // 创建发布者，发布关节状态消息（话题名固定为/joint_states，ROS约定）
    ros::Publisher joint_state_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 10);

    // 关节名称（需与URDF中的关节名一致，例如mbot_base.urdf中的轮子关节）
    std::vector<std::string> joint_names = {"left_wheel_joint", "right_wheel_joint"};

    // 循环频率（10Hz）
    ros::Rate loop_rate(10);

    // 关节位置（模拟轮子转动，随时间递增）
    double left_pos = 0.0;
    double right_pos = 0.0;
    double delta = 0.1;  // 每次更新的角度（弧度）

    while (ros::ok()) {
        // 初始化关节状态消息
        sensor_msgs::JointState joint_state;
        joint_state.header.stamp = ros::Time::now();  // 时间戳
        joint_state.name = joint_names;               // 关节名称

        // 更新关节位置（模拟转动）
        left_pos += delta;
        right_pos += delta;
        joint_state.position = {left_pos, right_pos};  // 关节位置（弧度）
        joint_state.velocity = {0.5, 0.5};             // 关节速度（弧度/秒，示例值）
        joint_state.effort = {10.0, 10.0};             // 关节力/力矩（示例值）

        // 发布消息
        joint_state_pub.publish(joint_state);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
