#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>  // tf库的变换广播器
#include <tf/LinearMath/Quaternion.h>  // tf的四元数
#include <urdf/model.h>                // URDF解析
#include <map>
#include <string>
#include <cmath>

// 存储关节信息的结构体
struct JointInfo {
    std::string name;
    std::string parent_link;
    std::string child_link;
    int type;  // 关节类型（用整数存储URDF的枚举值）
    urdf::Vector3 axis;       // 旋转轴
    urdf::Pose origin;        // 初始变换
};

class RobotStatePublisher {
private:
    ros::NodeHandle nh_;
    ros::Subscriber joint_state_sub_;
    tf::TransformBroadcaster tf_broadcaster_;  // tf的广播器
    std::map<std::string, JointInfo> joints_;

public:
    RobotStatePublisher() {
        // 1. 从参数服务器获取URDF模型
        std::string urdf_string;
        if (!nh_.getParam("robot_description", urdf_string)) {
            ROS_ERROR("Failed to get 'robot_description' from parameter server!");
            ros::shutdown();
            return;
        }

        // 2. 解析URDF
        urdf::Model urdf_model;
        if (!urdf_model.initString(urdf_string)) {
            ROS_ERROR("Failed to parse URDF!");
            ros::shutdown();
            return;
        }

        // 3. 提取关节信息（存储类型为整数，避免枚举兼容性问题）
        for (const auto& joint_pair : urdf_model.joints_) {
            const urdf::JointConstSharedPtr& joint = joint_pair.second;
            JointInfo info;
            info.name = joint->name;
            info.parent_link = joint->parent_link_name;
            info.child_link = joint->child_link_name;
            info.type = joint->type;  // 用整数存储URDF关节类型（REVOLUTE=1, CONTINUOUS=2）
            if (info.type == urdf::Joint::REVOLUTE || info.type == urdf::Joint::CONTINUOUS) {
                info.axis = joint->axis;
            }
            info.origin = joint->parent_to_joint_origin_transform;
            joints_[info.name] = info;
        }

        // 4. 订阅关节状态
        joint_state_sub_ = nh_.subscribe("/joint_states", 10, &RobotStatePublisher::jointStateCallback, this);
        ROS_INFO("RobotStatePublisher (tf) initialized.");
    }

    // 关节状态回调：计算并发布TF
    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg) {
        for (size_t i = 0; i < msg->name.size(); ++i) {
            std::string joint_name = msg->name[i];
            if (joints_.find(joint_name) == joints_.end()) {
                ROS_WARN("Joint '%s' not found in URDF!", joint_name.c_str());
                continue;
            }
            JointInfo& joint = joints_[joint_name];

            // 初始化TF变换
            tf::Transform transform;
            tf::Quaternion origin_quat;

            // 1. 初始平移（URDF定义的原点）
            transform.setOrigin(tf::Vector3(
                joint.origin.position.x,
                joint.origin.position.y,
                joint.origin.position.z
            ));

            // 2. 初始旋转（URDF定义的原点姿态）
            origin_quat.setX(joint.origin.rotation.x);
            origin_quat.setY(joint.origin.rotation.y);
            origin_quat.setZ(joint.origin.rotation.z);
            origin_quat.setW(joint.origin.rotation.w);
            transform.setRotation(origin_quat);

            // 3. 叠加关节旋转（仅旋转关节）
            if (joint.type == urdf::Joint::REVOLUTE || joint.type == urdf::Joint::CONTINUOUS) {
                // 旋转轴（单位化）
                tf::Vector3 axis(joint.axis.x, joint.axis.y, joint.axis.z);
                axis.normalize();

                // 旋转角度（从关节状态获取）
                double angle = msg->position[i];

                // 用tf的四元数设置轴角旋转（tf1支持setAxisAngle）
                tf::Quaternion joint_quat(axis, angle); 
               

                // 总旋转 = 初始旋转 * 关节旋转
                transform.setRotation(origin_quat * joint_quat);
            }

            // 发布TF变换
            tf_broadcaster_.sendTransform(tf::StampedTransform(
                transform,
                msg->header.stamp,  // 时间戳同步
                joint.parent_link,  // 父坐标系
                joint.child_link    // 子坐标系
            ));
        }
    }
};

int main(int argc, char**argv) {
    ros::init(argc, argv, "robot_state_publisher");
    RobotStatePublisher publisher;
    ros::spin();
    return 0;
}
