#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "mujoco/mujoco.h"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <string>

class MujocoPublisherCpp : public rclcpp::Node {
public:
    MujocoPublisherCpp() : Node("mujoco_publisher_cpp") {
        // 读取现有config目录的模型（你的Python包已有的模型，无需重复放）
        this->declare_parameter<std::string>("model_path", "config/humanoid.xml");
        std::string model_rel_path = this->get_parameter("model_path").as_string();
        
        // 拼接绝对路径（自动找到你的mujoco_ros_demo包的share目录）
        std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("mujoco_ros_demo");
        std::string model_path = pkg_share_dir + "/" + model_rel_path;
        
        // 加载MuJoCo模型
        char error[1000];
        model_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!model_) {
            RCLCPP_FATAL(this->get_logger(), "Failed to load model: %s (path: %s)", error, model_path.c_str());
            rclcpp::shutdown();
        }
        data_ = mj_makeData(model_);
        
        // 发布关节角度（话题名与Python节点一致，可替换或共存）
        pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/mujoco/joint_angles", 10);
        
        // 50Hz定时器运行仿真
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&MujocoPublisherCpp::sim_step, this));
            
        RCLCPP_INFO(this->get_logger(), "C++ MuJoCo Publisher started (model: %s)", model_path.c_str());
    }

    ~MujocoPublisherCpp() {
        mj_deleteData(data_);
        mj_deleteModel(model_);
    }

private:
    void sim_step() {
        mj_step(model_, data_);
        auto msg = std_msgs::msg::Float64MultiArray();
        int num_joints = std::min(6, model_->nq);  // 取前6个关节（适配你的模型）
        for (int i = 0; i < num_joints; ++i) {
            msg.data.push_back(data_->qpos[i]);
        }
        pub_->publish(msg);
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    mjModel* model_;
    mjData* data_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MujocoPublisherCpp>());
    rclcpp::shutdown();
    return 0;
}