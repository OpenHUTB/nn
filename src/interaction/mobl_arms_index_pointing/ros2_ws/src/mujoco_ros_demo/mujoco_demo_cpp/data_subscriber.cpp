#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class DataSubscriberCpp : public rclcpp::Node {
public:
    DataSubscriberCpp() : Node("data_subscriber_cpp") {
        // 订阅与Python节点相同的话题
        sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/mujoco/joint_angles",
            10,
            std::bind(&DataSubscriberCpp::callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "C++ Data Subscriber started");
    }

private:
    void callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty joint angles");
            return;
        }
        
        // 打印关节角度并计算平均值
        double sum = 0.0;
        RCLCPP_INFO(this->get_logger(), "C++ Received joint angles:");
        for (size_t i = 0; i < msg->data.size(); ++i) {
            sum += msg->data[i];
            RCLCPP_INFO(this->get_logger(), "Joint %zu: %.2f rad", i+1, msg->data[i]);
        }
        RCLCPP_INFO(this->get_logger(), "Average angle: %.2f rad", sum / msg->data.size());
    }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DataSubscriberCpp>());
    rclcpp::shutdown();
    return 0;
}