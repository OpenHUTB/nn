#include "rclcpp/rclcpp.hpp"
#include "my_interfaces/srv/robot_ctrl.hpp"
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>

// 全局变量
std::string current_action_key = "stop";
bool action_switch_flag = false;

// 非阻塞键盘读取
char read_keypress_non_blocking() {
    struct termios oldt, newt;
    int ch;
    int oldf;

    // 获取终端设置
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    // 禁用 canonical 模式和回显
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    // 设置非阻塞模式
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    // 恢复终端设置
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    return (ch != EOF) ? ch : 0;
}

class RobotController : public rclcpp::Node {
public:
    RobotController() : Node("robot_controller") {
        // 创建服务客户端
        client_ = this->create_client<my_interfaces::srv::RobotCtrl>("robot_control");
        
        // 等待服务可用
        while (!client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "等待服务时被中断");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "等待服务 'robot_control' 启动...");
        }
        
        RCLCPP_INFO(this->get_logger(), "操作提示：w=行走 | s=下蹲 | q=退出 | 其他键=停止");
    }

    void send_request(const std::string &action) {
        auto request = std::make_shared<my_interfaces::srv::RobotCtrl::Request>();
        request->command = action;

        // 发送异步请求
        auto result_future = client_->async_send_request(
            request,
            std::bind(&RobotController::result_callback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "已发送指令：%s", action.c_str());
    }

private:
    rclcpp::Client<my_interfaces::srv::RobotCtrl>::SharedPtr client_;

    void result_callback(rclcpp::Client<my_interfaces::srv::RobotCtrl>::SharedFuture future) {
        auto response = future.get();
        RCLCPP_INFO(this->get_logger(), "指令执行成功：%s", response->message.c_str());
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotController>();

    // 初始发送停止指令
    node->send_request(current_action_key);

    try {
        while (rclcpp::ok()) {
            // 读取键盘输入
            char key = read_keypress_non_blocking();
            if (key) {
                if (key == 'w' && current_action_key != "walk") {
                    current_action_key = "walk";
                    action_switch_flag = true;
                } else if (key == 's' && current_action_key != "squat") {
                    current_action_key = "squat";
                    action_switch_flag = true;
                } else if (key == 'q') {  // 退出
                    RCLCPP_INFO(node->get_logger(), "收到退出指令，程序终止");
                    break;
                } else if (key != 'w' && key != 's') {  // 其他键停止
                    if (current_action_key != "stop") {
                        current_action_key = "stop";
                        action_switch_flag = true;
                    }
                }
            }

            // 发送指令
            if (action_switch_flag) {
                node->send_request(current_action_key);
                action_switch_flag = false;
            }

            // 处理ROS事件
            rclcpp::spin_some(node);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "异常：%s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}