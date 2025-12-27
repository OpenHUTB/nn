#include "rclcpp/rclcpp.hpp"
#include "my_interfaces/srv/robot_ctrl.hpp"
#include "my_interfaces/msg/robot_state.hpp"
#include <mujoco/mujoco.h>
#include <fstream>
#include <string>
#include <thread>

// 全局变量
std::string current_action = "stop";

class MujocoRos2Sim : public rclcpp::Node {
public:
    MujocoRos2Sim() : Node("mujoco_ros2_sim") {
        RCLCPP_INFO(this->get_logger(), "节点已启动：mujoco_ros2_sim!");

        // 加载模型
        char error[1000];
        model_ = mj_loadXML("RobotH.xml", nullptr, error, 1000);
        if (!model_) {
            RCLCPP_ERROR(this->get_logger(), "模型加载失败: %s", error);
            exit(1);
        }
        data_ = mj_makeData(model_);

        // 加载轨迹数据（这里简化处理，实际需要根据npz格式解析）
        load_trajectory_data();

        // 创建viewer
        viewer_ = mjViewerCreate();
        mjv_defaultCamera(&cam_);
        mjv_defaultOption(&opt_);
        mjr_defaultContext(&con_);
        mjv_makeScene(model_, &scene_, 1000);
        mjr_makeContext(model_, &con_, mjFONTSCALE_150);
        mjViewerSetup(viewer_, &scene_, &cam_, &opt_, &con_);

        // 创建服务
        service_ = this->create_service<my_interfaces::srv::RobotCtrl>(
            "robot_control",
            std::bind(&MujocoRos2Sim::control_callback, this, std::placeholders::_1, std::placeholders::_2)
        );

        // 创建发布者
        publisher_ = this->create_publisher<my_interfaces::msg::RobotState>("robot_state", 10);

        // 启动模拟线程
        sim_thread_ = std::thread(&MujocoRos2Sim::run_sim, this);
    }

    ~MujocoRos2Sim() {
        if (sim_thread_.joinable()) {
            sim_thread_.join();
        }
        mjViewerClose(viewer_);
        mjr_freeContext(&con_);
        mjv_freeScene(&scene_);
        mj_deleteData(data_);
        mj_deleteModel(model_);
    }

private:
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    mjViewer* viewer_ = nullptr;
    mjvScene scene_;
    mjvCamera cam_;
    mjvOption opt_;
    mjrContext con_;
    rclcpp::Service<my_interfaces::srv::RobotCtrl>::SharedPtr service_;
    rclcpp::Publisher<my_interfaces::msg::RobotState>::SharedPtr publisher_;
    std::thread sim_thread_;

    // 轨迹数据（示例）
    std::vector<double> walk_qpos, walk_qvel;
    std::vector<double> squat_qpos, squat_qvel;

    void load_trajectory_data() {
        // 实际应用中需要解析npz文件
        // 这里仅做示例初始化
        walk_qpos.push_back(0.0);
        walk_qvel.push_back(0.1);
        squat_qpos.push_back(-0.5);
        squat_qvel.push_back(0.0);
    }

    void control_callback(
        const std::shared_ptr<my_interfaces::srv::RobotCtrl::Request> request,
        std::shared_ptr<my_interfaces::srv::RobotCtrl::Response> response
    ) {
        RCLCPP_INFO(this->get_logger(), "收到指令: %s", request->command.c_str());
        current_action = request->command;
        response->result = true;
        response->message = "指令已接收";
    }

    void publish_status() {
        auto msg = my_interfaces::msg::RobotState();
        msg.action = current_action;
        publisher_->publish(msg);
    }

    void run_sim() {
        while (rclcpp::ok()) {
            // 控制动作
            if (current_action == "squat" && !squat_qpos.empty()) {
                data_->qpos[0] = squat_qpos[0];
                data_->qvel[0] = squat_qvel[0];
            } else if (current_action == "walk" && !walk_qpos.empty()) {
                data_->qpos[0] = walk_qpos[0];
                data_->qvel[0] = walk_qvel[0];
            } else if (current_action == "stop") {
                data_->qpos[0] = 0;
                data_->qvel[0] = 0;
            }

            // 模拟一步
            mj_step(model_, data_);

            // 渲染
            mjv_updateScene(model_, data_, &opt_, nullptr, &cam_, mjCAT_ALL, &scene_);
            mjViewerRender(viewer_, &scene_, &con_);

            // 发布状态
            publish_status();

            // 小延迟
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MujocoRos2Sim>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}