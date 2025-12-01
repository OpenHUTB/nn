#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "plumbing_pub_sub/runningAction.h"

typedef actionlib::SimpleActionServer<plumbing_pub_sub::runningAction> Server;

// 收到action的goal后调用该回调函数
void execute(const plumbing_pub_sub::runningGoalConstPtr& goal, Server* as)
{
    ros::Rate r(1);
    plumbing_pub_sub::runningFeedback feedback;

    ROS_INFO("The robot %d is arriaved.", goal->running_id);

    // 假设洗盘子的进度，并且按照1hz频率发布进度feedback
    for(int i=1; i<=10; i++)
    {
        feedback.percent_complete = i * 10;
        as->publishFeedback(feedback);
        r.sleep();
    }

    // 当action完成后，向客户端返回结果
    ROS_INFO("run %d is ok.", goal->running_id);
    as->setSucceeded();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "running_server");
    ros::NodeHandle n;

    // 定义一个服务器
    Server server(n, "running", boost::bind(&execute, _1, &server), false);
    
    // 服务器开始运行
    server.start();

    ros::spin();

    return 0;
}
