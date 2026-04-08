#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "plumbing_pub_sub/Carry_bricksAction.h"

typedef actionlib::SimpleActionServer<plumbing_pub_sub::Carry_bricksAction> Server;

// 收到action的goal后调用该回调函数
void execute(const plumbing_pub_sub::Carry_bricksGoalConstPtr& goal, Server* as)
{
    ros::Rate r(1);
    plumbing_pub_sub::Carry_bricksFeedback feedback;

    ROS_INFO("The robot %d is working.", goal->Carry_id);

    // 假设洗盘子的进度，并且按照1hz频率发布进度feedback
    for(int i=1; i<=10; i++)
    {
        feedback.percent_complete = i * 10;
        as->publishFeedback(feedback);
        r.sleep();
    }

    // 当action完成后，向客户端返回结果
    ROS_INFO("Carry_bricks %d finish working.", goal->Carry_id);
    as->setSucceeded();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "Carry_bricks_server");
    ros::NodeHandle n;

    // 定义一个服务器
    Server server(n, "Carry_bricks", boost::bind(&execute, _1, &server), false);
    
    // 服务器开始运行
    server.start();

    ros::spin();

    return 0;
}
