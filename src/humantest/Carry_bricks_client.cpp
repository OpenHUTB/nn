#include <actionlib/client/simple_action_client.h>
#include "plumbing_pub_sub/Carry_bricksAction.h"

typedef actionlib::SimpleActionClient<plumbing_pub_sub::Carry_bricksAction> Client;

// 当action完成后会调用该回调函数一次
void doneCb(const actionlib::SimpleClientGoalState& state,
        const plumbing_pub_sub::Carry_bricksResultConstPtr& result)
{
    ROS_INFO("Yay! The bricks are all carried");
    ros::shutdown();
}

// 当action激活后会调用该回调函数一次
void activeCb()
{
    ROS_INFO("Goal just went active");
}

// 收到feedback后调用该回调函数
void feedbackCb(const plumbing_pub_sub::Carry_bricksFeedbackConstPtr& feedback)
{
    ROS_INFO(" percent_complete : %f ", feedback->percent_complete);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "Carry_bricks_client");

    // 定义一个客户端
    Client client("Carry_bricks", true);

    // 等待服务器端
    ROS_INFO("Waiting for action server to start.");
    client.waitForServer();
    ROS_INFO("Action server started, sending goal.");

    // 创建一个action的goal
    plumbing_pub_sub::Carry_bricksGoal goal;
    goal.Carry_id = 1;

    // 发送action的goal给服务器端，并且设置回调函数
    client.sendGoal(goal,  &doneCb, &activeCb, &feedbackCb);

    ros::spin();

    return 0;
}
