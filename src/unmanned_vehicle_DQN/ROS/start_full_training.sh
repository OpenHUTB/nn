#!/bin/bash
echo "=========================================="
echo "🚗 CARLA DQN完整训练系统启动"
echo "=========================================="

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 检查Carla是否运行
echo "检查Carla是否运行..."
if ! pgrep -f CarlaUE4 > /dev/null; then
    echo "⚠️  Carla未运行，请先启动Carla!"
    echo "    在另一个终端运行: ~/carla/CarlaUE4.sh"
    read -p "按回车继续（如果Carla已启动）或Ctrl+C退出..."
fi

# 进入ROS工作空间
cd ros_ws

# 编译（如果需要）
echo "编译ROS包..."
catkin_make

# 设置环境
source devel/setup.bash

# 检查roscore是否运行
if ! rostopic list > /dev/null 2>&1; then
    echo "启动roscore..."
    gnome-terminal -- bash -c "roscore; exec bash"
    sleep 3
fi

# 启动完整训练系统
echo "启动完整训练系统..."
echo "训练轮次: 100轮（可在launch文件中修改）"
gnome-terminal -- bash -c "source ~/projects/my_carla_project/ros_ws/devel/setup.bash && roslaunch carla_dqn train_full.launch; exec bash"

echo ""
echo "✅ 完整训练系统启动完成！"
echo ""
echo "📊 监控训练状态:"
echo "  1. 查看训练日志: 在当前终端查看"
echo "  2. 查看图像: 自动打开image_view窗口"
echo "  3. 查看曲线: 运行: rqt_plot"
echo ""
echo "📈 查看奖励曲线:"
echo "  rosrun rqt_plot rqt_plot /carla/full_training/reward"
echo ""
echo "🛑 停止系统: 按Ctrl+C关闭所有窗口"
echo "=========================================="
