# 作业选题：自身位置到 CARLA 地图中随机目标点的全局规划 - 模拟用户预订前往目的地的行程



## 环境配置
* 平台：Ubuntu 20.04
* 软件：Python 3.8 
## 运行大致流程
1.安装ros2  版本：foxy（若有ros1记得避免环境变量冲突）
2.在工作根目录下激活ros2环境  source /opt/ros/foxy/setup.bash
3.安装colcon
3.编译节点文件  # 在工作空间根目录，执行以下命令（确保已经source过环境）
   colcon build
4.启动节点  ros2 run carla_global_planner carla_global_planner_node.py
 如果脚本卡住不动且无报错：说明节点正在运行
（报错缺少模块需安装carla，需较大内存）
5.新开终端  启动 CARLA 仿真，让节点完成功能闭环
cd ~/carla/CARLA_0.9.15  # 替换为你的 CARLA 实际路径
./CarlaUE4.sh -quality-level=Low -RenderOffScreen


## 参考


