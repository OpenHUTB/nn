# ROS 1 证据（BLOCKED）

本机未安装 WSL2 与 ROS 1，无法执行 `roscore` / `roslaunch`，因此没有截图。

安装 WSL2 需要管理员权限并重启系统，按任务书要求已向用户申请授权，未擅自执行。

可用的替代验证：

```bash
python -m pytest tests/test_ros_nodes.py -q
```

该测试用 `rospy` 桩模块实例化节点并跑控制周期，可提前发现消息字段错误。
