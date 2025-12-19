#!/home/pan-j-l/my_ros_project/venv_py37/bin/python3
# 适配你的虚拟环境Python解释器路径（关键！避免环境依赖问题）
import rospy
import torch
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu, LaserScan
# 导入自定义感知/决策模块（路径适配你的scripts/models目录）
from models.perception_module import PerceptionModule
from models.decision_module import DecisionModule

class PerceptionDecisionNode:
    def __init__(self):
        # 节点初始化
        rospy.init_node('perception_decision_node', anonymous=True)
        self.node_name = rospy.get_name()
        rospy.loginfo(f"[{self.node_name}] 节点初始化成功！")

     
        image_shape_str = rospy.get_param('~image_shape', "[3, 128, 128]")  # 读字符串
        self.image_shape = tuple(map(int, image_shape_str.strip('[]').replace(' ', '').split(',')))  # 核心转换

        # 2. 处理数值型参数（字符串→int/float）
        self.feature_dim = int(rospy.get_param('~feature_dim', 128))  # 转整数
        self.cmd_dim = int(rospy.get_param('~cmd_dim', 2))            # 转整数
        self.timer_freq = float(rospy.get_param('~timer_freq', 10.0)) # 转浮点数
        
        # 打印加载的参数（日志格式和你预期的一致）
        rospy.loginfo(f"[{self.node_name}] 加载参数完成：")
        rospy.loginfo(f"  - 图像形状：{self.image_shape}")
        rospy.loginfo(f"  - 特征维度：{self.feature_dim}")
        rospy.loginfo(f"  - 指令维度：{self.cmd_dim}")
        rospy.loginfo(f"  - 定时器频率：{self.timer_freq}Hz")

        # 初始化感知/决策模块
        try:
            self.perception = PerceptionModule(
                image_input_shape=self.image_shape,
                output_feature_dim=self.feature_dim
            )
            self.decision = DecisionModule(
                input_feature_dim=self.feature_dim,
                output_cmd_dim=self.cmd_dim
            )
            rospy.loginfo(f"[{self.node_name}] 感知/决策模块初始化成功！")
        except Exception as e:
            rospy.logfatal(f"[{self.node_name}] 模块初始化失败：{str(e)}")
            exit(1)  # 模块初始化失败直接退出

        # 初始化控制指令发布者（发布到/cmd_vel话题）
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.loginfo(f"[{self.node_name}] 发布者初始化成功：/cmd_vel话题")

        # 初始化定时器（按设定频率执行数据处理逻辑）
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.timer_freq),
            self.timer_callback
        )
        rospy.loginfo(f"[{self.node_name}] 定时器初始化成功：{self.timer_freq}Hz")
        rospy.loginfo(f"[{self.node_name}] 节点开始运行（按Ctrl+C退出）...")

    def timer_callback(self, event):
        """定时器回调：核心逻辑（模拟数据→感知→决策→发布指令）"""
        # ========== 跳过传感器等待：直接生成模拟数据 ==========
        # 模拟IMU数据（1个batch，6维：线加速度x/y/z + 角速度x/y/z）
        self.imu_data = torch.randn(1, 6)
        # 模拟图像数据（1个batch，对应image_shape参数）
        self.image_data = torch.randn(1, *self.image_shape)
        # 模拟激光雷达数据（1个batch，360个扫描点）
        self.lidar_data = torch.randn(1, 360)

        try:
            # 1. 感知模块：融合多传感器数据生成特征
            fused_feature = self.perception.process(
                self.imu_data,
                self.image_data,
                self.lidar_data
            )

            # 2. 决策模块：根据融合特征生成控制指令
            control_cmd = self.decision.get_control_cmd(fused_feature)

            # 3. 封装并发布ROS控制指令（Twist消息）
            twist_msg = Twist()
            twist_msg.linear.x = control_cmd['linear_x']  # 线速度x
            twist_msg.angular.z = control_cmd['angular_z'] # 角速度z
            self.cmd_pub.publish(twist_msg)

            # 4. 打印日志（格式和你预期的完全一致）
            rospy.loginfo(
                f"[{self.node_name}] 生成控制指令：线速度x={control_cmd['linear_x']:.2f}m/s，"
                f"角速度z={control_cmd['angular_z']:.2f}rad/s"
            )

        except Exception as e:
            # 异常捕获：避免单个错误导致节点崩溃
            rospy.logerr(f"[{self.node_name}] 指令生成失败：{str(e)}")

if __name__ == '__main__':
    try:
        # 创建节点实例并运行
        node = PerceptionDecisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        # 捕获Ctrl+C中断，友好退出
        rospy.loginfo(f"[{rospy.get_name()}] 节点被中断，正常退出！")
    except Exception as e:
        # 捕获其他致命错误
        rospy.logfatal(f"节点启动失败：{str(e)}")
        exit(1)