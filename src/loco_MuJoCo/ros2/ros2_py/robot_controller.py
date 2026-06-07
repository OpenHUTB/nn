import sys
import select
import rclpy
from rclpy.node import Node
from my_interfaces.srv import RobotCtrl

# 全局变量（提前初始化，避免未绑定）
current_action_key = 'stop'
action_switch_flag = False


class RobotController(Node):
    def __init__(self, name):
        super().__init__(name)
        # 创建服务客户端，连接机器人控制服务
        self.client = self.create_client(RobotCtrl, 'robot_control')
        # 等待服务上线
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务 "robot_control" 启动...')
        self.get_logger().info("操作提示：w=行走 | s=下蹲 | q=退出 | 其他键=停止")

    def result_callback(self, future):
        """服务调用结果的回调函数"""

        response = future.result()
        self.get_logger().info(f'指令执行成功：{response.message}')


    def send_request(self, action):
        """发送控制指令到服务端"""

        request = RobotCtrl.Request()
        request.command = action  # 传递动作指令
        self.client.call_async(request).add_done_callback(self.result_callback)
        self.get_logger().info(f'已发送指令：{action}')



def read_keypress_non_blocking():
    """非阻塞读取终端按键"""
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        key = sys.stdin.read(1)
        return key.strip()
    return None


def main(args=None):
    # 关键：声明使用全局变量（修复核心错误）
    global current_action_key, action_switch_flag
    
    rclpy.init(args=args)
    robot_controller = RobotController("robot_controller")

    # 初始发送停止指令（此时current_action_key已绑定全局值'stop'）
    robot_controller.send_request(current_action_key)

    try:
        # 主循环：非阻塞读取按键 + 处理ROS事件
        while rclpy.ok():
            # 1. 读取终端按键
            key = read_keypress_non_blocking()
            if key:
                # 按键逻辑
                if key == 'w' and current_action_key != "walk":
                    current_action_key = "walk"
                    action_switch_flag = True
                elif key == 's' and current_action_key != "squat":
                    current_action_key = "squat"
                    action_switch_flag = True
                elif key == 'q':  # q键退出
                    robot_controller.get_logger().info("收到退出指令，程序终止")
                    break
                elif key not in ['w', 's']:  # 其他键停止
                    if current_action_key != "stop":
                        current_action_key = "stop"
                        action_switch_flag = True

            # 2. 检测动作切换并发送指令
            if action_switch_flag:
                robot_controller.send_request(current_action_key)
                action_switch_flag = False  # 重置标记

            # 3. 非阻塞处理ROS回调
            rclpy.spin_once(robot_controller, timeout_sec=0.01)

    except KeyboardInterrupt:
        robot_controller.get_logger().info("用户强制退出")
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()
        print("程序已退出")


if __name__ == '__main__':
    main()