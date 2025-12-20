#!/usr/bin/env python3
"""
CARLA全局路径规划节点 - 增强版
提供从起始点到随机目标点的路径规划服务，并将规划结果通过ROS消息发布和可视化
新增功能：
- 路径长度估算
- 连接状态监控
- 改进的错误处理
- 性能统计
- 配置参数化
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from carla_global_planner.srv import PlanGlobalPath
from utilities.planner import compute_route_waypoints

import carla
import random
import time  # 新增：用于性能统计
import math  # 新增：用于距离计算
from tf_transformations import quaternion_from_euler


class GlobalPlannerNode(Node):
    """全局路径规划节点类 - 增强版"""

    def __init__(self):
        super().__init__('carla_global_planner_node')

        # 新增：配置参数定义
        self.declare_parameters(
            namespace='',
            parameters=[
                ('carla_host', 'localhost'),  # CARLA服务器地址
                ('carla_port', 2000),  # CARLA服务器端口
                ('carla_timeout', 5.0),  # CARLA客户端超时时间
                ('min_waypoints', 50),  # 最小路点数量
                ('max_planning_attempts', 10),  # 最大规划尝试次数
                ('waypoint_resolution', 1.0),  # 路点分辨率
                ('path_line_width', 0.6),  # 路径可视化线宽
                ('enable_performance_stats', True),  # 是否启用性能统计
                ('publish_goal_marker', True),  # 是否发布目标点标记
            ]
        )

        # 新增：性能统计变量
        self.planning_count = 0
        self.total_planning_time = 0.0
        self.last_planning_time = 0.0

        # 初始化CARLA客户端和世界对象
        self.client = None
        self.world = None
        self.map = None
        self.carla_connected = False  # 新增：连接状态标志
        self._initialize_carla_client()

        # 初始化ROS发布器和服务
        self.marker_pub = self.create_publisher(
            Marker, 'visualization_marker', 10)

        # 新增：发布标记数组（用于同时显示路径和目标点）
        self.marker_array_pub = self.create_publisher(
            MarkerArray, 'visualization_marker_array', 10)

        self.srv = self.create_service(
            PlanGlobalPath, 'plan_to_random_goal', self.plan_path_cb)

        # 新增：定时器用于连接状态检查
        self.connection_check_timer = self.create_timer(
            5.0, self._check_carla_connection)

        # 新增：定时器用于性能统计输出
        if self.get_parameter('enable_performance_stats').value:
            self.stats_timer = self.create_timer(
                30.0, self._publish_performance_stats)

        self.get_logger().info("CARLA全局路径规划服务已启动 [增强版]")

    def _initialize_carla_client(self):
        """初始化CARLA客户端连接 - 增强版错误处理"""
        host = self.get_parameter('carla_host').value
        port = self.get_parameter('carla_port').value
        timeout = self.get_parameter('carla_timeout').value

        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)

            # 新增：测试连接
            version = self.client.get_server_version()
            self.world = self.client.get_world()
            self.map = self.world.get_map()

            self.carla_connected = True  # 新增：设置连接状态
            self.get_logger().info(
                f"成功连接到CARLA服务器 {host}:{port} "
                f"(版本: {version})"
            )

        except Exception as e:
            self.carla_connected = False  # 新增：设置连接状态
            self.get_logger().error(f"CARLA客户端初始化失败: {str(e)}")
            # 不再抛出异常，允许节点继续运行但处于降级模式
            self.get_logger().warn("节点将在降级模式下运行，等待CARLA连接恢复")

    def _check_carla_connection(self):
        """新增：定期检查CARLA连接状态"""
        if not self.carla_connected:
            self.get_logger().info("尝试重新连接CARLA...")
            self._initialize_carla_client()
        else:
            try:
                # 测试连接是否仍然有效
                _ = self.world.get_weather()
            except Exception as e:
                self.get_logger().warning(f"CARLA连接丢失: {str(e)}")
                self.carla_connected = False

    def _publish_performance_stats(self):
        """新增：发布性能统计信息"""
        if self.planning_count > 0:
            avg_time = self.total_planning_time / self.planning_count
            self.get_logger().info(
                f"路径规划性能统计 - "
                f"总次数: {self.planning_count}, "
                f"平均用时: {avg_time:.3f}s, "
                f"上次用时: {self.last_planning_time:.3f}s"
            )

    def plan_path_cb(self, request, response):
        """
        路径规划服务回调函数 - 增强版
        接收起始点，规划到随机目标点的路径并返回
        新增性能统计和更好的错误处理
        """
        start_time = time.time()  # 新增：开始计时

        # 新增：检查CARLA连接状态
        if not self.carla_connected or not self.map:
            self.get_logger().error("CARLA连接未建立或地图未初始化，无法规划路径")
            response.success = False  # 假设服务定义中有success字段
            return response

        try:
            # 将ROS坐标转换为CARLA坐标
            start_location = self._ros_to_carla_location(request.start)
            start_wp = self.map.get_waypoint(start_location)

            if not start_wp:  # 新增：检查起始路点有效性
                self.get_logger().error("无法在起始位置找到有效的路点")
                return response

            # 规划足够长的路径
            min_waypoints = self.get_parameter('min_waypoints').value
            max_attempts = self.get_parameter('max_planning_attempts').value

            route, goal_wp = self._get_valid_route(start_wp, min_waypoints, max_attempts)
            if not route:
                self.get_logger().error("无法生成有效的路径")
                return response

            # 新增：计算路径总长度
            total_distance = self._calculate_path_distance(route)

            # 构建路径消息并可视化
            path_msg = self._build_path_message(route)
            self._visualize_path_and_goal(path_msg, goal_wp)  # 修改：同时可视化路径和目标点

            response.path = path_msg

            # 新增：更新性能统计
            planning_time = time.time() - start_time
            self.last_planning_time = planning_time
            self.total_planning_time += planning_time
            self.planning_count += 1

            self.get_logger().info(
                f"成功规划路径 - "
                f"路点数: {len(route)}, "
                f"总距离: {total_distance:.2f}m, "
                f"用时: {planning_time:.3f}s"
            )
            return response

        except Exception as e:
            self.get_logger().error(f"路径规划过程中发生错误: {str(e)}")
            return response

    def _ros_to_carla_location(self, odom_msg):
        """将ROS里程计消息中的位置转换为CARLA坐标"""
        return carla.Location(
            x=odom_msg.pose.pose.position.x,
            y=-odom_msg.pose.pose.position.y,  # CARLA与ROS的Y轴方向相反
            z=odom_msg.pose.pose.position.z
        )

    def _get_valid_route(self, start_wp, min_waypoints=50, max_attempts=10):
        """
        获取有效的路径 - 增强版
        返回路径和目标路点元组
        尝试多次生成路径，直到满足最小路点数量要求或达到最大尝试次数
        """
        best_route = None
        best_goal = None
        best_length = 0

        for attempt in range(max_attempts):
            # 选择随机可行的目标点
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                self.get_logger().warning("未找到可用的生成点")
                continue  # 修改：继续尝试而不是返回None

            goal_transform = random.choice(spawn_points)
            goal_wp = self.map.get_waypoint(goal_transform.location)

            # 新增：确保目标点与起始点不同
            distance_to_goal = start_wp.transform.location.distance(goal_wp.transform.location)
            if distance_to_goal < 10.0:  # 如果距离太近，跳过此次尝试
                continue

            # 计算路径
            self.get_logger().debug(  # 修改：改为debug级别以减少日志噪音
                f"路径规划尝试 {attempt + 1}/{max_attempts}: "
                f"从({start_wp.transform.location.x:.2f}, {start_wp.transform.location.y:.2f}) "
                f"到({goal_wp.transform.location.x:.2f}, {goal_wp.transform.location.y:.2f}) "
                f"直线距离: {distance_to_goal:.2f}m"
            )

            resolution = self.get_parameter('waypoint_resolution').value
            route = compute_route_waypoints(
                self.map, start_wp, goal_wp, resolution=resolution)

            # 新增：保存最佳路径（最长的）
            if len(route) > best_length:
                best_route = route
                best_goal = goal_wp
                best_length = len(route)

            if len(route) >= min_waypoints:
                return route, goal_wp  # 找到满足要求的路径，直接返回

        # 返回最佳路径（即使不满足最小长度要求）
        if best_route:
            self.get_logger().warning(
                f"达到最大尝试次数({max_attempts})，返回最长路径 "
                f"(长度: {best_length}, 要求: {min_waypoints})"
            )
            return best_route, best_goal

        return None, None

    def _calculate_path_distance(self, route):
        """新增：计算路径总长度"""
        if len(route) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(route) - 1):
            current_wp = route[i][0]
            next_wp = route[i + 1][0]

            # 计算两个路点之间的欧几里得距离
            dx = next_wp.transform.location.x - current_wp.transform.location.x
            dy = next_wp.transform.location.y - current_wp.transform.location.y
            dz = next_wp.transform.location.z - current_wp.transform.location.z

            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            total_distance += distance

        return total_distance

    def _build_path_message(self, route):
        """将CARLA路径转换为ROS Path消息"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for waypoint, _ in route:
            pose = PoseStamped()
            pose.header = path_msg.header

            # 设置位置（转换Y轴方向）
            pose.pose.position.x = waypoint.transform.location.x
            pose.pose.position.y = -waypoint.transform.location.y
            pose.pose.position.z = waypoint.transform.location.z

            # 转换旋转角度为四元数
            yaw_rad = waypoint.transform.rotation.yaw * (3.1415 / 180.0)
            q = quaternion_from_euler(0, 0, -yaw_rad)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            path_msg.poses.append(pose)

        return path_msg

    def _visualize_path_and_goal(self, path_msg, goal_wp):
        """修改：可视化路径和目标点"""
        # 原有的单独路径可视化保持不变
        self._visualize_path(path_msg)

        # 新增：如果启用了目标点标记，则发布目标点
        if self.get_parameter('publish_goal_marker').value and goal_wp:
            self._visualize_goal_point(path_msg.header, goal_wp)

    def _visualize_path(self, path_msg):
        """可视化路径，先删除旧标记再发布新标记"""
        # 删除旧标记
        delete_marker = self._create_path_marker(
            path_msg.header, action=Marker.DELETE)
        self.marker_pub.publish(delete_marker)

        # 发布新标记
        line_width = self.get_parameter('path_line_width').value
        new_marker = self._create_path_marker(
            path_msg.header,
            action=Marker.ADD,
            points=[pose.pose.position for pose in path_msg.poses],
            line_width=line_width
        )
        self.marker_pub.publish(new_marker)

    def _visualize_goal_point(self, header, goal_wp):
        """新增：可视化目标点"""
        goal_marker = Marker()
        goal_marker.header = header
        goal_marker.ns = "carla_goal"
        goal_marker.id = 1  # 不同的ID避免与路径标记冲突
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD

        # 设置目标点位置
        goal_marker.pose.position.x = goal_wp.transform.location.x
        goal_marker.pose.position.y = -goal_wp.transform.location.y
        goal_marker.pose.position.z = goal_wp.transform.location.z + 1.0  # 稍微提高以便可见

        # 设置目标点外观（红色球体）
        goal_marker.scale.x = 2.0
        goal_marker.scale.y = 2.0
        goal_marker.scale.z = 2.0
        goal_marker.color.a = 0.8
        goal_marker.color.r = 1.0  # 红色
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0

        self.marker_pub.publish(goal_marker)

    def _create_path_marker(self, header, action=Marker.ADD, points=None, line_width=0.6):
        """创建路径可视化标记 - 增强版"""
        marker = Marker()
        marker.header = header
        marker.ns = "carla_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = action

        # 视觉外观设置
        marker.scale.x = line_width  # 修改：使用参数化线宽
        marker.color.a = 1.0  # 透明度
        marker.color.r = 0.0
        marker.color.g = 1.0  # 绿色
        marker.color.b = 0.0

        # 新增：设置生存时间，自动清理旧标记
        marker.lifetime.sec = 60  # 60秒后自动删除

        # 添加点集（仅用于ADD动作）
        if points and action == Marker.ADD:
            marker.points = points

        return marker

    def __del__(self):
        """新增：析构函数，用于清理资源"""
        if hasattr(self, 'connection_check_timer'):
            self.connection_check_timer.cancel()
        if hasattr(self, 'stats_timer'):
            self.stats_timer.cancel()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    try:
        node = GlobalPlannerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("接收到中断信号，正在关闭节点...")
    except Exception as e:
        print(f"节点运行失败: {str(e)}")
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass  # 忽略关闭时的异常


if __name__ == '__main__':
    main()