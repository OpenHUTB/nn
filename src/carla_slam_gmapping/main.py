#!/usr/bin/env python3
"""
CARLA全局路径规划节点 - 增强版
提供从起始点到随机目标点的路径规划服务，并将规划结果通过ROS消息发布和可视化
新增slam算法相关功能
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from carla_global_planner.srv import PlanGlobalPath
from utilities.planner import compute_route_waypoints
from nav_msgs.msg import Odometry
import carla
import random
import time  # 新增：用于性能统计
import math  # 新增：用于距离计算
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf_transformations import quaternion_from_euler
from typing import Optional, Tuple, List, Any


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
            # SLAM功能相关
            self.laser_sub = self.create_subscription(
                LaserScan, '/carla/ego_vehicle/lidar', self.laser_callback, 10)

            self.map_pub = self.create_publisher(
                OccupancyGrid, '/map', 10)

            # TF broadcaster for SLAM
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            # SLAM相关发布器
            self.odom_pub = self.create_publisher(Odometry, '/carla/ego_vehicle/odometry', 10)
            self.odom_timer = self.create_timer(0.1, self.publish_odom)  # 10Hz发布里程计
            self.vehicle = None  # 存储车辆actor
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
    
        不足：原方法过长，包含多个职责，难以维护和测试。
        优化：拆分为子方法，提高可读性和可维护性。
        """
        start_time = time.time()

        if not self._validate_connection(response):
            return response

        try:
            start_wp = self._get_start_waypoint(request.start)
            if not start_wp:
                self.get_logger().error("无法在起始位置找到有效的路点")
                return response

            route, goal_wp = self._plan_route(start_wp)
            if not route:
                self.get_logger().error("无法生成有效的路径")
                return response

            path_msg = self._process_route(route, goal_wp)
            response.path = path_msg

            self._update_performance_stats(time.time() - start_time, len(route), self._calculate_path_distance(route))

            return response

        except Exception as e:
            self.get_logger().error(f"路径规划过程中发生错误: {str(e)}")
            return response

    def _validate_connection(self, response) -> bool:
        """验证CARLA连接状态"""
        if not self.carla_connected or not self.map:
            self.get_logger().error("CARLA连接未建立或地图未初始化，无法规划路径")
            response.success = False
            return False
        return True

    def _get_start_waypoint(self, start_msg) -> Optional[carla.Waypoint]:
        """获取起始路点"""
        start_location = self._ros_to_carla_location(start_msg)
        return self.map.get_waypoint(start_location)

    def _plan_route(self, start_wp: carla.Waypoint) -> Tuple[Optional[List], Optional[carla.Waypoint]]:
        """规划路径"""
        min_waypoints = self.get_parameter('min_waypoints').value
        max_attempts = self.get_parameter('max_planning_attempts').value
        return self._get_valid_route(start_wp, min_waypoints, max_attempts)

    def _process_route(self, route: List, goal_wp: Optional[carla.Waypoint]) -> Path:
        """处理路径：构建消息并可视化"""
        path_msg = self._build_path_message(route)
        self._visualize_path_and_goal(path_msg, goal_wp)
        return path_msg
    def laser_callback(self, msg: LaserScan):
        """
        处理激光雷达数据，用于SLAM
        """
        # 这里实现激光雷达数据的处理逻辑
        # 可以转发给gmapping或其他SLAM算法处理
        pass
    def publish_tf_transform(self, position, orientation, frame_id, child_frame_id):
        """
        发布TF变换
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        
        t.transform.rotation.x = orientation[0]
        t.transform.rotation.y = orientation[1]
        t.transform.rotation.z = orientation[2]
        t.transform.rotation.w = orientation[3]
        
        self.tf_broadcaster.sendTransform(t)
    def publish_odom(self):
        """发布里程计数据用于SLAM"""
        if not self.vehicle:
            # 获取ego车辆
            actors = self.world.get_actors()
            for actor in actors:
                if actor.type_id.startswith('vehicle') and 'ego' in actor.attributes.get('role_name', ''):
                    self.vehicle = actor
                    break
        
        if self.vehicle:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            
            # 创建odom消息
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'ego_vehicle'
            
            # 位置（转换CARLA坐标系到ROS）
            odom.pose.pose.position.x = transform.location.x
            odom.pose.pose.position.y = -transform.location.y
            odom.pose.pose.position.z = transform.location.z
            
            # 方向（转换为四元数）
            yaw_rad = math.radians(transform.rotation.yaw)
            q = quaternion_from_euler(0, 0, yaw_rad)
            odom.pose.pose.orientation.x = q[0]
            odom.pose.pose.orientation.y = q[1]
            odom.pose.pose.orientation.z = q[2]
            odom.pose.pose.orientation.w = q[3]
            
            # 速度
            odom.twist.twist.linear.x = velocity.x
            odom.twist.twist.linear.y = -velocity.y
            odom.twist.twist.linear.z = velocity.z
            
            self.odom_pub.publish(odom)
            
            # 发布TF变换 odom -> ego_vehicle
            self.publish_tf_transform(
                [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z],
                [q[0], q[1], q[2], q[3]],
                'odom',
                'ego_vehicle'
            )

    def _update_performance_stats(self, planning_time: float, waypoint_count: int, total_distance: float):
        """更新性能统计"""
        self.last_planning_time = planning_time
        self.total_planning_time += planning_time
        self.planning_count += 1
        self.get_logger().info(
            f"成功规划路径 - "
            f"路点数: {waypoint_count}, "
            f"总距离: {total_distance:.2f}m, "
            f"用时: {planning_time:.3f}s"
        )

    def _ros_to_carla_location(self, odom_msg):
        """将ROS里程计消息中的位置转换为CARLA坐标"""
        return carla.Location(
            x=odom_msg.pose.pose.position.x,
            y=-odom_msg.pose.pose.position.y,  # CARLA与ROS的Y轴方向相反
            z=odom_msg.pose.pose.position.z
        )

    def _get_valid_route(self, start_wp, min_waypoints=50, max_attempts=10):
        """
        获取有效的路径 -
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

        if best_route:
            self.get_logger().warning(
                f"达到最大尝试次数({max_attempts})，返回最长路径 "
                f"(长度: {best_length}, 要求: {min_waypoints})"
            )
            return best_route, best_goal

        return None, None

    def _calculate_path_distance(self, route):
        """计算路径总长度 - 优化版"""
        if len(route) < 2:
            return 0.0
        return sum(
            math.sqrt(
                (next_wp.transform.location.x - current_wp.transform.location.x) ** 2 +
                (next_wp.transform.location.y - current_wp.transform.location.y) ** 2 +
                (next_wp.transform.location.z - current_wp.transform.location.z) ** 2
            )
            for (current_wp, _), (next_wp, _) in zip(route, route[1:])
        )

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

   
        marker.lifetime.sec = 60  # 60秒后自动删除

    
        if points and action == Marker.ADD:
            marker.points = points

        return marker

    def __del__(self):
        """新增：析构函数，用于清理资源"""
        if hasattr(self, 'connection_check_timer'):
            self.connection_check_timer.cancel()
        if hasattr(self, 'stats_timer'):
            self.stats_timer.cancel()
        if hasattr(self, 'odom_timer'):
            self.odom_timer.cancel()


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