#!/usr/bin/env python3
"""
CARLA全局路径规划节点 - Docker兼容增强版
针对容器化部署优化：
- 网络参数动态配置（支持Docker环境变量注入）
- 资源占用控制（定时器周期可配置）
- 结构化日志输出（便于容器日志收集）
- 完善的信号处理（支持容器优雅退出）
- 依赖版本兼容处理
- 配置文件支持（可选从文件加载参数）
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from carla_global_planner.srv import PlanGlobalPath
from utilities.planner import compute_route_waypoints

import carla
import random
import time
import math
import signal
import os
import psutil
from tf_transformations import quaternion_from_euler
from ament_index_python.packages import get_package_share_directory
import yaml


class GlobalPlannerNode(Node):
    """全局路径规划节点类 - Docker兼容版"""

    def __init__(self):
        super().__init__('carla_global_planner_node')

        # 加载配置参数（环境变量 > 配置文件 > 默认值）
        self._load_config()

        # 性能与资源监控变量
        self.planning_count = 0
        self.total_planning_time = 0.0
        self.last_planning_time = 0.0
        self.avg_memory_usage = 0.0

        # CARLA客户端核心变量
        self.client = None
        self.world = None
        self.map = None
        self.carla_connected = False
        self.reconnect_interval = self.get_parameter('reconnect_interval').value

        # 初始化CARLA客户端（带重试机制）
        self._initialize_carla_client(retries=3)

        # QoS配置（适配容器网络）
        qos_profile = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ROS发布器
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', qos_profile)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', qos_profile)
        self.path_pub = self.create_publisher(Path, 'planned_path', qos_profile)

        # ROS服务
        self.srv = self.create_service(
            PlanGlobalPath, 
            'plan_to_random_goal', 
            self.plan_path_cb,
            qos_profile=QoSProfile(depth=1)
        )

        # 定时器
        self.connection_check_timer = self.create_timer(
            self.reconnect_interval,
            self._check_carla_connection
        )

        if self.get_parameter('enable_performance_stats').value:
            self.stats_timer = self.create_timer(
                self.get_parameter('stats_publish_interval').value,
                self._publish_performance_stats
            )

        self.resource_monitor_timer = self.create_timer(
            10.0,
            self._monitor_resource_usage
        )

        # 信号处理（容器优雅退出）
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigint)

        self.get_logger().info(
            f"CARLA全局路径规划服务[Docker版]启动成功\n"
            f"CARLA服务器: {self.get_parameter('carla_host').value}:{self.get_parameter('carla_port').value}\n"
            f"性能统计: {'启用' if self.get_parameter('enable_performance_stats').value else '禁用'}\n"
            f"资源监控: 启用（内存阈值: {self.get_parameter('max_memory_threshold_mb').value}MB）"
        )

    def _load_config(self):
        """多源参数加载（环境变量 > 配置文件 > 默认值）"""
        # 基础参数默认值
        default_params = [
            ('carla_host', 'carla'),  # Docker环境默认使用服务名
            ('carla_port', 2000),
            ('carla_timeout', 10.0),  # 容器网络超时延长
            ('min_waypoints', 50),
            ('max_planning_attempts', 10),
            ('waypoint_resolution', 1.0),
            ('path_line_width', 0.6),
            ('enable_performance_stats', True),
            ('publish_goal_marker', True),
            ('reconnect_interval', 5.0),
            ('stats_publish_interval', 30.0),
            ('max_memory_threshold_mb', 512),
            ('config_file_path', '')
        ]

        # 从环境变量读取参数
        env_params = {
            'carla_host': os.getenv('CARLA_HOST'),
            'carla_port': os.getenv('CARLA_PORT'),
            'carla_timeout': os.getenv('CARLA_TIMEOUT'),
            'reconnect_interval': os.getenv('RECONNECT_INTERVAL'),
            'max_memory_threshold_mb': os.getenv('MAX_MEMORY_THRESHOLD_MB')
        }

        # 声明参数
        for param_name, default_val in default_params:
            if env_params.get(param_name) is not None:
                val = env_params[param_name]
                # 类型转换
                if isinstance(default_val, int):
                    val = int(val)
                elif isinstance(default_val, float):
                    val = float(val)
                elif isinstance(default_val, bool):
                    val = val.lower() in ['true', '1', 'yes']
                self.declare_parameter(param_name, val)
            else:
                self.declare_parameter(param_name, default_val)

        # 加载配置文件
        config_path = self.get_parameter('config_file_path').value
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                for param_name, val in config.items():
                    if self.has_parameter(param_name):
                        self.set_parameters([rclpy.parameter.Parameter(
                            param_name, rclpy.Parameter.Type.from_parameter_value(val))])
                        self.get_logger().info(f"从配置文件加载参数: {param_name} = {val}")
            except Exception as e:
                self.get_logger().warning(f"配置文件加载失败: {str(e)}")

    def _initialize_carla_client(self, retries=3):
        """CARLA客户端初始化（带重试机制）"""
        host = self.get_parameter('carla_host').value
        port = self.get_parameter('carla_port').value
        timeout = self.get_parameter('carla_timeout').value

        for attempt in range(retries):
            try:
                self.client = carla.Client(host, port)
                self.client.set_timeout(timeout)
                time.sleep(1.0)  # 容器网络延迟等待

                version = self.client.get_server_version()
                self.world = self.client.get_world()
                self.map = self.world.get_map()

                self.carla_connected = True
                self.get_logger().info(
                    f"成功连接CARLA服务器（第{attempt+1}/{retries}次尝试）\n"
                    f"服务器版本: {version}\n"
                    f"地图: {self.map.name}"
                )
                return

            except Exception as e:
                self.carla_connected = False
                if attempt < retries - 1:
                    self.get_logger().warning(
                        f"第{attempt+1}次连接失败（将重试）: {str(e)}"
                    )
                    time.sleep(self.reconnect_interval)
                else:
                    self.get_logger().error(
                        f"连接CARLA失败（已达最大重试次数）: {str(e)}"
                    )

    def _check_carla_connection(self):
        """CARLA连接检查"""
        if not self.carla_connected:
            self.get_logger().info("尝试重新连接CARLA服务器...")
            self._initialize_carla_client(retries=1)
        else:
            try:
                _ = self.world.get_weather()
            except Exception as e:
                self.get_logger().warning(f"CARLA连接丢失: {str(e)}")
                self.carla_connected = False

    def _monitor_resource_usage(self):
        """资源监控（内存占用）"""
        try:
            process = psutil.Process(os.getpid())
            mem_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.avg_memory_usage = (self.avg_memory_usage * 0.9) + (mem_usage_mb * 0.1)

            max_mem = self.get_parameter('max_memory_threshold_mb').value
            if mem_usage_mb > max_mem:
                self.get_logger().warning(
                    f"内存占用超过阈值！当前: {mem_usage_mb:.1f}MB, 阈值: {max_mem}MB"
                )

        except Exception as e:
            self.get_logger().error(f"资源监控失败: {str(e)}")

    def _publish_performance_stats(self):
        """性能统计输出"""
        if self.planning_count == 0:
            return

        avg_time = self.total_planning_time / self.planning_count
        stats_msg = (
            f"[PERF_STATS] "
            f"total_plans={self.planning_count}, "
            f"avg_time={avg_time:.3f}s, "
            f"last_time={self.last_planning_time:.3f}s, "
            f"avg_mem={self.avg_memory_usage:.1f}MB, "
            f"carla_connected={self.carla_connected}"
        )
        self.get_logger().info(stats_msg)

    def _handle_sigterm(self, signum, frame):
        """处理SIGTERM信号"""
        self.get_logger().info("接收到容器停止信号（SIGTERM），正在清理资源...")
        self._cleanup_resources()
        rclpy.shutdown()

    def _handle_sigint(self, signum, frame):
        """处理SIGINT信号"""
        self.get_logger().info("接收到中断信号（SIGINT），正在清理资源...")
        self._cleanup_resources()
        rclpy.shutdown()

    def _cleanup_resources(self):
        """资源清理"""
        # 停止定时器
        if hasattr(self, 'connection_check_timer'):
            self.connection_check_timer.cancel()
        if hasattr(self, 'stats_timer'):
            self.stats_timer.cancel()
        if hasattr(self, 'resource_monitor_timer'):
            self.resource_monitor_timer.cancel()

        # 发布删除标记
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.ns = "carla_path"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETE
        self.marker_pub.publish(delete_marker)

        # 关闭CARLA连接
        if self.client:
            try:
                self.client = None
                self.world = None
                self.map = None
            except Exception as e:
                self.get_logger().warning(f"CARLA客户端清理失败: {str(e)}")

    def plan_path_cb(self, request, response):
        """路径规划服务回调"""
        start_time = time.time()
        response.success = False
        response.error_msg = ""

        if not self.carla_connected or not self.map:
            error_msg = "CARLA连接未建立或地图未初始化"
            self.get_logger().error(f"[PLAN_FAILED] {error_msg}")
            response.error_msg = error_msg
            return response

        try:
            start_location = self._ros_to_carla_location(request.start)
            start_wp = self.map.get_waypoint(start_location)
            if not start_wp:
                error_msg = f"起始位置无效（x:{start_location.x:.2f}, y:{start_location.y:.2f}）"
                self.get_logger().error(f"[PLAN_FAILED] {error_msg}")
                response.error_msg = error_msg
                return response

            min_waypoints = self.get_parameter('min_waypoints').value
            max_attempts = self.get_parameter('max_planning_attempts').value
            route, goal_wp = self._get_valid_route(start_wp, min_waypoints, max_attempts)

            if not route:
                error_msg = f"达到最大尝试次数（{max_attempts}次），未生成有效路径"
                self.get_logger().error(f"[PLAN_FAILED] {error_msg}")
                response.error_msg = error_msg
                return response

            total_distance = self._calculate_path_distance(route)
            path_msg = self._build_path_message(route)
            
            self.path_pub.publish(path_msg)
            self._visualize_path_and_goal(path_msg, goal_wp)

            response.path = path_msg
            response.success = True
            response.path_length = total_distance

            # 更新性能统计
            planning_time = time.time() - start_time
            self.last_planning_time = planning_time
            self.total_planning_time += planning_time
            self.planning_count += 1

            self.get_logger().info(
                f"[PLAN_SUCCESS] 路点数: {len(route)}, "
                f"距离: {total_distance:.2f}m, "
                f"用时: {planning_time:.3f}s"
            )
            return response

        except Exception as e:
            error_msg = f"规划过程出错: {str(e)}"
            self.get_logger().error(f"[PLAN_FAILED] {error_msg}")
            response.error_msg = error_msg
            return response

    def _ros_to_carla_location(self, odom_msg):
        """ROS到CARLA坐标转换"""
        return carla.Location(
            x=odom_msg.pose.pose.position.x,
            y=-odom_msg.pose.pose.position.y,  # Y轴方向转换
            z=odom_msg.pose.pose.position.z
        )

    def _get_valid_route(self, start_wp, min_waypoints=50, max_attempts=10):
        """获取有效路径"""
        best_route = None
        best_goal = None
        best_length = 0

        for attempt in range(max_attempts):
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                self.get_logger().warning("未找到可用的生成点")
                continue

            goal_transform = random.choice(spawn_points)
            goal_wp = self.map.get_waypoint(goal_transform.location)

            distance_to_goal = start_wp.transform.location.distance(goal_wp.transform.location)
            if distance_to_goal < 10.0:
                continue

            self.get_logger().debug(
                f"路径规划尝试 {attempt + 1}/{max_attempts}: "
                f"距离: {distance_to_goal:.2f}m"
            )

            resolution = self.get_parameter('waypoint_resolution').value
            route = compute_route_waypoints(
                self.map, start_wp, goal_wp, resolution=resolution)

            if len(route) > best_length:
                best_route = route
                best_goal = goal_wp
                best_length = len(route)

            if len(route) >= min_waypoints:
                return route, goal_wp

        if best_route:
            self.get_logger().warning(
                f"返回最长路径 (长度: {best_length}, 要求: {min_waypoints})"
            )
            return best_route, best_goal

        return None, None

    def _calculate_path_distance(self, route):
        """计算路径总长度"""
        if len(route) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(route) - 1):
            current_wp = route[i][0]
            next_wp = route[i + 1][0]

            dx = next_wp.transform.location.x - current_wp.transform.location.x
            dy = next_wp.transform.location.y - current_wp.transform.location.y
            dz = next_wp.transform.location.z - current_wp.transform.location.z

            total_distance += math.sqrt(dx * dx + dy * dy + dz * dz)

        return total_distance

    def _build_path_message(self, route):
        """构建路径消息"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for waypoint, _ in route:
            pose = PoseStamped()
            pose.header = path_msg.header

            pose.pose.position.x = waypoint.transform.location.x
            pose.pose.position.y = -waypoint.transform.location.y
            pose.pose.position.z = waypoint.transform.location.z

            yaw_rad = waypoint.transform.rotation.yaw * (math.pi / 180.0)
            q = quaternion_from_euler(0, 0, -yaw_rad)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            path_msg.poses.append(pose)

        return path_msg

    def _visualize_path_and_goal(self, path_msg, goal_wp):
        """可视化路径和目标点"""
        self._visualize_path(path_msg)

        if self.get_parameter('publish_goal_marker').value and goal_wp:
            self._visualize_goal_point(path_msg.header, goal_wp)

    def _visualize_path(self, path_msg):
        """可视化路径"""
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
        """可视化目标点"""
        goal_marker = Marker()
        goal_marker.header = header
        goal_marker.ns = "carla_goal"
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD

        goal_marker.pose.position.x = goal_wp.transform.location.x
        goal_marker.pose.position.y = -goal_wp.transform.location.y
        goal_marker.pose.position.z = goal_wp.transform.location.z + 1.0

        goal_marker.scale.x = 2.0
        goal_marker.scale.y = 2.0
        goal_marker.scale.z = 2.0
        goal_marker.color.a = 0.8
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0

        self.marker_pub.publish(goal_marker)

    def _create_path_marker(self, header, action=Marker.ADD, points=None, line_width=0.6):
        """创建路径标记"""
        marker = Marker()
        marker.header = header
        marker.ns = "carla_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = action

        marker.scale.x = line_width
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.lifetime.sec = 60

        if points and action == Marker.ADD:
            marker.points = points

        return marker


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    try:
        node = GlobalPlannerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"节点运行失败: {str(e)}")
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
