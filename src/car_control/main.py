#!/usr/bin/env python3
"""
CARLA全局路径规划节点
提供从起始点到随机目标点的路径规划服务，并将规划结果通过ROS消息发布和可视化
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from carla_global_planner.srv import PlanGlobalPath
from utilities.planner import compute_route_waypoints

import carla
import random
from tf_transformations import quaternion_from_euler


class GlobalPlannerNode(Node):
    """全局路径规划节点类"""
    
    def __init__(self):
        super().__init__('carla_global_planner_node')
        
        # 初始化CARLA客户端和世界对象
        self.client = None
        self.world = None
        self.map = None
        self._initialize_carla_client()
        
        # 初始化ROS发布器和服务
        self.marker_pub = self.create_publisher(
            Marker, 'visualization_marker', 10)
        
        self.srv = self.create_service(
            PlanGlobalPath, 'plan_to_random_goal', self.plan_path_cb)
        
        self.get_logger().info("CARLA全局路径规划服务已启动")

    def _initialize_carla_client(self):
        """初始化CARLA客户端连接"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            self.get_logger().info("成功连接到CARLA服务器")
        except Exception as e:
            self.get_logger().error(f"CARLA客户端初始化失败: {str(e)}")
            # 抛出异常以终止节点初始化，因为CARLA连接是核心功能
            raise

    def plan_path_cb(self, request, response):
        """
        路径规划服务回调函数
        接收起始点，规划到随机目标点的路径并返回
        """
        if not self.map:
            self.get_logger().error("CARLA地图未初始化，无法规划路径")
            return response
            
        try:
            # 将ROS坐标转换为CARLA坐标
            start_location = self._ros_to_carla_location(request.start)
            start_wp = self.map.get_waypoint(start_location)
            
            # 规划足够长的路径
            route = self._get_valid_route(start_wp)
            if not route:
                self.get_logger().error("无法生成有效的路径")
                return response
                
            # 构建路径消息并可视化
            path_msg = self._build_path_message(route)
            self._visualize_path(path_msg)
            
            response.path = path_msg
            self.get_logger().info(f"成功规划路径，包含{len(route)}个路点")
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
        获取有效的路径
        尝试多次生成路径，直到满足最小路点数量要求或达到最大尝试次数
        """
        for attempt in range(max_attempts):
            # 选择随机可行的目标点
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                self.get_logger().warning("未找到可用的生成点")
                return None
                
            goal_transform = random.choice(spawn_points)
            goal_wp = self.map.get_waypoint(goal_transform.location)
            
            # 计算路径
            self.get_logger().info(
                f"路径规划尝试 {attempt+1}/{max_attempts}: "
                f"从({start_wp.transform.location.x:.2f}, {start_wp.transform.location.y:.2f}) "
                f"到({goal_wp.transform.location.x:.2f}, {goal_wp.transform.location.y:.2f})"
            )
            
            route = compute_route_waypoints(
                self.map, start_wp, goal_wp, resolution=1.0)
            
            if len(route) >= min_waypoints:
                return route
                
        self.get_logger().warning(
            f"达到最大尝试次数({max_attempts})，返回当前最长路径")
        return route  # 返回最后一次尝试的路径，即使它短于要求

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

    def _visualize_path(self, path_msg):
        """可视化路径，先删除旧标记再发布新标记"""
        # 删除旧标记
        delete_marker = self._create_marker(
            path_msg.header, action=Marker.DELETE)
        self.marker_pub.publish(delete_marker)
        
        # 发布新标记
        new_marker = self._create_marker(
            path_msg.header, 
            action=Marker.ADD,
            points=[pose.pose.position for pose in path_msg.poses]
        )
        self.marker_pub.publish(new_marker)

    def _create_marker(self, header, action=Marker.ADD, points=None):
        """创建路径可视化标记"""
        marker = Marker()
        marker.header = header
        marker.ns = "carla_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = action
        
        # 视觉外观设置
        marker.scale.x = 0.6  # 线宽
        marker.color.a = 1.0  # 透明度
        marker.color.r = 0.0
        marker.color.g = 1.0  # 绿色
        marker.color.b = 0.0
        
        # 添加点集（仅用于ADD动作）
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
