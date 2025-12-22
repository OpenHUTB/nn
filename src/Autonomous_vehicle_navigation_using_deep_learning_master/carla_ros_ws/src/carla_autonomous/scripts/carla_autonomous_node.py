#!/usr/bin/env python3
"""
CARLA自动驾驶ROS节点 - 完整修复版本
"""

import rospy
import os
import sys
import time
import threading
import numpy as np

# TensorFlow配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ROS消息
from std_msgs.msg import Header, Float32, Bool, Int32, String, Float32MultiArray
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
import tf
from cv_bridge import CvBridge

# 导入服务消息
try:
    from carla_autonomous.srv import Reset, ResetResponse
    from carla_autonomous.srv import StartEpisode, StartEpisodeResponse
    from carla_autonomous.srv import Stop, StopResponse
except ImportError:
    # 如果服务消息还未生成，创建简单的替代
    rospy.logwarn("服务消息未找到，使用简单替代")
    from std_srvs.srv import Empty, EmptyResponse, Trigger, TriggerResponse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, os.path.abspath(src_dir))

# 导入项目模块
try:
    from car_env import CarEnv
    from route_visualizer import RouteVisualizer
    from vehicle_tracker import VehicleTracker
    from traffic_manager import TrafficManager
    from model_manager import ModelManager
    import config as cfg
    rospy.loginfo("成功导入项目模块")
except ImportError as e:
    rospy.logerr(f"导入项目模块失败: {e}")
    rospy.logerr(f"Python路径: {sys.path}")
    import traceback
    rospy.logerr(traceback.format_exc())
    sys.exit(1)

class CarlaAutonomousROS:
    """CARLA自动驾驶ROS节点"""
    
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('carla_autonomous_node', anonymous=True)
        
        # ROS参数
        self.rate = rospy.Rate(cfg.FPS_LIMIT if cfg.FPS_LIMIT > 0 else 30)
        self.cv_bridge = CvBridge()
        
        # 加载配置
        self.trajectory = cfg.get_current_trajectory()
        if not self.trajectory:
            rospy.logerr("无法加载轨迹配置")
            sys.exit(1)
        
        rospy.loginfo(f"使用轨迹: {self.trajectory['description']}")
            
        # ROS发布器
        self.vehicle_state_pub = rospy.Publisher('/carla/vehicle_state', Odometry, queue_size=10)
        self.vehicle_control_pub = rospy.Publisher('/carla/vehicle_control', Twist, queue_size=10)
        self.camera_image_pub = rospy.Publisher('/carla/camera/image', Image, queue_size=10)
        self.seg_image_pub = rospy.Publisher('/carla/camera/segmentation', Image, queue_size=10)
        self.path_pub = rospy.Publisher('/carla/planned_path', Path, queue_size=10)
        self.marker_pub = rospy.Publisher('/carla/visualization', MarkerArray, queue_size=10)
        self.status_pub = rospy.Publisher('/carla/status', String, queue_size=10)
        self.reward_pub = rospy.Publisher('/carla/reward', Float32, queue_size=10)
        
        # ROS服务 - 使用简单的替代方案
        try:
            # 尝试使用自定义服务
            self.reset_srv = rospy.Service('/carla/reset', Reset, self.handle_reset)
            self.start_srv = rospy.Service('/carla/start_episode', StartEpisode, self.handle_start_episode)
            self.stop_srv = rospy.Service('/carla/stop', Stop, self.handle_stop)
            rospy.loginfo("使用自定义服务")
        except:
            # 使用标准ROS服务
            rospy.loginfo("使用标准ROS服务（Empty/Trigger）")
            self.reset_srv = rospy.Service('/carla/reset', Empty, self.handle_reset_simple)
            self.start_srv = rospy.Service('/carla/start_episode', Trigger, self.handle_start_simple)
            self.stop_srv = rospy.Service('/carla/stop', Trigger, self.handle_stop_simple)
        
        # ROS订阅器
        rospy.Subscriber('/carla/control_cmd', Twist, self.control_callback)
        
        # 初始化变量
        self.env = None
        self.model_manager = None
        self.visualizer = None
        self.tracker = None
        self.traffic_mgr = None
        
        # 控制标志
        self.running = False
        self.current_episode = 0
        self.total_reward = 0.0
        self.step_count = 0
        
        # 手动控制标志
        self.manual_control_active = False
        self.last_control_msg = None
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 初始化CARLA环境
        if not self.init_carla_environment():
            rospy.logerr("CARLA环境初始化失败，节点将退出")
            sys.exit(1)
        
        rospy.loginfo("CARLA自动驾驶ROS节点初始化完成")
    
    def init_carla_environment(self):
        """初始化CARLA环境"""
        try:
            rospy.loginfo("正在初始化CARLA环境...")
            
            # 创建环境
            self.env = CarEnv(self.trajectory['start'], self.trajectory['end'])
            rospy.loginfo("CARLA环境创建成功")
            
            # 创建管理器
            self.model_manager = ModelManager()
            self.visualizer = RouteVisualizer(self.env.world)
            self.tracker = VehicleTracker(self.env.world)
            
            # 加载模型
            rospy.loginfo("正在加载模型...")
            if not self.model_manager.load_models():
                rospy.logerr("模型加载失败")
                return False
            
            # 生成交通
            if cfg.ENABLE_TRAFFIC:
                rospy.loginfo("正在生成交通...")
                self.traffic_mgr = TrafficManager(self.env.client, self.env.world)
                self.traffic_mgr.generate_traffic(
                    num_vehicles=cfg.TRAFFIC_VEHICLES,
                    num_walkers=cfg.TRAFFIC_WALKERS,
                    safe_mode=cfg.TRAFFIC_SAFE_MODE
                )
            
            rospy.loginfo("CARLA环境初始化成功")
            return True
            
        except Exception as e:
            rospy.logerr(f"初始化CARLA环境失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False
    
    def control_callback(self, msg):
        """控制命令回调"""
        if not self.env or not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            rospy.logwarn("无法处理控制命令：车辆未就绪")
            return
        
        try:
            self.last_control_msg = msg
            self.manual_control_active = True
            
            # 记录控制命令
            rospy.logdebug(f"收到控制命令: throttle={msg.linear.x:.2f}, steer={msg.angular.z:.2f}, brake={msg.linear.z:.2f}")
            
        except Exception as e:
            rospy.logwarn(f"控制命令处理失败: {e}")
    
    def apply_control(self):
        """应用当前控制"""
        if not self.env or not self.env.vehicle:
            return
        
        try:
            if self.manual_control_active and self.last_control_msg:
                # 手动控制模式
                import carla
                control = carla.VehicleControl()
                control.throttle = max(0.0, min(1.0, self.last_control_msg.linear.x))
                control.steer = max(-1.0, min(1.0, self.last_control_msg.angular.z))
                control.brake = max(0.0, min(1.0, self.last_control_msg.linear.z))
                
                self.env.vehicle.apply_control(control)
            elif self.running:
                # 自主驾驶模式 - 由run_episode处理
                pass
            else:
                # 停止车辆
                import carla
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                self.env.vehicle.apply_control(control)
                
        except Exception as e:
            rospy.logwarn(f"应用控制失败: {e}")
    
    def handle_reset(self, req):
        """重置环境服务处理（自定义服务）"""
        with self.lock:
            try:
                rospy.loginfo("收到重置环境请求")
                
                # 重置环境
                if self.env:
                    current_state = self.env.reset()
                    
                    # 转换为Float32MultiArray
                    state_array = Float32MultiArray()
                    state_array.data = current_state
                    
                    self.publish_status("环境重置成功")
                    
                    return ResetResponse(state_array)
                else:
                    rospy.logerr("环境未初始化")
                    return ResetResponse(Float32MultiArray())
                    
            except Exception as e:
                rospy.logerr(f"重置环境失败: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
                return ResetResponse(Float32MultiArray())
    
    def handle_reset_simple(self, req):
        """重置环境服务处理（标准服务）"""
        with self.lock:
            try:
                rospy.loginfo("重置环境")
                if self.env:
                    self.env.reset()
                    self.publish_status("环境重置成功")
                return EmptyResponse()
            except Exception as e:
                rospy.logerr(f"重置失败: {e}")
                return EmptyResponse()
    
    def handle_start_episode(self, req):
        """开始episode服务处理（自定义服务）"""
        with self.lock:
            if self.running:
                rospy.logwarn("已经有episode在运行")
                return StartEpisodeResponse(False, "已有episode在运行")
            
            try:
                self.running = True
                self.current_episode += 1
                self.total_reward = 0.0
                self.step_count = 0
                self.manual_control_active = False
                
                rospy.loginfo(f"开始Episode {self.current_episode}")
                
                # 在新线程中运行episode
                episode_thread = threading.Thread(target=self.run_episode)
                episode_thread.daemon = True
                episode_thread.start()
                
                return StartEpisodeResponse(True, f"Episode {self.current_episode} 已开始")
                
            except Exception as e:
                rospy.logerr(f"开始episode失败: {e}")
                self.running = False
                return StartEpisodeResponse(False, str(e))
    
    def handle_start_simple(self, req):
        """开始episode服务处理（标准服务）"""
        with self.lock:
            try:
                if not self.running:
                    self.running = True
                    self.current_episode += 1
                    self.total_reward = 0.0
                    self.step_count = 0
                    self.manual_control_active = False
                    
                    # 在新线程中运行episode
                    episode_thread = threading.Thread(target=self.run_episode)
                    episode_thread.daemon = True
                    episode_thread.start()
                    
                    return TriggerResponse(True, f"Episode {self.current_episode} started")
                else:
                    return TriggerResponse(False, "Already running")
            except Exception as e:
                return TriggerResponse(False, str(e))
    
    def handle_stop(self, req):
        """停止服务处理（自定义服务）"""
        with self.lock:
            rospy.loginfo("收到停止请求")
            self.running = False
            self.manual_control_active = False
            return StopResponse(True, "已停止")
    
    def handle_stop_simple(self, req):
        """停止服务处理（标准服务）"""
        with self.lock:
            rospy.loginfo("停止")
            self.running = False
            self.manual_control_active = False
            return TriggerResponse(True, "Stopped")
    
    def run_episode(self):
        """运行一个episode"""
        rospy.loginfo(f"开始运行Episode {self.current_episode}")
        
        try:
            # 重置环境
            if not self.env:
                rospy.logerr("环境未初始化")
                return
            
            current_state = self.env.reset()
            
            # 设置车辆跟踪
            if self.env.vehicle:
                self.tracker.set_follow_view(self.env.vehicle)
            
            # 主循环
            done = False
            while not done and not rospy.is_shutdown() and self.running:
                if self.step_count >= cfg.MAX_STEPS_PER_EPISODE:
                    rospy.loginfo("达到最大步数")
                    break
                
                # 模型预测动作（仅在自主驾驶模式）
                if not self.manual_control_active:
                    action = self.model_manager.predict_action(current_state)
                    rospy.logdebug(f"自主动作: {cfg.ACTION_NAMES[action] if action < len(cfg.ACTION_NAMES) else action}")
                    
                    # 执行动作
                    new_state, reward, done, _ = self.env.step(action, current_state)
                    
                    # 更新统计
                    self.total_reward += reward
                    self.step_count += 1
                    
                    # 发布奖励
                    reward_msg = Float32()
                    reward_msg.data = reward
                    self.reward_pub.publish(reward_msg)
                    
                    # 更新状态
                    current_state = new_state
                else:
                    # 手动控制模式，等待用户操作
                    time.sleep(0.1)
                    continue
                
                # 检查是否完成
                if done:
                    status_msg = f"Episode {self.current_episode} 完成，总奖励: {self.total_reward:.2f}"
                    rospy.loginfo(status_msg)
                    break
                
                # 控制循环频率
                self.rate.sleep()
            
            # 发布最终状态
            status_msg = f"Episode {self.current_episode} 结束，步数: {self.step_count}，总奖励: {self.total_reward:.2f}"
            self.publish_status(status_msg)
            rospy.loginfo(status_msg)
            
            # 重置运行标志
            self.running = False
            
        except Exception as e:
            rospy.logerr(f"运行episode时出错: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.running = False
    
    def publish_vehicle_state(self):
        """发布车辆状态"""
        if not self.env or not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return
        
        try:
            # 获取车辆状态
            vehicle = self.env.vehicle
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            
            # 创建Odometry消息
            odom = Odometry()
            odom.header = Header()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"
            odom.child_frame_id = "base_link"
            
            # 位置
            odom.pose.pose.position = Point(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z
            )
            
            # 方向（四元数）
            roll = transform.rotation.roll * np.pi / 180
            pitch = transform.rotation.pitch * np.pi / 180
            yaw = transform.rotation.yaw * np.pi / 180
            quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            odom.pose.pose.orientation = Quaternion(*quat)
            
            # 速度
            odom.twist.twist.linear = Vector3(
                x=velocity.x,
                y=velocity.y,
                z=velocity.z
            )
            
            # 发布
            self.vehicle_state_pub.publish(odom)
            
        except Exception as e:
            rospy.logwarn(f"发布车辆状态失败: {e}")
    
    def publish_camera_images(self):
        """发布相机图像"""
        if not self.env or not hasattr(self.env, 'cam') or self.env.cam is None:
            return
        
        try:
            # RGB图像
            cv_image = np.frombuffer(self.env.cam.raw_data, dtype=np.uint8)
            cv_image = cv_image.reshape((self.env.cam.height, self.env.cam.width, 4))
            cv_image = cv_image[:, :, :3]  # 移除alpha通道
            
            image_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            image_msg.header.stamp = rospy.Time.now()
            image_msg.header.frame_id = "camera"
            self.camera_image_pub.publish(image_msg)
            
            # 分割图像
            if hasattr(self.env, 'seg_array') and self.env.seg_array is not None:
                seg_msg = self.cv_bridge.cv2_to_imgmsg(self.env.seg_array, encoding="bgr8")
                seg_msg.header.stamp = rospy.Time.now()
                seg_msg.header.frame_id = "camera"
                self.seg_image_pub.publish(seg_msg)
                
        except Exception as e:
            rospy.logdebug(f"发布图像失败: {e}")
    
    def publish_planned_path(self):
        """发布规划路径"""
        if not self.env or not hasattr(self.env, 'path'):
            return
        
        try:
            path_msg = Path()
            path_msg.header = Header()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "map"
            
            for waypoint in self.env.path:
                pose = Pose()
                pose.position = Point(
                    x=waypoint.transform.location.x,
                    y=waypoint.transform.location.y,
                    z=waypoint.transform.location.z
                )
                path_msg.poses.append(pose)
            
            self.path_pub.publish(path_msg)
            
        except Exception as e:
            rospy.logdebug(f"发布规划路径失败: {e}")
    
    def publish_visualization_markers(self):
        """发布可视化标记"""
        if not self.env or not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return
        
        try:
            marker_array = MarkerArray()
            
            # 车辆标记
            vehicle_marker = Marker()
            vehicle_marker.header.frame_id = "map"
            vehicle_marker.header.stamp = rospy.Time.now()
            vehicle_marker.ns = "vehicle"
            vehicle_marker.id = 0
            vehicle_marker.type = Marker.CUBE
            vehicle_marker.action = Marker.ADD
            
            transform = self.env.vehicle.get_transform()
            vehicle_marker.pose.position = Point(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + 0.5
            )
            
            quat = tf.transformations.quaternion_from_euler(
                0, 0, transform.rotation.yaw * np.pi / 180
            )
            vehicle_marker.pose.orientation = Quaternion(*quat)
            
            vehicle_marker.scale.x = 2.0
            vehicle_marker.scale.y = 1.0
            vehicle_marker.scale.z = 1.5
            vehicle_marker.color.r = 0.0
            vehicle_marker.color.g = 1.0
            vehicle_marker.color.b = 0.0
            vehicle_marker.color.a = 0.8
            vehicle_marker.lifetime = rospy.Duration(0.1)
            
            marker_array.markers.append(vehicle_marker)
            
            # 目标点标记
            goal_marker = Marker()
            goal_marker.header.frame_id = "map"
            goal_marker.header.stamp = rospy.Time.now()
            goal_marker.ns = "goal"
            goal_marker.id = 1
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            
            goal_marker.pose.position = Point(
                x=self.trajectory['end'][0],
                y=self.trajectory['end'][1],
                z=self.trajectory['end'][2] + 1.0
            )
            
            goal_marker.scale.x = 2.0
            goal_marker.scale.y = 2.0
            goal_marker.scale.z = 2.0
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 0.8
            goal_marker.lifetime = rospy.Duration(0.1)
            
            marker_array.markers.append(goal_marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logdebug(f"发布可视化标记失败: {e}")
    
    def publish_status(self, status_msg):
        """发布状态信息"""
        try:
            status = String()
            status.data = status_msg
            self.status_pub.publish(status)
        except Exception as e:
            rospy.logwarn(f"发布状态失败: {e}")
    
    def publish_all_data(self):
        """发布所有数据"""
        try:
            self.publish_vehicle_state()
            self.publish_camera_images()
            self.publish_planned_path()
            self.publish_visualization_markers()
            
            # 应用控制
            self.apply_control()
            
        except Exception as e:
            rospy.logwarn(f"发布数据时出错: {e}")
    
    def run(self):
        """主运行循环"""
        rospy.loginfo("CARLA自动驾驶ROS节点开始运行")
        
        # 等待CARLA服务器
        time.sleep(2.0)
        
        # 发布初始状态
        self.publish_status("节点就绪")
        
        # 主循环
        while not rospy.is_shutdown():
            try:
                if self.env and hasattr(self.env, 'vehicle') and self.env.vehicle:
                    # 发布数据
                    self.publish_all_data()
                
                # 控制循环频率
                self.rate.sleep()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                rospy.logerr(f"主循环出错: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
                time.sleep(1.0)
        
        # 清理
        self.cleanup()
        rospy.loginfo("CARLA自动驾驶ROS节点已停止")
    
    def cleanup(self):
        """清理资源"""
        rospy.loginfo("正在清理资源...")
        
        try:
            # 停止车辆跟踪
            if self.tracker:
                self.tracker.cleanup()
            
            # 清理交通
            if self.traffic_mgr:
                self.traffic_mgr.cleanup()
            
            # 清理环境
            if self.env:
                self.env.cleanup()
                
        except Exception as e:
            rospy.logwarn(f"清理资源时出错: {e}")

if __name__ == '__main__':
    try:
        rospy.loginfo("启动CARLA自动驾驶ROS节点...")
        node = CarlaAutonomousROS()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点启动失败: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())