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
import math
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
        self.last_reset_time = 0.0
        
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
        self.speed_threshold = 1.0  # 最小速度阈值 (km/h)
        self.stuck_counter = 0
        self.max_stuck_steps = 10  # 连续多少步速度为0算作卡住
        
        rospy.loginfo("CARLA自动驾驶ROS节点初始化完成")
        rospy.loginfo("初始化完成，等待5秒后开始测试...")
        time.sleep(5)
        
        # 创建一个测试线程
        test_thread = threading.Thread(target=self.test_action_mapping)
        test_thread.daemon = True
        test_thread.start()
        

        rospy.loginfo("CARLA自动驾驶ROS节点初始化完成")
    
    def init_carla_environment(self, max_retries=3):
        """初始化CARLA环境，支持重试"""
        for attempt in range(max_retries):
            try:
                rospy.loginfo(f"尝试连接CARLA服务器 (尝试 {attempt+1}/{max_retries})...")
                
                # 创建环境
                self.env = CarEnv(self.trajectory['start'], self.trajectory['end'])
                rospy.loginfo("CARLA环境创建成功")
                
                # 确保path存在
                if not hasattr(self.env, 'path') or not self.env.path:
                    rospy.loginfo("计算规划路径...")
                    try:
                        # 调用reset方法会计算路径
                        initial_state = self.env.reset()
                        rospy.loginfo(f"规划路径计算完成，包含 {len(self.env.path)} 个点")
                    except Exception as e:
                        rospy.logwarn(f"计算规划路径失败: {e}")
                
                # 创建管理器
                self.model_manager = ModelManager()
                self.visualizer = RouteVisualizer(self.env.world)
                self.tracker = VehicleTracker(self.env.world)
                
                # 加载模型
                rospy.loginfo("正在加载模型...")
                if not self.model_manager.load_models():
                    rospy.logwarn("模型加载失败，使用默认控制")
                
                # 生成交通
                if cfg.ENABLE_TRAFFIC:
                    rospy.loginfo("正在生成交通...")
                    self.traffic_mgr = TrafficManager(self.env.client, self.env.world)
                    self.traffic_mgr.generate_traffic(
                        num_vehicles=min(cfg.TRAFFIC_VEHICLES, 5),
                        num_walkers=min(cfg.TRAFFIC_WALKERS, 10),
                        safe_mode=cfg.TRAFFIC_SAFE_MODE
                    )
                
                rospy.loginfo("CARLA环境初始化成功")
                self.publish_status("CARLA环境就绪")
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    rospy.logwarn(f"初始化失败，{5}秒后重试...: {e}")
                    time.sleep(5)
                else:
                    rospy.logerr(f"CARLA环境初始化最终失败: {e}")
                    return False
        return False

    def test_action_mapping(self):
        """测试动作到控制命令的映射"""
        rospy.loginfo("=== 开始动作映射测试 ===")
        
        if not self.env or not self.env.vehicle:
            rospy.logerr("环境或车辆未就绪")
            return
        
        import carla
        
        # 测试每个可能的动作
        for action_idx in range(len(cfg.ACTION_NAMES)):
            action_name = cfg.ACTION_NAMES[action_idx]
            rospy.loginfo(f"测试动作 {action_idx}: {action_name}")
            
            try:
                # 使用环境的方法执行动作
                # 创建一个初始状态
                current_state = [0.0, 0.0, 0.0, 0.0]  # 根据car_env.py中的状态维度设置
                
                # 检查车辆是否被销毁
                if not self.env.vehicle or not self.env.vehicle.is_alive:
                    rospy.logerr("车辆已被销毁，跳过测试")
                    break
                    
                state, reward, done, info = self.env.step(action_idx, current_state)
                
                # 获取应用的控制
                control = self.env.vehicle.get_control()
                rospy.loginfo(f"  控制命令: 油门={control.throttle:.2f}, 刹车={control.brake:.2f}, "
                            f"转向={control.steer:.2f}, 倒车={control.reverse}")
                
                # 检查车辆速度
                velocity = self.env.vehicle.get_velocity()
                speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                rospy.loginfo(f"  速度: {speed:.2f} km/h")
                
                # 重置控制
                stop_control = carla.VehicleControl()
                stop_control.brake = 1.0
                self.env.vehicle.apply_control(stop_control)
                time.sleep(0.5)
                
            except Exception as e:
                # 修复：使用正确的rospy日志方法
                rospy.logerr(f"  测试失败: {e}")
            
            time.sleep(0.5)
        
        rospy.loginfo("=== 动作映射测试完成 ===")
    def control_callback(self, msg):
        """控制命令回调"""
        if not self.env or not hasattr(self.env, 'vehicle'):
            rospy.logdebug("车辆未就绪，忽略控制命令")
            return
        
        try:
            # 更新控制消息
            self.last_control_msg = msg
            self.manual_control_active = True
            
            # 确保车辆存在且有效
            if not self.env.vehicle or not self.env.vehicle.is_alive:
                rospy.logwarn("车辆不存在或已销毁")
                return
            
            # 应用控制
            import carla
            control = carla.VehicleControl()
            
            # 解析控制消息
            throttle = float(msg.linear.x)  # 油门/前进
            steer = float(msg.angular.z)    # 转向
            brake = float(msg.linear.z)     # 刹车
            
            # 处理倒车逻辑
            if throttle < 0:
                # 倒车模式
                control.throttle = abs(throttle)  # 油门值取绝对值
                control.brake = 0.0
                control.reverse = True  # 设置倒车标志
                rospy.logdebug(f"倒车控制: 油门={control.throttle:.2f}, 转向={steer:.2f}")
            elif brake > 0:
                # 刹车模式
                control.throttle = 0.0
                control.brake = brake
                control.reverse = False
                rospy.logdebug(f"刹车控制: 刹车={control.brake:.2f}")
            else:
                # 前进模式
                control.throttle = throttle
                control.brake = 0.0
                control.reverse = False
                rospy.logdebug(f"前进控制: 油门={control.throttle:.2f}, 转向={steer:.2f}")
            
            # 转向控制
            control.steer = max(-1.0, min(1.0, steer))
            
            # 手刹（紧急刹车时使用）
            control.hand_brake = (brake > 0.8)
            
            # 应用控制
            self.env.vehicle.apply_control(control)
            
        except Exception as e:
            if "destroyed actor" not in str(e):
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
        """重置环境服务处理"""
        with self.lock:
            try:
                current_time = time.time()
                if hasattr(self, 'last_reset_time'):
                    if current_time - self.last_reset_time < 2.0:
                        rospy.logwarn("重置过于频繁，请等待2秒")
                        return ResetResponse(Float32MultiArray())
                else:
                    self.last_reset_time = 0.0  # 初始化属性    
                rospy.loginfo("收到重置环境请求")
                self.last_reset_time = current_time
                
                # 停止当前episode
                self.running = False
                self.manual_control_active = False
                time.sleep(0.5)
                
                # 重置环境
                if self.env:
                    try:
                        current_state = self.env.reset()
                        
                        # 重置后确保path存在
                        if not hasattr(self.env, 'path') or not self.env.path:
                            rospy.loginfo("重置后重新计算规划路径...")
                            try:
                                traj = self.env.trajectory(draw=False)
                                self.env.path = []
                                for el in traj:
                                    self.env.path.append(el[0])
                                rospy.loginfo(f"规划路径重新计算完成，包含 {len(self.env.path)} 个点")
                            except Exception as e:
                                rospy.logwarn(f"重新计算规划路径失败: {e}")
                        
                        # 转换为Float32MultiArray
                        state_array = Float32MultiArray()
                        state_array.data = current_state
                        
                        self.publish_status("环境重置成功")
                        rospy.loginfo("环境重置成功")
                        
                        return ResetResponse(state_array)
                    except Exception as e:
                        rospy.logerr(f"重置环境时出错: {e}")
                else:
                    rospy.logerr("环境未初始化")
                
                return ResetResponse(Float32MultiArray())
                    
            except Exception as e:
                rospy.logerr(f"重置环境失败: {e}")
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
        """运行一个episode - 优化控制频率"""
        import carla
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
            
            # 主循环 - 大幅提高控制频率
            done = False
            control_interval = 0.01  # 从0.05减少到0.01，100Hz控制频率
            last_control_time = time.time()
            
            # 性能计数器
            control_count = 0
            total_control_time = 0
            
            while not done and not rospy.is_shutdown() and self.running:
                if self.step_count >= cfg.MAX_STEPS_PER_EPISODE:
                    rospy.loginfo("达到最大步数")
                    break
                
                current_time = time.time()
                elapsed = current_time - last_control_time
                
                # 只在控制间隔到达时执行动作
                if elapsed >= control_interval:
                    control_start = time.time()
                    
                    # 检查车辆是否被销毁
                    if not self.env.vehicle or not self.env.vehicle.is_alive:
                        rospy.logwarn("车辆已被销毁，终止episode")
                        done = True
                        break
                    
                    # 模型预测动作（仅在自主驾驶模式）
                    if not self.manual_control_active:
                        action = self.model_manager.predict_action(current_state)
                        
                        # 执行动作
                        try:
                            new_state, reward, done, _ = self.env.step(action, current_state)
                        except Exception as e:
                            rospy.logerr(f"执行动作时出错: {e}")
                            done = True
                            break
                        
                        # 更新统计
                        self.total_reward += reward
                        self.step_count += 1
                        
                        # 发布奖励
                        reward_msg = Float32()
                        reward_msg.data = reward
                        self.reward_pub.publish(reward_msg)
                        
                        # 更新状态
                        current_state = new_state
                        
                        # 获取并记录速度和位置
                        if self.env.vehicle:
                            velocity = self.env.vehicle.get_velocity()
                            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                            
                            # 每50步记录一次性能数据
                            if self.step_count % 50 == 0:
                                # 计算路径跟踪误差
                                if hasattr(self.env, 'path') and self.env.path:
                                    waypoint_ind = self.env.get_closest_waypoint(self.env.path, self.env.vehicle.get_transform())
                                    if waypoint_ind < len(self.env.path):
                                        current_waypoint = self.env.path[waypoint_ind]
                                        vehicle_pos = self.env.vehicle.get_transform().location
                                        waypoint_pos = current_waypoint.transform.location
                                        lateral_error = math.sqrt(
                                            (vehicle_pos.x - waypoint_pos.x)**2 + 
                                            (vehicle_pos.y - waypoint_pos.y)**2
                                        )
                                        rospy.loginfo(f"步骤 {self.step_count}, 速度: {speed:.1f} km/h, 横向误差: {lateral_error:.2f} m")
                            
                            # 检查是否卡住
                            if speed < self.speed_threshold:
                                self.stuck_counter += 1
                                if self.stuck_counter >= self.max_stuck_steps:
                                    rospy.logwarn("车辆卡住，尝试恢复...")
                                    # 尝试短暂全油门
                                    control = carla.VehicleControl()
                                    control.throttle = 1.0
                                    control.steer = 0.0
                                    control.brake = 0.0
                                    self.env.vehicle.apply_control(control)
                                    time.sleep(0.1)  # 减少恢复等待时间
                                    self.stuck_counter = 0
                            else:
                                self.stuck_counter = 0
                    
                    control_time = time.time() - control_start
                    total_control_time += control_time
                    control_count += 1
                    
                    # 每100步记录平均控制延迟
                    if control_count % 100 == 0 and control_count > 0:
                        avg_control_time = total_control_time / control_count
                        rospy.loginfo(f"平均控制延迟: {avg_control_time*1000:.1f} ms, 频率: {1.0/avg_control_time:.1f} Hz")
                        total_control_time = 0
                        control_count = 0
                    
                    last_control_time = current_time
                    
                    # 检查是否完成
                    if done:
                        status_msg = f"Episode {self.current_episode} 完成，总奖励: {self.total_reward:.2f}"
                        rospy.loginfo(status_msg)
                        break
                else:
                    # 精确等待，避免CPU占用过高
                    sleep_time = control_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.001))
            
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

    def publish_planned_path(self):
        """发布规划路径"""
        if not self.env:
            return
        
        try:
            path_msg = Path()
            path_msg.header = Header()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "map"
            
            # 检查是否有path属性
            if hasattr(self.env, 'path') and self.env.path:
                point_count = 0
                for waypoint in self.env.path:
                    # 每隔5个点取一个点，避免路径太密集
                    if point_count % 5 == 0:
                        pose = Pose()
                        pose.position = Point(
                            x=waypoint.transform.location.x,
                            y=waypoint.transform.location.y,
                            z=waypoint.transform.location.z
                        )
                        path_msg.poses.append(pose)
                    point_count += 1
                
                # 定期记录
                if rospy.get_time() % 30 < 0.1:  # 每30秒记录一次
                    rospy.loginfo(f"发布规划路径，包含 {len(path_msg.poses)} 个点")
            
            self.path_pub.publish(path_msg)
            
        except Exception as e:
            rospy.logdebug(f"发布规划路径失败: {e}")
    
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
    
    def publish_vehicle_state(self):
        """发布车辆状态信息"""
        if not self.env or not hasattr(self.env, 'vehicle') or self.env.vehicle is None:
            return
        
        try:
            odom_msg = Odometry()
            odom_msg.header = Header()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "map"
            odom_msg.child_frame_id = "vehicle"
            
            # 获取车辆变换
            transform = self.env.vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            
            # 设置位置
            odom_msg.pose.pose.position = Point(
                x=location.x,
                y=location.y,
                z=location.z
            )
            
            # 设置朝向
            quat = tf.transformations.quaternion_from_euler(
                0, 0, rotation.yaw * np.pi / 180
            )
            odom_msg.pose.pose.orientation = Quaternion(*quat)
            
            # 获取车辆速度（如果可用）
            velocity = self.env.vehicle.get_velocity()
            odom_msg.twist.twist.linear = Vector3(
                x=velocity.x,
                y=velocity.y,
                z=velocity.z
            )
            
            # 获取车辆角速度（如果可用）
            angular_velocity = self.env.vehicle.get_angular_velocity()
            odom_msg.twist.twist.angular = Vector3(
                x=angular_velocity.x,
                y=angular_velocity.y,
                z=angular_velocity.z
            )
            
            # 发布里程计消息
            self.vehicle_state_pub.publish(odom_msg)
            
            # 发布TF变换
            br = tf.TransformBroadcaster()
            br.sendTransform(
                (location.x, location.y, location.z),
                quat,
                rospy.Time.now(),
                "vehicle",
                "map"
            )
            
        except Exception as e:
            rospy.logdebug(f"发布车辆状态失败: {e}")

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
        
        # 控制发布计数器
        last_path_publish = 0
        last_marker_publish = 0
        last_camera_publish = 0
        last_state_publish = 0
        
        # 主循环
        while not rospy.is_shutdown():
            try:
                current_time = rospy.get_time()
                
                if self.env:
                    # 发布车辆状态（中等频率，20Hz）
                    if current_time - last_state_publish > 0.05:
                        self.publish_vehicle_state()
                        last_state_publish = current_time
                    
                    # 发布相机图像（较低频率，10Hz）
                    if current_time - last_camera_publish > 0.1:
                        self.publish_camera_images()
                        last_camera_publish = current_time
                    
                    # 发布规划路径（低频，每5秒一次）
                    if current_time - last_path_publish > 5.0:
                        self.publish_planned_path()
                        last_path_publish = current_time
                    
                    # 发布可视化标记（低频，每1秒一次）
                    if current_time - last_marker_publish > 1.0:
                        self.publish_visualization_markers()
                        last_marker_publish = current_time
                
                # 控制循环频率
                self.rate.sleep()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                rospy.logerr(f"主循环出错: {e}")
                time.sleep(0.1)  # 减少错误等待时间
        
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