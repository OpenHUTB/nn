    def control_cmd_callback(self, msg):
        """处理控制命令"""
        rospy.loginfo(f"Received control command: linear={msg.linear.x}, angular={msg.angular.z}")
        
        # 发布状态
        status_msg = String()
        status_msg.data = f"Control: throttle={msg.linear.x}, steer={msg.angular.z}"
        self.status_pub.publish(status_msg)
        
        # 如果连接到CARLA，应用控制
        if self.carla_connected and hasattr(self, 'vehicle'):
            try:
                import carla
                
                # 创建CARLA控制命令
                control = carla.VehicleControl()
                control.throttle = max(0.0, min(1.0, msg.linear.x))  # 限制在0-1之间
                control.brake = max(0.0, min(1.0, -msg.linear.x)) if msg.linear.x < 0 else 0.0
                control.steer = max(-1.0, min(1.0, msg.angular.z))
                control.hand_brake = False
                control.manual_gear_shift = False
                
                # 应用控制
                self.vehicle.apply_control(control)
                
            except Exception as e:
                rospy.logerr(f"Error applying CARLA control: {e}")
    
    def start_simulation_callback(self, msg):
        """处理启动仿真命令"""
        if msg.data:
            rospy.loginfo("Starting simulation...")
            self.publish_status("Simulation starting")
            
            # 如果使用CARLA模式，启动代理
            if self.carla_connected and hasattr(self, 'agent'):
                try:
                    # 这里可以启动自动驾驶代理
                    rospy.loginfo("CARLA agent ready")
                except Exception as e:
                    rospy.logerr(f"Error starting CARLA agent: {e}")
    
    def stop_simulation_callback(self, msg):
        """处理停止仿真命令"""
        if msg.data:
            rospy.loginfo("Stopping simulation...")
            self.publish_status("Simulation stopping")
            
            # 停止车辆
            if self.carla_connected and hasattr(self, 'vehicle'):
                try:
                    control = carla.VehicleControl()
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.steer = 0.0
                    self.vehicle.apply_control(control)
                except Exception as e:
                    rospy.logerr(f"Error stopping vehicle: {e}")
    
    def image_callback(self, msg):
        """处理ROS图像消息"""
        if self.simulation_mode == 'ros_only':
            try:
                # 转换为OpenCV格式
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                
                # 使用YOLO进行对象检测（简化版本）
                detections = self.detect_objects_simple(cv_image)
                
                # 发布检测结果
                if detections:
                    detection_msg = String()
                    detection_msg.data = f"Detected {len(detections)} objects: {', '.join(detections)}"
                    self.detection_pub.publish(detection_msg)
                    
                    # 可视化并发布图像
                    vis_image = self.draw_detections(cv_image.copy(), detections)
                    ros_image = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
                    ros_image.header = msg.header
                    self.detection_image_pub.publish(ros_image)
                
                rospy.logdebug("Processed ROS image with object detection")
                
            except Exception as e:
                rospy.logerr(f"Error processing image: {e}")
    
    def detect_objects_simple(self, image):
        """简单的对象检测（用于演示）"""
        import cv2
        import numpy as np
        
        # 这里实现一个简化的检测逻辑
        # 在实际应用中，应该使用你的YOLOv3模型
        
        # 检查图像中是否有特定的颜色区域
        # 这是一个简单的演示，实际应该加载YOLO模型
        
        detections = []
        
        try:
            # 尝试加载YOLO模型
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            cfg_path = os.path.join(model_dir, 'yolov3.cfg')
            weights_path = os.path.join(model_dir, 'yolov3.weights')
            names_path = os.path.join(model_dir, 'coco.names')
            
            if os.path.exists(cfg_path) and os.path.exists(weights_path):
                # 读取类别名称
                with open(names_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                
                # 这里可以添加实际的YOLO检测代码
                # 为了简化，我们只返回一些模拟检测
                detections = ['car', 'person', 'traffic light']
            else:
                rospy.logwarn("YOLO model files not found, using simulated detections")
                detections = ['simulated_car', 'simulated_person']
                
        except Exception as e:
            rospy.logerr(f"Error in detection: {e}")
            detections = ['error']
        
        return detections
    
    def draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        import cv2
        
        height, width = image.shape[:2]
        
        for i, obj in enumerate(detections):
            # 计算位置
            x = int(width * 0.1)
            y = int(height * 0.1 * (i + 1))
            
            # 绘制文本
            cv2.putText(image, obj, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制边界框
            cv2.rectangle(image, (x-5, y-25), (x+100, y+5), (0, 255, 0), 2)
        
        # 添加水印
        cv2.putText(image, "ROS-CARLA Bridge", (10, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def update_carla_state(self):
        """更新CARLA状态（如果连接）"""
        if self.carla_connected and hasattr(self, 'vehicle'):
            try:
                # 获取车辆变换
                transform = self.vehicle.get_transform()
                velocity = self.vehicle.get_velocity()
                
                # 计算速度（m/s）
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                
                # 发布速度
                speed_msg = Float32()
                speed_msg.data = speed
                self.velocity_pub.publish(speed_msg)
                
                # 发布里程计
                odom = self.create_odometry_message(transform, velocity)
                self.odometry_pub.publish(odom)
                
                # 更新状态
                status_msg = String()
                status_msg.data = f"CARLA: Speed={speed:.2f}m/s, Location=({transform.location.x:.1f}, {transform.location.y:.1f})"
                self.status_pub.publish(status_msg)
                
            except Exception as e:
                rospy.logerr(f"Error updating CARLA state: {e}")
    
    def create_odometry_message(self, transform, velocity):
        """创建ROS里程计消息"""
        from nav_msgs.msg import Odometry
        import tf.transformations as tf_trans
        import numpy as np
        
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "carla_map"
        odom.child_frame_id = "carla_vehicle"
        
        # 位置
        odom.pose.pose.position.x = transform.location.x
        odom.pose.pose.position.y = -transform.location.y  # CARLA使用左手坐标系
        odom.pose.pose.position.z = transform.location.z
        
        # 方向（四元数）
        roll = np.deg2rad(transform.rotation.roll)
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(-transform.rotation.yaw)
        
        q = tf_trans.quaternion_from_euler(roll, pitch, yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        # 速度
        odom.twist.twist.linear.x = velocity.x
        odom.twist.twist.linear.y = -velocity.y
        odom.twist.twist.linear.z = velocity.z
        
        return odom
    
    def publish_status(self, message):
        """发布状态消息"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)
    
    def run(self):
        """主循环"""
        rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo(f"{self.node_name} running at {self.publish_rate}Hz")
        
        while not rospy.is_shutdown() and self.running:
            try:
                # 更新CARLA状态
                if self.carla_connected:
                    self.update_carla_state()
                
                # 发布心跳
                self.publish_status(f"{self.node_name} running. CARLA: {self.carla_connected}")
                
                rate.sleep()
                
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                time.sleep(1.0)
    
    def shutdown(self):
        """关闭节点"""
        rospy.loginfo(f"Shutting down {self.node_name}")
        self.running = False
        
        # 清理CARLA资源
        if self.carla_connected:
            try:
                if hasattr(self, 'vehicle'):
                    self.vehicle.destroy()
                    rospy.loginfo("CARLA vehicle destroyed")
            except Exception as e:
                rospy.logerr(f"Error during CARLA cleanup: {e}")

def main():
    try:
        # 创建桥接器实例
        bridge = CARLAROSBridge()
        
        # 注册关闭处理
        rospy.on_shutdown(bridge.shutdown)
        
        # 运行主循环
        bridge.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by ROS")
    except Exception as e:
        rospy.logerr(f"Fatal error in main: {e}")
    finally:
        rospy.loginfo("CARLA ROS Bridge shutdown complete")

if __name__ == '__main__':
    main()
