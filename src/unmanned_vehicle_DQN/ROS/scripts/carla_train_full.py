#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA DQN训练节点 - 完整使用原始代码
"""
import rospy
import sys
import os
import time
import threading
import numpy as np

# 设置路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

# ROS消息
from std_msgs.msg import String, Float32, Int32, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FullCarlaTrainer:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('carla_dqn_full_trainer')
        
        # 获取参数 - 使用Hyperparameters.py中的值作为默认值
        try:
            import Hyperparameters as hp
            default_episodes = hp.EPISODES
        except:
            default_episodes = 100
            
        self.episodes = rospy.get_param('~episodes', default_episodes)
        self.model_name = rospy.get_param('~model_name', 'YY_Enhanced_ObstacleAvoidance')
        
        # 创建发布器
        self.status_pub = rospy.Publisher('/carla/full_training/status', String, queue_size=10)
        self.reward_pub = rospy.Publisher('/carla/full_training/reward', Float32, queue_size=10)
        self.episode_pub = rospy.Publisher('/carla/full_training/episode', Int32, queue_size=10)
        self.epsilon_pub = rospy.Publisher('/carla/full_training/epsilon', Float32, queue_size=10)
        
        # 图像发布器
        self.image_pub = rospy.Publisher('/carla/full_training/image', Image, queue_size=1)
        self.bridge = CvBridge()
        
        # 导入原始代码
        self.import_modules()
        
        rospy.loginfo(f"✅ 完整训练节点初始化完成")
        rospy.loginfo(f"   训练轮次: {self.episodes}")
        rospy.loginfo(f"   模型名称: {self.model_name}")
        
        self.status_pub.publish(f"节点启动，准备训练{self.episodes}轮")
        
    def import_modules(self):
        """导入所有需要的模块"""
        try:
            # 导入主训练函数
            import main
            self.main_module = main
            rospy.loginfo("✅ 导入main.py模块")
            
            # 导入超参数
            import Hyperparameters
            self.hp = Hyperparameters
            rospy.loginfo(f"✅ 导入超参数: EPISODES={Hyperparameters.EPISODES}")
            
            # 导入环境
            import Environment
            self.Environment = Environment
            rospy.loginfo("✅ 导入环境模块")
            
            # 导入模型
            import Model
            self.Model = Model
            rospy.loginfo("✅ 导入模型模块")
            
            self.import_success = True
            
        except Exception as e:
            rospy.logerr(f"❌ 导入失败: {e}")
            self.import_success = False
            
    def publish_training_status(self, episode, reward, epsilon):
        """发布训练状态"""
        # 发布状态
        status_msg = String()
        status_msg.data = f"Episode {episode}: Reward={reward:.2f}, Epsilon={epsilon:.3f}"
        self.status_pub.publish(status_msg)
        
        # 发布奖励
        reward_msg = Float32()
        reward_msg.data = reward
        self.reward_pub.publish(reward_msg)
        
        # 发布回合数
        episode_msg = Int32()
        episode_msg.data = episode
        self.episode_pub.publish(episode_msg)
        
        # 发布探索率
        epsilon_msg = Float32()
        epsilon_msg.data = epsilon
        self.epsilon_pub.publish(epsilon_msg)
        
    def publish_image(self, cv_image):
        """发布图像"""
        try:
            if cv_image is not None and len(cv_image.shape) == 3:
                # 确保图像是8位
                if cv_image.dtype != np.uint8:
                    cv_image = cv_image.astype(np.uint8)
                
                # 调整大小以便显示
                display_image = cv2.resize(cv_image, (640, 480))
                
                # 转换为ROS消息
                ros_image = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = "carla_camera"
                
                self.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logwarn(f"发布图像失败: {e}")
    
    def run_original_training(self):
        """运行原始训练代码"""
        try:
            rospy.loginfo("🚗 开始原始CARLA DQN训练")
            
            # 这里直接调用你原来的main.py中的函数
            # 但需要稍作修改以适应ROS
            
            # 1. 设置GPU内存
            import tensorflow as tf
            gpu_options = tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=self.hp.MEMORY_FRACTION
            )
            tf.compat.v1.keras.backend.set_session(
                tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
            
            # 2. 创建目录
            for dir_name in ['models', 'expert_data', 'logs', 'training_stats']:
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
            
            # 3. 创建智能体和环境
            rospy.loginfo("创建增强版智能体和环境...")
            agent = self.Model.EnhancedDQNAgent(
                use_dueling=True, 
                use_per=True,
                use_curriculum=True,
                use_multi_objective=True,
                use_attention=True,
                use_enhanced_model=True
            )
            
            env = self.Environment.CarEnv(obstacle_detection_mode='advanced')
            
            # 4. 设置训练策略
            agent.setup_training_strategies(env)
            
            # 5. 启动训练线程
            trainer_thread = threading.Thread(target=agent.train_in_loop, daemon=True)
            trainer_thread.start()
            
            while not agent.training_initialized:
                time.sleep(0.01)
            
            # 6. 训练循环
            rospy.loginfo(f"开始{self.episodes}轮训练...")
            
            success_count = 0
            scores = []
            
            for episode in range(1, self.episodes + 1):
                if rospy.is_shutdown():
                    rospy.loginfo("ROS关闭，停止训练")
                    break
                
                # 重置环境
                if agent.curriculum_manager:
                    config = agent.curriculum_manager.get_current_config()
                    current_state = env.reset(episode, curriculum_config=config)
                else:
                    current_state = env.reset(episode)
                
                # 重置统计
                score = 0
                step = 1
                done = False
                
                # 获取最大步数
                if agent.curriculum_manager:
                    config = agent.curriculum_manager.get_current_config()
                    max_steps = config['max_episode_steps']
                else:
                    max_steps = self.hp.SECONDS_PER_EPISODE * self.hp.FPS
                
                # 单轮训练
                while not done and step < max_steps:
                    # 选择动作
                    if np.random.random() > self.hp.EPSILON:
                        qs = agent.get_qs(current_state)
                        action = np.argmax(qs)
                    else:
                        action = np.random.randint(0, 5)
                        time.sleep(1 / self.hp.FPS)
                    
                    # 执行动作
                    new_state, reward, done, _ = env.step(action)
                    
                    # 判断是否为重要经验
                    is_obstacle_experience = (env.last_ped_distance < 10.0)
                    is_success_experience = (reward > 2.0 and not done)
                    
                    # 更新经验回放
                    agent.update_replay_memory(
                        (current_state, action, reward, new_state, done),
                        is_obstacle=is_obstacle_experience,
                        is_success=is_success_experience
                    )
                    
                    score += reward
                    current_state = new_state
                    step += 1
                    
                    # 每10步发布一次图像
                    if step % 10 == 0:
                        self.publish_image(current_state)
                    
                    if done:
                        break
                
                # 发布本轮结果
                self.publish_training_status(episode, score, self.hp.EPSILON)
                
                # 记录分数
                scores.append(score)
                success = score > 8
                if success:
                    success_count += 1
                
                # 清理环境
                env.cleanup_actors()
                
                # 衰减探索率
                if self.hp.EPSILON > self.hp.MIN_EPSILON:
                    self.hp.EPSILON *= self.hp.EPSILON_DECAY
                    self.hp.EPSILON = max(self.hp.MIN_EPSILON, self.hp.EPSILON)
                
                # 每10轮保存一次检查点
                if episode % 10 == 0:
                    model_path = f"models/{self.model_name}_checkpoint_ep{episode}.model"
                    agent.save_model(model_path)
                    rospy.loginfo(f"检查点已保存: {model_path}")
                
                # 显示进度
                rospy.loginfo(f"Episode {episode}/{self.episodes}: Score={score:.2f}, "
                             f"Success={success_count}, Epsilon={self.hp.EPSILON:.3f}")
            
            # 训练完成
            rospy.loginfo("🎉 训练完成！")
            
            # 保存最终模型
            if scores:
                final_avg = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                final_model = f"models/{self.model_name}_final_avg{final_avg:.2f}.model"
                agent.save_model(final_model)
                rospy.loginfo(f"最终模型已保存: {final_model}")
            
            # 清理
            agent.terminate = True
            trainer_thread.join()
            env.cleanup_actors()
            
            self.status_pub.publish("训练完成")
            
        except Exception as e:
            rospy.logerr(f"训练失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.status_pub.publish(f"训练失败: {str(e)}")
    
    def run(self):
        """主运行函数"""
        rospy.loginfo("=" * 60)
        rospy.loginfo("CARLA DQN完整训练系统")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"训练轮次: {self.episodes}")
        rospy.loginfo(f"使用模型: {self.model_name}")
        rospy.loginfo("")
        rospy.loginfo("📡 发布的话题:")
        rospy.loginfo("  /carla/full_training/status   - 训练状态")
        rospy.loginfo("  /carla/full_training/reward   - 实时奖励")
        rospy.loginfo("  /carla/full_training/episode  - 当前回合")
        rospy.loginfo("  /carla/full_training/epsilon  - 探索率")
        rospy.loginfo("  /carla/full_training/image    - 训练图像")
        rospy.loginfo("")
        rospy.loginfo("⏳ 5秒后开始训练...")
        
        time.sleep(5)
        
        # 在新线程中运行训练
        if self.import_success:
            train_thread = threading.Thread(target=self.run_original_training)
            train_thread.daemon = True
            train_thread.start()
        else:
            rospy.logerr("导入失败，无法开始训练")
            return
        
        # 保持节点运行
        rospy.spin()

def main():
    trainer = FullCarlaTrainer()
    trainer.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("训练节点被中断")
    except Exception as e:
        rospy.logerr(f"节点异常: {e}")
