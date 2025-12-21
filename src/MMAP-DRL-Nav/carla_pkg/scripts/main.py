#!/usr/bin/env python
# coding=utf-8
# ROS1节点版：CARLA DQN训练/测试脚本
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse  
import rospy  # 新增：ROS1核心库
from std_msgs.msg import Float32MultiArray  # 新增：发布控制指令
from sensor_msgs.msg import Image           # 新增：发布相机图像
from cv_bridge import CvBridge              # 新增：图像格式转换
# 注意：修改导入路径（因为文件都在同一目录，去掉models/envs前缀）
from dqn_agent import DQNAgent
from pruning import ModelPruner
from quantization import quantize_model
from carla_environment import CarlaEnvironment 
import yaml

# --------------------------
# 新增：ROS节点核心类（封装CARLA+ROS通信）
# --------------------------
class CarlaDQNROSNode:
    def __init__(self):
        # 1. 初始化ROS节点
        rospy.init_node('carla_dqn_node', anonymous=False)
        rospy.loginfo("【ROS】CARLA DQN节点启动成功！")
        
        # 2. 创建ROS发布者
        self.control_pub = rospy.Publisher('/carla/control', Float32MultiArray, queue_size=10)
        self.image_pub = rospy.Publisher('/carla/camera', Image, queue_size=10)
        self.bridge = CvBridge()
        
        # 3. 初始化配置（默认加载config.yaml，可通过ROS参数覆盖）
        self.config_path = rospy.get_param("~config_path", "configs/config.yaml")
        self.config = load_config(self.config_path)
        rospy.loginfo(f"【ROS】加载配置文件：{self.config_path}")
        
        # 4. 初始化CARLA环境（替换原有环境初始化）
        self.env = CarlaEnvironment()
        # 重写CARLA环境的相机回调，发布到ROS
        self.env.camera.listen(lambda data: self._camera_callback(data))
        rospy.loginfo("【ROS】CARLA环境初始化成功！")
        
        # 5. DQN智能体（后续在train/test中初始化）
        self.agent = None
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        rospy.loginfo(f"【ROS】状态形状：{self.state_shape}，动作维度：{self.action_size}")
        
        # 6. ROS循环频率
        self.rate = rospy.Rate(5)  # 5Hz，适配虚拟机性能

    # 新增：相机回调→发布ROS图像
    def _camera_callback(self, carla_image):
        img_array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((carla_image.height, carla_image.width, 4))[:, :, :3]
        ros_img = self.bridge.cv2_to_imgmsg(img_array, encoding='bgr8')
        self.image_pub.publish(ros_img)

    # 新增：发布控制指令到ROS
    def publish_control(self, throttle, steer):
        control_msg = Float32MultiArray()
        control_msg.data = [throttle, steer]
        self.control_pub.publish(control_msg)
        rospy.loginfo(f"【ROS】发布控制指令：油门={throttle:.2f}，转向={steer:.2f}")

# --------------------------
# 原有逻辑：解析命令行参数（兼容ROS）
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='CARLA DQN 训练/测试脚本（ROS版）')
    parser.add_argument('--mode', type=str, required=False, choices=['train', 'test'],
                        help='运行模式：train（训练）/ test（测试），可通过ROS参数覆盖')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径，可通过ROS参数覆盖')
    return parser.parse_args()

def load_config(config_path='configs/config.yaml'):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        rospy.loginfo(f"成功加载配置文件：{config_path}")  # 替换print为rospy.loginfo
        return config
    except Exception as e:
        rospy.logerr(f"加载配置文件失败：{e}")  # 替换print为rospy.logerr
        raise  

# --------------------------
# 原有逻辑：训练函数（适配ROS）
# --------------------------
def train_model(node, config):
    rospy.loginfo("=== 开始DQN训练（ROS版）===")
    try:
        # 初始化DQN智能体
        node.agent = DQNAgent(state_shape=node.state_shape, action_size=node.action_size, config=config)
        rospy.loginfo("DQN智能体初始化成功")

        episodes = config['train']['episodes']
        rospy.loginfo(f"开始训练：共{episodes}轮Episode")
        for e in range(episodes):
            # ROS节点关闭时停止训练
            if rospy.is_shutdown():
                rospy.loginfo("【ROS】节点已关闭，停止训练")
                break
            
            state = node.env.reset()
            state = state.astype(np.float32) / 255.0
            done = False
            total_reward = 0
            step = 0

            while not done and step < 500 and not rospy.is_shutdown():
                step += 1
                action = node.agent.act(state)
                next_state, reward, done, _ = node.env.step(action)
                
                # 数据预处理
                next_state = next_state.astype(np.float32) / 255.0  
                reward = np.clip(reward, -10, 10)  

                # 记忆存储
                node.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # 经验回放
                if len(node.agent.memory) > config['train']['batch_size']:
                    node.agent.replay(config['train']['batch_size'])

                # 发布控制指令到ROS（从action解析油门/转向，需匹配你的action映射）
                # 替换下面的映射规则为你实际的action→控制指令逻辑
                if action == 0:
                    throttle, steer = 0.4, 0.0
                elif action == 1:
                    throttle, steer = 0.3, -0.2
                elif action == 2:
                    throttle, steer = 0.3, 0.2
                else:
                    throttle, steer = 0.0, 0.0
                node.publish_control(throttle, steer)

                # ROS频率控制
                node.rate.sleep()

            # 打印训练日志（每5轮一次）
            if (e + 1) % 5 == 0:
                rospy.loginfo(f"Episode {e+1:4d}/{episodes}, Total Reward: {total_reward:6.1f}, 探索率: {node.agent.epsilon:.4f}")

        # 模型优化（剪枝+量化）
        rospy.loginfo("开始模型剪枝...")
        pruner = ModelPruner(node.agent.model)
        pruner.prune_model(amount=0.2)
        rospy.loginfo("模型剪枝完成（移除20%权重）")

        rospy.loginfo("开始模型量化...")
        node.agent.model = quantize_model(node.agent.model)
        rospy.loginfo("模型量化完成")

        # 导出ONNX模型
        rospy.loginfo("导出模型为ONNX格式...")
        export_to_onnx(node.agent.model, node.state_shape, config.get('model', {}).get('onnx_path', 'model.onnx'))
        rospy.loginfo("模型导出成功！")

        # 保存模型权重
        torch.save(node.agent.model.state_dict(), "dqn_carla_final.pth")
        rospy.loginfo("模型权重已保存：dqn_carla_final.pth")

    except Exception as e:
        rospy.logerr(f"训练过程出错：{e}")
        raise

# --------------------------
# 原有逻辑：ONNX导出（保留）
# --------------------------
def export_to_onnx(model, state_shape, file_path='model.onnx'):
    dummy_input = torch.randn(1, 3, state_shape[0], state_shape[1]).to(next(model.parameters()).device)
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            file_path, 
            opset_version=12,
            input_names=["input_image"],
            output_names=["action_q_values"],
            dynamic_axes={"input_image": {0: "batch_size"}, "action_q_values": {0: "batch_size"}}
        )
    except Exception as e:
        rospy.logerr(f"ONNX导出失败：{e}")
        raise

# --------------------------
# 原有逻辑：测试函数（适配ROS）
# --------------------------
def test_model(node, config):
    rospy.loginfo("=== 开始测试（ROS版）===")
    try:
        node.agent = DQNAgent(state_shape=node.state_shape, action_size=node.action_size, config=config)
        # 加载训练好的模型
        node.agent.model.load_state_dict(torch.load("dqn_carla_final.pth"))
        node.agent.model.eval()  # 评估模式
        rospy.loginfo("模型加载成功，进入评估模式")
        
        rospy.loginfo("开始测试（10轮）...")
        for e in range(10):
            if rospy.is_shutdown():
                rospy.loginfo("【ROS】节点已关闭，停止测试")
                break
            
            state = node.env.reset()
            state = state.astype(np.float32) / 255.0
            done = False
            total_reward = 0
            step = 0
            while not done and step < 500 and not rospy.is_shutdown():
                step += 1
                action = node.agent.act(state)
                next_state, reward, done, _ = node.env.step(action)
                next_state = next_state.astype(np.float32) / 255.0
                state = next_state
                total_reward += reward

                # 发布控制指令到ROS
                if action == 0:
                    throttle, steer = 0.4, 0.0
                elif action == 1:
                    throttle, steer = 0.3, -0.2
                elif action == 2:
                    throttle, steer = 0.3, 0.2
                else:
                    throttle, steer = 0.0, 0.0
                node.publish_control(throttle, steer)

                node.rate.sleep()
            rospy.loginfo(f"Test Episode {e+1}, Total Reward: {total_reward:.1f}")
        node.env.close()
    except Exception as e:
        rospy.logerr(f"测试过程出错：{e}")
        raise

# --------------------------
# 程序入口（ROS适配版）
# --------------------------
if __name__ == "__main__":
    try:
        # 1. 解析命令行参数
        args = parse_args()  
        
        # 2. 创建ROS节点实例
        node = CarlaDQNROSNode()
        
        # 3. 优先使用ROS参数，其次命令行参数
        mode = rospy.get_param("~mode", args.mode)
        if not mode:
            rospy.logerr("请指定运行模式：--mode train/test 或设置ROS参数 ~mode")
            exit(1)
        
        rospy.loginfo(f"当前运行模式：{mode}")

        # 4. 执行训练/测试
        if mode == 'train':
            train_model(node, node.config)
        elif mode == 'test':
            test_model(node, node.config)
        else:
            rospy.logerr(f"无效模式：{mode}，仅支持 train / test")
        
        # 5. ROS保持运行（直到手动关闭）
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("【ROS】节点已手动停止！")
    except Exception as e:
        rospy.logerr(f"\n程序异常退出：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
