import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse  
from models.dqn_agent import DQNAgent
from models.pruning import ModelPruner
from models.quantization import quantize_model
from envs.carla_environment import CarlaEnvironment 
import yaml

# --------------------------
# 解析命令行参数（保留你的逻辑）
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='CARLA DQN 训练/测试脚本')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='运行模式：train（训练）/ test（测试）')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    return parser.parse_args()

def load_config(config_path='configs/config.yaml'):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"成功加载配置文件：{config_path}")
        return config
    except Exception as e:
        print(f"加载配置文件失败：{e}")
        raise  

def train_model(config):
    print("=== 开始DQN训练 ===")
    try:
        # 初始化环境（已集成无抖动后方摄像头）
        print("初始化CARLA环境...")
        env = CarlaEnvironment()
        print("CARLA环境初始化成功（车辆已生成，摄像头挂载完成）")

        # 关键：获取完整图像形状（128,128,3），适配新版DQN
        state_shape = env.observation_space.shape  
        action_size = env.action_space.n
        print(f"状态形状：{state_shape}，动作维度：{action_size}")  
        
        # 初始化DQN（传state_shape，匹配新版DQN）
        agent = DQNAgent(state_shape=state_shape, action_size=action_size, config=config)
        print("DQN智能体初始化成功")

        # 优化器（新版DQN已内置，此处仅打印日志）
        print(f"优化器初始化成功（学习率：{config['train']['learning_rate']}）")

        episodes = config['train']['episodes']
        print(f"开始训练：共{episodes}轮Episode")
        for e in range(episodes):
            state = env.reset()
            state = state.astype(np.float32) / 255.0  # 图像归一化
            done = False
            total_reward = 0
            step = 0

            while not done and step < 500:  # 限制步数，避免死循环
                step += 1
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # 数据预处理
                next_state = next_state.astype(np.float32) / 255.0  
                reward = np.clip(reward, -10, 10)  

                # 记忆存储
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # 经验回放
                if len(agent.memory) > config['train']['batch_size']:
                    agent.replay(config['train']['batch_size'])

            # 打印训练日志（每5轮一次）
            if (e + 1) % 5 == 0:
                print(f"Episode {e+1:4d}/{episodes}, Total Reward: {total_reward:6.1f}, 探索率: {agent.epsilon:.4f}")

        # 模型优化（剪枝+量化）
        print("开始模型剪枝...")
        pruner = ModelPruner(agent.model)
        pruner.prune_model(amount=0.2)
        print("模型剪枝完成（移除20%权重）")

        print("开始模型量化...")
        agent.model = quantize_model(agent.model)
        print("模型量化完成")

        # 导出ONNX模型（适配图像输入）
        print("导出模型为ONNX格式...")
        export_to_onnx(agent.model, state_shape, config.get('model', {}).get('onnx_path', 'model.onnx'))
        print("模型导出成功！")

        # 保存模型权重
        torch.save(agent.model.state_dict(), "dqn_carla_final.pth")
        print("模型权重已保存：dqn_carla_final.pth")

    except Exception as e:
        print(f"训练过程出错：{e}")
        raise

# 适配图像输入的ONNX导出函数
def export_to_onnx(model, state_shape, file_path='model.onnx'):
    # 图像输入维度：(1, 3, H, W)，匹配CNN输入
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
        print(f"ONNX导出失败：{e}")
        raise

# 测试函数（保留）
def test_model(config):
    print("=== 开始测试 ===")
    try:
        env = CarlaEnvironment()
        state_shape = env.observation_space.shape
        action_size = env.action_space.n
        agent = DQNAgent(state_shape=state_shape, action_size=action_size, config=config)
        # 加载训练好的模型
        agent.model.load_state_dict(torch.load("dqn_carla_final.pth"))
        agent.model.eval()  # 评估模式
        
        print("开始测试（10轮）...")
        for e in range(10):
            state = env.reset()
            state = state.astype(np.float32) / 255.0
            done = False
            total_reward = 0
            step = 0
            while not done and step < 500:
                step += 1
                action = agent.act(state)  # 关闭探索，仅用模型预测
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32) / 255.0
                state = next_state
                total_reward += reward
            print(f"Test Episode {e+1}, Total Reward: {total_reward:.1f}")
        env.close()
    except Exception as e:
        print(f"测试过程出错：{e}")
        raise

# --------------------------
# 程序入口（保留你的命令行逻辑）
# --------------------------
if __name__ == "__main__":
    try:
        args = parse_args()  
        print(f"当前运行模式：{args.mode}")

        if args.mode == 'train':
            config = load_config(args.config)
            train_model(config)
        elif args.mode == 'test':
            config = load_config(args.config)
            test_model(config)
        else:
            print(f"无效模式：{args.mode}，仅支持 train / test")
    except Exception as e:
        print(f"\n程序异常退出：{e}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈
        exit(1)
