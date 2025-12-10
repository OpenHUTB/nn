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
# 解析命令行参数（保留原有逻辑）
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
        # 初始化环境
        print("初始化CARLA环境...")
        env = CarlaEnvironment()
        print("CARLA环境初始化成功（车辆已生成）")

        # 关键修复1：获取完整图像形状，而非单维度
        state_shape = env.observation_space.shape  # (128, 128, 3)
        action_size = env.action_space.n
        print(f"状态形状：{state_shape}，动作维度：{action_size}")  # 修正打印文案
        
        # 关键修复2：传state_shape参数，匹配新版DQN
        agent = DQNAgent(state_shape=state_shape, action_size=action_size, config=config)
        print("DQN智能体初始化成功")

        # 优化器（新版DQN已内置优化器，此处可注释/保留，避免重复初始化）
        # optimizer = optim.Adam(agent.model.parameters(), lr=config['train']['learning_rate'])
        # criterion = nn.MSELoss()
        print(f"优化器初始化成功（学习率：{config['train']['learning_rate']}）")

        episodes = config['train']['episodes']
        print(f"开始训练：共{episodes}轮Episode")
        for e in range(episodes):
            state = env.reset()
            state = state.astype(np.float32) / 255.0  # 新增：图像归一化
            done = False
            total_reward = 0
            step = 0

            while not done and step < 500:  # 新增：限制步数，避免死循环
                step += 1
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # 数据预处理
                next_state = next_state.astype(np.float32) / 255.0  # 归一化
                reward = np.clip(reward, -10, 10)  # 奖励裁剪

                # 记忆
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # 经验回放
                if len(agent.memory) > config['train']['batch_size']:
                    agent.replay(config['train']['batch_size'])

            # 打印轮次日志
            if (e + 1) % 5 == 0:
                print(f"Episode {e+1:4d}/{episodes}, Total Reward: {total_reward:6.1f}, 探索率: {agent.epsilon:.4f}")

        # 剪枝和量化
        print("开始模型剪枝...")
        pruner = ModelPruner(agent.model)
        pruner.prune_model(amount=0.2)
        print("模型剪枝完成（移除20%权重）")

        print("开始模型量化...")
        agent.model = quantize_model(agent.model)
        print("模型量化完成")

        # 导出模型为 ONNX 格式（关键修复：适配图像输入维度）
        print("导出模型为ONNX格式...")
        export_to_onnx(agent.model, state_shape, config.get('model', {}).get('onnx_path', 'model.onnx'))
        print("模型导出成功！")

    except Exception as e:
        print(f"训练过程出错：{e}")
        raise

# 关键修复：适配图像输入的ONNX导出函数
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
    print("测试功能尚未实现，请补充代码后使用")

# --------------------------
# 程序入口（保留原有命令行逻辑）
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
        exit(1)  
