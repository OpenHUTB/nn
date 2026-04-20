# 强制打印脚本标识，确认运行的是新版本
print("=====================================")
print("✅ 运行的是最终版训练脚本（train_final.py）")
print("=====================================")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from models.dqn_agent import DQNAgent
from models.pruning import ModelPruner
from models.quantization import quantize_model
from envs.carla_environment import CarlaEnvironment 

def load_config(config_path='configs/config.yaml'):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✅ 成功加载配置文件：{config_path}")
        return config
    except Exception as e:
        raise ValueError(f"❌ 加载配置文件失败：{e}")

def train_model(config):
    """训练DQN模型（适配CARLA真实图像输入）"""
    # 1. 初始化设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备：{device}")

    # 2. 初始化CARLA环境
    print("🔧 初始化CARLA环境...")
    env = CarlaEnvironment()
    state_shape = env.observation_space.shape  # (128, 128, 3) 完整图像形状
    action_size = env.action_space.n
    print(f"✅ CARLA环境初始化成功 | 状态形状：{state_shape} | 动作维度：{action_size}")

    # 3. 初始化DQN智能体（仅传state_shape，绝对不含state_size）
    print("🔧 初始化DQN智能体...")
    agent = DQNAgent(
        state_shape=state_shape,  # 唯一正确的参数名
        action_size=action_size,
        config=config
    )
    print("✅ DQN智能体初始化成功")

    # 4. 训练参数
    episodes = config['train']['episodes']
    batch_size = config['train']['batch_size']
    reward_history = []  # 记录奖励历史

    # 5. 开始训练
    print(f"🚀 开始训练：共{episodes}轮Episode")
    for e in range(episodes):
        # 重置环境，获取初始状态
        state = env.reset()
        # 图像归一化（0-255 → 0-1）
        state = state.astype(np.float32) / 255.0
        done = False
        total_reward = 0
        step = 0

        while not done:
            step += 1
            # 选择动作
            action = agent.act(state)
            # 执行动作，获取环境反馈
            next_state, reward, done, _ = env.step(action)
            
            # 数据预处理
            next_state = next_state.astype(np.float32) / 255.0  # 归一化
            reward = np.clip(reward, -10, 10)  # 奖励裁剪，避免极端值

            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            # 更新状态
            state = next_state
            total_reward += reward

            # 经验回放（批量更新）
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # 防止单轮步数过多
            if step > 500:
                done = True

        # 记录奖励，打印训练日志
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else total_reward
        print(f"📊 Episode {e+1}/{episodes} | 总奖励：{total_reward:.2f} | 最近10轮平均：{avg_reward:.2f} | 探索率：{agent.epsilon:.4f}")

    # 6. 模型优化（剪枝+量化）
    print("\n🔧 开始模型优化（剪枝+量化）...")
    try:
        pruner = ModelPruner(agent.model)
        pruner.prune_model(amount=0.2)  # 剪枝20%参数
        agent.model = quantize_model(agent.model)  # 量化模型
        print("✅ 模型优化完成")
    except Exception as e:
        print(f"⚠️  模型优化失败（可忽略）：{e}")

    # 7. 导出ONNX模型
    try:
        export_to_onnx(agent.model, state_shape, device)
    except Exception as e:
        print(f"⚠️  ONNX导出失败（可忽略）：{e}")
    
    # 8. 保存模型权重
    torch.save(agent.model.state_dict(), "dqn_carla_model_final.pth")
    print("✅ 模型权重已保存：dqn_carla_model_final.pth")

    # 9. 清理环境
    env.close()
    print("\n🎉 训练完成！")

def export_to_onnx(model, state_shape, device, file_path='model_final.onnx'):
    """导出ONNX模型（适配CNN图像输入）"""
    # 构建dummy input：(1, 3, 128, 128)
    dummy_input = torch.randn(1, 3, state_shape[0], state_shape[1]).to(device)
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        opset_version=12,
        input_names=["input_image"],
        output_names=["action_q_values"],
        dynamic_axes={"input_image": {0: "batch_size"}, "action_q_values": {0: "batch_size"}}
    )
    print(f"✅ ONNX模型已导出：{file_path}")

if __name__ == "__main__":
    # 加载配置
    config = load_config()
    # 启动训练
    try:
        train_model(config)
    except Exception as e:
        print(f"❌ 训练过程出错：{e}")
        import traceback
        traceback.print_exc()  # 打印详细错误栈
        raise