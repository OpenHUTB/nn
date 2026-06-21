# BipedalWalker-v3 模型保存与加载工具

import torch
from stable_baselines3 import PPO

def save_trained_model(model, save_path="bipedalwalker_ppo_model.zip"):
    """
    保存训练好的PPO模型
    :param model: 训练完成的PPO模型实例
    :param save_path: 模型保存路径
    """
    model.save(save_path)
    print(f"✅ 模型已成功保存至: {save_path}")


def load_trained_model(load_path="bipedalwalker_ppo_model.zip", env=None):
    """
    加载已训练的PPO模型
    :param load_path: 模型文件路径
    :param env: 模型对应的环境（可选）
    :return: 加载好的PPO模型实例
    """
    model = PPO.load(load_path, env=env)
    print(f"✅ 模型已成功从 {load_path} 加载")
    return model


def save_torch_state_dict(model, save_path="bipedalwalker_ppo_state.pth"):
    """
    保存模型的PyTorch状态字典（备用方案）
    :param model: PPO模型实例
    :param save_path: 状态字典保存路径
    """
    torch.save(model.policy.state_dict(), save_path)
    print(f"✅ 模型状态字典已保存至: {save_path}")


def load_torch_state_dict(model, load_path="bipedalwalker_ppo_state.pth"):
    """
    从PyTorch状态字典加载模型权重
    :param model: PPO模型实例
    :param load_path: 状态字典路径
    :return: 加载权重后的模型实例
    """
    model.policy.load_state_dict(torch.load(load_path))
    print(f"✅ 模型状态字典已成功加载")
    return model