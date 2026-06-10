"""
模型保存、加载和部署工具

支持模型的序列化和推理服务
"""

import os
import json
import pickle
from typing import Any, Dict, Optional, Callable
import numpy as np
from datetime import datetime


class ModelSerializer:
    """模型序列化器"""
    
    @staticmethod
    def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None):
        """
        保存模型
        
        参数:
            model: 要保存的模型对象
            filepath: 保存路径
            metadata: 元数据字典
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        save_data = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        elif filepath.endswith('.npz'):
            np.savez_compressed(filepath, **save_data)
        else:
            with open(filepath + '.pkl', 'wb') as f:
                pickle.dump(save_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> tuple:
        """
        加载模型
        
        返回:
            (model, metadata)
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
        elif filepath.endswith('.npz'):
            save_data = np.load(filepath, allow_pickle=True)
        else:
            with open(filepath + '.pkl', 'rb') as f:
                save_data = pickle.load(f)
        
        model = save_data['model']
        metadata = save_data.get('metadata', {})
        
        print(f"模型已从 {filepath} 加载")
        return model, metadata


class QTableModel:
    """Q表模型包装器"""
    
    def __init__(self, q_table: np.ndarray, state_dim: int, action_dim: int):
        self.q_table = q_table
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def get_action(self, state: int, training: bool = False) -> int:
        """获取最优动作"""
        return int(np.argmax(self.q_table[state]))
    
    def get_q_values(self, state: int) -> np.ndarray:
        """获取Q值"""
        return self.q_table[state]
    
    def save(self, filepath: str):
        """保存Q表"""
        np.save(filepath, self.q_table)
        metadata = {
            'type': 'QTable',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        with open(filepath + '.meta.json', 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'QTableModel':
        """加载Q表"""
        q_table = np.load(filepath)
        with open(filepath + '.meta.json', 'r') as f:
            metadata = json.load(f)
        return cls(q_table, metadata['state_dim'], metadata['action_dim'])


class PolicyModel:
    """策略模型包装器"""
    
    def __init__(self, policy_weights: Dict, action_dim: int, state_dim: int):
        self.policy_weights = policy_weights
        self.action_dim = action_dim
        self.state_dim = state_dim
    
    def get_action(self, state: np.ndarray, training: bool = False) -> int:
        """获取动作"""
        probs = self._forward(state)
        if training:
            return int(np.random.choice(self.action_dim, p=probs))
        return int(np.argmax(probs))
    
    def _forward(self, state: np.ndarray) -> np.ndarray:
        """前向传播"""
        x = state.flatten().reshape(1, -1)
        for i, (w, b) in enumerate(zip(self.policy_weights['W'], self.policy_weights['b'])):
            x = np.dot(x, w) + b
            if i < len(self.policy_weights['W']) - 1:
                x = np.maximum(0, x)
        exp_x = np.exp(x - np.max(x))
        return (exp_x / np.sum(exp_x)).flatten()
    
    def save(self, filepath: str):
        """保存策略"""
        np.savez(filepath, **self.policy_weights)
        metadata = {
            'type': 'Policy',
            'action_dim': self.action_dim,
            'state_dim': self.state_dim
        }
        with open(filepath + '.meta.json', 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'PolicyModel':
        """加载策略"""
        weights = np.load(filepath, allow_pickle=True).item()
        with open(filepath + '.meta.json', 'r') as f:
            metadata = json.load(f)
        return cls(weights, metadata['action_dim'], metadata['state_dim'])


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """预测动作"""
        if hasattr(self.model, 'get_action'):
            action = self.model.get_action(state, training=False)
            return np.array([action])
        elif hasattr(self.model, 'select_action'):
            action = self.model.select_action(state, training=False)
            if isinstance(action, tuple):
                return np.array([action[0]])
            return np.array([action])
        else:
            raise ValueError("模型没有 predict/get_action/select_action 方法")
    
    def batch_predict(self, states: np.ndarray) -> np.ndarray:
        """批量预测"""
        actions = []
        for state in states:
            actions.append(self.predict(state)[0])
        return np.array(actions)


class RLAgentServer:
    """RL智能体服务器（简单的REST API）"""
    
    def __init__(self, model, host: str = 'localhost', port: int = 8000):
        self.model = model
        self.host = host
        self.port = port
        self.engine = InferenceEngine(model)
    
    def predict(self, state: list) -> dict:
        """预测接口"""
        state_array = np.array(state)
        action = self.engine.predict(state_array)[0]
        q_values = None
        
        if hasattr(self.model, 'get_q_values'):
            q_values = self.model.get_q_values(state).tolist()
        
        return {
            'action': int(action),
            'q_values': q_values,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset(self) -> dict:
        """重置接口"""
        return {'status': 'reset', 'timestamp': datetime.now().isoformat()}
    
    def get_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_type': type(self.model).__name__,
            'host': self.host,
            'port': self.port,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        ModelSerializer.save_model(self.model, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.model, _ = ModelSerializer.load_model(filepath)
        self.engine = InferenceEngine(self.model)


def export_for_deployment(model: Any, output_dir: str, format: str = 'onnx'):
    """
    导出模型用于部署
    
    参数:
        model: 模型对象
        output_dir: 输出目录
        format: 导出格式 ('onnx', 'torchscript', 'savedmodel')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if format == 'onnx':
        try:
            import torch
            import torch.onnx
            
            if hasattr(model, 'q_network'):
                model.q_network = model.q_network
                dummy_input = torch.randn(1, model.state_dim)
                torch.onnx.export(
                    model.q_network,
                    dummy_input,
                    os.path.join(output_dir, 'model.onnx'),
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                print(f"模型已导出到: {output_dir}/model.onnx")
        except ImportError:
            print("PyTorch未安装，无法导出ONNX格式")
    
    elif format == 'savedmodel':
        metadata = {
            'model_type': type(model).__name__,
            'export_time': datetime.now().isoformat(),
            'format': format
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        ModelSerializer.save_model(model, os.path.join(output_dir, 'model.pkl'))
        print(f"模型已导出到: {output_dir}")


def create_inference_script(model_path: str, output_path: str):
    """
    创建独立推理脚本
    
    参数:
        model_path: 模型路径
        output_path: 输出脚本路径
    """
    script_content = f'''#!/usr/bin/env python3
"""
独立推理脚本
模型: {model_path}
"""

import numpy as np
import pickle

def load_model(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['model']

def main():
    model = load_model('{model_path}')
    
    print("RL推理服务已启动")
    print("输入状态 (用空格分隔的数字): ")
    
    while True:
        try:
            state_input = input("> ")
            if state_input.lower() in ['exit', 'quit', 'q']:
                break
            
            state = np.array([float(x) for x in state_input.split()])
            
            if hasattr(model, 'get_action'):
                action = model.get_action(state, training=False)
            else:
                action = model.select_action(state, training=False)
            
            if isinstance(action, tuple):
                action = action[0]
            
            print(f"推荐动作: {{action}}")
            
            if hasattr(model, 'get_q_values'):
                q_values = model.get_q_values(state)
                print(f"Q值: {{q_values}}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {{e}}")

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"推理脚本已创建: {output_path}")
