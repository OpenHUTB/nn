"""
模型管理器 - 加载和管理自动驾驶模型
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import carla
import config as cfg

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.braking_model = None
        self.driving_model = None
        self.models_loaded = False
        
        # 设置TensorFlow
        self._setup_tensorflow()
    
    def _setup_tensorflow(self):
        """设置TensorFlow配置"""
        print(f"TensorFlow版本: {tf.__version__}")
        
        # GPU配置
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ 找到 {len(gpus)} 个GPU，已启用内存增长")
            except RuntimeError as e:
                print(f"⚠️ GPU设置错误: {e}")
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                print("使用CPU运行")
        else:
            print("ℹ️ 未找到GPU，使用CPU运行")
    
    def load_models(self):
        """加载所有模型"""
        print("\n" + "="*40)
        print("加载自动驾驶模型")
        print("="*40)
        
        # 加载刹车模型
        self.braking_model = self._load_single_model(
            cfg.MODEL_PATHS['braking'],
            "刹车模型"
        )
        
        # 加载驾驶模型
        self.driving_model = self._load_single_model(
            cfg.MODEL_PATHS['driving'],
            "驾驶模型"
        )
        
        self.models_loaded = self.braking_model is not None and self.driving_model is not None
        
        if self.models_loaded:
            print("✅ 所有模型加载成功")
        else:
            print("❌ 模型加载失败")
            
        return self.models_loaded
    
    def _load_single_model(self, model_path, model_name):
        """加载单个模型"""
        if not os.path.exists(model_path):
            print(f"❌ {model_name}文件不存在: {model_path}")
            return None
        
        try:
            model = load_model(model_path)
            print(f"✅ {model_name}加载成功: {os.path.basename(model_path)}")
            return model
        except Exception as e:
            print(f"❌ {model_name}加载失败: {e}")
            return None
    
    def predict_action(self, current_state, vehicle_state=None):
        """预测动作"""
        if not self.models_loaded:
            print("⚠️ 模型未加载，使用默认动作")
            return 0  # 默认刹车
        
        try:
            # 预处理状态数据
            braking_state = self._preprocess_state(current_state, "braking")
            driving_state = self._preprocess_state(current_state, "driving")
            
            # 首先检查是否需要刹车
            braking_qs = self.braking_model.predict(braking_state, verbose=0)[0]
            braking_action = np.argmax(braking_qs)
            
            # 如果刹车模型判断为安全，再使用驾驶模型
            if braking_action == 1:  # 安全，可以行驶
                # 检查交通灯
                if vehicle_state and self._check_traffic_light(vehicle_state):
                    print("🚦 红灯 - 停车")
                    return 0
                
                # 使用驾驶模型选择具体动作
                driving_qs = self.driving_model.predict(driving_state, verbose=0)[0]
                driving_action = np.argmax(driving_qs)
                
                # 驾驶模型输出0-4，对应动作1-5
                return driving_action + 1
            else:
                # 刹车
                return 0
                
        except Exception as e:
            print(f"❌ 预测错误: {e}")
            return 0  # 出错时刹车
    
    def _preprocess_state(self, state_data, model_type):
        """预处理状态数据"""
        try:
            if model_type == "braking":
                # 刹车模型使用前两个状态
                state_array = np.array(state_data[:2])
            else:
                # 驾驶模型使用后两个状态
                state_array = np.array(state_data[2:])
            
            # 确保是二维数组
            if len(state_array.shape) == 1:
                state_array = state_array.reshape(1, -1)
            
            return state_array
        except Exception as e:
            print(f"状态预处理错误: {e}")
            return np.array([[0, 0]])
    
    def _check_traffic_light(self, vehicle_state):
        """检查交通灯状态（简化版本）"""
        # 这里可以扩展为实际的交通灯检测
        # 目前返回False表示没有红灯
        return False
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            'braking_model_loaded': self.braking_model is not None,
            'driving_model_loaded': self.driving_model is not None,
            'models_loaded': self.models_loaded,
            'braking_model_path': cfg.MODEL_PATHS['braking'] if self.braking_model else None,
            'driving_model_path': cfg.MODEL_PATHS['driving'] if self.driving_model else None
        }
        return info