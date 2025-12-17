# TrainingStrategies.py
import os
import math
import pickle
import numpy as np
from datetime import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# 课程学习管理器
class CurriculumManager:
    def __init__(self, env):
        self.env = env
        self.current_stage = 0
        self.stage_thresholds = [0.3, 0.5, 0.7, 0.85]  # 成功率阈值
        self.stage_configs = [
            # 阶段0: 入门
            {
                'pedestrian_cross': 2,      # 十字路口行人数量（减少）
                'pedestrian_normal': 1,     # 普通路段行人数量（减少）
                'pedestrian_speed_min': 0.5,  # 行人最低速度
                'pedestrian_speed_max': 1.0,  # 行人最高速度
                'max_episode_steps': 1200,   # 最大步数 (20秒 * 60FPS)
                'success_threshold': 0.3     # 进入下一阶段成功率
            },
            # 阶段1: 初级
            {
                'pedestrian_cross': 4,      # 逐步增加
                'pedestrian_normal': 2,
                'pedestrian_speed_min': 0.7,
                'pedestrian_speed_max': 1.3,
                'max_episode_steps': 1800,   # 30秒
                'success_threshold': 0.5
            },
            # 阶段2: 中级
            {
                'pedestrian_cross': 6,
                'pedestrian_normal': 3,
                'pedestrian_speed_min': 0.8,
                'pedestrian_speed_max': 1.5,
                'max_episode_steps': 2400,   # 40秒
                'success_threshold': 0.7
            },
            # 阶段3: 高级 (正常难度)
            {
                'pedestrian_cross': 8,
                'pedestrian_normal': 4,
                'pedestrian_speed_min': 1.0,
                'pedestrian_speed_max': 2.0,
                'max_episode_steps': 3600,   # 60秒
                'success_threshold': 0.85
            },
            # 阶段4: 专家 (挑战)
            {
                'pedestrian_cross': 10,     # 适当减少
                'pedestrian_normal': 5,
                'pedestrian_speed_min': 1.2,
                'pedestrian_speed_max': 2.5,
                'max_episode_steps': 3600,
                'success_threshold': 0.9
            }
        ]
        
        # 训练历史
        self.success_history = deque(maxlen=20)  # 记录最近20轮的成功情况
        self.reward_history = deque(maxlen=50)   # 记录最近50轮的奖励
        
    def update_stage(self, success, reward):
        """更新训练阶段"""
        # 记录历史
        self.success_history.append(1 if success else 0)
        self.reward_history.append(reward)
        
        # 计算最近成功率
        if len(self.success_history) >= 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0
            
            # 减少打印频率
            if len(self.success_history) % 10 == 0:
                print(f"课程学习 - 当前阶段: {self.current_stage}, 成功率: {success_rate:.2f}, 平均奖励: {avg_reward:.2f}")
            
            # 检查是否可以进入下一阶段
            if self.current_stage < len(self.stage_configs) - 1:
                next_stage_threshold = self.stage_configs[self.current_stage]['success_threshold']
                if success_rate >= next_stage_threshold and avg_reward > 5:
                    self.current_stage += 1
                    print(f"🎉 课程学习: 进阶到阶段 {self.current_stage}!")
                    return True
                    
            # 如果表现太差，退回上一阶段
            if self.current_stage > 0 and success_rate < 0.2:
                self.current_stage -= 1
                print(f"⚠️ 课程学习: 退回阶段 {self.current_stage}")
                return True
        
        return False
    
    def get_current_config(self):
        """获取当前阶段的配置"""
        return self.stage_configs[min(self.current_stage, len(self.stage_configs) - 1)]
    
    def apply_to_environment(self):
        """将当前阶段配置应用到环境"""
        config = self.get_current_config()
        return config


# 多目标优化器
class MultiObjectiveOptimizer:
    def __init__(self):
        # 定义优化目标及其权重（可动态调整）
        self.objectives = {
            'safety': {
                'weight': 0.35,  # 稍微降低安全权重
                'description': '安全避障和避免碰撞',
                'metrics': ['collision_avoidance', 'pedestrian_distance']
            },
            'efficiency': {
                'weight': 0.30,  # 提高效率权重
                'description': '快速到达目的地',
                'metrics': ['progress_speed', 'total_time']
            },
            'comfort': {
                'weight': 0.20,
                'description': '平稳驾驶体验',
                'metrics': ['smoothness', 'steering_changes']
            },
            'rule_following': {
                'weight': 0.15,
                'description': '遵守交通规则',
                'metrics': ['lane_keeping', 'speed_limit']
            }
        }
        
        # 指标跟踪
        self.metrics_history = {
            'safety': [],
            'efficiency': [],
            'comfort': [],
            'rule_following': []
        }
        
    def compute_composite_reward(self, metrics):
        """计算综合奖励值"""
        composite = 0
        
        for obj_name, obj_info in self.objectives.items():
            if obj_name in metrics:
                # 归一化处理每个目标的贡献
                normalized_value = self._normalize_metric(metrics[obj_name], obj_name)
                composite += normalized_value * obj_info['weight']
                
                # 记录指标历史
                self.metrics_history[obj_name].append(normalized_value)
        
        # 特殊惩罚项
        if metrics.get('collision', False):
            composite -= 8  # 减少碰撞惩罚
        if metrics.get('off_road', False):
            composite -= 3  # 减少偏离道路惩罚
        if metrics.get('dangerous_action', False):
            composite -= 2  # 减少危险动作惩罚
            
        return composite
    
    def _normalize_metric(self, value, metric_name):
        """归一化指标值到[0, 1]范围"""
        # 不同指标的归一化方式不同
        normalization_rules = {
            'safety': lambda x: min(max(x / 10, 0), 1),  # 假设安全分满分10
            'efficiency': lambda x: min(max(x / 100, 0), 1),  # 效率分满分100
            'comfort': lambda x: min(max((x + 5) / 10, 0), 1),  # 舒适度[-5, 5] -> [0, 1]
            'rule_following': lambda x: min(max(x, 0), 1)  # 规则遵循度[0, 1]
        }
        
        if metric_name in normalization_rules:
            return normalization_rules[metric_name](value)
        return min(max(value, 0), 1)  # 默认截断到[0, 1]
    
    def adjust_weights(self, performance_feedback):
        """根据性能反馈动态调整权重"""
        # 如果某个目标表现持续较差，增加其权重
        recent_performance = {}
        for obj in self.objectives:
            if len(self.metrics_history[obj]) >= 10:
                recent_avg = np.mean(self.metrics_history[obj][-10:])
                recent_performance[obj] = recent_avg
        
        if recent_performance:
            # 找到表现最差的目标
            worst_obj = min(recent_performance, key=recent_performance.get)
            best_obj = max(recent_performance, key=recent_performance.get)
            
            # 如果最差目标表现低于阈值，增加其权重
            if recent_performance[worst_obj] < 0.3:
                adjustment = 0.02  # 减少调整幅度
                self.objectives[worst_obj]['weight'] += adjustment
                self.objectives[best_obj]['weight'] -= adjustment
                
                # 确保权重总和为1
                total = sum(obj['weight'] for obj in self.objectives.values())
                for obj in self.objectives:
                    self.objectives[obj]['weight'] /= total
                
                if adjustment != 0:
                    print(f"动态权重调整: {worst_obj}权重↑ {adjustment:.3f}, {best_obj}权重↓ {adjustment:.3f}")
    
    def get_performance_report(self):
        """生成性能报告"""
        report = "多目标优化性能报告:\n"
        report += "=" * 50 + "\n"
        
        for obj_name, obj_info in self.objectives.items():
            history = self.metrics_history[obj_name]
            if history:
                avg = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)
                report += f"{obj_name}(权重:{obj_info['weight']:.2f}): 平均得分={avg:.3f}\n"
                report += f"  描述: {obj_info['description']}\n"
        
        return report


# 模仿学习管理器
class ImitationLearningManager:
    def __init__(self, expert_data_path=None):
        self.expert_data_path = expert_data_path
        self.expert_data = []
        self.is_pretrained = False
        
    def load_expert_data(self, path):
        """加载专家示范数据"""
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.expert_data = pickle.load(f)
                print(f"已加载 {len(self.expert_data)} 条专家示范数据")
                return True
            else:
                print(f"专家数据文件不存在: {path}")
                return False
        except Exception as e:
            print(f"加载专家数据失败: {e}")
            return False
    
    def collect_expert_demonstration(self, env, num_episodes=10):
        """收集专家示范数据（可以手动控制或使用规则控制器）"""
        print(f"开始收集专家示范数据 ({num_episodes}个episodes)...")
        
        demonstrations = []
        
        for episode in range(num_episodes):
            print(f"收集专家示范 Episode {episode + 1}/{num_episodes}")
            
            state = env.reset(episode)
            done = False
            episode_data = []
            
            step_count = 0
            max_steps = 60 * 60  # 最大60秒
            
            while not done and step_count < max_steps:
                # 这里可以使用规则控制器或手动控制
                # 示例：简单的规则控制器
                action = self._rule_based_controller(env)
                
                new_state, reward, done, _ = env.step(action)
                
                # 保存示范数据
                episode_data.append({
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': new_state.copy(),
                    'done': done
                })
                
                state = new_state
                step_count += 1
            
            demonstrations.extend(episode_data)
            env.cleanup_actors()
        
        # 保存专家数据
        self.expert_data = demonstrations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"expert_data_{timestamp}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(demonstrations, f)
        
        print(f"专家示范数据已保存到: {save_path}, 共 {len(demonstrations)} 条记录")
        return True
    
    def _rule_based_controller(self, env):
        """基于规则的控制器（作为专家示范）"""
        # 获取车辆状态
        vehicle_location = env.vehicle.get_location()
        velocity = env.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
        
        # 简单规则：保持速度在20-35 km/h，避免障碍物
        if speed_kmh < 20:
            return 2  # 加速
        elif speed_kmh > 35:
            return 0  # 减速
        else:
            # 检查前方障碍物
            has_obstacle_ahead = self._check_obstacle_ahead(env)
            if has_obstacle_ahead:
                return 0  # 减速
            else:
                return 1  # 保持
        
        return 1  # 默认保持
    
    def _check_obstacle_ahead(self, env):
        """检查前方是否有障碍物（简化版本）"""
        # 这里可以添加更复杂的障碍物检测逻辑
        # 暂时返回False
        return False
    
    def pretrain_with_behavioral_cloning(self, model, epochs=20):
        """使用行为克隆进行预训练"""
        if not self.expert_data:
            print("没有专家数据可用，跳过预训练")
            return model
        
        print(f"开始行为克隆预训练 ({epochs}个epochs)...")
        
        # 准备训练数据
        states = []
        actions = []
        
        for demo in self.expert_data:
            states.append(demo['state'])
            actions.append(demo['action'])
        
        # 将状态归一化
        states = np.array(states) / 255.0
        
        # 将动作转换为one-hot编码
        actions_onehot = tf.keras.utils.to_categorical(actions, num_classes=5)
        
        # 备份原始编译设置
        original_loss = model.loss
        original_optimizer = model.optimizer
        original_metrics = model.metrics_names
        
        # 重新编译模型用于分类任务
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练模型模仿专家行为
        history = model.fit(
            states, actions_onehot,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"预训练完成 - 最终准确率: {history.history['accuracy'][-1]:.3f}")
        
        # 恢复原始编译设置
        model.compile(
            optimizer=original_optimizer,
            loss=original_loss,
            metrics=original_metrics
        )
        
        self.is_pretrained = True
        return model
    
    def train_with_dagger(self, model, env, iterations=5, episodes_per_iter=5):
        """使用DAgger算法进行训练"""
        print(f"开始DAgger训练 ({iterations}次迭代，每次{episodes_per_iter}个episodes)...")
        
        aggregated_data = self.expert_data.copy()
        
        for iteration in range(iterations):
            print(f"\nDAgger 迭代 {iteration + 1}/{iterations}")
            
            # 使用当前策略收集数据
            new_demos = []
            
            for episode in range(episodes_per_iter):
                print(f"  收集数据 Episode {episode + 1}/{episodes_per_iter}")
                
                state = env.reset(episode)
                done = False
                step_count = 0
                max_steps = 60 * 60
                
                while not done and step_count < max_steps:
                    # 使用当前策略选择动作
                    qs = model.predict(np.array(state).reshape(-1, *state.shape) / 255, verbose=0)[0]
                    action = np.argmax(qs)
                    
                    # 执行动作
                    new_state, reward, done, _ = env.step(action)
                    
                    # 专家纠正（这里可以添加专家纠正逻辑）
                    # 如果策略动作与专家建议不同，使用专家动作
                    expert_action = self._rule_based_controller(env)
                    
                    # 保存数据（使用专家纠正后的动作）
                    new_demos.append({
                        'state': state.copy(),
                        'action': expert_action,  # 使用专家动作
                        'reward': reward,
                        'next_state': new_state.copy(),
                        'done': done
                    })
                    
                    state = new_state
                    step_count += 1
                
                env.cleanup_actors()
            
            # 合并数据
            aggregated_data.extend(new_demos)
            
            # 在合并数据上重新训练
            states = [d['state'] for d in aggregated_data]
            actions = [d['action'] for d in aggregated_data]
            
            states = np.array(states) / 255.0
            actions_onehot = tf.keras.utils.to_categorical(actions, num_classes=5)
            
            # 备份原始编译设置
            original_loss = model.loss
            original_optimizer = model.optimizer
            original_metrics = model.metrics_names
            
            # 重新编译用于分类
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 训练模型
            history = model.fit(
                states, actions_onehot,
                batch_size=32,
                epochs=10,
                validation_split=0.1,
                verbose=0
            )
            
            # 恢复原始编译设置
            model.compile(
                optimizer=original_optimizer,
                loss=original_loss,
                metrics=original_metrics
            )
            
            print(f"  训练完成 - 准确率: {history.history['accuracy'][-1]:.3f}")
        
        print("DAgger训练完成!")
        return model


# 优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, max_size=10000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.max_size = max_size
        self.alpha = alpha  # 优先级程度 (0 = 均匀采样, 1 = 完全优先级)
        self.beta_start = beta_start  # 重要性采样权重起始值
        self.beta_frames = beta_frames  # beta线性增长的帧数
        self.frame = 1
        
        # 使用循环缓冲区
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
    def __len__(self):
        return len(self.buffer)
    
    def beta(self):
        """线性递增的beta值，用于重要性采样权重"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, experience, error=None):
        """添加经验到缓冲区"""
        if error is None:
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = (abs(error) + 1e-5) ** self.alpha
            
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        """从缓冲区中采样一批经验"""
        if len(self.buffer) == 0:
            return [], [], []
            
        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        
        # 获取样本
        samples = [self.buffer[i] for i in indices]
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta())
        weights /= weights.max()  # 归一化
        
        # 更新帧计数器
        self.frame += 1
        
        return indices, samples, weights
    
    def update_priorities(self, indices, errors):
        """更新采样经验的优先级"""
        for idx, error in zip(indices, errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha