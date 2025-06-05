import os
import numpy as np
import tensorflow as tf


class RLQGAgent:
    """
    基于深度强化学习的棋盘游戏智能体（Q-Learning）
    - 使用卷积神经网络（CNN）提取棋盘特征
    - 通过Q值预测合法动作
    """

    def __init__(self):
        """初始化模型目录和TensorFlow会话"""
        # 获取模型保存目录（与当前脚本同级的 Reversi 文件夹）
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        os.makedirs(self.model_dir, exist_ok=True)  # 创建目录（如果不存在）

        # TensorFlow 会话和模型相关属性
        self.sess = None
        self.saver = None
        self.input_states = None  # 输入占位符
        self.q_values = None      # 输出Q值
        self.build_model()        # 初始化模型

    def build_model(self):
        """
        构建卷积神经网络模型
        输入: [batch_size, 8, 8, 3] 棋盘状态（玩家棋子、对手棋子、合法位置）
        输出: [batch_size, 64]    每个位置的Q值
        """
        self.sess = tf.Session()
        
        # 定义输入占位符
        self.input_states = tf.placeholder(
            tf.float32,
            shape=[None, 8, 8, 3],
            name="input_states"
        )

        # 卷积层1: 提取局部特征
        conv1 = tf.layers.conv2d(
            inputs=self.input_states,
            filters=32,               # 输出通道数
            kernel_size=3,            # 卷积核大小
            padding="same",           # 保持输出尺寸
            activation=tf.nn.relu
        )

        # 卷积层2: 提取高级特征
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu
        )

        # 展平层
        flat = tf.layers.flatten(conv2)

        # 全连接层: 提取语义特征
        dense = tf.layers.dense(
            inputs=flat,
            units=512,
            activation=tf.nn.relu
        )

        # 输出层: 64个动作的Q值
        self.q_values = tf.layers.dense(
            inputs=dense,
            units=64,
            name="q_values"
        )

        # 初始化变量和模型保存器
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def place(self, state, enables):
        """
        根据当前状态选择合法动作
        Args:
            state: 当前棋盘状态 (8x8x3)
            enables: 合法动作掩码 (64位布尔数组)
        Returns:
            action: 0-63 的合法动作索引
        """
        # 状态预处理：转换为张量并归一化
        state_input = np.array(state).reshape(1, 8, 8, 3).astype(np.float32)

        # 获取Q值
        q_values = self.sess.run(
            self.q_values,
            feed_dict={self.input_states: state_input}
        )[0]  # 取第一个样本的Q值

        # 过滤合法动作
        legal_actions = np.where(enables)[0]
        if len(legal_actions) == 0:
            raise ValueError("No legal actions available!")

        legal_q = q_values[legal_actions]  # 合法动作对应的Q值

        # 特殊情况处理：所有Q值为0时随机选择
        if np.all(legal_q == 0):
            return np.random.choice(legal_actions)

        # 找到最大Q值对应的所有候选动作
        max_q = np.max(legal_q)
        candidates = np.where(legal_q == max_q)[0]

        # 随机选择最优动作（处理多个最大值的情况）
        return legal_actions[candidates[np.random.choice(len(candidates))]]

    def save_model(self):
        """保存模型参数"""
        try:
            self.saver.save(self.sess, os.path.join(self.model_dir, "parameter.ckpt"))
        except Exception as e:
            print(f"[ERROR] Model save failed: {e}")

    def load_model(self):
        """加载模型参数"""
        try:
            self.saver.restore(self.sess, os.path.join(self.model_dir, "parameter.ckpt"))
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")

    def close(self):
        """关闭会话"""
        if self.sess:
            self.sess.close()


# 示例用法
if __name__ == "__main__":
    agent = RLQGAgent()
    
    # 测试输入（示例）
    dummy_state = np.zeros((8, 8, 3))
    dummy_enables = np.random.randint(0, 2, size=64, dtype=bool)
    
    try:
        action = agent.place(dummy_state, dummy_enables)
        print(f"Selected action: {action}")
    finally:
        agent.close()
