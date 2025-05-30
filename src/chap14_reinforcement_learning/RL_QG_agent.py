import tensorflow as tf
import os
import numpy as np

class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
    #    pass    # 删掉这句话，并填写相应代码
        os.makedirs(self.model_dir, exist_ok=True)
        self.sess = None
        self.saver = None
        self.input_states = None
        self.Q_values = None

    def init_model(self):

        # 定义自己的 网络
        self.sess = tf.Session()
        # 定义输入状态，假设为8x8棋盘，3个通道（如当前玩家棋子、对手棋子、可行位置）
        self.input_states = tf.placeholder(tf.float32, shape=[None, 8, 8, 3], name="input_states")
        # 构建卷积神经网络
        conv1 = tf.layers.conv2d(inputs=self.input_states, filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, padding="same", activation=tf.nn.relu)
        # 扁平化层
        flat = tf.layers.flatten(conv2)
        # 全连接层
        dense = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
        # 输出层，64个动作的Q值
        self.Q_values = tf.layers.dense(inputs=dense, units=64, name="q_values")
        # 初始化变量和Saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # 补全代码
        

    def place(self,state,enables):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
       # action = 123456789    # 删掉这句话，并填写相应代码
       # 状态预处理
        state_input = np.array(state).reshape(1, -1).astype(np.float32)  # 转换为(1,64)形状
        
        # 前向传播获取Q值
        q_vals = self.sess.run(self.q_values, feed_dict={self.input_state: state_input})
        
        # 过滤合法动作并选择最优
        legal_q = q_vals[0][enables]
        if np.sum(legal_q) == 0:  # 所有合法动作Q值都为0的特殊情况处理
            return np.random.choice(np.where(enables)[0])
        
        max_q = np.max(legal_q)
        candidates = np.where(legal_q == max_q)[0]
        
        # 随机选择最优动作（解决多个最大值的情况）
        action = np.random.choice(candidates)

        return action

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    # 定义自己需要的函数
