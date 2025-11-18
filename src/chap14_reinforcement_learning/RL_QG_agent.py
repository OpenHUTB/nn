
import os
import numpy as np
import tensorflow as tf                 #要用到1.x的版本，我电脑上是2.x的版本
tf.compat.v1.disable_eager_execution()  # 禁用即时执行，兼容1.x语法

class RL_QG_agent:
    """黑白棋强化学习智能体，基于Q学习和卷积神经网络"""
    
    def __init__(self):
        """初始化智能体参数和模型路径"""
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        os.makedirs(self.model_dir, exist_ok=True)  # 创建模型保存目录
        
        # TensorFlow组件
        self.sess = None          # 会话
        self.saver = None         # 模型保存器
        self.input_states = None  # 输入张量
        self.Q_values = None      # Q值输出张量

    def init_model(self):
        """构建卷积神经网络模型"""
        self.sess = tf.compat.v1.Session()
        
        # 输入：[批次大小, 8, 8, 3]（3通道：黑棋、白棋、当前玩家）
        self.input_states = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 8, 8, 3], name="input_states"
        )
        
        # 卷积层1：32个3x3卷积核
        conv1 = tf.compat.v1.layers.conv2d(
            inputs=self.input_states,
            filters=32,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu
        )
        
        # 卷积层2：64个3x3卷积核
        conv2 = tf.compat.v1.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu
        )
        
        # 扁平化
        flat = tf.compat.v1.layers.flatten(conv2)
        
        # 全连接层
        dense = tf.compat.v1.layers.dense(
            inputs=flat,
            units=512,
            activation=tf.nn.relu
        )
        
        # 输出层：64个Q值（对应8x8棋盘）
        self.Q_values = tf.compat.v1.layers.dense(
            inputs=dense,
            units=64,
            name="q_values"
        )
        
        # 初始化变量和保存器
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

    def place(self, state, enables):
        """根据当前状态和合法动作选择最优落子位置"""
        # 状态预处理：[1, 8, 8, 3]
        state_input = np.array(state).reshape(1, 8, 8, 3).astype(np.float32)
        
        # 计算所有位置的Q值
        q_vals = self.sess.run(self.Q_values, feed_dict={self.input_states: state_input})
        
        # 提取合法动作的Q值
        legal_q = q_vals[0][enables]
        
        # 处理无有效Q值的情况（随机选择）
        if np.sum(legal_q) == 0:
            return np.random.choice(enables)
        
        # 选择Q值最大的动作（若有多个，随机选一个）
        max_q = np.max(legal_q)
        best_indices = np.where(legal_q == max_q)[0]
        return enables[np.random.choice(best_indices)]

    def save_model(self):
        """保存模型参数"""
        try:
            model_path = os.path.join(self.model_dir, 'parameter.ckpt')
            self.saver.save(self.sess, model_path)
            print("模型已保存至", self.model_dir)
        except Exception as e:
            print("保存模型时出错:", e)

    def load_model(self):
        """加载模型参数"""
        if self.sess is None:
            self.init_model()  # 未初始化则先构建模型
        
        model_path = os.path.join(self.model_dir, 'parameter.ckpt')
        if not os.path.exists(model_path + '.index'):
            print("模型文件不存在，使用初始化模型")
            return
        
        try:
            self.saver.restore(self.sess, model_path)
            print("模型已从", self.model_dir, "加载")
        except Exception as e:
            print("加载模型时出错:", e)