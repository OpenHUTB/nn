# 操作系统接口模块 - 提供与操作系统交互的功能
import os
# 数值计算库 - Python科学计算的核心库
import numpy as np
# 深度学习框架 - Google开发的开源机器学习平台
import tensorflow as tf

class RL_QG_agent: # 定义了一个名为 RL_QG_agent 的类
    def __init__(self): # __init__  方法是类的构造函数，用于初始化类的实例
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi") # self.model_dir用于存储模型文件的目录路径。os.path.dirname(os.path.abspath(__file__))获取当前脚本文件的绝对路径，并提取其所在的目录
        # 用于初始化与模型保存、TensorFlow会话以及输入和输出张量相关的属性
        os.makedirs(self.model_dir, exist_ok = True)    # 创建模型保存目录（如果目录不存在则自动创建）
        self.sess = None                                # TensorFlow会话对象初始化占位
        self.saver = None                               # TensorFlow模型保存器初始化占位
        self.input_states = None                        # 神经网络输入占位符初始化占位
        self.Q_values = None                            # 神经网络输出的Q值初始化占位


    def init_model(self):
        '''
        self.input_states	网络输入（状态图像），大小为 [batch_size, 8, 8, 3]
        conv1, conv2	两层卷积网络，用于提取棋盘局部特征
        flat	卷积输出扁平化，供全连接层使用
        dense	一个隐藏层，用于提取高层语义特征
        self.Q_values	输出层，返回每个位置对应的 Q 值（动作的价值）
        self.target_Q	训练目标 Q 值（用于计算 loss）
        self.loss	使用 MSE 作为损失函数
        self.optimizer	使用 Adam 优化器进行参数更新
        self.saver	用于模型保存和恢复

        '''
        # 定义自己的 网络
        self.sess = tf.Session()
        # 定义输入状态，假设为8x8棋盘，3个通道（如当前玩家棋子、对手棋子、可行位置）
        self.input_states = tf.placeholder(tf.float32, shape = [None, 8, 8, 3], name = "input_states")
        # 构建卷积神经网络
        # 第1个卷积层：提取局部空间特征
        conv1 = tf.layers.conv2d(# 自动调整子图参数，优化布局避免元素重叠
            inputs = self.input_states,   # 输入张量，形状应为 [batch_size, height, width, channels]
            filters = 32,                 # 输出通道数：32个卷积核
            kernel_size = 3,              # 卷积核大小 3x3
            padding = "same",             # 输出大小与输入相同
            activation = tf.nn.relu       # ReLU 激活函数
            )

    # 第2个卷积层：提取更高级特征
        conv2 = tf.layers.conv2d(

            inputs = conv1,               # 输入：上层卷积层（conv1）的输出特征图
            filters = 64,                 # 输出通道数：64个卷积核
            kernel_size = 3,             #指的是卷积核的大小为 3×3
            padding = "same",            #这种填充方式能保证输出特征图的尺寸和输入特征图的尺寸相同

            activation = tf.nn.relu      # 使用 ReLU 激活函数，引入非线性
            )
        
        # 扁平化层
        flat = tf.layers.flatten(conv2)
        # 全连接层
        dense = tf.layers.dense(inputs = flat, units = 512, activation = tf.nn.relu)
        # 输出层，64个动作的Q值
        self.Q_values = tf.layers.dense(inputs = dense, units = 64, name = "q_values")
        # 初始化变量和Saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # 补全代码
        
    def place(self,state,enables):
        """
           根据当前棋盘状态和合法落子位置，选择最优落子位置。

           :param state: 当前棋盘状态，形状为 (8, 8, 3) 的数组
           :param enables: 合法落子位置的索引列表
           :return: 选择的落子位置索引，范围为 0-63
           """
        # 状态预处理
        state_input = np.array(state).reshape(1, 8, 8, 3).astype(np.float32)  # 转换为(1,64)形状
        
        # 前向传播获取Q值
        q_vals = self.sess.run(self.q_values, feed_dict = {self.input_states: state_input})
        
        # 过滤合法动作并选择最优
        
        legal_q = q_vals[0][enables]  # 从 Q 值矩阵中提取当前状态下的合法动作的 Q 值
        if np.sum(legal_q) == 0:  # 所有合法动作Q值都为 0 的特殊情况处理
            return np.random.choice(np.where(enables)[0])   # 随机选择一个合法动作
        
        max_q = np.max(legal_q)  # 找到合法动作中 Q 值最大的值
        candidates = np.where(legal_q == max_q)[0]  # 找到所有 Q 值等于最大 Q 值的动作索引
        
        # 随机选择最优动作 （解决多个最大值的情况）
        return np.random.choice(candidates)
    #save_model  和  load_model，用于保存和加载 TensorFlow 模型的参数
    # 保存模型
    def save_model(self):  
    # 使用TensorFlow的Saver对象保存模型参数
    # 参数保存为checkpoint格式，包含所有可训练变量的值
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))
    # 重新导入模型
    def load_model(self):
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    # 定义自己需要的函数
