# 导入必要的库
import os           # 操作系统接口，用于文件路径处理和目录操作
import numpy as np  # 数值计算库，用于数组操作和数学计算
import tensorflow as tf  # 深度学习框架，用于构建和训练神经网络

class RL_QG_agent:
    """黑白棋强化学习智能体，基于Q学习和卷积神经网络实现落子策略"""
    
    def __init__(self):
        """初始化智能体，设置模型保存路径和TensorFlow相关组件"""
        # 确定模型保存目录：当前脚本所在目录下的Reversi文件夹
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        os.makedirs(self.model_dir, exist_ok=True)  # 创建目录（若不存在）
        
        # TensorFlow相关组件占位符
        self.sess = None          # 会话对象，管理TensorFlow图的执行
        self.saver = None         # 模型保存器，用于保存和加载参数
        self.input_states = None  # 网络输入张量（棋盘状态）
        self.Q_values = None      # 网络输出张量（各位置Q值）


    def init_model(self):
        """构建卷积神经网络模型，用于预测黑白棋落子位置的Q值"""
        self.sess = tf.Session()  # 创建TensorFlow会话
        
        # 定义网络输入：[批次大小, 棋盘高度, 棋盘宽度, 通道数]
        self.input_states = tf.placeholder(
            tf.float32,              # 输入数据类型为32位浮点数
            shape=[None, 8, 8, 3],   # 输入张量的形状
            name="input_states"      # 该张量在计算图中的名称
        )
        
        # ========== 卷积层1：提取局部棋子模式特征 ==========
        # 32个3x3卷积核，捕捉相邻棋子的局部关系
        # 输出形状：[None, 8, 8, 32]
        conv1 = tf.layers.conv2d(
            inputs=self.input_states,
            filters=32,         # 32个卷积核，生成32个特征图
            kernel_size=3,      # 3x3卷积核，捕捉局部区域
            padding="same",     # 同尺寸填充，保持输出尺寸与输入一致
            activation=tf.nn.relu  # ReLU激活函数，引入非线性
        )

        # ========== 卷积层2：提取全局布局特征 ==========
        # 64个3x3卷积核，捕捉更复杂的棋子布局模式
        # 输出形状：[None, 8, 8, 64]
        conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 64,         # 特征图数量翻倍，增强特征表达能力
            kernel_size = 3,      # 使用3×3的卷积核，平衡特征提取能力与参数量
            padding = "same",     # 保持输出特征图尺寸与输入一致（补零填充）
            activation = tf.nn.relu   # ReLU激活函数，引入非线性并抑制负梯
        )

        # ========== 扁平化层：将多维特征转为一维向量 ==========
        # 输入形状[None, 8, 8, 64] → 输出形状[None, 4096]
        flat = tf.layers.flatten(conv2)

        # ========== 全连接层：学习特征间的全局关系 ==========
        # 512个神经元，通过ReLU激活学习非线性组合
        # 输出形状：[None, 512]
        dense = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

        # ========== 输出层：预测各位置的Q值 ==========
        # 64个神经元对应棋盘64个位置，直接输出Q值（无激活函数）
        self.Q_values = tf.layers.dense(inputs=dense, units=64, name="q_values")

        # 初始化所有变量并创建模型保存器
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def place(self, state, enables):
        """
        根据当前棋盘状态和合法落子位置，选择最优落子位置
        
        :param state: 当前棋盘状态，形状为(8, 8, 3)的NumPy数组
                      3个通道分别表示：黑棋位置、白棋位置、当前玩家
        :param enables: 合法落子位置的索引列表（0-63）
        :return: 选择的落子位置索引（0-63）
        """
        # 状态预处理：转换为适合网络输入的形状 [1, 8, 8, 3]
        state_input = np.array(state).reshape(1, 8, 8, 3).astype(np.float32)
        
        # 前向传播获取所有位置的Q值
        q_vals = self.sess.run(self.Q_values, feed_dict={self.input_states: state_input})
        
        # 提取合法位置的Q值
        legal_q = q_vals[0][enables]  # 形状与enables长度一致
        
        # 处理所有合法Q值为0的特殊情况（随机选择）
        if np.sum(legal_q) == 0:
            return np.random.choice(enables)  # 从合法位置中随机选
        
        # 选择Q值最大的位置（若有多个，随机选一个）
        max_q = np.max(legal_q)                # 最大Q值
        best_indices = np.where(legal_q == max_q)[0]  # 所有最大Q值的索引
        return enables[np.random.choice(best_indices)]  # 映射回原始位置


    def save_model(self):
        """保存训练好的模型参数到指定目录"""
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))
        print("模型已保存至", self.model_dir)


    def load_model(self):
        """从指定目录加载预训练的模型参数"""
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))
        print("模型已从", self.model_dir, "加载")
