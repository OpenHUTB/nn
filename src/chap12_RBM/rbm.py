# python: 2.7
# encoding: utf-8
# 导入numpy模块并命名为np
import numpy as np # 导入NumPy库用于高效数值计算
import sys

class RBM:
    """Restricted Boltzmann Machine (RBM) 实现
    
    RBM是一种生成式神经网络，可学习输入数据的概率分布，常用于降维、特征提取和生成新数据。
    它由一个可见层(输入层)和一个隐藏层组成，层间全连接但层内无连接。
    """

    def __init__(self, n_hidden=2, n_observe=784):
        """
        初始化受限玻尔兹曼机（RBM）模型参数

        Args:
            n_hidden (int): 隐藏层单元数量（默认 2）
            n_observe (int): 可见层单元数量（默认 784，如 MNIST 图像 28x28）

        Raises:
            ValueError: 若输入参数非正整数则抛出异常
        """
        # 参数验证：确保隐藏层和可见层单元数量为正整数
        if not (isinstance(n_hidden, int) and n_hidden > 0):
            raise ValueError("隐藏层单元数量 n_hidden 必须为正整数")
        if not (isinstance(n_observe, int) and n_observe > 0):
            raise ValueError("可见层单元数量 n_observe 必须为正整数")
            
        # 模型架构参数
        self.n_hidden = n_hidden  # 隐藏层神经元个数，控制特征表达能力
        self.n_observe = n_observe  # 可见层神经元个数，对应输入数据维度
        
        # 权重矩阵初始化：连接可见层和隐藏层的参数
        # 使用Xavier初始化，有助于缓解梯度消失/爆炸问题
        init_std = np.sqrt(2.0 / (self.n_observe + self.n_hidden))
        self.W = np.random.normal(0, init_std, size=(self.n_observe, self.n_hidden))
        
        # 偏置向量初始化：分别对应可见层和隐藏层
        self.b_h = np.zeros(n_hidden)  # 隐藏层偏置，影响神经元激活难易度
        self.b_v = np.zeros(n_observe)  # 可见层偏置，影响数据重构质量

    def _sigmoid(self, x):
        """Sigmoid激活函数，将输入值映射到(0,1)区间，表示神经元激活概率"""
        return 1.0 / (1 + np.exp(-x))

    def _sample_binary(self, probs):
        """
        伯努利采样：根据给定概率生成0或1
        
        Args:
            probs (np.ndarray): 每个神经元的激活概率
            
        Returns:
            np.ndarray: 二值化的采样结果(0或1)
        """
        return np.random.binomial(1, probs)
    
    def train(self, data):
        """
        使用Contrastive Divergence (CD) 算法训练RBM
        
        Args:
            data (np.ndarray): 训练数据，形状为 (n_samples, n_observe)
        """
        # 数据预处理：展平多维输入为二维数组
        data_flat = data.reshape(data.shape[0], -1)  
        n_samples = data_flat.shape[0]  # 样本总数
        
        # 训练超参数设置
        learning_rate = 0.1  # 控制参数更新步长，过大可能导致震荡，过小收敛缓慢
        epochs = 10  # 训练轮数，每轮遍历整个数据集一次
        batch_size = 100  # 批处理大小，每次更新使用的样本数，影响训练稳定性和速度
        
        # 主训练循环
        for epoch in range(epochs):
            np.random.shuffle(data_flat)  # 每轮打乱数据顺序，增加训练随机性
            
            # 小批量梯度下降
            for i in range(0, n_samples, batch_size):
                # 获取当前批次数据并转换为浮点型
                batch = data_flat[i:i + batch_size]
                v0 = batch.astype(np.float64)  # 初始可见层状态(输入数据)
                
                # 正相传播(Positive phase)：计算隐藏层激活概率和采样
                h0_prob = self._sigmoid(np.dot(v0, self.W) + self.b_h)  # 隐藏层激活概率
                h0_sample = self._sample_binary(h0_prob)  # 基于概率的二值采样
                
                # 负相传播(Negative phase)：重构可见层并再次计算隐藏层
                v1_prob = self._sigmoid(np.dot(h0_sample, self.W.T) + self.b_v)  # 重构可见层概率
                v1_sample = self._sample_binary(v1_prob)  # 重构可见层采样
                h1_prob = self._sigmoid(np.dot(v1_sample, self.W) + self.b_h)  # 重构后的隐藏层概率
                
                # 计算梯度：基于CD算法(对比散度)
                # 正梯度项：输入数据与初始隐藏层状态的关联
                # 负梯度项：重构数据与重构隐藏层状态的关联
                dW = np.dot(v0.T, h0_sample) - np.dot(v1_sample.T, h1_prob)
                db_v = np.sum(v0 - v1_sample, axis=0)  # 可见层偏置梯度
                db_h = np.sum(h0_sample - h1_prob, axis=0)  # 隐藏层偏置梯度
                
                # 参数更新：按批量平均梯度调整权重和偏置
                self.W += learning_rate * dW / batch_size
                self.b_v += learning_rate * db_v / batch_size
                self.b_h += learning_rate * db_h / batch_size

    def sample(self):
        """
        从训练好的模型中采样生成新数据(Gibbs采样)
        
        Returns:
            np.ndarray: 生成的图像数据，形状为(28, 28)
        """
        # 随机初始化可见层状态(图像)
        v = np.random.binomial(1, 0.5, self.n_observe)
        
        # 进行Gibbs采样迭代，通过交替采样可见层和隐藏层来逼近数据分布
        for _ in xrange(1000):  # 使用xrange避免生成完整列表，节省内存
            # 基于当前可见层状态计算隐藏层
            h_prob = self._sigmoid(np.dot(v, self.W) + self.b_h)
            h_sample = self._sample_binary(h_prob)
            
            # 基于隐藏层状态重构可见层
            v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)
            v = self._sample_binary(v_prob)
        
        # 将一维向量重塑为28×28的图像格式
        return v.reshape(28, 28)

# 使用MNIST手写数字数据集训练RBM并生成图像
if __name__ == '__main__':
    try:
        # 加载二值化的MNIST数据(60000张28×28手写数字图像)
        mnist = np.load('mnist_bin.npy')  # 确保数据文件存在
    except IOError:
        print("无法加载MNIST数据文件，请确保mnist_bin.npy文件在正确的路径下")
        sys.exit(1)
    
    # 打印数据形状信息
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print(f"数据形状: {mnist.shape}")  # 输出: (60000, 28, 28)
    
    # 初始化RBM模型：2个隐藏节点学习数据的二维表示
    rbm = RBM(2, img_size)
    
    # 训练模型
    rbm.train(mnist)
    
    # 从模型中采样生成一张新的"手写数字"图像
    generated_image = rbm.sample()
    print(f"生成的图像形状: {generated_image.shape}")  # 输出: (28, 28)
