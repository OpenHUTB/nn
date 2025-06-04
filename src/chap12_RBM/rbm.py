# python: 2.7
# encoding: utf-8

import numpy as np

class RBM:
    """Restricted Boltzmann Machine (RBM).
    受限玻尔兹曼机是一种生成式神经网络，由可见层和隐藏层组成，层内无连接，层间全连接。
    模型通过能量函数定义概率分布，并使用Contrastive Divergence算法训练。
    """

    def __init__(self, n_hidden=2, n_observe=784):
        """初始化模型参数
        参数:
            n_hidden: 隐藏层神经元数量
            n_observe: 可见层神经元数量（如MNIST图像为28x28=784）
        """
        # 确保隐藏层和可见层的单元数量为正整数
        if n_hidden <= 0 or n_observe <= 0:
            raise ValueError("Number of hidden and visible units must be positive integers.")

        self.n_hidden = n_hidden
        self.n_observe = n_observe
        
        # 使用Xavier初始化权重矩阵，有助于缓解梯度消失/爆炸问题
        # 初始化标准差: sqrt(2/(输入单元数+输出单元数))
        init_std = np.sqrt(2.0 / (self.n_observe + self.n_hidden))
        self.W = np.random.normal(0, init_std, size=(self.n_observe, self.n_hidden))
        
        # 初始化偏置为零
        self.b_h = np.zeros(n_hidden)  # 隐藏层偏置
        self.b_v = np.zeros(n_observe) # 可见层偏置
    
    def _sigmoid(self, x):
        """Sigmoid激活函数，将输入值映射到(0,1)区间，表示神经元激活概率
        公式: σ(x) = 1 / (1 + e^(-x))
        """
        return 1.0 / (1 + np.exp(-x))

    def _sample_binary(self, probs):
        """伯努利采样：根据给定概率生成0或1的二元值
        参数:
            probs: 概率数组，每个元素表示对应位置生成1的概率
        返回:
            采样结果：0或1组成的数组
        """
        return np.random.binomial(1, probs)
    
    def train(self, data):
        """使用Contrastive Divergence (CD)算法训练RBM
        CD-k算法流程：
        1. 从数据中取一个样本v0
        2. 正向传播计算隐藏层h0的概率和采样值
        3. 反向传播重构可见层v1
        4. 再次计算隐藏层h1的概率（k=1时）
        5. 更新参数：ΔW ∝ <v0·h0> - <v1·h1>
        """
        # 将数据展平为二维数组 [n_samples, n_observe]
        data_flat = data.reshape(data.shape[0], -1)
        n_samples = data_flat.shape[0]
        
        # 训练参数设置
        learning_rate = 0.1  # 学习率，控制参数更新步长
        epochs = 10         # 训练轮数
        batch_size = 100    # 批处理大小
        
        # 开始训练
        for epoch in range(epochs):
            # 打乱数据顺序，提高训练稳定性
            np.random.shuffle(data_flat)
            
            # 小批量梯度下降
            for i in range(0, n_samples, batch_size):
                # 获取当前批次数据
                batch = data_flat[i:i + batch_size]
                v0 = batch.astype(np.float64)  # 确保数据类型为float64
                
                # 正相（Positive phase）：计算数据的期望
                # 前向传播：v0 → h0
                h0_prob = self._sigmoid(np.dot(v0, self.W) + self.b_h)  # 隐藏层激活概率
                h0_sample = self._sample_binary(h0_prob)                  # 隐藏层采样值
                
                # 负相（Negative phase）：计算模型的期望（重构误差）
                # 反向传播：h0 → v1 → h1
                v1_prob = self._sigmoid(np.dot(h0_sample, self.W.T) + self.b_v)  # 重构可见层概率
                v1_sample = self._sample_binary(v1_prob)                          # 重构可见层采样值
                h1_prob = self._sigmoid(np.dot(v1_sample, self.W) + self.b_h)    # 重构后的隐藏层概率
                
                # 计算梯度（基于CD-1算法）
                # 权重梯度：数据期望与模型期望的差值
                dW = np.dot(v0.T, h0_prob) - np.dot(v1_sample.T, h1_prob)
                # 可见层偏置梯度：原始数据与重构数据的差值
                db_v = np.sum(v0 - v1_sample, axis=0)
                # 隐藏层偏置梯度：原始隐藏层激活与重构隐藏层激活的差值
                db_h = np.sum(h0_prob - h1_prob, axis=0)
                
                # 更新参数（除以batch_size实现均值梯度）
                self.W += learning_rate * dW / batch_size
                self.b_v += learning_rate * db_v / batch_size
                self.b_h += learning_rate * db_h / batch_size
    
    def sample(self):
        """从训练好的模型中采样生成新数据（Gibbs采样）
        通过交替条件采样，逐步逼近模型的平稳分布
        """
        # 初始化可见层为随机二元状态（0.5概率为1）
        v = np.random.binomial(1, 0.5, self.n_observe)
        
        # 进行1000次Gibbs采样迭代，使系统达到平稳分布
        for _ in range(1000):
            # 基于当前可见层状态v，计算隐藏层激活概率
            h_prob = self._sigmoid(np.dot(v, self.W) + self.b_h)
            
            # 从隐藏层概率分布中采样
            h_sample = self._sample_binary(h_prob)
            
            # 基于隐藏层采样结果，重构可见层概率
            v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)
            
            # 从可见层概率分布中采样，更新可见层状态
            v = self._sample_binary(v_prob)
        
        # 将最终的可见层向量重塑为28×28的图像格式
        return v.reshape(28, 28)


# 使用MNIST数据集训练RBM模型
if __name__ == '__main__':
    # 加载二值化的MNIST数据，形状为 (60000, 28, 28)
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols  # 计算单张图片展开后的长度
    print(mnist.shape)  # 打印数据维度

    # 初始化RBM对象：2个隐藏节点，784个可见节点（对应28×28图像）
    rbm = RBM(2, img_size)

    # 使用MNIST数据进行训练
    rbm.train(mnist)

    # 从模型中采样一张图像
    s = rbm.sample()
