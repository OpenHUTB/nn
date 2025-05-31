# python: 2.7
# encoding: utf-8

import numpy as np


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """初始化模型参数（受限玻尔兹曼机）"""

        # 请补全此处代码

        # 确保隐藏层和可见层的单元数量为正整数
        if n_hidden <= 0 or n_observe <= 0:
            raise ValueError("Number of hidden and visible units must be positive integers.")
        self.n_hidden = n_hidden     # 隐藏层神经元个数
        self.n_observe = n_observe    # 可见层神经元个数
        # 初始化权重和偏置
        init_std = np.sqrt(2.0 / (self.n_observe + self.n_hidden))     # Xavier初始化
        self.W = np.random.normal(0, init_std, size=(self.n_observe, self.n_hidden))  # 权重矩阵
        # self.W = np.random.normal(0, 0.01, size=(n_observe, n_hidden))  # 另一种初始化方式
        self.b_h = np.zeros(n_hidden)   # 隐藏层偏置
        self.b_v = np.zeros(n_observe)  # 可见层偏置
        pass
    def _sigmoid(self, x):
        """Sigmoid激活函数，用于将输入映射到概率空间"""
        return 1.0 / (1 + np.exp(-x))

    def _sample_binary(self, probs):
        """伯努利采样：根据给定概率生成0或1（用于模拟神经元激活）"""
        return np.random.binomial(1, probs)
    def train(self, data):
        """使用Contrastive Divergence算法对模型进行训练"""
    
        # 请补全此处代码
         # 将数据展平为二维数组 [n_samples, n_observe]
        data_flat = data.reshape(data.shape[0], -1)
        n_samples = data_flat.shape[0]
        
        # 定义训练参数
        learning_rate = 0.1
        epochs = 10
        batch_size = 100

       # 开始训练轮数
        for epoch in range(epochs):
            # 打乱数据顺序
            np.random.shuffle(data_flat)
            for i in range(0, n_samples, batch_size):
                batch = data_flat[i:i + batch_size]
                v0 = batch.astype(np.float64)  # 确保数据类型正确

                # 正相传播：从v0计算隐藏层激活概率
                h0_prob = self._sigmoid(np.dot(v0, self.W) + self.b_h)
                h0_sample = self._sample_binary(h0_prob)

                # 负相传播：从隐藏层重构可见层，再计算隐藏层概率
                v1_prob = self._sigmoid(np.dot(h0_sample, self.W.T) + self.b_v)
                v1_sample = self._sample_binary(v1_prob)
                h1_prob = self._sigmoid(np.dot(v1_sample, self.W) + self.b_h)

                # 计算梯度
                dW = np.dot(v0.T, h0_sample) - np.dot(v1_sample.T, h1_prob)
                db_v = np.sum(v0 - v1_sample, axis=0)
                db_h = np.sum(h0_sample - h1_prob, axis=0)

                # 更新参数
                self.W += learning_rate * dW / batch_size
                self.b_v += learning_rate * db_v / batch_size
                self.b_h += learning_rate * db_h / batch_size
        pass

    def sample(self):
        """从训练好的模型中采样生成新数据（Gibbs采样）"""

        # 请补全此处代码
        # 初始化随机可见层
        v = np.random.binomial(1, 0.5, self.n_observe)
        # 进行1000次Gibbs采样
        for _ in range(1000):
            h_prob = self._sigmoid(np.dot(v, self.W) + self.b_h)
            h_sample = self._sample_binary(h_prob)
            v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)
            v = self._sample_binary(v_prob)
        return v.reshape(28, 28)  # 恢复图像形状
        pass


# 使用MNIST数据集训练RBM模型
if __name__ == '__main__':
    # 加载二值化的MNIST数据，形状为 (60000, 28, 28)
    mnist = np.load('mnist_bin.npy')  # 60000x28x28
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols  # 计算单张图片展开后的长度
    print mnist.shape  # 打印数据维度

    # 初始化RBM对象：2个隐藏节点，784个可见节点（28×28图像）
    rbm = RBM(2, img_size)

    # 使用MNIST数据进行训练
    rbm.train(mnist)

    # 从模型中采样一张图像
    s = rbm.sample()
