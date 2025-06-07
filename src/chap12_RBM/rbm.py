# python: 2.7
# encoding: utf-8
# 导入numpy模块并命名为np
import numpy as np# 导入NumPy库用于高效数值计算
import sys
class RBM:
    """Restricted Boltzmann Machine."""

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
        # 初始化模型参数
        self.n_hidden = n_hidden
        self.n_observe = n_observe
        # 权重矩阵 (可见层到隐藏层)
        self.W = np.random.normal(
        loc = 0.0,                # 均值
        scale = 0.1,              # 标准差（常见初始化方法）
        size = (n_observe, n_hidden))
        # 可见层偏置（1 x n_observe）
        self.Wv = np.zeros((1, n_observe))
        # 隐藏层偏置（1 x n_hidden）
        self.Wh = np.zeros((1, n_hidden))
        # 可选：使用 Xavier/Glorot 初始化替代
        # self.W = np.random.randn(n_observe, n_hidden) * np.sqrt(1.0 / n_observe)
        # self.Wv = np.zeros((1, n_observe))
        # self.Wh = np.zeros((1, n_hidden))

        # 请补全此处代码
        # 确保隐藏层和可见层的单元数量为正整数
        # 神经网络模型的一部分，用于初始化隐藏层和可见层的权重和偏置
        """
        参数说明：
        n_observe (int): 可见层（输入层）神经元的数量，即输入特征维度
        n_hidden (int): 隐藏层神经元的数量
        """

        self.n_hidden = n_hidden     # 隐藏层神经元个数
        self.n_observe = n_observe   # 可见层神经元个数

        # 初始化权重和偏置
        # 使用 Xavier 初始化方法：标准差 = sqrt(2 / (输入维度 + 输出维度))
        init_std = np.sqrt(2.0 / (self.n_observe + self.n_hidden))  # Xavier初始化标准差

        self.W = np.random.normal(
            0, init_std, size=(self.n_observe, self.n_hidden)
        )  # 初始化权重矩阵（可见层 -> 隐藏层）

        # 可选替代方案：使用更小的固定标准差进行初始化。
        # self.W = np.random.normal(0, 0.01, size=(n_observe, n_hidden))

        self.b_h = np.zeros(n_hidden)   # 初始化隐藏层偏置向量

        self.b_v = np.zeros(n_observe)  # 初始化可见层偏置向量
        # pass
    
    def _sigmoid(self, x):
        """Sigmoid激活函数，用于将输入映射到概率空间"""
        return 1.0 / (1 + np.exp(-x))

    def _sample_binary(self, probs):
        """伯努利采样：根据给定概率生成0或1（用于模拟神经元激活）"""
        return np.random.binomial(1, probs)
    
    def train(self, data):
        """
         使用Contrastive Divergence算法对模型进行训练
         参数说明：
         data (numpy.ndarray): 训练数据，形状为 (n_samples, n_observe)。
        """
    
        # 请补全此处代码
        # 将数据展平为二维数组 [n_samples, n_observe]
        data_flat = data.reshape(data.shape[0], -1)  
        n_samples = data_flat.shape[0]  # 样本数量

        # 定义训练参数
        learning_rate = 0.1 # 学习率，控制参数更新的步长
        epochs = 10 # 训练轮数，整个数据集将被遍历10次v
        batch_size = 100 # 批处理大小，每次更新参数使用的样本数量

       # 开始训练轮数
        for epoch in range(epochs):
            # 打乱数据顺序
            np.random.shuffle(data_flat) # 使用小批量梯度下降法
            for i in range(0, n_samples, batch_size):# 获取当前批次的数据
                batch = data_flat[i:i + batch_size] # 将批次数据转换为 float64 类型，确保数值计算的精度
                v0 = batch.astype(np.float64)  # 确保数据类型正确

                # 正相传播：从v0计算隐藏层激活概率
                h0_prob = self._sigmoid(np.dot(v0, self.W) + self.b_h) # 通过输入层向量v0与权重矩阵self.W的点积，再加上隐藏层偏置self.b_h
                h0_sample = self._sample_binary(h0_prob) #这段代码的作用是借助 self._sample_binary 方法，对 h0_prob 进行二值采样，进而得到 h0_sample。在深度学习领域，当处理二值变量或者进行二值掩码操作时，常常会用到这样的采样。

                # 负相传播：从隐藏层重构可见层，再计算隐藏层概率
                v1_prob = self._sigmoid(np.dot(h0_sample, self.W.T) + self.b_v)  #将上述结果传入 Sigmoid 激活函数进行非线性变换，得到最终的概率值 v1_prob
                v1_sample = self._sample_binary(v1_prob)        # 对可见层进行二值采样
                h1_prob = self._sigmoid(np.dot(v1_sample, self.W) + self.b_h)       #计算隐藏单元被激活的概率

                # 计算梯度      
                dW = np.dot(v0.T, h0_sample) - np.dot(v1_sample.T, h1_prob)          # 计算权重矩阵的梯度
                db_v = np.sum(v0 - v1_sample, axis=0)                                # 计算可见层偏置的梯度
                db_h = np.sum(h0_sample - h1_prob, axis=0)                           # 计算隐藏层偏置的梯度

                # 更新参数
                self.W += learning_rate * dW / batch_size                            # 更新权重矩阵
                self.b_v += learning_rate * db_v / batch_size                        # 更新可见层偏置
                self.b_h += learning_rate * db_h / batch_size                         # 更新隐藏层偏置

    def sample(self):
        """从训练好的模型中采样生成新数据（Gibbs采样）"""
        # 初始化可见层：使用伯努利分布随机生成二值向量（每个像素有50%概率为1）
        # n_observe是可见层神经元数量（28x28=784）
        v = np.random.binomial(1, 0.5, self.n_observe)

        # 进行1000次 Gibbs采样迭代，以逐步趋近真实数据分布
        for _ in xrange(1000):
            # 基于当前的可见层v，计算隐藏层神经元被激活的概率（前向传播）
            h_prob = self._sigmoid(np.dot(v, self.W) + self.b_h)

            # 根据激活概率采样得到隐藏层的状态（伯努利采样）
            h_sample = self._sample_binary(h_prob)

            # 基于隐藏层的采样结果，重新估算可见层的激活概率（反向传播）
            v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)

            # 根据估算的概率采样新的可见层状态
            v = self._sample_binary(v_prob)

        # 将最终的可见层向量重塑为 28×28 的图像格式
        return v.reshape(28, 28)

# 使用 MNIST 数据集训练 RBM 模型
if __name__ == '__main__':
    try:
    # 加载二值化的MNIST数据，形状为 (60000, 28, 28)
      mnist = np.load('mnist_bin.npy')  # 60000x28x28
    except IOError:
      print("无法加载MNIST数据文件，请确保mnist_bin.npy文件在正确的路径下")
      sys.exit(1)
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols  # 计算单张图片展开后的长度
    print(mnist.shape)  # 打印数据维度

    # 初始化 RBM 对象：2个隐藏节点，784个可见节点（28×28 图像）
    rbm = RBM(2, img_size)
    # 使用 MNIST 数据进行训练
    rbm.train(mnist)

    # 从模型中采样一张图像
    s = rbm.sample()
