import numpy as np
import matplotlib.pyplot as plt

# 生成混合高斯分布数据
def generate_data(n_samples=1000):
    np.random.seed(42)
    # 真实参数
    # 定义三个高斯分布的均值(中心点)
    mu_true = np.array([ 
        [0, 0],  # 第一个高斯分布的均值
        [5, 5],  # 第二个高斯分布的均值
        [-5, 5]  # 第三个高斯分布的均值
    ])
    # 定义三个高斯分布的协方差矩阵
    sigma_true = np.array([
        [[1, 0], [0, 1]],  # 第一个分布：圆形分布(各向同性)
        [[2, 0.5], [0.5, 1]],   # 第二个分布：倾斜的椭圆
        [[1, -0.5], [-0.5, 2]]  # 第三个分布：反向倾斜的椭圆
    ])
    # 定义每个高斯分布的混合权重(必须和为1)
    weights_true = np.array([0.3, 0.4, 0.3])
    # 获取混合成分的数量(这里是3)
    n_components = len(weights_true)
    
    # 生成一个合成数据集，该数据集由多个多元正态分布的样本组成
    samples_per_component = (weights_true * n_samples).astype(int)
    X_list = []  # 用于存储每个高斯分布生成的数据点
    y_true = []  # 用于存储每个数据点对应的真实分布标签
    for i in range(n_components):  # 从第i个高斯分布生成样本
        X_i = np.random.multivariate_normal(mu_true[i], sigma_true[i], samples_per_component[i])
        X_list.append(X_i)  # 将生成的样本添加到列表
        y_true.extend([i] * samples_per_component[i])  # 添加对应标签
    
    # 合并并打乱数据
    X = np.vstack(X_list)  #将多个子数据集合并为一个完整数据集
    y_true = np.array(y_true)  #将Python列表转换为NumPy数组
    shuffle_idx = np.random.permutation(n_samples) #生成0到n_samples-1的随机排列
    return X[shuffle_idx], y_true[shuffle_idx] #使用相同的随机索引同时打乱特征和标签

# 自定义logsumexp函数
def logsumexp(log_p, axis=1, keepdims=False):
    """优化后的logsumexp实现，包含数值稳定性增强和特殊case处理
    数学原理：log(sum(exp(log_p))) = max(log_p) + log(sum(exp(log_p - max(log_p))))
    该技巧通过减去最大值避免指数运算时的数值溢出或下溢
    """
    log_p = np.asarray(log_p)
    
    # 处理空输入情况
    if log_p.size == 0:  # 检查输入的对数概率数组是否为空
        return np.array(-np.inf, dtype=log_p.dtype)  # 返回与输入相同数据类型的负无穷值
    
    # 计算最大值（处理全-inf输入）
    max_val = np.max(log_p, axis=axis, keepdims=True)  # 计算沿指定轴的最大值
    if np.all(np.isneginf(max_val)):  # 检查是否所有最大值都是负无穷
        return max_val.copy() if keepdims else max_val.squeeze(axis=axis)  # 根据keepdims返回适当形式
    
    # 计算修正后的指数和（处理-inf输入）
    safe_log_p = np.where(np.isneginf(log_p), -np.inf, log_p - max_val)  # 安全调整对数概率
    sum_exp = np.sum(np.exp(safe_log_p), axis=axis, keepdims=keepdims)  # 计算调整后的指数和
    
    # 计算最终结果
    result = max_val + np.log(sum_exp)
    
    # 处理全-inf输入的特殊case
    if np.any(np.isneginf(log_p)) and not np.any(np.isfinite(log_p)):  #判断是否所有有效值都是-inf
        result = max_val.copy() if keepdims else max_val.squeeze(axis=axis) #根据keepdims参数的值返回 max_val 的适当形式。
    
    return result  #返回处理后的结果，保持与正常情况相同的接口

# 高斯混合模型类
class GaussianMixtureModel:
    """高斯混合模型(GMM)实现，使用期望最大化(EM)算法进行参数估计
    模型假设：数据由K个多元高斯分布混合生成，即：
    p(x) = Σ_{k=1}^K π_k * N(x|μ_k, Σ_k)
    其中，π_k是混合系数(权重)，μ_k和Σ_k分别是第k个高斯分布的均值和协方差矩阵
    """
    def __init__(self, n_components=3, max_iter=100, tol=1e-6):
        
        # 初始化模型参数
        self.n_components = n_components  # 高斯分布数量
        self.max_iter = max_iter          # EM算法最大迭代次数
        self.tol = tol                    # 收敛阈值
    
    def fit(self, X):
        """使用EM算法训练模型
        EM算法分为两个步骤迭代进行：
        1. E步：根据当前参数计算隐变量(各样本属于每个高斯成分的概率)
        2. M步：根据隐变量更新模型参数(π, μ, Σ)
        重复这两个步骤直到收敛
        """
        n_samples, n_features = X.shape
        
        # 初始化混合系数（均匀分布）
        self.pi = np.ones(self.n_components) / self.n_components
        
        # 随机选择样本点作为初始均值
        self.mu = X[np.random.choice(n_samples, self.n_components, replace=False)]
        
        # 初始化协方差矩阵为单位矩阵
        self.sigma = np.array([np.eye(n_features) for _ in range(self.n_components)])

        log_likelihood = -np.inf  # 初始化对数似然值为负无穷
        for iter in range(self.max_iter): # 开始EM算法的主循环
            # E步：计算后验概率（也称为"响应度"）
            # 对于每个样本x_i，计算它属于每个高斯成分k的概率γ_{i,k} = P(z_i=k|x_i;θ)
            
            log_prob = np.zeros((n_samples, self.n_components)) # 初始化对数概率矩阵，形状为(样本数 × 成分数)
            for k in range(self.n_components): # 遍历每个高斯成分
                # 计算第k个高斯分布的对数概率密度：log[π_k * N(x|μ_k,Σ_k)]
                # 这里拆分为两部分：混合权重的对数 + 高斯概率密度的对数
                log_prob[:, k] = np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.sigma[k]) 
            log_prob_sum = logsumexp(log_prob, axis=1, keepdims=True) # 使用logsumexp实现数值稳定的概率求和
            gamma = np.exp(log_prob - log_prob_sum) # 计算后验概率矩阵gamma(也称为响应度矩阵)
            # 这里使用了对数概率的减法，等价于：gamma = exp(log_prob) / sum(exp(log_prob))
            
            # M步：更新参数（基于当前后验概率重新估计模型参数）
            Nk = np.sum(gamma, axis=0) # 计算每个高斯成分的"有效样本数"（即属于该成分的样本概率之和）
            self.pi = Nk / n_samples # 更新混合权重π：各成分的样本占比
            
            new_mu = np.zeros_like(self.mu) # 初始化新均值和新协方差矩阵的存储空间
            new_sigma = np.zeros_like(self.sigma)
            
            for k in range(self.n_components): # 遍历每个高斯成分更新参数
                # 更新均值：加权平均，权重为后验概率gamma
                # 公式：μ_k = (Σ_i γ_{i,k} * x_i) / N_k
                new_mu[k] = np.sum(gamma[:, k, None] * X, axis=0) / Nk[k]
                
                # 更新协方差矩阵
                # 公式：Σ_k = (Σ_i γ_{i,k} * (x_i-μ_k)(x_i-μ_k)^T) / N_k
                X_centered = X - new_mu[k]  # 中心化：每个样本减去当前估计的均值
                weighted_X = gamma[:, k, None] * X_centered  # 加权：每个样本乘以其属于该成分的概率
                new_sigma[k] = (X_centered.T @ weighted_X) / Nk[k]  # 计算加权协方差矩阵
                new_sigma[k] += np.eye(n_features) * 1e-6  # 添加小的正则化项，确保协方差矩阵正定
            
            # 计算对数似然（用于判断收敛）
            # 对数似然：log P(X|θ) = Σ_i log[Σ_k π_k * N(x_i|μ_k,Σ_k)]
            current_log_likelihood = np.sum(log_prob_sum)
            if iter > 0 and abs(current_log_likelihood - log_likelihood) < self.tol:
                break  # 对数似然变化小于阈值时，认为算法收敛
            log_likelihood = current_log_likelihood
            
            # 更新模型参数为新估计值
            self.mu = new_mu
            self.sigma = new_sigma
        
        # 计算最终聚类结果：每个样本分配给概率最大的成分
        self.labels_ = np.argmax(gamma, axis=1)
        return self

    def _log_gaussian(self, X, mu, sigma):
        """计算多元高斯分布的对数概率密度
        数学公式：log N(x|μ,Σ) = -0.5*D*log(2π) - 0.5*log|Σ| - 0.5*(x-μ)^TΣ^(-1)(x-μ)
        其中：
        - D是特征维度
        - |Σ|是协方差矩阵的行列式
        - (x-μ)^TΣ^(-1)(x-μ)是马氏距离的平方
        """
        # 获取特征维度数量
        n_features = mu.shape[0]

        # 将每个样本减去均值，进行中心化处理
        X_centered = X - mu

        # 计算协方差矩阵的对数行列式（log determinant）和符号
        # 如果协方差矩阵不可逆或行列式为负，说明可能存在数值问题
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            # 添加微小扰动确保协方差矩阵正定（数值稳定性）
            sigma += np.eye(n_features) * 1e-6
            sign, logdet = np.linalg.slogdet(sigma)

        # 计算协方差矩阵的逆
        inv = np.linalg.inv(sigma)
        # 计算高斯分布中的指数项（二次型），对应 (x - μ)^T Σ⁻¹ (x - μ)
        exponent = -0.5 * np.einsum('...i,...i->...', X_centered @ inv, X_centered)

        # 返回多维高斯分布的对数概率密度值
        # 公式为：-0.5 * D * log(2π) - 0.5 * log|Σ| + exponent
        return -0.5 * n_features * np.log(2 * np.pi) - 0.5 * logdet + exponent

# 主程序
if __name__ == "__main__":
    X, y_true = generate_data()
    
    # 训练GMM模型
    gmm = GaussianMixtureModel(n_components=3)
    gmm.fit(X)
    y_pred = gmm.labels_
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10)
    plt.title("True Clusters") # 子图标题

    # 设置坐标轴标签
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.7) # 添加网格线，线型为虚线，透明度为0.7
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=10)
    plt.title("GMM Predicted Clusters") # 子图标题

    # 设置坐标轴标签
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.7) # 添加网格线，线型为虚线，透明度为0.7

    plt.show() # 显示图形
