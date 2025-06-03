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
    X_list = []
    y_true = []
    for i in range(n_components):
        X_i = np.random.multivariate_normal(mu_true[i], sigma_true[i], samples_per_component[i])
        X_list.append(X_i)
        y_true.extend([i] * samples_per_component[i])
    
    # 合并并打乱数据
    X = np.vstack(X_list)
    y_true = np.array(y_true)
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y_true[shuffle_idx]

# 自定义logsumexp函数
def logsumexp(log_p, axis=1, keepdims=False):
    #max_val = np.max(log_p, axis=axis, keepdims=True)
    #return max_val + np.log(np.sum(np.exp(log_p - max_val), axis=axis, keepdims=keepdims))
    """优化后的logsumexp实现，包含数值稳定性增强和特殊case处理"""
    log_p = np.asarray(log_p)
    
    # 处理空输入情况
    if log_p.size == 0:
        return np.array(-np.inf, dtype=log_p.dtype)
    
    # 计算最大值（处理全-inf输入）
    max_val = np.max(log_p, axis=axis, keepdims=True)
    if np.all(np.isneginf(max_val)):
        return max_val.copy() if keepdims else max_val.squeeze(axis=axis)
    
    # 计算修正后的指数和（处理-inf输入）
    safe_log_p = np.where(np.isneginf(log_p), -np.inf, log_p - max_val)
    sum_exp = np.sum(np.exp(safe_log_p), axis=axis, keepdims=keepdims)
    
    # 计算最终结果
    result = max_val + np.log(sum_exp)
    
    # 处理全-inf输入的特殊case
    if np.any(np.isneginf(log_p)) and not np.any(np.isfinite(log_p)):
        result = max_val.copy() if keepdims else max_val.squeeze(axis=axis) #根据keepdims参数的值返回 max_val 的适当形式。
    
    return result

# 高斯混合模型类
class GaussianMixtureModel:
    def __init__(self, n_components=3, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X):
        n_samples, n_features = X.shape
        self.pi = np.ones(self.n_components) / self.n_components
        self.mu = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.sigma = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        log_likelihood = -np.inf
        for iter in range(self.max_iter):
            # E步：计算后验概率
            log_prob = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                log_prob[:, k] = np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.sigma[k])
            log_prob_sum = logsumexp(log_prob, axis=1, keepdims=True)
            gamma = np.exp(log_prob - log_prob_sum)
            
            # M步：更新参数
            Nk = np.sum(gamma, axis=0)
            self.pi = Nk / n_samples
            new_mu = np.zeros_like(self.mu)
            new_sigma = np.zeros_like(self.sigma)
            
            for k in range(self.n_components):
                # 更新均值
                new_mu[k] = np.sum(gamma[:, k, None] * X, axis=0) / Nk[k]
                # 更新协方差
                X_centered = X - new_mu[k]
                weighted_X = gamma[:, k, None] * X_centered
                new_sigma[k] = (X_centered.T @ weighted_X) / Nk[k]
                new_sigma[k] += np.eye(n_features) * 1e-6  # 正则化
            
            # 计算对数似然
            current_log_likelihood = np.sum(log_prob_sum)
            if iter > 0 and abs(current_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = current_log_likelihood
            
            self.mu = new_mu
            self.sigma = new_sigma
        
        # 计算最终聚类结果
        self.labels_ = np.argmax(gamma, axis=1)
        return self

    def _log_gaussian(self, X, mu, sigma):
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
        exponent = -0.5 * np.sum(X_centered @ inv * X_centered, axis=1)

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
