# 导入NumPy库，用于科学计算和数值操作
import numpy as np
# 导入matplotlib.pyplot模块，用于数据可视化和绘图
import matplotlib
import matplotlib.pyplot as plt
# matplotlib 3.9 将 boxplot 的 labels 参数重命名为 tick_labels
_MPL_GE_39 = tuple(int(x) for x in matplotlib.__version__.split('.')[:2]) >= (3, 9)
_BOXPLOT_LABEL_KEY = 'tick_labels' if _MPL_GE_39 else 'labels'
# 导入argparse，用于命令行参数配置
import argparse
# 导入csv，用于保存收敛日志
import csv
# 导入Path，便于跨平台路径处理
from pathlib import Path
#添加类型提示支持
from typing import Tuple, List 

# 生成混合高斯分布数据
def generate_data(n_samples = 1000, random_state = 42):
    """生成混合高斯分布数据集
    
    Args:
        n_samples: 样本数量 (default=1000)
        random_state: 随机种子 (default=42)
        
    Returns:
        Tuple: (X, y_true)
            X: 特征矩阵 (n_samples, 2)
            y_true: 真实标签 (n_samples,)
    """
    np.random.seed(random_state)  # 固定随机种子以确保结果可复现
    # 定义三个高斯分布的中心点
    mu_true = np.array([ 
        [0, 0],  # 第一个高斯分布的均值
        [5, 5],  # 第二个高斯分布的均值
        [-5, 5]  # 第三个高斯分布的均值
    ])
    
    # 定义三个高斯分布的协方差矩阵
    sigma_true = np.array([
        [[1, 0], [0, 1]],  # 第一个分布：圆形分布(各向同性)
        [[2, 0.5], [0.5, 1]],  # 第二个分布：倾斜的椭圆
        [[1, -0.5], [-0.5, 2]]  # 第三个分布：反向倾斜的椭圆
    ])
    
    # 定义每个高斯分布的混合权重(必须和为1)
    weights_true = np.array([0.3, 0.4, 0.3])
    
    # 获取混合成分的数量(这里是3)
    n_components = len(weights_true)
    
    # 生成一个合成数据集，该数据集由多个多元正态分布的样本组成
    # 计算每个分布应生成的样本数（浮点转整数可能有误差）
    samples_per_component = (weights_true * n_samples).astype(int)
    
    # 确保样本总数正确（由于浮点转换可能有误差）
    total_samples = np.sum(samples_per_component)
    if total_samples < n_samples:
        # 将缺少的样本添加到权重最大的成分中
        samples_per_component[np.argmax(weights_true)] += n_samples - total_samples
    
    # 用于存储每个高斯分布生成的数据点
    X_list = []  
    
    # 用于存储每个数据点对应的真实分布标签
    y_true = []  
    
    # 从第i个高斯分布生成样本
    for i in range(n_components): 
        #生成多元正态分布样本
        X_i = np.random.multivariate_normal(mu_true[i], sigma_true[i], samples_per_component[i])
        # 将生成的样本添加到列表
        X_list.append(X_i) 
        # 添加对应标签（0、1、2表示三个分布）
        y_true.extend([i] * samples_per_component[i]) 
    
    # 合并并打乱数据并打乱顺序（模拟无标签数据）
    # 使用函数将X_list中的数组沿垂直方向拼接成一个二维数组X
    X = np.vstack(X_list)  
    # 将Python列表转换为NumPy数组
    y_true = np.array(y_true) 
    # 生成0到n_samples-1的随机排列
    shuffle_idx = np.random.permutation(n_samples) 
    # 通过随机索引同时打乱矩阵和标签，确保对应关系不变
    return X[shuffle_idx], y_true[shuffle_idx]

# 自定义logsumexp函数.LogSumExp（LSE）函数 是一种常见的数值计算函数，主要用于对数域求和的场景。
def logsumexp(log_p, axis=1, keepdims=False):
    """优化后的logsumexp实现，包含数值稳定性增强和特殊case处理
    
    计算log(sum(exp(log_p)))，通过减去最大值避免数值溢出
    数学公式: log(sum(exp(log_p))) = max(log_p) + log(sum(exp(log_p - max(log_p))))
    
    参数：
    log_p: 输入的对数概率（可能为负无穷）。
    axis: 沿着哪个轴进行计算，默认为1（即按行计算）。
    keepdims: 是否保持维度，默认为False。

    返回：
    计算结果的log(sum(exp(log_p)))，返回与输入数组相同形状的结果。
    """
    log_p = np.asarray(log_p)                                               # 将对数概率列表转换为NumPy数组
    
    # 处理空输入情况
    if log_p.size == 0:                                                     # 检查输入的对数概率数组是否为空
        return np.array(-np.inf, dtype = log_p.dtype)                         # 返回与输入相同数据类型的负无穷值
    
    # 计算最大值（处理全-inf输入）
    max_val = np.max(log_p, axis = axis, keepdims = True)                       # 计算沿指定轴的最大值
    if np.all(np.isneginf(max_val)):                                        # 检查是否所有最大值都是负无穷
        return max_val.copy() if keepdims else max_val.squeeze(axis = axis)   # 根据keepdims返回适当形式
    
    # 计算修正后的指数和（处理-inf输入）
    # 安全计算指数和：先减去最大值，再计算指数
    safe_log_p = np.where(np.isneginf(log_p), -np.inf, log_p - max_val)     # 安全调整对数概率
    sum_exp = np.sum(np.exp(safe_log_p), axis = axis, keepdims = keepdims)  # 计算调整后的指数和
    
    # 计算最终结果
    result = max_val + np.log(sum_exp)
    
    # 处理全-inf输入的特殊case
    if np.any(np.isneginf(log_p)) and not np.any(np.isfinite(log_p)):       # 判断是否所有有效值都是-inf
        result = max_val.copy() if keepdims else max_val.squeeze(axis = axis) # 根据keepdims参数的值返回max_val的适当形式
    return result                                                           # 返回处理后的结果，保持与正常情况相同的接口

# 高斯混合模型类
class GaussianMixtureModel:
    """高斯混合模型(GMM)实现
    
    参数:
        n_components: int, 高斯分布数量 (默认=3)
        max_iter: int, EM算法最大迭代次数 (默认=100)
        tol: float, 收敛阈值 (默认=1e-6)
        random_state: int, 随机种子 (可选)
    """
    def __init__(self, n_components = 3, max_iter = 100, tol = 1e-6, random_state = None, init = 'random'):
        # 初始化模型参数
        self.n_components = n_components  # 高斯分布数量
        self.max_iter = max_iter          # EM算法最大迭代次数
        self.tol = tol                    # 收敛阈值
        self.init = init                  # 初始化策略：'random'（随机）或 'kmeans++'（智能距离权重采样）
        self.log_likelihoods = []         # 存储每轮迭代的对数似然值
        self.n_iters_ = 0                 # 实际收敛所用的迭代次数
        self.aic_ = None                  # AIC 值（训练后计算）
        self.bic_ = None                  # BIC 值（训练后计算）

        # 初始化随机数生成器（使用 numpy 新式 Generator，线程安全且可重复）
        self.rng = np.random.default_rng(random_state)

    def fit(self, X):
        """使用EM算法训练模型

        EM算法流程：
        1. 初始化模型参数（混合权重π、均值μ、协方差矩阵Σ）
        2. 重复以下步骤直到收敛：
           - E步：计算每个样本属于各高斯成分的后验概率（责任度）
           - M步：基于后验概率更新模型参数
        """
        X = np.asarray(X)  # 将输入数据 X 转换为 NumPy 数组格式，确保后续操作的兼容性
        n_samples, n_features = X.shape # 获取数据的样本数量和特征维度
        
        # 初始化混合系数（均匀分布）
        self.pi = np.ones(self.n_components) / self.n_components
        
        # 选择初始均值：'kmeans++' 用距离权重采样，'random' 用均匀随机采样
        if self.init == 'kmeans++':
            self.mu = self._kmeans_plus_plus_init(X)
        else:
            indices = self.rng.choice(n_samples, self.n_components, replace=False)
            self.mu = X[indices].copy()
        
        # 初始化协方差矩阵为单位矩阵
        self.sigma = np.array([np.eye(n_features) for _ in range(self.n_components)])

        log_likelihood = -np.inf  # 初始化对数似然值为负无穷
        
        # EM算法主循环：交替执行E步(期望)和M步(最大化)
        for iter in range(self.max_iter):
            # E步：计算后验概率（每个样本属于各个高斯成分的概率）
            log_prob = np.zeros((n_samples, self.n_components)) # 初始化对数概率矩阵
            
            # 对每个高斯成分，计算样本的对数概率密度
            for k in range(self.n_components):
                # 对数概率 = log(混合权重) + log(高斯概率密度)
                log_prob[:, k] = np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.sigma[k]) # 计算第k个高斯混合成分的对数概率密度，并存储在log_prob的第k列
            
            # 使用logsumexp计算归一化因子，确保数值稳定性
            log_prob_sum = logsumexp(log_prob, axis = 1, keepdims = True)
            
            # 计算后验概率（responsibility）：gamma_{ik} = P(z_i=k|x_i)
            gamma = np.exp(log_prob - log_prob_sum)

            # M步：更新模型参数（基于后验概率）
            Nk = np.sum(gamma, axis=0) # 每个高斯成分的"有效样本数"
            
            # 更新混合权重
            # 计算类别先验概率（class prior），即每个类别在样本中的比例
            # Nk: 当前类别k的样本数量
            # n_samples: 总样本数量
           # 结果self.pi表示类别k在总体中的出现频率，用于后续的概率计算
            self.pi = Nk / n_samples
            
            # 初始化新均值和新协方差矩阵
            new_mu = np.zeros_like(self.mu)# 创建一个与 self.mu 形状相同且全为零的数组，作为新的均值向量
            new_sigma = np.zeros_like(self.sigma)# 创建一个与 self.sigma 形状相同且全为零的数组，作为新的协方差矩阵

            # 对每个高斯成分更新参数
            for k in range(self.n_components):
                # 更新均值：加权平均
                new_mu[k] = np.sum(gamma[:, k, None] * X, axis=0) / Nk[k]

                # 更新协方差矩阵
                X_centered = X - new_mu[k]  # 中心化数据
                weighted_X = gamma[:, k, None] * X_centered  # 加权中心化数据
                
                # 使用einsum高效计算协方差矩阵
                # 等价于: new_sigma_k = (X_centered.T @ diag(gamma[:,k]) @ X_centered) / Nk[k]
                # 更稳定的协方差计算方式
                new_sigma_k = np.einsum('ki,kj->ij', gamma[:, k, None] * X_centered, X_centered) / Nk[k]

                # 统一正则化处理，确保协方差矩阵正定
                new_sigma_k += np.eye(n_features) * 1e-6
                
                new_sigma[k] = new_sigma_k  # 存储更新后的协方差矩阵

            # 计算对数似然（模型对数据的拟合程度）
            current_log_likelihood = np.sum(log_prob_sum)  # 所有样本的对数似然之和
            self.log_likelihoods.append(current_log_likelihood)  # 记录当前对数似然
            
            # 检查收敛条件：如果对数似然变化小于阈值，则停止迭代
            if iter > 0 and abs(current_log_likelihood - log_likelihood) < self.tol:
                break
                
            log_likelihood = current_log_likelihood   # 更新记录的上一次迭代的对数似然值
            
            # 更新模型参数

            # 更新模型的均值参数（self.mu）为计算得到的新均值（new_mu）
            # new_mu通常是通过优化算法（如EM算法、梯度下降）得到的当前最优估计值
            self.mu = new_mu
            # 更新模型的协方差参数（self.sigma）为计算得到的新协方差（new_sigma）
            # new_sigma需保证为正定矩阵，常见实现中会通过Cholesky分解等方法确保数值稳定性
            self.sigma = new_sigma
        
        # 记录实际收敛所用的迭代次数（for 循环结束后 iter 保留最后一次值）
        self.n_iters_ = iter + 1
        # 最终聚类结果：每个样本分配到概率最大的高斯成分
        self.labels_ = np.argmax(gamma, axis=1)
        
        # 计算 AIC 和 BIC 准则
        self._compute_aic_bic(X)
        
        # 基于软聚类结果确定最终的硬聚类标签
        return self

    def _compute_aic_bic(self, X):
        """计算 AIC（Akaike Information Criterion）和 BIC（Bayesian Information Criterion）
        
        AIC = 2k - 2ln(L)
        BIC = k * ln(n) - 2ln(L)
        
        其中：
            k: 模型参数数量（每个成分有 d 个均值 + d*(d+1)/2 个协方差参数 + 权重参数）
            n: 样本数量
            L: 模型似然值
        """
        n_samples, n_features = X.shape
        n_components = self.n_components
        
        # 计算模型参数数量
        # 每个成分: n_features(均值) + n_features*(n_features+1)/2(协方差矩阵下三角)
        # 权重参数: n_components - 1（权重和为1，最后一个由前n-1个决定）
        params_per_component = n_features + n_features * (n_features + 1) // 2
        total_params = n_components * params_per_component + (n_components - 1)
        
        # 最终对数似然值
        log_likelihood = self.log_likelihoods[-1]
        
        # AIC = 2k - 2ln(L)
        self.aic_ = 2 * total_params - 2 * log_likelihood
        
        # BIC = k * ln(n) - 2ln(L)
        self.bic_ = total_params * np.log(n_samples) - 2 * log_likelihood

    def bic(self):
        """返回 BIC 值（需先调用 fit 训练）"""
        if self.bic_ is None:
            raise ValueError("请先调用 fit 方法训练模型")
        return self.bic_

    def aic(self):
        """返回 AIC 值（需先调用 fit 训练）"""
        if self.aic_ is None:
            raise ValueError("请先调用 fit 方法训练模型")
        return self.aic_

    def _kmeans_plus_plus_init(self, X):
        """k-means++ 初始化：以平方距离为权重的概率采样，使初始中心点尽量分散

        算法步骤：
          1. 从数据集随机均匀地选取第一个中心点
          2. 对剩余每个待选中心点：
             a. 计算每个样本到已选中最近中心的平方距离 D²(x)
             b. 以 D²(x)/Σ D²(x) 为概率分布采样下一个中心点
          3. 重复步骤 2 直到选出 n_components 个中心

        相比纯随机初始化，k-means++ 能有效避免多个中心点聚集在同一区域，
        从而减少 EM 算法陷入局部最优的概率，加快收敛速度。
        """
        n_samples = X.shape[0]
        # 步骤 1：随机选取第一个中心
        first_idx = self.rng.integers(0, n_samples)
        centers = [X[first_idx].copy()]

        for _ in range(1, self.n_components):
            # 向量化计算每个样本到最近已选中心的平方距离
            # center_arr: (k, n_features)；X: (n, n_features)
            center_arr = np.array(centers)                              # (k, f)
            diff = X[:, np.newaxis, :] - center_arr[np.newaxis, :, :]  # (n, k, f)
            sq_dists = np.sum(diff ** 2, axis=2)                        # (n, k)
            min_sq_dists = sq_dists.min(axis=1)                         # (n,)

            # 防止全零（极端情况）导致除零
            total = min_sq_dists.sum()
            if total == 0:
                probs = np.ones(n_samples) / n_samples
            else:
                probs = min_sq_dists / total

            # 按概率采样下一个中心
            next_idx = self.rng.choice(n_samples, p=probs)
            centers.append(X[next_idx].copy())

        return np.array(centers)

    def _log_gaussian(self, X, mu, sigma):
        """计算多维高斯分布的对数概率密度
        
        参数:
            X: 输入数据点/样本集，形状为(n_samples, n_features)
            mu: 高斯分布的均值向量，形状为(n_features,)
            sigma: 高斯分布的协方差矩阵，形状为(n_features, n_features)
            
        返回:
            log_prob: 每个样本的对数概率密度，形状为(n_samples,)
        """
        # 获取特征维度数（协方差矩阵的维度）
        n_features = mu.shape[0]

        # 数据归一化：将数据减去均值，得到中心化数据
        # 高斯分布公式中的(x-μ)项
        X_centered = X - mu  # 形状保持(n_samples, n_features)

        # 计算协方差矩阵的行列式符号和对数值
        # sign: 行列式的符号（正负）
        # logdet: 行列式的自然对数值
        sign, logdet = np.linalg.slogdet(sigma)  # 数值稳定的行列式计算方法

        # 处理协方差矩阵可能奇异（不可逆）的情况
        if sign <= 0:  # 行列式非正（理论上协方差矩阵应是正定的）
            # 添加一个小的对角扰动项（单位矩阵乘以1e-6）
            # 确保矩阵可逆且正定，提高数值稳定性
            sigma += np.eye(n_features) * 1e-6  # 正则化处理
            
            # 重新计算调整后的协方差矩阵的行列式
            sign, logdet = np.linalg.slogdet(sigma)

            # 使用solve方法计算逆矩阵，更稳定高效
            inv = np.linalg.solve(sigma, np.eye(n_features))
            
            # 计算二次型：(x-μ)^T·Σ^(-1)·(x-μ)
            # 使用einsum高效计算多个样本的二次型
            exponent = -0.5 * np.sum(X_centered @ inv * X_centered, axis=1)

            # 返回对数概率密度
            # 公式：log_p(x) = -0.5*D*log(2π) - 0.5*log|Σ| - 0.5*(x-μ)^T·Σ^(-1)·(x-μ)
            return -0.5 * n_features * np.log(2 * np.pi) - 0.5 * logdet + exponent
        else:
            # 处理非奇异协方差矩阵
            inv = np.linalg.inv(sigma) #计算协方差矩阵的逆
            exponent = -0.5 * np.einsum('...i,...i->...', X_centered @ inv, X_centered) #计算指数部分（二次型）
            return -0.5 * n_features * np.log(2 * np.pi) - 0.5 * logdet + exponent #组合对数概率密度
        
    def plot_convergence(self, save_path = None, show = True):
        """可视化对数似然的收敛过程"""
        # 检查是否有对数似然值记录
        if not self.log_likelihoods:
            raise ValueError("请先调用fit方法训练模型")

        # 创建一个图形窗口，设置大小为10x6英寸
        plt.figure(figsize=(10, 6))
        # 绘制对数似然值随迭代次数的变化曲线
        # 使用蓝色实线绘制，范围从1到len(self.log_likelihoods)
        plt.plot(range(1, len(self.log_likelihoods) + 1), self.log_likelihoods, 'b-')
        # 设置x轴的标签为“迭代次数”
        plt.xlabel('迭代次数')
        # 设置y轴的标签为“对数似然值”
        plt.ylabel('对数似然值')
        # 设置图表的标题为“EM算法收敛曲线”
        plt.title('EM算法收敛曲线')
        # 启用网格线，增强可读性
        plt.grid(True, alpha=0.5) 
        # 如提供保存路径，则写入文件
        if save_path is not None:
            plt.savefig(save_path, dpi=140, bbox_inches='tight')
        # 是否显示窗口
        if show:
            plt.show()
        else:
            plt.close()

# ============================================================
# 辅助函数：最优标签匹配准确率（枚举全排列，适用于小 k）
# ============================================================
def _cluster_accuracy(y_true, y_pred, n_classes):
    """枚举所有标签排列，取最高匹配准确率（k<=8 均可）"""
    from itertools import permutations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    best = 0.0
    for perm in permutations(range(n_classes)):
        mapped = np.array([perm[p] for p in y_pred])
        acc = np.mean(mapped == y_true)
        if acc > best:
            best = acc
    return best


# ============================================================
# 模型选择工具：基于 BIC 自动选择最佳成分数量
# ============================================================
def select_best_components(X, min_components=2, max_components=10, random_state=42):
    """基于 BIC 准则自动选择 GMM 的最佳高斯成分数量
    
    参数:
        X: 数据集，形状为(n_samples, n_features)
        min_components: 最小成分数量（默认=2）
        max_components: 最大成分数量（默认=10）
        random_state: 随机种子
    
    返回:
        best_gmm: 最佳成分数量的 GMM 模型
        results: 各成分数量对应的 BIC 值列表
    """
    results = []
    best_bic = np.inf
    best_gmm = None
    
    print(f"基于 BIC 选择最佳成分数量 [{min_components}~{max_components}]...")
    
    for n_components in range(min_components, max_components + 1):
        gmm = GaussianMixtureModel(
            n_components=n_components,
            max_iter=100,
            tol=1e-6,
            random_state=random_state,
            init='kmeans++'
        )
        gmm.fit(X)
        bic = gmm.bic()
        results.append({
            'n_components': n_components,
            'bic': bic,
            'aic': gmm.aic(),
            'n_iters': gmm.n_iters_
        })
        
        print(f"  成分数={n_components}: BIC={bic:.2f}, AIC={gmm.aic():.2f}, 迭代={gmm.n_iters_}")
        
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    
    print(f"\n最佳成分数量: {best_gmm.n_components} (BIC={best_bic:.2f})")
    return best_gmm, results


# ============================================================
# 主程序：随机初始化 vs k-means++ 初始化 对比实验 + BIC 模型选择
# ============================================================
if __name__ == "__main__":
    # 设置中文字体（Windows 优先 Microsoft YaHei，Linux/Mac 回退 SimHei）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 命令行参数
    parser = argparse.ArgumentParser(
        description="GMM 初始化方法对比：随机初始化 vs k-means++ 初始化")
    parser.add_argument("--n-samples",    type=int,   default=1000,    help="样本数量")
    parser.add_argument("--n-components", type=int,   default=3,       help="高斯成分数量")
    parser.add_argument("--max-iter",     type=int,   default=100,     help="EM最大迭代次数")
    parser.add_argument("--tol",          type=float, default=1e-6,    help="收敛阈值")
    parser.add_argument("--n-trials",     type=int,   default=50,      help="对比实验重复次数")
    parser.add_argument("--out-dir",      type=str,   default="outputs", help="输出目录")
    parser.add_argument("--no-show",      action="store_true",         help="不弹出图像窗口，仅保存文件")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 生成固定数据集 ----------
    print("生成混合高斯分布数据...")
    X, y_true = generate_data(n_samples=args.n_samples, random_state=42)
    print(f"数据形状: {X.shape}，类别数: {args.n_components}")

    # ---------- 多次实验：记录收敛迭代次数、最终对数似然、聚类准确率 ----------
    print(f"\n正在运行 {args.n_trials} 次对比实验（每种方法各 {args.n_trials} 次）...")
    random_iters, random_lls, random_accs = [], [], []
    kpp_iters,    kpp_lls,    kpp_accs    = [], [], []

    for seed in range(args.n_trials):
        for init_method, iters_list, lls_list, accs_list in [
            ('random',   random_iters, random_lls, random_accs),
            ('kmeans++', kpp_iters,    kpp_lls,    kpp_accs),
        ]:
            gmm = GaussianMixtureModel(
                n_components=args.n_components,
                max_iter=args.max_iter,
                tol=args.tol,
                random_state=seed,
                init=init_method,
            )
            gmm.fit(X)
            iters_list.append(gmm.n_iters_)
            lls_list.append(gmm.log_likelihoods[-1])
            accs_list.append(_cluster_accuracy(y_true, gmm.labels_, args.n_components))

    # ---------- 打印统计结果 ----------
    print("\n========== 实验结果统计（{} 次）==========".format(args.n_trials))
    print(f"{'指标':<22} {'随机初始化':>14} {'k-means++':>14}  {'提升':>8}")
    print("-" * 62)
    iter_imp  = (np.mean(random_iters) - np.mean(kpp_iters)) / np.mean(random_iters) * 100
    ll_imp    = (np.mean(kpp_lls)    - np.mean(random_lls))  / abs(np.mean(random_lls)) * 100
    acc_imp   = (np.mean(kpp_accs)   - np.mean(random_accs)) / np.mean(random_accs)  * 100
    print(f"{'收敛迭代次数 (均值)':<22} {np.mean(random_iters):>14.1f} {np.mean(kpp_iters):>14.1f}  {-iter_imp:>+7.1f}%")
    print(f"{'收敛迭代次数 (中位数)':<22} {np.median(random_iters):>14.1f} {np.median(kpp_iters):>14.1f}")
    print(f"{'最终对数似然 (均值)':<22} {np.mean(random_lls):>14.2f} {np.mean(kpp_lls):>14.2f}  {ll_imp:>+7.1f}%")
    print(f"{'聚类准确率 (均值)':<22} {np.mean(random_accs):>14.4f} {np.mean(kpp_accs):>14.4f}  {acc_imp:>+7.1f}%")
    print(f"{'聚类准确率 (最低)':<22} {np.min(random_accs):>14.4f} {np.min(kpp_accs):>14.4f}")
    print("=" * 62)

    # ============================================================
    # 图1：多次实验对比基准图（3 列）
    # ============================================================
    COLORS = ['#4C72B0', '#DD8452']
    LABELS = ['随机初始化', 'k-means++']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"GMM 初始化方法对比（{args.n_trials} 次随机实验）\n"
        "k-means++ 在收敛速度、解质量和稳定性上均优于随机初始化",
        fontsize=13, fontweight='bold')

    # 子图1：收敛迭代次数箱线图
    ax = axes[0]
    bp = ax.boxplot([random_iters, kpp_iters], **{_BOXPLOT_LABEL_KEY: LABELS},
                    patch_artist=True, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('收敛迭代次数')
    ax.set_title('收敛速度对比\n（值越小越好）')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.97,
            f'均值: {np.mean(random_iters):.1f}  →  {np.mean(kpp_iters):.1f}  ({-iter_imp:+.1f}%)',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    # 子图2：最终对数似然箱线图
    ax = axes[1]
    bp = ax.boxplot([random_lls, kpp_lls], **{_BOXPLOT_LABEL_KEY: LABELS},
                    patch_artist=True, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('最终对数似然值')
    ax.set_title('收敛质量对比\n（值越大越好）')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.03,
            f'均值: {np.mean(random_lls):.1f}  →  {np.mean(kpp_lls):.1f}  ({ll_imp:+.1f}%)',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    # 子图3：聚类准确率分布直方图
    ax = axes[2]
    lo = min(min(random_accs), min(kpp_accs)) - 0.02
    hi = max(max(random_accs), max(kpp_accs)) + 0.02
    bins = np.linspace(lo, hi, 16)
    ax.hist(random_accs, bins=bins, alpha=0.65, color=COLORS[0], label=LABELS[0])
    ax.hist(kpp_accs,    bins=bins, alpha=0.65, color=COLORS[1], label=LABELS[1])
    ax.set_xlabel('聚类准确率')
    ax.set_ylabel('频次')
    ax.set_title('聚类准确率分布\n（分布越靠右越好）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.97,
            f'均值: {np.mean(random_accs):.4f}  →  {np.mean(kpp_accs):.4f}  ({acc_imp:+.1f}%)',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    plt.tight_layout()
    bench_path = out_dir / "comparison_benchmark.png"
    plt.savefig(bench_path, dpi=140, bbox_inches='tight')
    print(f"\n[已保存] 对比基准图: {bench_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # ============================================================
    # 图2：单次运行聚类散点图对比（真实 / 随机 / k-means++）
    # ============================================================
    gmm_rand = GaussianMixtureModel(
        n_components=args.n_components, max_iter=args.max_iter,
        tol=args.tol, random_state=42, init='random')
    gmm_rand.fit(X)

    gmm_kpp = GaussianMixtureModel(
        n_components=args.n_components, max_iter=args.max_iter,
        tol=args.tol, random_state=42, init='kmeans++')
    gmm_kpp.fit(X)

    acc_rand = _cluster_accuracy(y_true, gmm_rand.labels_, args.n_components)
    acc_kpp  = _cluster_accuracy(y_true, gmm_kpp.labels_,  args.n_components)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GMM 聚类结果对比（seed=42）", fontsize=13, fontweight='bold')

    plot_data = [
        (y_true,          f'真实标签（Ground Truth）'),
        (gmm_rand.labels_, f'随机初始化\n迭代 {gmm_rand.n_iters_} 轮，准确率 {acc_rand:.3f}'),
        (gmm_kpp.labels_,  f'k-means++ 初始化\n迭代 {gmm_kpp.n_iters_} 轮，准确率 {acc_kpp:.3f}'),
    ]
    for ax, (labels, title) in zip(axes, plot_data):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    cluster_path = out_dir / "cluster_comparison.png"
    plt.savefig(cluster_path, dpi=140, bbox_inches='tight')
    print(f"[已保存] 聚类散点图: {cluster_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # ============================================================
    # 图3：EM 收敛曲线对比
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EM 算法收敛曲线对比（seed=42）", fontsize=13, fontweight='bold')

    for ax, (gmm_obj, title, color) in zip(axes, [
        (gmm_rand, f'随机初始化（共 {gmm_rand.n_iters_} 轮）', COLORS[0]),
        (gmm_kpp,  f'k-means++ 初始化（共 {gmm_kpp.n_iters_} 轮）', COLORS[1]),
    ]):
        iters_x = range(1, len(gmm_obj.log_likelihoods) + 1)
        ax.plot(iters_x, gmm_obj.log_likelihoods, '-o',
                color=color, linewidth=2, markersize=4)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('对数似然值')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    conv_path = out_dir / "convergence_comparison.png"
    plt.savefig(conv_path, dpi=140, bbox_inches='tight')
    print(f"[已保存] 收敛曲线图: {conv_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # ---------- 保存迭代日志（两种方法） ----------
    log_path = out_dir / "iteration_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "iteration", "log_likelihood"])
        for i, ll in enumerate(gmm_rand.log_likelihoods, start=1):
            writer.writerow(["random", i, ll])
        for i, ll in enumerate(gmm_kpp.log_likelihoods, start=1):
            writer.writerow(["kmeans++", i, ll])

    # ============================================================
    # 图4：BIC/AIC 模型选择曲线
    # ============================================================
    print("\n--- 基于 BIC 的模型选择 ---")
    best_gmm, bic_results = select_best_components(X, min_components=2, max_components=8, random_state=42)

    n_components_list = [r['n_components'] for r in bic_results]
    bic_values = [r['bic'] for r in bic_results]
    aic_values = [r['aic'] for r in bic_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_components_list, bic_values, '-o', color='#E74C3C', label='BIC', linewidth=2, markersize=6)
    ax.plot(n_components_list, aic_values, '-s', color='#3498DB', label='AIC', linewidth=2, markersize=6)
    
    # 标记最佳成分数
    best_n = best_gmm.n_components
    best_bic = best_gmm.bic()
    ax.scatter(best_n, best_bic, color='#E74C3C', s=150, zorder=5, edgecolors='black', label=f'最佳 k={best_n}')
    
    ax.set_xlabel('高斯成分数量 k')
    ax.set_ylabel('准则值（越小越好）')
    ax.set_title('BIC/AIC 模型选择曲线（自动确定最佳聚类数）', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    bic_path = out_dir / "bic_model_selection.png"
    plt.savefig(bic_path, dpi=140, bbox_inches='tight')
    print(f"[已保存] BIC模型选择图: {bic_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # ---------- 保存 BIC/AIC 结果日志 ----------
    bic_log_path = out_dir / "bic_aic_log.csv"
    with bic_log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "BIC", "AIC", "iterations"])
        for r in bic_results:
            writer.writerow([r['n_components'], r['bic'], r['aic'], r['n_iters']])

    print(f"\n所有输出已保存至: {out_dir.resolve()}")




