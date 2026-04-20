import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import argparse
import csv
from pathlib import Path
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# 数据生成
# ─────────────────────────────────────────────

def generate_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """生成混合高斯分布数据集。

    Args:
        n_samples:    样本总数（默认 1000）
        random_state: 随机种子（默认 42）

    Returns:
        X:      特征矩阵，形状 (n_samples, 2)
        y_true: 真实标签，形状 (n_samples,)
    """
    np.random.seed(random_state)

    mu_true = np.array([
        [0,  0],
        [5,  5],
        [-5, 5],
    ])
    sigma_true = np.array([
        [[1,  0.0], [0.0,  1]],
        [[2,  0.5], [0.5,  1]],
        [[1, -0.5], [-0.5, 2]],
    ])
    weights_true = np.array([0.3, 0.4, 0.3])
    n_components = len(weights_true)

    samples_per_component = (weights_true * n_samples).astype(int)
    # 补足因浮点截断少掉的样本
    samples_per_component[np.argmax(weights_true)] += n_samples - samples_per_component.sum()

    X_list, y_list = [], []
    for i in range(n_components):
        X_i = np.random.multivariate_normal(mu_true[i], sigma_true[i], samples_per_component[i])
        X_list.append(X_i)
        y_list.extend([i] * samples_per_component[i])

    X = np.vstack(X_list)
    y_true = np.array(y_list)
    idx = np.random.permutation(n_samples)
    return X[idx], y_true[idx]


# ─────────────────────────────────────────────
# 数值稳定的 logsumexp
# ─────────────────────────────────────────────

def logsumexp(log_p: np.ndarray, axis: int = 1, keepdims: bool = False) -> np.ndarray:
    """数值稳定的 log(sum(exp(log_p)))。

    通过减去最大值避免上溢/下溢：
        log(Σ exp(xᵢ)) = max(x) + log(Σ exp(xᵢ − max(x)))
    """
    log_p = np.asarray(log_p)
    if log_p.size == 0:
        return np.array(-np.inf, dtype=log_p.dtype)

    max_val = np.max(log_p, axis=axis, keepdims=True)
    if np.all(np.isneginf(max_val)):
        return max_val.copy() if keepdims else max_val.squeeze(axis=axis)

    safe = np.where(np.isneginf(log_p), -np.inf, log_p - max_val)
    # 中间计算始终保持维度，最后再按 keepdims 决定是否压缩
    sum_exp = np.sum(np.exp(safe), axis=axis, keepdims=True)
    result = max_val + np.log(sum_exp)          # 形状与 max_val 一致
    if not keepdims:
        result = result.squeeze(axis=axis)
    return result


# ─────────────────────────────────────────────
# 高斯混合模型
# ─────────────────────────────────────────────

class GaussianMixtureModel:
    """EM 算法实现的高斯混合模型（GMM）。

    Args:
        n_components:  高斯成分数量（默认 3）
        max_iter:      EM 最大迭代次数（默认 100）
        tol:           对数似然收敛阈值（默认 1e-6）
        random_state:  随机种子（可选）
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihoods: list = []
        # 统一使用 self.rng，避免与全局 np.random 状态混用
        self.rng = np.random.default_rng(random_state)

        # 训练后才有效的属性
        self.pi: np.ndarray
        self.mu: np.ndarray
        self.sigma: np.ndarray
        self.labels_: np.ndarray

    # ── 训练 ──────────────────────────────────

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """使用 EM 算法拟合数据。

        E 步：计算每个样本属于各成分的后验概率（responsibility）。
        M 步：用后验概率加权更新 π、μ、Σ。

        Args:
            X: 训练数据，形状 (n_samples, n_features)

        Returns:
            self（支持链式调用）
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 初始化参数
        self.pi = np.ones(self.n_components) / self.n_components
        # 修复：统一使用 self.rng，不再调用全局 np.random.choice
        init_idx = self.rng.choice(n_samples, self.n_components, replace=False)
        self.mu = X[init_idx].copy()
        self.sigma = np.array([np.eye(n_features)] * self.n_components, dtype=float)

        log_likelihood = -np.inf

        for _ in range(self.max_iter):
            # ── E 步 ──────────────────────────
            log_prob = np.column_stack([
                np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.sigma[k])
                for k in range(self.n_components)
            ])  # (n_samples, n_components)

            log_prob_sum = logsumexp(log_prob, axis=1, keepdims=True)
            # clip 防止极小值引起 NaN
            gamma = np.clip(np.exp(log_prob - log_prob_sum), 0, 1)

            # ── M 步 ──────────────────────────
            Nk = gamma.sum(axis=0)                    # (n_components,)
            self.pi = Nk / n_samples

            new_mu = np.zeros_like(self.mu)
            new_sigma = np.zeros_like(self.sigma)

            for k in range(self.n_components):
                w = gamma[:, k]                       # (n_samples,)
                new_mu[k] = (w[:, None] * X).sum(axis=0) / Nk[k]

                X_c = X - new_mu[k]                   # 中心化
                # Σ_k = Σᵢ γᵢₖ (xᵢ−μₖ)(xᵢ−μₖ)ᵀ / Nₖ
                new_sigma[k] = (w[:, None] * X_c).T @ X_c / Nk[k]
                new_sigma[k] += np.eye(n_features) * 1e-6   # 正则化，保证正定

            # ── 收敛检测 ──────────────────────
            current_ll = float(log_prob_sum.sum())
            self.log_likelihoods.append(current_ll)

            if abs(current_ll - log_likelihood) < self.tol:
                break
            log_likelihood = current_ll

            self.mu = new_mu
            self.sigma = new_sigma

        self.labels_ = np.argmax(gamma, axis=1)
        return self

    # ── 预测 ──────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """对新样本预测所属成分。

        Args:
            X: 输入数据，形状 (n_samples, n_features)

        Returns:
            labels: 预测标签，形状 (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        log_prob = np.column_stack([
            np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.sigma[k])
            for k in range(self.n_components)
        ])
        return np.argmax(log_prob, axis=1)

    def score(self, X: np.ndarray) -> float:
        """计算数据的平均对数似然（越大越好）。

        Args:
            X: 输入数据，形状 (n_samples, n_features)

        Returns:
            平均对数似然（标量）
        """
        X = np.asarray(X, dtype=float)
        log_prob = np.column_stack([
            np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.sigma[k])
            for k in range(self.n_components)
        ])
        return float(logsumexp(log_prob, axis=1).mean())

    # ── 内部方法 ──────────────────────────────

    def _log_gaussian(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """多元高斯分布的对数概率密度。

        log p(x) = −D/2·log(2π) − 1/2·log|Σ| − 1/2·(x−μ)ᵀΣ⁻¹(x−μ)

        Args:
            X:     数据矩阵，(n_samples, n_features)
            mu:    均值向量，(n_features,)
            sigma: 协方差矩阵，(n_features, n_features)

        Returns:
            对数概率密度，(n_samples,)
        """
        n_features = mu.shape[0]
        # 修复：操作局部副本，不修改调用方传入的 sigma
        sigma = sigma.copy()

        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            sigma += np.eye(n_features) * 1e-6
            sign, logdet = np.linalg.slogdet(sigma)

        X_c = X - mu
        # 用 solve 代替显式求逆，数值更稳定
        inv_sigma_X = np.linalg.solve(sigma, X_c.T).T      # (n_samples, n_features)
        exponent = -0.5 * np.einsum("ki,ki->k", X_c, inv_sigma_X)
        return -0.5 * n_features * np.log(2 * np.pi) - 0.5 * logdet + exponent

    # ── 可视化 ────────────────────────────────

    def plot_convergence(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """绘制 EM 算法收敛曲线。"""
        if not self.log_likelihoods:
            raise ValueError("请先调用 fit() 训练模型")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.log_likelihoods) + 1), self.log_likelihoods, "b-o", ms=4)
        plt.xlabel("迭代次数")
        plt.ylabel("对数似然值")
        plt.title("EM 算法收敛曲线")
        plt.grid(True, alpha=0.5)
        if save_path is not None:
            plt.savefig(save_path, dpi=140, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()


# ─────────────────────────────────────────────
# 聚类准确率（匈牙利算法对齐标签）
# ─────────────────────────────────────────────

def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_components: int) -> float:
    """用最优标签映射计算聚类准确率。

    由于 GMM 预测的标签编号可能与真实标签不一致，
    通过枚举所有排列找到最佳对应关系。
    """
    from itertools import permutations

    best_acc = 0.0
    for perm in permutations(range(n_components)):
        mapped = np.array([perm[p] for p in y_pred])
        acc = np.mean(mapped == y_true)
        best_acc = max(best_acc, acc)
    return best_acc


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM 无监督聚类实验")
    parser.add_argument("--n-samples",    type=int,   default=1000,  help="样本数量")
    parser.add_argument("--n-components", type=int,   default=3,     help="高斯成分数量")
    parser.add_argument("--max-iter",     type=int,   default=100,   help="EM 最大迭代次数")
    parser.add_argument("--tol",          type=float, default=1e-6,  help="收敛阈值")
    parser.add_argument("--random-state", type=int,   default=42,    help="随机种子")
    parser.add_argument("--out-dir",      type=str,   default="outputs", help="输出目录")
    parser.add_argument("--no-show",      action="store_true",       help="不弹出图像窗口")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成数据
    print("生成混合高斯分布数据...")
    X, y_true = generate_data(n_samples=args.n_samples, random_state=args.random_state)
    print(f"数据形状: X={X.shape}, y={y_true.shape}")

    # 2. 训练模型
    gmm = GaussianMixtureModel(
        n_components=args.n_components,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.random_state,
    )
    gmm.fit(X)
    y_pred = gmm.labels_

    # 3. 评估
    acc = cluster_accuracy(y_true, y_pred, args.n_components)
    print(f"迭代轮数: {len(gmm.log_likelihoods)}")
    print(f"最终对数似然: {gmm.log_likelihoods[-1]:.4f}")
    print(f"平均对数似然(score): {gmm.score(X):.4f}")
    print(f"聚类准确率: {acc * 100:.2f}%")

    # 4. 聚类对比图
    show = not args.no_show
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, labels, title in zip(
        axes,
        [y_true, y_pred],
        ["True Clusters", "GMM Predicted Clusters"],
    ):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    cluster_path = out_dir / "cluster_comparison.png"
    plt.savefig(cluster_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    # 5. 收敛曲线
    conv_path = out_dir / "convergence_curve.png"
    gmm.plot_convergence(save_path=conv_path, show=show)

    # 6. 迭代日志
    log_path = out_dir / "iteration_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "log_likelihood"])
        for i, ll in enumerate(gmm.log_likelihoods, start=1):
            writer.writerow([i, f"{ll:.6f}"])

    print(f"\n输出目录 : {out_dir.resolve()}")
    print(f"聚类图   : {cluster_path.name}")
    print(f"收敛图   : {conv_path.name}")
    print(f"日志     : {log_path.name}")