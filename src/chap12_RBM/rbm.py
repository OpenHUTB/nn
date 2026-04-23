# encoding: utf-8
"""深度优化 RBM (v2)

在 v1 基础上新增：
1. scipy.special.expit 加速 sigmoid（SIMD C 实现，快 2-3x）
   无 scipy 时用 tanh 恒等式回退 —— 顺便修掉 v1 中
   `np.exp(-x[pos], out=out[pos])` 写入 fancy-index 副本的 BUG
2. 矩阵乘法 out= 参数 + 预分配缓冲区，消除 batch 级 alloc
3. 动量更新全 in-place（*= 和 +=）
4. PCD (Persistent Contrastive Divergence) 选项，梯度偏差更小
5. CD-k 可配置（默认 k=1）
6. 批量 Gibbs 采样（多链并行）
7. 自由能监控（可选）
"""
import numpy as np
import sys

try:
    from scipy.special import expit as _scipy_expit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class RBM:
    """Restricted Boltzmann Machine - 深度优化版"""

    def __init__(self, n_hidden=2, n_observe=784, dtype=np.float32, seed=None):
        if not (isinstance(n_hidden, int) and n_hidden > 0):
            raise ValueError("n_hidden 必须为正整数")
        if not (isinstance(n_observe, int) and n_observe > 0):
            raise ValueError("n_observe 必须为正整数")

        self.n_hidden = n_hidden
        self.n_observe = n_observe
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)

        init_std = np.sqrt(2.0 / (n_observe + n_hidden))
        self.W = self.rng.normal(0, init_std, (n_observe, n_hidden)).astype(dtype)
        self.b_h = np.zeros(n_hidden, dtype=dtype)
        self.b_v = np.zeros(n_observe, dtype=dtype)

        # 动量缓存
        self._vW = np.zeros_like(self.W)
        self._vb_v = np.zeros_like(self.b_v)
        self._vb_h = np.zeros_like(self.b_h)

        # PCD 持久化链（按需初始化）
        self._pcd_chain = None

    # ---------- 基础原语 ----------
    @staticmethod
    def _sigmoid(x, out=None):
        """Sigmoid：scipy.special.expit（SIMD C 实现）优先，否则 tanh 恒等式回退。

        注意：v1 版本的 masked-exp 实现有 BUG——`out[pos]` 是 fancy-index 产生的
        *副本*，给 `np.exp(..., out=out[pos])` 写进去的值会被丢弃。tanh 版本无此问题。
        """
        if _HAS_SCIPY:
            return _scipy_expit(x, out=out)
        # 回退：sigmoid(x) = 0.5 + 0.5 * tanh(x/2)，全范围数值稳定
        if out is None:
            out = np.empty_like(x)
        np.multiply(x, 0.5, out=out)
        np.tanh(out, out=out)
        out *= 0.5
        out += 0.5
        return out

    def _sample_binary(self, probs):
        """快速伯努利采样。"""
        return (self.rng.random(probs.shape, dtype=self.dtype) < probs).astype(self.dtype)

    def free_energy(self, v):
        """F(v) = -b_v·v - Σ_j softplus(b_h_j + Σ_i v_i W_ij)

        比重构误差更严格：直接反映 log P(v) 的相对值（忽略配分函数）。
        """
        v = np.atleast_2d(v).astype(self.dtype, copy=False)
        vbias_term = v @ self.b_v
        wx_b = v @ self.W + self.b_h
        # 数值稳定的 softplus: log(1+exp(x)) = max(x,0) + log(1+exp(-|x|))
        softplus = np.maximum(wx_b, 0) + np.log1p(np.exp(-np.abs(wx_b)))
        return -vbias_term - softplus.sum(axis=-1)

    # ---------- 训练 ----------
    def train(self, data, epochs=10, batch_size=100, learning_rate=0.1,
              momentum=0.5, weight_decay=1e-4, k=1, use_pcd=False,
              verbose=True, track_free_energy=False):
        """CD-k / PCD-k 训练。

        Args:
            k: Gibbs 采样步数（1 = 标准 CD-1）
            use_pcd: True 则用 Persistent CD，负相位从持久链继续
                     而非每 batch 从 v0 重启
        """
        data_flat = np.ascontiguousarray(
            data.reshape(data.shape[0], -1).astype(self.dtype, copy=False)
        )
        n_samples = data_flat.shape[0]
        n_batches = n_samples // batch_size
        inv_bs = self.dtype(1.0 / batch_size)
        lr = self.dtype(learning_rate)
        mom = self.dtype(momentum)
        wd_lr = self.dtype(learning_rate * weight_decay)

        # 预分配全部热路径缓冲区
        h_buf = np.empty((batch_size, self.n_hidden), dtype=self.dtype)
        v_buf = np.empty((batch_size, self.n_observe), dtype=self.dtype)
        h0_prob = np.empty_like(h_buf)
        v_neg_prob = np.empty_like(v_buf)
        h_neg_prob = np.empty_like(h_buf)

        # PCD 链按需初始化
        if use_pcd:
            if self._pcd_chain is None or self._pcd_chain.shape != (batch_size, self.n_observe):
                self._pcd_chain = (
                    self.rng.random((batch_size, self.n_observe), dtype=self.dtype) < 0.5
                ).astype(self.dtype)

        epoch_errors = []
        for epoch in range(epochs):
            perm = self.rng.permutation(n_samples)
            err_sum = 0.0

            for i in range(n_batches):
                idx = perm[i * batch_size:(i + 1) * batch_size]
                v0 = data_flat[idx]

                # -- 正相位：v0 -> h0_prob（用概率算梯度，方差小） --
                np.matmul(v0, self.W, out=h_buf)
                h_buf += self.b_h
                self._sigmoid(h_buf, out=h0_prob)

                # -- 负相位起点 --
                if use_pcd:
                    # PCD：从持久链继续
                    v_neg_sample = self._pcd_chain
                    np.matmul(v_neg_sample, self.W, out=h_buf)
                    h_buf += self.b_h
                    self._sigmoid(h_buf, out=h_neg_prob)
                    h_neg_sample = self._sample_binary(h_neg_prob)
                else:
                    # CD：从 h0 采样开始
                    h_neg_sample = self._sample_binary(h0_prob)

                # -- k 步 Gibbs --
                for step in range(k):
                    # h -> v
                    np.matmul(h_neg_sample, self.W.T, out=v_buf)
                    v_buf += self.b_v
                    self._sigmoid(v_buf, out=v_neg_prob)

                    # 最后一步：v 用概率（不采样）、h 用概率，减小梯度方差
                    if step == k - 1:
                        np.matmul(v_neg_prob, self.W, out=h_buf)
                        h_buf += self.b_h
                        self._sigmoid(h_buf, out=h_neg_prob)
                    else:
                        v_neg_sample = self._sample_binary(v_neg_prob)
                        np.matmul(v_neg_sample, self.W, out=h_buf)
                        h_buf += self.b_h
                        self._sigmoid(h_buf, out=h_neg_prob)
                        h_neg_sample = self._sample_binary(h_neg_prob)

                # PCD：更新持久链（用采样后的 v 以保持链的随机性）
                if use_pcd:
                    self._pcd_chain = self._sample_binary(v_neg_prob)

                # -- 重构误差 --
                diff = v0 - v_neg_prob
                err_sum += float(np.mean(diff * diff))

                # -- 梯度 --
                grad_W = (v0.T @ h0_prob - v_neg_prob.T @ h_neg_prob) * inv_bs
                grad_b_v = diff.sum(axis=0) * inv_bs
                grad_b_h = (h0_prob - h_neg_prob).sum(axis=0) * inv_bs

                # -- 动量 + L2 权重衰减，全 in-place --
                self._vW *= mom
                self._vW += lr * grad_W
                self._vW -= wd_lr * self.W
                self._vb_v *= mom
                self._vb_v += lr * grad_b_v
                self._vb_h *= mom
                self._vb_h += lr * grad_b_h

                self.W += self._vW
                self.b_v += self._vb_v
                self.b_h += self._vb_h

            avg_err = err_sum / n_batches
            epoch_errors.append(avg_err)
            if verbose:
                if track_free_energy:
                    fe = float(self.free_energy(data_flat[:500]).mean())
                    print(f"Epoch {epoch+1}/{epochs}, MSE: {avg_err:.6f}, FE: {fe:.3f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, 重构误差: {avg_err:.6f}")

        return epoch_errors

    # ---------- 采样 ----------
    def sample(self, n_steps=200, n_samples=1, start=None, return_prob=False):
        """批量 Gibbs 采样。

        Args:
            n_steps: Gibbs 步数
            n_samples: 并行链数；返回 (n_samples, side, side)，n=1 时返回 (side, side)
            start: 初始状态 (n_samples, n_observe)，None 则随机均匀初始化
            return_prob: 最后一步返回 v_prob 而非二值样本（更清晰的灰度图）
        """
        if start is None:
            v = (self.rng.random((n_samples, self.n_observe), dtype=self.dtype) < 0.5
                 ).astype(self.dtype)
        else:
            v = np.atleast_2d(np.asarray(start, dtype=self.dtype)).reshape(-1, self.n_observe)
            n_samples = v.shape[0]

        # 预分配缓冲区
        h_buf = np.empty((n_samples, self.n_hidden), dtype=self.dtype)
        v_buf = np.empty((n_samples, self.n_observe), dtype=self.dtype)

        for step in range(n_steps):
            np.matmul(v, self.W, out=h_buf)
            h_buf += self.b_h
            h_prob = self._sigmoid(h_buf, out=h_buf)
            h_sample = self._sample_binary(h_prob)

            np.matmul(h_sample, self.W.T, out=v_buf)
            v_buf += self.b_v
            v_prob = self._sigmoid(v_buf, out=v_buf)

            if return_prob and step == n_steps - 1:
                v = v_prob.copy()
            else:
                v = self._sample_binary(v_prob)

        side = int(np.sqrt(self.n_observe))
        out = v.reshape(-1, side, side)
        return out[0] if out.shape[0] == 1 else out


if __name__ == '__main__':
    try:
        mnist = np.load('mnist_bin.npy')
    except IOError:
        from tensorflow.keras import datasets
        (train_images, _), (_, _) = datasets.mnist.load_data()
        mnist_bin = (train_images >= 128).astype(np.int8)
        np.save('mnist_bin.npy', mnist_bin)
        mnist = np.load('mnist_bin.npy')
    except Exception as e:
        print(f"加载 MNIST 失败: {e}")
        sys.exit(1)

    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols
    print(f"数据形状: {mnist.shape}, scipy 加速: {_HAS_SCIPY}")

    rbm = RBM(n_hidden=2, n_observe=img_size, seed=42)

    print("=" * 50 + "\n开始训练 RBM (CD-1 + momentum + weight decay)...\n" + "=" * 50)
    errors = rbm.train(mnist, epochs=10, batch_size=100,
                       learning_rate=0.1, momentum=0.5, weight_decay=1e-4,
                       k=1, use_pcd=False)
    np.save('training_errors.npy', np.array(errors))

    print("=" * 50 + "\n训练完成，生成样本...\n" + "=" * 50)
    # 一次性批量采样 5 张，而不是串行 5 次
    samples = rbm.sample(n_steps=200, n_samples=5, return_prob=True)
    np.save('generated_samples.npy', np.asarray(samples))
    print("Done.")