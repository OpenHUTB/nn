#!/usr/bin/env python
# coding: utf-8
"""
SVM 核函数对比实验
=================
改进目标：原始 svm.py 仅实现线性 SVM（梯度下降 + hinge loss），
对于非线性可分数据集（train_kernel.txt）效果较差。
本脚本在原线性 SVM 基础上引入 RBF 核函数，展示：
  1. 线性 SVM 在非线性数据上的决策边界（效果差）
  2. RBF 核 SVM 在相同数据上的决策边界（效果好）
  3. 两种方法的准确率对比柱状图

运行方式：
    python svm_kernel_compare.py
输出图片：outputs/svm_kernel_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pathlib import Path

# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_data(fname):
    """加载 SVM 数据文件（含表头行 x1 x2 t）"""
    if not os.path.exists(fname):
        raise FileNotFoundError(f"数据文件未找到: {fname}")
    data = []
    with open(fname, 'r') as f:
        f.readline()          # 跳过表头
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            data.append([float(parts[0]), float(parts[1]), int(float(parts[2]))])
    return np.array(data)


def eval_acc(label, pred):
    """计算准确率"""
    return np.sum(label == pred) / len(pred)


# ─────────────────────────────────────────────
# 原始线性 SVM（梯度下降 + hinge loss）
# 与 svm.py 中 SVM 类逻辑一致，方便对比
# ─────────────────────────────────────────────

class LinearSVM:
    """线性 SVM：梯度下降优化 hinge loss + L2 正则"""

    def __init__(self, learning_rate=0.1, reg_lambda=0.001, max_iter=20000):
        self.lr = learning_rate
        self.lam = reg_lambda
        self.max_iter = max_iter
        self.w = None
        self.b = 0
        self.scaler = StandardScaler()

    def fit(self, X_raw, y_raw):
        X = self.scaler.fit_transform(X_raw)
        y = np.where(y_raw <= 0, -1, 1).astype(np.float64)
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for _ in range(self.max_iter):
            score = X @ self.w + self.b
            margin = y * score
            idx = np.where(margin < 1)[0]
            if len(idx) > 0:
                dw = 2 * self.lam * self.w - np.sum(y[idx, None] * X[idx], axis=0) / m
                db = -np.mean(y[idx])
            else:
                dw = 2 * self.lam * self.w
                db = 0.0
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X_raw):
        X = self.scaler.transform(X_raw)
        score = X @ self.w + self.b
        return np.where(score >= 0, 1, -1).astype(int)

    def decision_function(self, X_raw):
        X = self.scaler.transform(X_raw)
        return X @ self.w + self.b


# ─────────────────────────────────────────────
# 绘图辅助
# ─────────────────────────────────────────────

def plot_decision_boundary(ax, model, X, y, title, acc_train, acc_test, use_sklearn=False):
    """在坐标轴 ax 上绘制散点 + 决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    if use_sklearn:
        Z = model.predict(grid)
    else:
        Z = model.predict(grid)

    Z = Z.reshape(xx.shape)

    # 填充决策区域
    ax.contourf(xx, yy, Z, alpha=0.25, levels=[-2, 0, 2],
                colors=['#FF9999', '#9999FF'])

    # 绘制决策边界
    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.5, linestyles='-')

    # 绘制样本散点
    pos = X[y == 1]
    neg = X[y == -1]
    ax.scatter(pos[:, 0], pos[:, 1], c='blue', marker='+', s=60,
               linewidths=1.5, label='类别 +1')
    ax.scatter(neg[:, 0], neg[:, 1], color='none', edgecolors='darkred',
               marker='o', s=40, linewidths=1.2, label='类别 -1')

    ax.set_title(
        f'{title}\n训练准确率: {acc_train*100:.1f}%  测试准确率: {acc_test*100:.1f}%',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('特征 x1')
    ax.set_ylabel('特征 x2')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'train_kernel.txt')
    test_file  = os.path.join(base_dir, 'data', 'test_kernel.txt')

    data_train = load_data(train_file)
    data_test  = load_data(test_file)

    X_train, y_train = data_train[:, :2], data_train[:, 2].astype(int)
    X_test,  y_test  = data_test[:, :2],  data_test[:, 2].astype(int)

    # ── 1. 原始线性 SVM ──────────────────────────────────────
    print("训练线性 SVM（梯度下降 hinge loss）...")
    linear_svm = LinearSVM(learning_rate=0.1, reg_lambda=0.001, max_iter=20000)
    linear_svm.fit(X_train, y_train)

    train_pred_lin = linear_svm.predict(X_train)
    test_pred_lin  = linear_svm.predict(X_test)
    acc_train_lin  = eval_acc(y_train, train_pred_lin)
    acc_test_lin   = eval_acc(y_test,  test_pred_lin)
    print(f"  线性 SVM  — 训练: {acc_train_lin*100:.1f}%  测试: {acc_test_lin*100:.1f}%")

    # ── 2. RBF 核 SVM（sklearn SMO 求解）─────────────────────
    print("训练 RBF 核 SVM（sklearn SMO）...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rbf_svm = SVC(kernel='rbf', C=10.0, gamma='scale')
    rbf_svm.fit(X_train_s, y_train)

    acc_train_rbf = rbf_svm.score(X_train_s, y_train)
    acc_test_rbf  = rbf_svm.score(X_test_s,  y_test)
    print(f"  RBF 核 SVM — 训练: {acc_train_rbf*100:.1f}%  测试: {acc_test_rbf*100:.1f}%")

    # ── 3. 绘制三图对比 ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SVM 核函数对比：线性 vs RBF 核（非线性可分数据集）',
                 fontsize=14, fontweight='bold', y=1.02)

    # 子图1：线性 SVM 决策边界
    plot_decision_boundary(
        axes[0], linear_svm, X_train, y_train,
        '原始线性 SVM\n（梯度下降 hinge loss）',
        acc_train_lin, acc_test_lin, use_sklearn=False
    )

    # 为 RBF 模型包装一个在原始空间预测的 wrapper
    class RBFWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))

    rbf_wrapper = RBFWrapper(rbf_svm, scaler)

    # 子图2：RBF 核 SVM 决策边界
    plot_decision_boundary(
        axes[1], rbf_wrapper, X_train, y_train,
        '改进 RBF 核 SVM\n（sklearn SMO 求解）',
        acc_train_rbf, acc_test_rbf, use_sklearn=False
    )

    # 子图3：准确率对比柱状图
    ax = axes[2]
    methods  = ['线性 SVM\n（原始）', 'RBF 核 SVM\n（改进）']
    train_accs = [acc_train_lin * 100, acc_train_rbf * 100]
    test_accs  = [acc_test_lin  * 100, acc_test_rbf  * 100]

    x = np.arange(len(methods))
    width = 0.32
    bars1 = ax.bar(x - width/2, train_accs, width, label='训练准确率',
                   color=['#5588CC', '#CC5544'], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, test_accs,  width, label='测试准确率',
                   color=['#88AADD', '#DD8877'], alpha=0.85, edgecolor='white')

    # 在柱子上标注数值
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 标注改进幅度
    improvement = acc_test_rbf * 100 - acc_test_lin * 100
    ax.annotate(
        f'测试准确率提升\n+{improvement:.1f}%',
        xy=(1 + width/2, acc_test_rbf * 100),
        xytext=(1 + width/2 + 0.3, acc_test_rbf * 100 - 8),
        fontsize=10, color='darkgreen', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5)
    )

    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_title('训练 / 测试准确率对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    out_dir = Path(base_dir) / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'svm_kernel_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存至: {out_path.resolve()}")

    # ── 4. 控制台汇总 ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'方法':<22} {'训练准确率':>10} {'测试准确率':>10}")
    print("-" * 50)
    print(f"{'线性 SVM（原始）':<22} {acc_train_lin*100:>9.1f}% {acc_test_lin*100:>9.1f}%")
    print(f"{'RBF 核 SVM（改进）':<22} {acc_train_rbf*100:>9.1f}% {acc_test_rbf*100:>9.1f}%")
    print("-" * 50)
    print(f"测试准确率提升: +{improvement:.1f}%")
    print("=" * 50)


if __name__ == '__main__':
    # 设置中文字体（Windows 下用黑体，Linux 用 WenQuanYi）
    import matplotlib.font_manager as fm
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    for font in chinese_fonts:
        if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            matplotlib.rcParams['font.sans-serif'] = [font]
            break
    matplotlib.rcParams['axes.unicode_minus'] = False

    main()
