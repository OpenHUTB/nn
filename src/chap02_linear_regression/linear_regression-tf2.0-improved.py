#!/usr/bin/env python
# coding: utf-8
"""
深度优化版线性回归 —— 修复版
修复了学习率获取错误
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model
from pathlib import Path
import json
import os
from datetime import datetime
import argparse

# 设置随机种子确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==================== 配置类 ====================
class Config:
    """全局配置"""
    # 数据参数
    train_file = os.getenv("LR_TRAIN_FILE", "train.txt")
    test_file = os.getenv("LR_TEST_FILE", "test.txt")

    # 基函数参数
    basis_type = os.getenv("LR_BASIS_TYPE", "gaussian")  # gaussian, polynomial, identity
    feature_num = int(os.getenv("LR_FEATURE_NUM", "10"))

    # 训练参数
    epochs = int(os.getenv("LR_EPOCHS", "10000"))
    batch_size = int(os.getenv("LR_BATCH_SIZE", "32"))
    initial_lr = float(os.getenv("LR_INITIAL_LR", "0.05"))
    decay_steps = int(os.getenv("LR_DECAY_STEPS", "2000"))
    decay_rate = float(os.getenv("LR_DECAY_RATE", "0.96"))

    # 正则化参数
    l2_lambda = float(os.getenv("LR_L2_LAMBDA", "0.001"))

    # 早停参数
    early_stopping_patience = int(os.getenv("LR_EARLY_STOPPING", "500"))
    min_delta = float(os.getenv("LR_MIN_DELTA", "1e-6"))

    # 输出参数
    print_interval = int(os.getenv("LR_PRINT_INTERVAL", "500"))
    output_dir = os.getenv("LR_OUTPUT_DIR", "outputs")
    save_model = os.getenv("LR_SAVE_MODEL", "True").lower() == "true"

    # 可视化参数
    plot_smooth_points = int(os.getenv("LR_PLOT_POINTS", "500"))


config = Config()


# ==================== 基函数 ====================
class BasisFunction:
    """基函数类 - 支持多种基函数"""

    @staticmethod
    def identity(x):
        """恒等基函数"""
        return np.expand_dims(x, axis=1)

    @staticmethod
    def polynomial(x, feature_num=10):
        """多项式基函数 - 添加数值稳定性处理"""
        x = np.expand_dims(x, axis=1)
        feat = [x]
        for i in range(2, feature_num + 1):
            # 对高次幂进行归一化，防止数值爆炸
            x_pow = x ** i
            if i > 5:  # 高次幂进行缩放
                x_pow = x_pow / (np.max(np.abs(x_pow)) + 1e-8)
            feat.append(x_pow)
        return np.concatenate(feat, axis=1)

    @staticmethod
    def gaussian(x, feature_num=10):
        """高斯基函数 — 改进版：自适应宽度"""
        x_min, x_max = x.min(), x.max()
        # 在数据范围内均匀放置中心点，两端各扩展一点
        centers = np.linspace(x_min - 0.2 * (x_max - x_min),
                              x_max + 0.2 * (x_max - x_min),
                              feature_num)
        # 自适应宽度：根据中心点间距动态调整
        width = 0.8 * (centers[1] - centers[0]) if feature_num > 1 else 1.0
        x_expanded = np.expand_dims(x, axis=1)
        x_expanded = np.concatenate([x_expanded] * feature_num, axis=1)
        out = (x_expanded - centers) / width
        return np.exp(-0.5 * out ** 2)

    @staticmethod
    def fourier(x, feature_num=10):
        """傅里叶基函数 - 适合周期性数据"""
        x_expanded = np.expand_dims(x, axis=1)
        freqs = np.linspace(0.5, 5, feature_num // 2)
        feat = []
        for freq in freqs:
            feat.append(np.sin(2 * np.pi * freq * x_expanded))
            feat.append(np.cos(2 * np.pi * freq * x_expanded))
        if len(feat) < feature_num:
            feat.append(np.ones_like(x_expanded))
        return np.concatenate(feat, axis=1)[:, :feature_num]

    @classmethod
    def get_basis(cls, basis_type):
        """获取基函数"""
        basis_map = {
            'identity': cls.identity,
            'polynomial': lambda x: cls.polynomial(x, config.feature_num),
            'gaussian': lambda x: cls.gaussian(x, config.feature_num),
            'fourier': lambda x: cls.fourier(x, config.feature_num)
        }
        return basis_map.get(basis_type, cls.gaussian)


# ==================== 数据加载与预处理 ====================
class DataLoader:
    """数据加载与预处理类"""

    def __init__(self, basis_func):
        self.basis_func = basis_func
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def load_data(self, filename):
        """
        载入数据
        数据格式: 每行两个值 (x, y)
        """
        x_list, y_list = [], []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    x_list.append(float(parts[0]))
                    y_list.append(float(parts[1]))
                except ValueError:
                    continue

        if len(x_list) == 0:
            raise ValueError(f"文件 {filename} 没有有效数据")

        xs = np.asarray(x_list, dtype=np.float32)
        ys = np.asarray(y_list, dtype=np.float32)
        return xs, ys

    def normalize_x(self, x, fit=True):
        """标准化输入特征"""
        if fit:
            self.x_mean = np.mean(x)
            self.x_std = np.std(x) + 1e-8
        return (x - self.x_mean) / self.x_std

    def normalize_y(self, y, fit=True):
        """标准化输出标签"""
        if fit:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y) + 1e-8
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_norm):
        """反标准化输出"""
        return y_norm * self.y_std + self.y_mean

    def prepare_data(self, filename, fit=True):
        """准备训练/测试数据"""
        x_raw, y_raw = self.load_data(filename)

        # 标准化
        x_norm = self.normalize_x(x_raw, fit=fit)
        y_norm = self.normalize_y(y_raw, fit=fit)

        # 基函数变换
        phi0 = np.expand_dims(np.ones_like(x_norm), axis=1)
        phi1 = self.basis_func(x_norm)
        X = np.concatenate([phi0, phi1], axis=1).astype(np.float32)

        return X, y_norm, x_raw, y_raw


# ==================== 改进的线性回归模型 ====================
class ImprovedLinearModel(Model):
    """带正则化的线性回归模型"""

    def __init__(self, ndim, l2_lambda=0.001, name=None):
        """
        Args:
            ndim: 输入维度
            l2_lambda: L2正则化系数
        """
        super(ImprovedLinearModel, self).__init__(name=name)
        self.l2_lambda = l2_lambda

        # Xavier初始化
        limit = np.sqrt(6.0 / ndim)
        self.w = tf.Variable(
            shape=[ndim, 1],
            initial_value=tf.random.uniform(
                [ndim, 1], minval=-limit, maxval=limit, dtype=tf.float32
            ),
            trainable=True,
            name="weight",
        )
        self.b = tf.Variable(
            initial_value=tf.zeros([1], dtype=tf.float32),
            trainable=True,
            name="bias",
        )

    @tf.function
    def call(self, x, training=False):
        x = tf.matmul(x, self.w)
        y = tf.squeeze(x, axis=1) + self.b
        return y

    def regularization_loss(self):
        """计算L2正则化损失"""
        return self.l2_lambda * 0.5 * tf.reduce_sum(tf.square(self.w))


# ==================== 训练管理类 ====================
class TrainingManager:
    """训练过程管理器"""

    def __init__(self, model, optimizer, early_stopping_patience=500, min_delta=1e-6):
        self.model = model
        self.optimizer = optimizer
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        self.best_loss = np.inf
        self.best_weights = None
        self.patience_counter = 0
        self.history = {'loss': [], 'val_loss': [], 'lr': []}
        self.step = 0

    def get_current_learning_rate(self):
        """安全获取当前学习率"""
        lr = self.optimizer.learning_rate
        # 如果学习率是可调用的调度器
        if callable(lr):
            return lr(self.optimizer.iterations).numpy()
        # 如果是普通变量
        elif hasattr(lr, 'numpy'):
            return lr.numpy()
        else:
            return float(lr)

    def train_step(self, x_batch, y_batch, training=True):
        """单步训练"""
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=training)
            loss = tf.reduce_mean(tf.keras.losses.MSE(y_batch, y_pred))
            loss += self.model.regularization_loss()

        grads = tape.gradient(loss, self.model.trainable_variables)
        # 梯度裁剪，防止梯度爆炸
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train_epoch(self, dataset, training=True):
        """训练一个epoch"""
        epoch_losses = []
        for x_batch, y_batch in dataset:
            loss = self.train_step(x_batch, y_batch, training)
            epoch_losses.append(loss.numpy())
            self.step += 1
        return np.mean(epoch_losses)

    def validate(self, x_val, y_val):
        """验证"""
        y_pred = self.model(x_val, training=False)
        loss = tf.reduce_mean(tf.keras.losses.MSE(y_val, y_pred))
        loss += self.model.regularization_loss()
        return loss.numpy()

    def early_stopping_check(self, val_loss):
        """早停检查"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            # 保存最佳权重
            self.best_weights = [w.numpy() for w in self.model.trainable_variables]
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience

    def restore_best_weights(self):
        """恢复最佳权重"""
        if self.best_weights is not None:
            for var, weight in zip(self.model.trainable_variables, self.best_weights):
                var.assign(weight)

    def record_history(self, loss, val_loss):
        """记录历史"""
        current_lr = self.get_current_learning_rate()
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)
        self.history['lr'].append(current_lr)


# ==================== 评估工具 ====================
class Evaluator:
    """模型评估工具"""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """计算多种评估指标"""
        y_true = y_true.numpy() if hasattr(y_true, 'numpy') else y_true
        y_pred = y_pred.numpy() if hasattr(y_pred, 'numpy') else y_pred

        # 标准差
        std = np.std(y_true - y_pred)

        # R²决定系数
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # 平均绝对误差
        mae = np.mean(np.abs(y_true - y_pred))

        # 均方根误差
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # 平均绝对百分比误差（处理接近0的值）
        y_true_safe = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        return {
            'std': std,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    @staticmethod
    def print_metrics(metrics, title="Evaluation"):
        """打印评估指标"""
        print(f"\n{title}:")
        print(f"  Standard Deviation: {metrics['std']:.4f}")
        print(f"  R² Score: {metrics['r2']:.6f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")


# ==================== 可视化工具 ====================
class Visualizer:
    """可视化工具类"""

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_predictions(self, x_train, y_train, x_test, y_test,
                         x_smooth, y_smooth, title="Regression Result"):
        """绘制预测结果"""
        plt.figure(figsize=(12, 8))

        # 绘制数据点
        plt.plot(x_train, y_train, 'ro', markersize=4, label='Training Data', alpha=0.7)
        plt.plot(x_test, y_test, 'bo', markersize=4, label='Test Data', alpha=0.7)

        # 绘制预测曲线
        plt.plot(x_smooth, y_smooth, 'k-', linewidth=2.5, label='Prediction', alpha=0.9)

        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5, color='gray')
        plt.legend(fontsize=11)
        plt.tight_layout()

        # 保存图片
        save_path = self.output_dir / 'regression_result.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.show()

    def plot_training_history(self, history):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 损失曲线
        epochs = range(1, len(history['loss']) + 1)
        axes[0].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=1.5)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 学习率曲线
        axes[1].plot(epochs, history['lr'], 'g-', linewidth=1.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        # 残差分布（最后10个epoch）
        if len(history['loss']) > 10:
            recent_loss = history['loss'][-10:]
            recent_val_loss = history['val_loss'][-10:]
            axes[2].bar(['Train', 'Validation'], [np.mean(recent_loss), np.mean(recent_val_loss)],
                        color=['blue', 'red'], alpha=0.7)
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Recent 10 Epochs Average Loss')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
        plt.show()

    def plot_residuals(self, y_true, y_pred, title="Residual Analysis"):
        """绘制残差分析图"""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 残差分布
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)

        # 残差直方图
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'residual_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Residual analysis saved to: {save_path}")
        plt.show()


# ==================== 模型保存与加载 ====================
class ModelManager:
    """模型管理类"""

    def __init__(self, model, output_dir="outputs"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, filename="linear_model"):
        """保存模型权重和配置"""
        save_path = self.output_dir / f"{filename}.weights.h5"
        self.model.save_weights(str(save_path))

        # 保存配置
        config_info = {
            'ndim': int(self.model.w.shape[0]),
            'l2_lambda': float(self.model.l2_lambda),
            'timestamp': datetime.now().isoformat()
        }
        config_path = self.output_dir / f"{filename}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)

        print(f"Model saved to: {save_path}")
        print(f"Config saved to: {config_path}")

    def load_model(self, filename="linear_model"):
        """加载模型权重"""
        load_path = self.output_dir / f"{filename}.weights.h5"
        if load_path.exists():
            self.model.load_weights(str(load_path))
            print(f"Model loaded from: {load_path}")
            return True
        else:
            print(f"Model file not found: {load_path}")
            return False


# ==================== 主函数 ====================
def main():
    """主训练流程"""
    print("=" * 70)
    print("Deeply Optimized Linear Regression")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Basis function: {config.basis_type}")
    print(f"  - Feature number: {config.feature_num}")
    print(f"  - Epochs: {config.epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Initial learning rate: {config.initial_lr}")
    print(f"  - L2 lambda: {config.l2_lambda}")
    print(f"  - Early stopping patience: {config.early_stopping_patience}")
    print("=" * 70)

    # 1. 数据准备
    print("\n[1/7] Loading data...")
    basis_func = BasisFunction.get_basis(config.basis_type)
    data_loader = DataLoader(basis_func)

    # 加载训练数据
    X_train, y_train_norm, x_train_raw, y_train_raw = data_loader.prepare_data(
        config.train_file, fit=True
    )

    # 加载测试数据（使用训练集的统计量）
    X_test, y_test_norm, x_test_raw, y_test_raw = data_loader.prepare_data(
        config.test_file, fit=False
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Feature dimension: {X_train.shape[1]}")

    # 2. 创建数据集
    print("\n[2/7] Creating datasets...")
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_norm))
    train_dataset = train_dataset.shuffle(1000).batch(config.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    # 3. 创建模型
    print("\n[3/7] Creating model...")
    model = ImprovedLinearModel(
        ndim=X_train.shape[1],
        l2_lambda=config.l2_lambda
    )

    # 4. 配置优化器
    print("\n[4/7] Configuring optimizer...")
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.initial_lr,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=True,
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    # 5. 训练
    print("\n[5/7] Training...")
    trainer = TrainingManager(
        model, optimizer,
        early_stopping_patience=config.early_stopping_patience,
        min_delta=config.min_delta
    )

    for epoch in range(config.epochs):
        # 训练一个epoch
        train_loss = trainer.train_epoch(train_dataset, training=True)

        # 验证
        val_loss = trainer.validate(X_test, y_test_norm)

        # 记录历史
        trainer.record_history(train_loss, val_loss)

        # 打印
        if (epoch + 1) % config.print_interval == 0:
            current_lr = trainer.get_current_learning_rate()
            print(f"  Epoch {epoch + 1:5d}/{config.epochs} | "
                  f"Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")

        # 早停检查
        if trainer.early_stopping_check(val_loss):
            print(f"\n  Early stopping triggered at epoch {epoch + 1}")
            break

    # 恢复最佳权重
    trainer.restore_best_weights()

    # 6. 评估
    print("\n[6/7] Evaluating model...")
    evaluator = Evaluator()

    # 训练集评估
    y_train_pred = model(X_train, training=False)
    y_train_denorm = data_loader.denormalize_y(y_train_norm)
    y_train_pred_denorm = data_loader.denormalize_y(y_train_pred)
    train_metrics = evaluator.calculate_metrics(y_train_denorm, y_train_pred_denorm)
    evaluator.print_metrics(train_metrics, "Training Set Evaluation")

    # 测试集评估
    y_test_pred = model(X_test, training=False)
    y_test_denorm = data_loader.denormalize_y(y_test_norm)
    y_test_pred_denorm = data_loader.denormalize_y(y_test_pred)
    test_metrics = evaluator.calculate_metrics(y_test_denorm, y_test_pred_denorm)
    evaluator.print_metrics(test_metrics, "Test Set Evaluation")

    # 7. 可视化
    print("\n[7/7] Generating visualizations...")
    visualizer = Visualizer(config.output_dir)

    # 生成平滑曲线
    x_min = min(x_train_raw.min(), x_test_raw.min())
    x_max = max(x_train_raw.max(), x_test_raw.max())
    x_smooth = np.linspace(x_min - 0.1 * (x_max - x_min),
                           x_max + 0.1 * (x_max - x_min),
                           config.plot_smooth_points)

    # 对平滑点做变换
    x_smooth_norm = data_loader.normalize_x(x_smooth, fit=False)
    phi0_s = np.expand_dims(np.ones_like(x_smooth_norm), axis=1)
    phi1_s = basis_func(x_smooth_norm)
    X_smooth = np.concatenate([phi0_s, phi1_s], axis=1).astype(np.float32)
    y_smooth_norm = model(X_smooth, training=False)
    y_smooth = data_loader.denormalize_y(y_smooth_norm)

    # 绘制预测结果
    visualizer.plot_predictions(
        x_train_raw, y_train_raw,
        x_test_raw, y_test_raw,
        x_smooth, y_smooth,
        title=f"Linear Regression ({config.basis_type.capitalize()} Basis)"
    )

    # 绘制训练历史
    visualizer.plot_training_history(trainer.history)

    # 绘制残差分析
    visualizer.plot_residuals(y_test_denorm, y_test_pred_denorm)

    # 8. 保存模型
    if config.save_model:
        print("\nSaving model...")
        model_manager = ModelManager(model, config.output_dir)
        model_manager.save_model()

    # 9. 保存训练指标
    print("\nSaving metrics...")
    metrics = {
        'config': {
            'basis_type': config.basis_type,
            'feature_num': config.feature_num,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'initial_lr': config.initial_lr,
            'l2_lambda': config.l2_lambda,
            'early_stopping_patience': config.early_stopping_patience
        },
        'train_metrics': {k: float(v) for k, v in train_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_val_loss': float(trainer.best_loss),
        'epochs_trained': len(trainer.history['loss']),
        'timestamp': datetime.now().isoformat()
    }

    metrics_path = Path(config.output_dir) / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print(f"Output directory: {config.output_dir}")
    print("Generated files:")
    print("  - regression_result.png")
    print("  - training_history.png")
    print("  - residual_analysis.png")
    print("  - linear_model.weights.h5")
    print("  - training_metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    # 支持命令行参数
    parser = argparse.ArgumentParser(description="Deeply Optimized Linear Regression")
    parser.add_argument("--basis", type=str, default="gaussian",
                        choices=['identity', 'polynomial', 'gaussian', 'fourier'],
                        help="Basis function type")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization lambda")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    # 覆盖配置
    config.basis_type = args.basis
    config.epochs = args.epochs
    config.initial_lr = args.lr
    config.l2_lambda = args.l2
    config.output_dir = args.output

    main()
