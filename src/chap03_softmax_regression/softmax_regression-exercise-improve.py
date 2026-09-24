#!/usr/bin/env python
# coding: utf-8

# # 优化版 Softmax Regression (修复版)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

# 设置随机种子确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 禁用TensorFlow的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==================== 配置参数 ====================
class Config:
    # 数据参数
    dot_num = 100  # 每类样本数量
    input_dim = 2
    num_classes = 3

    # 训练参数
    batch_size = int(os.getenv("SOFTMAX_BATCH_SIZE", "32"))
    epochs = int(os.getenv("SOFTMAX_EPOCHS", "100"))
    learning_rate = float(os.getenv("SOFTMAX_LEARNING_RATE", "0.1"))
    validation_split = float(os.getenv("SOFTMAX_VALIDATION_SPLIT", "0.2"))

    # 正则化参数
    l2_lambda = float(os.getenv("SOFTMAX_L2_LAMBDA", "0.001"))

    # 早停参数
    patience = int(os.getenv("SOFTMAX_PATIENCE", "10"))
    min_delta = float(os.getenv("SOFTMAX_MIN_DELTA", "0.001"))

    # 日志和保存
    log_interval = int(os.getenv("SOFTMAX_LOG_INTERVAL", "10"))
    metrics_out = os.getenv("SOFTMAX_METRICS_OUT", "outputs/softmax_metrics.json")
    checkpoint_dir = os.getenv("SOFTMAX_CHECKPOINT_DIR", "checkpoints")
    log_dir = os.getenv("SOFTMAX_LOG_DIR", "logs")

    # 数值稳定性参数
    epsilon = 1e-12


config = Config()


# ==================== 数据生成 ====================
def generate_data():
    """生成三分类高斯分布数据"""
    # 类别0: 均值(3,6)
    x0 = np.random.normal(3.0, 1, config.dot_num)
    y0 = np.random.normal(6.0, 1, config.dot_num)
    label0 = np.zeros(config.dot_num)  # 类别0

    # 类别1: 均值(6,3)
    x1 = np.random.normal(6.0, 1, config.dot_num)
    y1 = np.random.normal(3.0, 1, config.dot_num)
    label1 = np.ones(config.dot_num)  # 类别1

    # 类别2: 均值(7,7)
    x2 = np.random.normal(7.0, 1, config.dot_num)
    y2 = np.random.normal(7.0, 1, config.dot_num)
    label2 = np.ones(config.dot_num) * 2  # 类别2

    # 合并数据
    X = np.vstack([np.column_stack([x0, y0]),
                   np.column_stack([x1, y1]),
                   np.column_stack([x2, y2])])
    y = np.concatenate([label0, label1, label2])

    # 打乱数据
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    return X.astype(np.float32), y.astype(np.int32)


# ==================== 数据预处理 ====================
class DataPreprocessor:
    """数据标准化处理器"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """计算均值和标准差"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        return self

    def transform(self, X):
        """应用标准化"""
        if self.mean is None or self.std is None:
            raise ValueError("必须先调用fit方法")
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """拟合并转换"""
        self.fit(X)
        return self.transform(X)


# ==================== 模型定义 ====================
class SoftmaxRegression(tf.Module):
    """带L2正则化的Softmax回归模型"""

    def __init__(self, input_dim=2, num_classes=3, l2_lambda=0.001, name=None):
        super().__init__(name=name)
        self.l2_lambda = l2_lambda
        self.num_classes = num_classes

        # 使用Xavier/Glorot初始化
        initializer = tf.keras.initializers.GlorotUniform()
        self.W = tf.Variable(
            initializer([input_dim, num_classes], dtype=tf.float32),
            name="W",
            trainable=True
        )
        self.b = tf.Variable(
            tf.zeros([num_classes], dtype=tf.float32),
            name="b",
            trainable=True
        )

    def __call__(self, x, return_logits=False):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, input_dim]
            return_logits: 是否返回logits（用于数值稳定的损失计算）
        Returns:
            概率分布或logits
        """
        logits = tf.matmul(x, self.W) + self.b

        if return_logits:
            return logits

        # 数值稳定的softmax
        return tf.nn.softmax(logits)

    def regularization_loss(self):
        """计算L2正则化损失"""
        return self.l2_lambda * 0.5 * tf.reduce_sum(tf.square(self.W))

    def get_config(self):
        """返回模型配置"""
        return {
            'input_dim': self.W.shape[0],
            'num_classes': self.num_classes,
            'l2_lambda': self.l2_lambda
        }


# ==================== 损失函数 ====================
@tf.function
def compute_loss(logits, labels, model=None):
    """
    计算交叉熵损失和准确率（数值稳定版本）
    Args:
        logits: 模型输出的logits [batch_size, num_classes]
        labels: 真实标签 [batch_size]
        model: 模型实例（用于正则化）
    Returns:
        loss, accuracy
    """
    # 转换标签为int32
    labels = tf.cast(labels, tf.int32)

    # 使用TensorFlow内置的稳定交叉熵
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
    )

    # 添加L2正则化
    if model is not None:
        loss += model.regularization_loss()

    # 计算准确率
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

    return loss, accuracy


# ==================== 训练步骤 ====================
@tf.function
def train_step(model, optimizer, x_batch, y_batch):
    """单步训练"""
    with tf.GradientTape() as tape:
        logits = model(x_batch, return_logits=True)
        loss, accuracy = compute_loss(logits, y_batch, model)

    # 计算梯度并更新
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, accuracy


@tf.function
def evaluate(model, x_val, y_val):
    """验证集评估"""
    logits = model(x_val, return_logits=True)
    loss, accuracy = compute_loss(logits, y_val, model)
    return loss, accuracy


# ==================== 学习率调度器 ====================
class LearningRateScheduler:
    """学习率调度器"""

    def __init__(self, initial_lr=0.1, decay_steps=100, decay_rate=0.96, staircase=True):
        self.schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )

    def get_optimizer(self):
        """返回配置好的优化器"""
        return tf.keras.optimizers.SGD(learning_rate=self.schedule, momentum=0.9)

    def get_learning_rate(self, step):
        """获取当前学习率"""
        return self.schedule(step)


# ==================== 早停机制 ====================
class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_accuracy = 0
        self.should_stop = False

    def update(self, val_loss, val_accuracy):
        """更新状态并判断是否停止"""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop

    def get_best_metrics(self):
        """返回最佳指标"""
        return self.best_loss, self.best_accuracy


# ==================== 模型管理 ====================
class ModelManager:
    """模型保存和加载管理"""

    def __init__(self, model, checkpoint_dir='checkpoints'):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 创建检查点
        self.checkpoint = tf.train.Checkpoint(
            model=model,
            step=tf.Variable(0, dtype=tf.int64)
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            str(self.checkpoint_dir),
            max_to_keep=3
        )

    def save(self, step, is_best=False):
        """保存模型"""
        self.checkpoint.step.assign(step)
        save_path = self.manager.save()

        if is_best:
            best_path = self.checkpoint_dir / 'best_model'
            self.checkpoint.write(str(best_path))

        return save_path

    def load_latest(self):
        """加载最新模型"""
        latest_checkpoint = self.manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            return int(self.checkpoint.step.numpy())
        return 0

    def load_best(self):
        """加载最佳模型"""
        best_path = self.checkpoint_dir / 'best_model'
        # 检查是否存在best_model的index文件
        if (self.checkpoint_dir / 'best_model.index').exists():
            self.checkpoint.restore(str(best_path))
            return True
        return False


# ==================== 训练监控 ====================
class TrainingMonitor:
    """训练过程监控"""

    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.train_log_dir = self.log_dir / 'train'
        self.val_log_dir = self.log_dir / 'validation'

        # 清理旧日志
        import shutil
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)

        self.train_writer = tf.summary.create_file_writer(str(self.train_log_dir))
        self.val_writer = tf.summary.create_file_writer(str(self.val_log_dir))

    def log_train_metrics(self, step, loss, accuracy, learning_rate=None):
        """记录训练指标"""
        with self.train_writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar('accuracy', accuracy, step=step)
            if learning_rate is not None:
                tf.summary.scalar('learning_rate', learning_rate, step=step)

    def log_val_metrics(self, step, loss, accuracy):
        """记录验证指标"""
        with self.val_writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar('accuracy', accuracy, step=step)

    def log_histogram(self, step, weights, biases):
        """记录参数分布"""
        with self.train_writer.as_default():
            tf.summary.histogram('weights', weights, step=step)
            tf.summary.histogram('biases', biases, step=step)


# ==================== 可视化工具 ====================
class Visualizer:
    """结果可视化"""

    @staticmethod
    def plot_data(X, y, title="Generated Data"):
        """绘制原始数据"""
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'red']
        markers = ['+', 'o', '*']
        labels = ['Class 0', 'Class 1', 'Class 2']

        for i in range(3):
            mask = y == i
            if np.any(mask):
                plt.scatter(X[mask, 0], X[mask, 1],
                            c=colors[i], marker=markers[i],
                            label=labels[i], s=80, alpha=0.7)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()

    @staticmethod
    def plot_decision_boundary(model, X, y, preprocessor=None, title="Decision Boundary"):
        """绘制决策边界"""
        plt.figure(figsize=(10, 8))

        # 确定边界范围
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))

        # 预测网格点
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        if preprocessor:
            grid = preprocessor.transform(grid)

        Z = model(grid)
        Z = tf.argmax(Z, axis=1).numpy()
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

        # 绘制原始数据
        colors = ['blue', 'green', 'red']
        markers = ['+', 'o', '*']
        for i in range(3):
            mask = y == i
            if np.any(mask):
                plt.scatter(X[mask, 0], X[mask, 1],
                            c=colors[i], marker=markers[i],
                            label=f'Class {i}', s=60, alpha=0.8)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()

    @staticmethod
    def plot_training_history(history):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(history['epoch'], history['train_acc'], 'b-', label='Train Accuracy')
        ax2.plot(history['epoch'], history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ==================== 训练流程 ====================
def train_model():
    """完整的训练流程"""

    # 1. 生成数据
    print("Generating data...")
    X_raw, y_raw = generate_data()

    # 2. 数据预处理
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X = preprocessor.fit_transform(X_raw)

    # 3. 划分训练集和验证集
    n = len(X)
    n_val = int(n * config.validation_split)
    indices = np.random.permutation(n)

    X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
    y_train, y_val = y_raw[indices[n_val:]], y_raw[indices[:n_val]]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # 4. 创建模型
    print("Creating model...")
    model = SoftmaxRegression(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        l2_lambda=config.l2_lambda
    )

    # 5. 配置优化器和调度器
    steps_per_epoch = max(1, len(X_train) // config.batch_size)
    lr_scheduler = LearningRateScheduler(
        initial_lr=config.learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = lr_scheduler.get_optimizer()

    # 6. 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(config.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    # 7. 初始化管理工具
    model_manager = ModelManager(model, config.checkpoint_dir)
    monitor = TrainingMonitor(config.log_dir)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    visualizer = Visualizer()

    # 8. 训练历史记录
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # 9. 绘制原始数据
    print("Plotting original data...")
    visualizer.plot_data(X_raw, y_raw, "Original Data Distribution")
    plt.savefig('original_data.png', dpi=100, bbox_inches='tight')
    plt.close()  # 关闭图形，避免阻塞

    # 10. 训练循环
    print("\nStarting training...")
    global_step = 0
    best_val_acc = 0

    for epoch in range(config.epochs):
        # 训练阶段
        epoch_train_losses = []
        epoch_train_accs = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
            loss, acc = train_step(model, optimizer, x_batch, y_batch)
            epoch_train_losses.append(loss.numpy())
            epoch_train_accs.append(acc.numpy())

            # 记录训练指标
            if global_step % config.log_interval == 0:
                # 修复：正确获取当前学习率
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(optimizer.iterations)
                monitor.log_train_metrics(global_step, loss.numpy(), acc.numpy(),
                                          current_lr.numpy() if hasattr(current_lr, 'numpy') else current_lr)

            global_step += 1

        # 计算平均训练指标
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_acc = np.mean(epoch_train_accs)

        # 验证阶段
        val_loss, val_acc = evaluate(model, X_val, y_val)
        val_loss = val_loss.numpy()
        val_acc = val_acc.numpy()

        # 记录验证指标
        monitor.log_val_metrics(epoch, val_loss, val_acc)

        # 记录历史
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{config.epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_manager.save(epoch, is_best=True)

        # 早停检查
        if early_stopping.update(val_loss, val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            print(f"Best validation accuracy: {early_stopping.best_accuracy:.4f}")
            break

    # 11. 加载最佳模型
    print("\nLoading best model...")
    if not model_manager.load_best():
        print("Warning: No best model found, using current model")

    # 12. 绘制训练历史
    print("Plotting training history...")
    visualizer.plot_training_history(history)
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 13. 绘制决策边界
    print("Plotting decision boundary...")
    visualizer.plot_decision_boundary(model, X_raw, y_raw, preprocessor, "Decision Boundary")
    plt.savefig('decision_boundary.png', dpi=100, bbox_inches='tight')
    plt.close()

    return model, preprocessor, history, best_val_acc


# ==================== 模型评估和保存 ====================
def save_metrics(metrics, output_path):
    """保存训练指标"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        return obj

    # 递归转换
    def convert_dict(d):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = convert_dict(value)
            elif isinstance(value, list):
                result[key] = [convert_to_serializable(item) for item in value]
            else:
                result[key] = convert_to_serializable(value)
        return result

    serializable_metrics = convert_dict(metrics)

    with out_path.open('w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nMetrics saved to: {out_path.resolve()}")


def evaluate_on_test(model, preprocessor, X_raw, y_raw):
    """在测试集上评估模型"""
    X = preprocessor.transform(X_raw)
    logits = model(X, return_logits=True)
    loss, accuracy = compute_loss(logits, y_raw, model)

    print("\n" + "=" * 50)
    print("Final Evaluation Results")
    print("=" * 50)
    print(f"Test Loss: {loss.numpy():.4f}")
    print(f"Test Accuracy: {accuracy.numpy():.4f}")
    print("=" * 50)

    return loss.numpy(), accuracy.numpy()


# ==================== 预测示例 ====================
def predict_sample(model, preprocessor, sample):
    """预测单个样本"""
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    sample_normalized = preprocessor.transform(sample)
    probs = model(sample_normalized)
    pred_class = tf.argmax(probs, axis=1).numpy()[0]
    confidence = tf.reduce_max(probs, axis=1).numpy()[0]

    return pred_class, confidence, probs.numpy()[0]


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("Optimized Softmax Regression for Multi-class Classification")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs: {config.epochs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - L2 regularization: {config.l2_lambda}")
    print(f"  - Early stopping patience: {config.patience}")
    print("=" * 60 + "\n")

    # 训练模型
    try:
        model, preprocessor, history, best_val_acc = train_model()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # 最终评估
    print("\nEvaluating on test set...")
    X_raw, y_raw = generate_data()  # 生成新的测试数据
    test_loss, test_acc = evaluate_on_test(model, preprocessor, X_raw, y_raw)

    # 保存指标
    metrics = {
        'final_test_loss': test_loss,
        'final_test_accuracy': test_acc,
        'best_validation_accuracy': best_val_acc,
        'epochs_trained': len(history['epoch']),
        'config': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'l2_lambda': config.l2_lambda,
            'patience': config.patience,
            'validation_split': config.validation_split
        },
        'history': {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc']
        },
        'timestamp': datetime.now().isoformat()
    }

    save_metrics(metrics, config.metrics_out)

    # 预测示例
    print("\n" + "=" * 50)
    print("Sample Predictions")
    print("=" * 50)

    # 从每个类别选一个样本进行预测
    X_test, y_test = generate_data()
    X_test_normalized = preprocessor.transform(X_test)

    for class_id in range(3):
        class_samples = X_test_normalized[y_test == class_id]
        if len(class_samples) > 0:
            sample = class_samples[0:1]  # 取第一个样本
            pred_class, confidence, probs = predict_sample(model, preprocessor, sample)
            print(f"\nTrue Class: {class_id}")
            print(f"Predicted Class: {pred_class}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Class Probabilities: {probs}")

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"TensorBoard: tensorboard --logdir {config.log_dir}")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print("Generated files:")
    print("  - original_data.png")
    print("  - training_history.png")
    print("  - decision_boundary.png")
    print(f"  - {config.metrics_out}")
    print("=" * 50)


if __name__ == "__main__":
    main()