import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 路径设置
base_dir = os.path.abspath("../data")  # 数据根目录，包含'train'和'test'文件夹
train_dir = os.path.join(base_dir, "train") # 训练集目录路径
test_dir = os.path.join(base_dir, "test")  # 测试集目录路径

# 模型参数设置
img_size = (128, 128)  # 图像调整尺寸为128x128像素
batch_size = 32 # 每个训练批次的样本数量
epochs = 70# 训练总轮数

# 图像数据预处理与增强（用于训练集）
train_datagen = ImageDataGenerator(
    rescale=1. /255,# 像素值归一化到0-1范围
    rotation_range=30,# 随机旋转角度范围±30度
    width_shift_range=0.1,# 水平随机平移范围10%
    height_shift_range=0.1,# 垂直随机平移范围10%
    shear_range=0.2,# 剪切变换强度
    zoom_range=0.2,# 随机缩放范围
    horizontal_flip=True# 启用水平翻转
)

# 测试集数据预处理（只做归一化，不进行增强）
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 创建训练数据生成器
train_gen = train_datagen.flow_from_directory(
    train_dir, # 训练集目录
    target_size=img_size, # 调整图像大小
    batch_size=batch_size, # 批次大小
    class_mode="categorical"  # 多分类模式
)

# 创建测试数据生成器
test_gen = test_datagen.flow_from_directory(
    test_dir,# 测试集目录
    target_size=img_size, # 调整图像大小
    batch_size=batch_size, # 批次大小
    class_mode="categorical"  # 多分类模式
)

# 导入迁移学习相关模块
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

# 加载预训练的MobileNetV2基础模型
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # 输入图像尺寸
    include_top=False,# 不包含原始顶层分类器
    weights='imagenet' # 使用在ImageNet上预训练的权重
)
base_model.trainable = False # 冻结基础模型权重，不参与训练

# 构建迁移学习模型
model = tf.keras.Sequential([
    base_model, # 预训练的特征提取器
    GlobalAveragePooling2D(),# 全局平均池化层，减少参数数量
    Dense(128, activation="relu"), # 全连接层，128个神经元，ReLU激活
    Dropout(0.5),# 丢弃层，丢弃率50%，防止过拟合
    Dense(train_gen.num_classes, activation="softmax")  # 输出层，使用softmax激活
])

# 编译模型
model.compile(
    optimizer="adam", # 使用Adam优化器
    loss="categorical_crossentropy", # 分类交叉熵损失函数
    metrics=["accuracy"] # 评估指标为准确率
)

# 打印模型结构摘要
model.summary()

# 设置训练回调函数
# 早停回调：监控验证集损失，连续5轮无改善则停止训练
early_stop = EarlyStopping(
    monitor='val_loss', # 监控验证集损失
    patience=5,# 容忍轮数
    restore_best_weights=True  # 恢复最佳权重
)

# 模型检查点回调：保存最佳模型
checkpoint = ModelCheckpoint(
    filepath=os.path.join(base_dir, "best_model.h5"),  # 模型保存路径
    monitor='val_accuracy',# 监控验证集准确率
    save_best_only=True,# 只保存最佳模型
    verbose=1# 显示保存信息
)

# 开始训练模型
history = model.fit(
    train_gen, # 训练数据生成器
    epochs=epochs,  # 训练轮数
    validation_data=test_gen,  # 验证数据
    callbacks=[early_stop, checkpoint]  # 使用回调函数
)

# 保存最终训练完成的模型
model.save(os.path.join(base_dir, "cnn_model.h5"))

# 训练完成提示
print("模型训练完成并已保存。")


# 错误分析函数
def analyze_errors(model, test_gen, class_labels, num_samples=16):
    """
    分析模型在测试集上的错误分类情况

    参数:
    - model: 训练好的模型
    - test_gen: 测试数据生成器
    - class_labels: 类别标签列表
    - num_samples: 要显示的错误样本数量
    """

    # 重置测试生成器
    test_gen.reset()

    # 获取所有预测和真实标签
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    # 计算准确率
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"测试集准确率: {accuracy:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # 混淆矩阵
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
    plt.show()

    # 找出错误分类的样本
    misclassified_indices = np.where(predicted_classes != true_classes)[0]

    print(f"\n总错误分类样本数: {len(misclassified_indices)}")
    print(f"总样本数: {len(true_classes)}")
    print(f"错误率: {len(misclassified_indices) / len(true_classes):.4f}")

     # 显示一些错误分类的样本
    if len(misclassified_indices) > 0:
        # 随机选择一些错误样本进行可视化
        if len(misclassified_indices) > num_samples:
            selected_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
        else:
            selected_indices = misclassified_indices

        # 获取文件名
        filenames = test_gen.filenames

        # 创建错误分类可视化
        plot_misclassified_samples(selected_indices, filenames, true_classes,
                                   predicted_classes, predictions, class_labels, test_gen)

    return misclassified_indices


def plot_misclassified_samples(indices, filenames, true_classes, predicted_classes,
                               predictions, class_labels, test_gen):
    """
    绘制错误分类的样本图像
    """
    # 计算网格大小
    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    # 重置生成器以获取原始图像
    test_gen.reset()
    all_images = []
    all_batches = len(test_gen)

    # 收集所有图像
    for i in range(all_batches):
        images, _ = test_gen[i]
        all_images.extend(images)

    for i, idx in enumerate(indices):
        if i < len(axes):
            ax = axes[i]

            # 显示图像
            ax.imshow(all_images[idx])

            # 设置标题
            true_label = class_labels[true_classes[idx]]
            pred_label = class_labels[predicted_classes[idx]]
            confidence = np.max(predictions[idx])

            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}"
            ax.set_title(title, fontsize=10, color='red')

            # 获取文件名（不包含路径）
            filename = os.path.basename(filenames[idx])
            ax.set_xlabel(f"File: {filename}", fontsize=8)

            ax.axis('off')

    # 隐藏多余的子图
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    plt.suptitle('错误分类样本示例', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'misclassified_samples.png'),
                bbox_inches='tight', dpi=300)
    plt.show()

    # 打印错误样本的详细信息
    print("\n错误分类样本详情:")
    print("-" * 80)
    for i, idx in enumerate(indices[:10]):  # 只显示前10个的详细信息
        true_label = class_labels[true_classes[idx]]
        pred_label = class_labels[predicted_classes[idx]]
        confidence = np.max(predictions[idx])
        filename = os.path.basename(filenames[idx])

        print(f"{i + 1:2d}. 文件: {filename:20s} | 真实: {true_label:15s} | "
              f"预测: {pred_label:15s} | 置信度: {confidence:.4f}")


# 在训练完成后调用错误分析
print("开始错误分析...")

# 获取类别标签
class_labels = list(train_gen.class_indices.keys())

# 加载最佳模型进行错误分析（如果有保存的最佳模型）
best_model_path = os.path.join(base_dir, "best_model.h5")
if os.path.exists(best_model_path):
    print("加载最佳模型进行错误分析...")
    best_model = tf.keras.models.load_model(best_model_path)
    misclassified_indices = analyze_errors(best_model, test_gen, class_labels)
else:
    print("使用最终训练模型进行错误分析...")
    misclassified_indices = analyze_errors(model, test_gen, class_labels)

print("\n错误分析完成！")