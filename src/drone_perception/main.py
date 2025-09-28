import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 路径设置
base_dir = os.path.abspath("../data")  # 数据根目录，包含'train'和'test'文件夹
train_dir = os.path.join(base_dir, "train")  # 训练集目录路径
test_dir = os.path.join(base_dir, "test")    # 测试集目录路径

# 模型参数设置
img_size = (128, 128)  # 图像调整尺寸为128x128像素
batch_size = 32        # 每个训练批次的样本数量
epochs = 70            # 训练总轮数

# 图像数据预处理与增强（用于训练集）
train_datagen = ImageDataGenerator(
    rescale=1./255,           # 像素值归一化到0-1范围
    rotation_range=30,        # 随机旋转角度范围±30度
    width_shift_range=0.1,    # 水平随机平移范围10%
    height_shift_range=0.1,   # 垂直随机平移范围10%
    shear_range=0.2,          # 剪切变换强度
    zoom_range=0.2,           # 随机缩放范围
    horizontal_flip=True      # 启用水平翻转
)

# 测试集数据预处理（只做归一化，不进行增强）
test_datagen = ImageDataGenerator(rescale=1./255)

# 创建训练数据生成器
train_gen = train_datagen.flow_from_directory(
    train_dir,                # 训练集目录
    target_size=img_size,     # 调整图像大小
    batch_size=batch_size,    # 批次大小
    class_mode="categorical"  # 多分类模式
)

# 创建测试数据生成器
test_gen = test_datagen.flow_from_directory(
    test_dir,                 # 测试集目录
    target_size=img_size,     # 调整图像大小
    batch_size=batch_size,    # 批次大小
    class_mode="categorical"  # 多分类模式
)

# 导入迁移学习相关模块
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

# 加载预训练的MobileNetV2基础模型
base_model = MobileNetV2(
    input_shape=(128, 128, 3),  # 输入图像尺寸
    include_top=False,          # 不包含原始顶层分类器
    weights='imagenet'          # 使用在ImageNet上预训练的权重
)
base_model.trainable = False   # 冻结基础模型权重，不参与训练

# 构建迁移学习模型
model = tf.keras.Sequential([
    base_model,                    # 预训练的特征提取器
    GlobalAveragePooling2D(),      # 全局平均池化层，减少参数数量
    Dense(128, activation="relu"), # 全连接层，128个神经元，ReLU激活
    Dropout(0.5),                  # 丢弃层，丢弃率50%，防止过拟合
    Dense(train_gen.num_classes, activation="softmax")  # 输出层，使用softmax激活
])

# 编译模型
model.compile(
    optimizer="adam",                      # 使用Adam优化器
    loss="categorical_crossentropy",      # 分类交叉熵损失函数
    metrics=["accuracy"]                  # 评估指标为准确率
)

# 打印模型结构摘要
model.summary()

# 设置训练回调函数
# 早停回调：监控验证集损失，连续5轮无改善则停止训练
early_stop = EarlyStopping(
    monitor='val_loss',      # 监控验证集损失
    patience=5,              # 容忍轮数
    restore_best_weights=True  # 恢复最佳权重
)

# 模型检查点回调：保存最佳模型
checkpoint = ModelCheckpoint(
    filepath=os.path.join(base_dir, "best_model.h5"),  # 模型保存路径
    monitor='val_accuracy',   # 监控验证集准确率
    save_best_only=True,      # 只保存最佳模型
    verbose=1                 # 显示保存信息
)

# 开始训练模型
history = model.fit(
    train_gen,                    # 训练数据生成器
    epochs=epochs,               # 训练轮数
    validation_data=test_gen,    # 验证数据
    callbacks=[early_stop, checkpoint]  # 使用回调函数
)

# 保存最终训练完成的模型
model.save(os.path.join(base_dir, "cnn_model.h5"))

# 训练完成提示
print("模型训练完成并已保存。")
