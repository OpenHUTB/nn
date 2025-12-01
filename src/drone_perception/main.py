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

# 导入其他模块的功能
from Data_classfication import split_dataset
from image_classification import ImageDataset, ImageClassifier, train_pytorch_model
from visual_navigation import run_visual_navigation
from 预测 import predict_image, predict_directory

# 路径设置
base_dir = os.path.abspath("./data")  # 修改为当前目录下的data
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
dataset_dir = os.path.join(base_dir, "dataset")

def setup_directories():
    """设置数据目录"""
    print("=" * 50)
    
    # 检查并创建目录
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 检查是否需要分割数据集
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print("训练集不存在或为空，开始自动分割数据集...")
        if os.path.exists(dataset_dir):
            success = split_dataset(dataset_dir, train_dir, test_dir, split_ratio=0.8)
            if not success:
                print("❌ 数据集分割失败，请检查原始数据集路径")
                return False
        else:
            print(f"❌ 原始数据集路径不存在: {dataset_dir}")
            print("请将数据集放入 ./data/dataset/ 目录")
            return False
    else:
        print("✅ 训练集已存在，跳过数据集分割步骤")
    
    return True

def train_tensorflow_model():
    """使用TensorFlow训练模型"""
    print("\n" + "=" * 50)
    print("开始TensorFlow模型训练...")
    
    # 模型参数设置
    img_size = (128, 128)
    batch_size = 32
    epochs = 70

    # 图像数据预处理与增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # 测试集数据预处理
    test_datagen = ImageDataGenerator(rescale=1./255)

    # 创建训练数据生成器
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    # 创建测试数据生成器
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    # 导入迁移学习相关模块
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D

    # 加载预训练的MobileNetV2基础模型
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # 构建迁移学习模型
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation="softmax")
    ])

    # 编译模型
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 打印模型结构摘要
    model.summary()

    # 设置训练回调函数
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(base_dir, "best_tensorflow_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # 开始训练模型
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=[early_stop, checkpoint]
    )

    # 保存最终训练完成的模型
    model.save(os.path.join(base_dir, "cnn_model.h5"))
    print("✅ TensorFlow模型训练完成并已保存。")
    
    return model, train_gen, test_gen

def train_pytorch_model_wrapper():
    """使用PyTorch训练模型"""
    print("\n" + "=" * 50)
    print("开始PyTorch模型训练...")
    
    # 调用tuxianfenlei.py中的训练函数
    model, train_losses, val_accuracies = train_pytorch_model(
        base_dir=base_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        img_size=(128, 128),
        batch_size=32,
        epochs=70
    )
    
    print("✅ PyTorch模型训练完成。")
    return model, train_losses, val_accuracies

def analyze_errors(model, test_gen, class_labels, num_samples=16):
    """
    分析模型在测试集上的错误分类情况
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

    return misclassified_indices

def run_error_analysis():
    """运行错误分析"""
    print("\n" + "=" * 50)
    print("开始错误分析...")
    
    # 加载最佳模型进行错误分析
    best_model_path = os.path.join(base_dir, "best_tensorflow_model.h5")
    if os.path.exists(best_model_path):
        print("加载最佳模型进行错误分析...")
        best_model = tf.keras.models.load_model(best_model_path)
        
        # 重新创建数据生成器以获取类别信息
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode="categorical",
            shuffle=False
        )
        
        class_labels = list(test_gen.class_indices.keys())
        misclassified_indices = analyze_errors(best_model, test_gen, class_labels)
    else:
        print("❌ 最佳模型文件不存在，无法进行错误分析")
    
    print("\n错误分析完成！")

def main():
    """主函数"""
    print("🚀 开始图像分类系统...")
    
    # 1. 设置数据目录
    if not setup_directories():
        return
    
    # 2. 训练TensorFlow模型
    tf_model, train_gen, test_gen = train_tensorflow_model()
    
    # 3. 训练PyTorch模型
    pytorch_model, train_losses, val_accuracies = train_pytorch_model_wrapper()
    
    # 4. 错误分析
    run_error_analysis()
    
    # 5. 启动视觉导航（可选）
    print("\n" + "=" * 50)
    choice = input("是否启动视觉导航系统？(y/n): ")
    if choice.lower() == 'y':
        run_visual_navigation()
    
    # 6. 提供预测功能（可选）
    print("\n" + "=" * 50)
    choice = input("是否进行图像预测？(y/n): ")
    if choice.lower() == 'y':
        test_image_path = input("请输入测试图像路径（或目录）: ")
        if os.path.exists(test_image_path):
            if os.path.isdir(test_image_path):
                results = predict_directory(
                    os.path.join(base_dir, "best_model.pth"),
                    test_image_path,
                    train_dir
                )
            else:
                result = predict_image(
                    os.path.join(base_dir, "best_model.pth"),
                    test_image_path,
                    train_dir
                )
        else:
            print("❌ 指定的路径不存在")
    
    print("\n🎉 所有任务完成！")

if __name__ == "__main__":
    main()
