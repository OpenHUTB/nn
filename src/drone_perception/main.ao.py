import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_trained_model():
   
    加载训练好的图像分类模型并在测试集上进行评估
  
    #  路径配置 
    base_dir = os.path.abspath("./saved_models")
    test_dir = os.path.join(base_dir, "test_dataset")
    
    # 图像参数（需与训练时保持一致）
    img_height, img_width = 150, 150
    batch_size = 32
    
    #  数据预处理 
    print("正在加载和预处理测试数据...")
    
    # 创建测试数据生成器
    test_datagen = ImageDataGenerator(
        rescale=1./255  # 像素值归一化
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,  # 保持数据顺序以对齐预测结果
        seed=42
    )
    
    print(f"找到 {test_generator.samples} 张测试图像")
    print(f"类别: {list(test_generator.class_indices.keys())}")
    
    # 加载预训练模型 
    print("\n正在加载预训练模型...")
    model_path = os.path.join(base_dir, "trained_model.h5")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print("模型加载成功!")
    
    #  模型评估 
    print("\n正在评估模型性能...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy*100:.2f}%")
    
    # 详细预测分析
    print("\n正在进行详细预测分析...")
    
    # 获取预测结果
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # 获取类别标签
    class_labels = list(test_generator.class_indices.keys())
    
    #  分类报告 
    print("\n" + "="*50)
    print("详细分类报告")
    print("="*50)
    print(classification_report(true_classes, predicted_classes, 
                              target_names=class_labels, digits=4))
    
    # 混淆矩阵 
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('模型混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    #  额外统计分析 
    print("\n" + "="*50)
    print("额外统计信息")
    print("="*50)
    
    # 计算每个类别的准确率
    class_accuracy = {}
    for i, class_name in enumerate(class_labels):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == i)
            class_accuracy[class_name] = class_acc
            print(f"{class_name:15s}: {class_acc:.4f} ({np.sum(class_mask)} 个样本)")
    
    # 识别最难分类的类别
    hardest_class = min(class_accuracy, key=class_accuracy.get)
    easiest_class = max(class_accuracy, key=class_accuracy.get)
    
    print(f"\n最难分类的类别: {hardest_class} (准确率: {class_accuracy[hardest_class]:.4f})")
    print(f"最易分类的类别: {easiest_class} (准确率: {class_accuracy[easiest_class]:.4f})")
    
    #  预测置信度分析 
    print("\n预测置信度分析:")
    confidence_scores = np.max(predictions, axis=1)
    print(f"平均预测置信度: {np.mean(confidence_scores):.4f}")
    print(f"预测置信度标准差: {np.std(confidence_scores):.4f}")
    print(f"最低预测置信度: {np.min(confidence_scores):.4f}")
    print(f"最高预测置信度: {np.max(confidence_scores):.4f}")
    
    # 低置信度预测比例
    low_confidence_threshold = 0.6
    low_confidence_ratio = np.sum(confidence_scores < low_confidence_threshold) / len(confidence_scores)
    print(f"低置信度预测比例 (<{low_confidence_threshold}): {low_confidence_ratio:.4f}")

def main():
    """主函数"""
    print("开始图像分类模型评估流程")
    print("="*60)
    
    try:
        evaluate_trained_model()
        print("\n" + "="*60)
        print("模型评估完成!")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        print("请检查: ")
        print("1. 模型文件路径是否正确")
        print("2. 测试数据目录结构是否正确")
        print("3. 图像尺寸是否与训练时一致")

if __name__ == "__main__":

    main()
