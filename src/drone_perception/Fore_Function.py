# predict_utils.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def load_model(model_path):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path, target_size=(128, 128)):
    """预处理单张图片"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_single_image(model, img_path, class_labels, target_size=(128, 128)):
    """对单张图片进行预测"""
    # 预处理图片
    img_array = preprocess_image(img_path, target_size)
    
    # 预测
    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    confidence = np.max(pred[0])
    
    return {
        'class_label': class_labels[class_idx],
        'class_index': class_idx,
        'confidence': float(confidence),
        'all_probabilities': pred[0].tolist()
    }

def get_class_labels(train_dir):
    """从训练目录获取类别标签"""
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")
    
    class_labels = sorted([d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))])
    return class_labels

# 如果作为脚本直接运行，提供测试功能
if __name__ == "__main__":
    # 测试代码
    model_path = "../data/best_model.h5"
    test_img_path = "test_image.jpg"  # 替换为实际路径
    train_dir = "../data/train"
    
    try:
        model = load_model(model_path)
        class_labels = get_class_labels(train_dir)
        result = predict_single_image(model, test_img_path, class_labels)
        
        print(f"🔍 预测结果: {result['class_label']}")
        print(f"🎯 置信度: {result['confidence']:.4f}")
        print(f"📊 所有类别概率: {dict(zip(class_labels, result['all_probabilities']))}")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
