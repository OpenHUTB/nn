import tensorflow as tf
from stable_baselines.common.policies import nature_cnn

def sequence_1d_cnn(scaled_images, **kwargs):
    """简化的1D序列CNN，处理时序数据"""
    activ = tf.nn.relu
    layer1 = activ(tf.layers.conv1d(scaled_images, 32, 8, strides=4,** kwargs))
    layer2 = activ(tf.layers.conv1d(layer1, 64, 4, strides=2, **kwargs))
    layer3 = activ(tf.layers.conv1d(layer2, 64, 3, strides=1,** kwargs))
    layer3 = tf.layers.flatten(layer3)
    return tf.layers.dense(layer3, 512, activation=activ, **kwargs)

def sequence_1d_cnn_ego_bypass_tc(scaled_images,** kwargs):
    """带ego车辆状态绕过（bypass）的1D CNN（可能包含时序卷积tc）"""
    # 这里简化处理，实际需根据项目需求补充ego状态融合逻辑
    cnn_features = sequence_1d_cnn(scaled_images, **kwargs)
    ego_features = tf.layers.dense(kwargs.get('ego_input'), 64, activation=tf.nn.relu)  # 假设传入ego状态
    return tf.concat([cnn_features, ego_features], axis=1)