#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import warnings
import pathlib

# 关闭oneDNN提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 关闭TF Lite警告
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow.lite.python.interpreter')


class PointHistoryClassifier(object):
    def __init__(
            self,
            model_path='model/point_history_classifier/point_history_classifier.tflite',
            score_th=0.5,
            invalid_value=0,
            num_threads=1,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        if not model_path.endswith('.tflite'):
            raise ValueError(f"路径不是.tflite文件：{model_path}")

        with open(model_path, 'rb') as f:
            model_data = f.read()
        self.interpreter = tf.lite.Interpreter(model_content=model_data, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value
        print(f"✅ 模型加载成功！")
        print(f"输入张量形状：{self.input_details[0]['shape']}")
        print(f"输出张量形状：{self.output_details[0]['shape']}")

    def __call__(self, point_history):
        if not isinstance(point_history, (list, np.ndarray)):
            raise TypeError("输入必须是列表或numpy数组")

        input_data = np.array([point_history], dtype=np.float32)
        if input_data.shape != tuple(self.input_details[0]['shape']):
            raise ValueError(
                f"输入形状不匹配！模型要求：{self.input_details[0]['shape']}，实际：{input_data.shape}"
            )

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        result_squeezed = np.squeeze(result)
        result_index = np.argmax(result_squeezed)

        print(f"\n原始预测得分：{result_squeezed}")
        print(f"最高得分索引：{result_index}，得分值：{result_squeezed[result_index]}")

        if result_squeezed[result_index] < self.score_th:
            result_index = self.invalid_value
            print(f"⚠️ 得分低于阈值({self.score_th})，返回无效值：{self.invalid_value}")

        return result_index


def preprocess_point_history(point_history):
    """归一化关键点数据到0~1"""
    point_history = np.array(point_history, dtype=np.float32)
    min_val = np.min(point_history)
    max_val = np.max(point_history)
    # 避免除零
    point_history = (point_history - min_val) / (max_val - min_val + 1e-6)
    return point_history


if __name__ == "__main__":
    # 配置
    MODEL_PATH = pathlib.Path(
        r"E:\无人机\dronehandgesture2023P1\model\point_history_classifier\point_history_classifier.tflite").resolve()
    SCORE_THRESHOLD = 0.5
    # 手势映射（根据实际训练标签调整）
    gesture_mapping = {0: "无手势/降落", 1: "起飞/前进"}

    print(f"当前模型路径：{MODEL_PATH}")
    try:
        # 实例化分类器
        classifier = PointHistoryClassifier(
            model_path=str(MODEL_PATH),
            score_th=SCORE_THRESHOLD,
            num_threads=4
        )

        # 1. 测试数据（随机生成32维，匹配模型输入）
        test_point_history = np.random.rand(32).astype(np.float32)
        # 2. 预处理（替换为真实数据时注释掉随机数，启用下面两行）
        # real_point_history = [0.1,0.2,...,0.3]  # 32个真实关键点数值
        # test_point_history = preprocess_point_history(real_point_history)

        # 分类推理
        result = classifier(test_point_history)
        print(f"\n🎯 最终分类结果：{result} → {gesture_mapping.get(result, '未知手势')}")

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback

        traceback.print_exc()