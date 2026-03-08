import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


class GestureClassifier:
    """机器学习手势分类器"""

    def __init__(self, model_type='svm', model_path=None):
        self.model_type = model_type
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None
        self.gesture_classes = None

        # 如果提供了模型路径，加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def create_model(self):
        """创建分类模型"""
        if self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")

    def load_dataset(self, dataset_path):
        """加载数据集"""
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['features']
        y = dataset['labels']
        self.gesture_classes = dataset['gesture_classes']

        return X, y

    def preprocess_data(self, X_train, X_test):
        """数据预处理"""
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def train(self, dataset_path, test_size=0.2, save_path=None):
        """训练模型"""
        # 加载数据
        X, y = self.load_dataset(dataset_path)

        # 检查数据
        print(f"数据集信息:")
        print(f"  总样本数: {len(X)}")
        print(f"  特征维度: {X.shape[1]}")

        unique_labels, counts = np.unique(y, return_counts=True)
        print("  各类样本数:")
        for label, count in zip(unique_labels, counts):
            class_name = self.gesture_classes[label]
            print(f"    {class_name}: {count}")

        if len(unique_labels) < 2:
            raise ValueError(f"需要至少2个类别才能训练，当前只有 {len(unique_labels)} 个类别")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 数据预处理
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)

        # 创建并训练模型
        self.create_model()
        print(f"训练 {self.model_type} 模型...")
        self.model.fit(X_train_scaled, y_train)

        # 评估模型
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)

        print(f"训练准确率: {train_acc:.4f}")
        print(f"测试准确率: {test_acc:.4f}")

        # 详细分类报告
        y_pred = self.model.predict(X_test_scaled)
        print("\n分类报告:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.gesture_classes))

        # 保存模型
        if save_path:
            self.save_model(save_path)

        return test_acc

    def predict(self, landmarks):
        """预测手势类别"""
        if self.model is None:
            raise ValueError("模型未训练或未加载")

        # 检查landmarks是否有效
        if len(landmarks) != 63:  # 21个关键点 * 3坐标
            return "none", 0.0

        # 预处理
        landmarks_scaled = self.scaler.transform([landmarks])

        # 预测
        probabilities = self.model.predict_proba(landmarks_scaled)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]

        predicted_class = self.gesture_classes[predicted_class_idx]

        return predicted_class, confidence

    def predict_batch(self, landmarks_list):
        """批量预测"""
        if not landmarks_list:
            return [], []

        # 预处理
        landmarks_array = np.array(landmarks_list)
        landmarks_scaled = self.scaler.transform(landmarks_array)

        # 预测
        probabilities = self.model.predict_proba(landmarks_scaled)
        predicted_indices = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        predicted_classes = [self.gesture_classes[idx] for idx in predicted_indices]

        return predicted_classes, confidences

    def save_model(self, save_path):
        """保存模型"""
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'gesture_classes': self.gesture_classes
        }

        joblib.dump(model_data, save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, model_path):
        """加载模型"""
        model_data = joblib.load(model_path)

        self.model_type = model_data['model_type']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.gesture_classes = model_data['gesture_classes']

        print(f"模型已从 {model_path} 加载")
        print(f"模型类型: {self.model_type}")
        print(f"手势类别: {self.gesture_classes}")


class EnsembleGestureClassifier:
    """集成学习手势分类器"""

    def __init__(self, model_paths=None):
        self.models = []
        self.gesture_classes = None

        if model_paths:
            self.load_models(model_paths)

    def load_models(self, model_paths):
        """加载多个模型"""
        for path in model_paths:
            classifier = GestureClassifier(model_path=path)
            self.models.append(classifier)

        # 使用第一个模型的类别
        if self.models:
            self.gesture_classes = self.models[0].gesture_classes

    def predict(self, landmarks, voting='soft'):
        """集成预测"""
        if not self.models:
            raise ValueError("没有加载任何模型")

        if len(landmarks) != 63:
            return "none", 0.0

        predictions = []
        confidences = []

        for model in self.models:
            pred_class, confidence = model.predict(landmarks)
            predictions.append(pred_class)
            confidences.append(confidence)

        if voting == 'hard':
            # 硬投票
            from collections import Counter
            most_common = Counter(predictions).most_common(1)[0]
            final_prediction = most_common[0]
            final_confidence = most_common[1] / len(predictions)
        else:
            # 软投票（平均置信度）
            class_confidences = {}
            for model, pred, conf in zip(self.models, predictions, confidences):
                if pred not in class_confidences:
                    class_confidences[pred] = []
                class_confidences[pred].append(conf)

            # 平均每个类别的置信度
            avg_confidences = {
                cls: np.mean(confs) for cls, confs in class_confidences.items()
            }

            # 选择平均置信度最高的类别
            final_prediction = max(avg_confidences.items(), key=lambda x: x[1])[0]
            final_confidence = avg_confidences[final_prediction]

        return final_prediction, final_confidence