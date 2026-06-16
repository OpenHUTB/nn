import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

class GestureDataset(Dataset):
    """手势数据集类"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DeepGestureModel(nn.Module):
    """深度学习手势识别模型"""
    def __init__(self, input_size, num_classes, model_type='cnn'):
        super(DeepGestureModel, self).__init__()
        self.model_type = model_type
        
        if model_type == 'cnn':
            # CNN模型：将关键点视为3x21的网格
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(128 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif model_type == 'transformer':
            # Transformer模型
            self.embedding = nn.Linear(input_size, 128)
            self.pos_encoding = self._generate_positional_encoding(21, 128)
            encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.fc = nn.Linear(128, num_classes)
        elif model_type == 'mlp':
            # 深度MLP模型
            self.layers = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
    
    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        if self.model_type == 'cnn':
            x = x.unsqueeze(1)  # 添加通道维度
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
        elif self.model_type == 'transformer':
            x = x.view(x.size(0), 21, 3)  # (batch, 21 points, 3 coords)
            x = self.embedding(x)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
            x = self.transformer_encoder(x)
            x = x.mean(dim=0)  # 全局平均池化
            x = self.fc(x)
        elif self.model_type == 'mlp':
            x = self.layers(x)
        return x

class DeepGestureClassifier:
    """深度学习手势分类器"""
    
    def __init__(self, model_type='cnn', model_path=None):
        self.model_type = model_type
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None
        self.gesture_classes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 0
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_dataset(self, dataset_path):
        """加载数据集"""
        with open(dataset_path, 'rb') as f:
            dataset = joblib.load(f)
        
        X = dataset['features']
        y = dataset['labels']
        self.gesture_classes = dataset['gesture_classes']
        self.num_classes = len(self.gesture_classes)
        
        return X, y
    
    def preprocess_data(self, X_train, X_test):
        """数据预处理"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def train(self, dataset_path, epochs=100, batch_size=32, lr=0.001, test_size=0.2, save_path=None):
        """训练深度学习模型"""
        X, y = self.load_dataset(dataset_path)
        
        print(f"数据集信息:")
        print(f"  总样本数: {len(X)}")
        print(f"  特征维度: {X.shape[1]}")
        
        unique_labels, counts = np.unique(y, return_counts=True)
        print("  各类样本数:")
        for label, count in zip(unique_labels, counts):
            class_name = self.gesture_classes[label]
            print(f"    {class_name}: {count}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
        
        train_dataset = GestureDataset(X_train_scaled, y_train)
        test_dataset = GestureDataset(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.model = DeepGestureModel(X.shape[1], self.num_classes, self.model_type).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        print(f"\n训练 {self.model_type} 深度学习模型...")
        print(f"设备: {self.device}")
        
        train_losses = []
        train_accs = []
        test_accs = []
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            scheduler.step()
            
            # 测试
            self.model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = test_correct / test_total
            test_accs.append(test_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        print(f"\n训练完成!")
        print(f"最终训练准确率: {train_accs[-1]:.4f}")
        print(f"最终测试准确率: {test_accs[-1]:.4f}")
        
        # 详细分类报告
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, target_names=self.gesture_classes))
        
        if save_path:
            self.save_model(save_path)
        
        return test_accs[-1], train_losses, train_accs, test_accs
    
    def predict(self, landmarks):
        """预测手势类别"""
        if self.model is None:
            raise ValueError("模型未训练或未加载")
        
        if len(landmarks) != 63:
            return "none", 0.0
        
        landmarks_scaled = self.scaler.transform([landmarks])
        landmarks_tensor = torch.FloatTensor(landmarks_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(landmarks_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        predicted_class = self.gesture_classes[predicted_class_idx]
        return predicted_class, confidence
    
    def predict_batch(self, landmarks_list):
        """批量预测"""
        if not landmarks_list:
            return [], []
        
        landmarks_array = np.array(landmarks_list)
        landmarks_scaled = self.scaler.transform(landmarks_array)
        landmarks_tensor = torch.FloatTensor(landmarks_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(landmarks_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_indices = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
        
        predicted_classes = [self.gesture_classes[idx] for idx in predicted_indices]
        return predicted_classes, confidences
    
    def save_model(self, save_path):
        """保存模型"""
        model_data = {
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'gesture_classes': self.gesture_classes,
            'num_classes': self.num_classes
        }
        torch.save(model_data, save_path)
        print(f"深度学习模型已保存到: {save_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        model_data = torch.load(model_path, map_location=self.device)
        
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.gesture_classes = model_data['gesture_classes']
        self.num_classes = model_data['num_classes']
        
        self.model = DeepGestureModel(63, self.num_classes, self.model_type).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"深度学习模型已从 {model_path} 加载")
        print(f"模型类型: {self.model_type}")
        print(f"手势类别: {self.gesture_classes}")