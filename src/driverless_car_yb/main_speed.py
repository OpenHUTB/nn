import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


class SpeedPredictor(nn.Module):
    """无人车速度预测模型：使用LSTM处理时序数据"""
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(SpeedPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,    # 输入特征数（如速度、加速度、方向盘角度等）
            hidden_size=hidden_size,  # LSTM隐藏层大小
            num_layers=num_layers,    # LSTM层数
            batch_first=True          # 输入格式为(batch, seq_len, input_size)
        )
        self.fc = nn.Linear(hidden_size, output_size)  # 输出层（预测未来速度）

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        # 取最后一个时间步的输出用于预测
        last_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)
        pred = self.fc(last_out)       # shape: (batch_size, output_size)
        return pred


def create_sequences(data, seq_len, pred_len=1):
    """
    将时序数据转换为输入序列和标签
    data: 原始数据（特征矩阵）
    seq_len: 输入序列长度（用过去多少个时间步预测未来）
    pred_len: 预测未来多少个时间步（这里简化为1）
    """
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:(i + seq_len)]  # 输入序列（过去seq_len个时间步的特征）
        y = data[i + seq_len:i + seq_len + pred_len, 0]  # 标签（未来1个时间步的速度，假设第0列是速度）
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def main():
    # 1. 数据准备（这里使用模拟数据，实际应用中替换为真实传感器数据）
    # 模拟特征：[速度(m/s), 加速度(m/s²), 方向盘角度(°), 油门开度(%), 刹车压力(bar)]
    num_samples = 10000
    time = np.linspace(0, 100, num_samples)
    speed = 10 + 5 * np.sin(time) + np.random.normal(0, 0.5, num_samples)  # 带噪声的正弦曲线模拟速度
    acceleration = np.gradient(speed, time)  # 加速度（速度的导数）
    steering = 10 * np.sin(time/2) + np.random.normal(0, 1, num_samples)  # 方向盘角度
    throttle = 30 + 10 * np.sin(time/3) + np.random.normal(0, 2, num_samples)  # 油门开度
    brake = np.where(speed < 8, 5 + np.random.normal(0, 1, num_samples), np.random.normal(0, 0.5, num_samples))  # 刹车压力

    # 组合成特征矩阵
    data = np.column_stack([speed, acceleration, steering, throttle, brake])

    # 2. 数据预处理
    scaler = StandardScaler()  # 标准化（均值为0，方差为1）
    data_scaled = scaler.fit_transform(data)

    # 创建序列数据（用过去10个时间步预测未来1个时间步的速度）
    seq_len = 10
    X, y = create_sequences(data_scaled, seq_len)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # 时序数据不打乱顺序

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # 3. 模型初始化
    input_size = X_train.shape[2]  # 特征数（这里是5）
    hidden_size = 64
    num_layers = 2
    model = SpeedPredictor(input_size, hidden_size, num_layers)

    # 4. 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失（回归任务）
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 模型训练
    epochs = 50
    batch_size = 32
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()  # 训练模式
        epoch_loss = 0
        # 分批训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()  # 清零梯度
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            epoch_loss += loss.item() * batch_X.size(0)  # 累计损失

        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)

        # 测试集验证
        model.eval()  # 评估模式
        with torch.no_grad():
            y_pred = model(X_test)
            test_loss = criterion(y_pred, y_test).item()
            test_losses.append(test_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

    # 6. 结果可视化
    # 反标准化（将预测结果转换为原始速度尺度）
    # 构造反标准化需要的虚拟特征矩阵（仅用于恢复速度的尺度）
    dummy = np.zeros_like(data_scaled[:len(y_test)])
    dummy[:, 0] = y_test.numpy().flatten()  # 测试集真实速度（标准化后）
    y_test_original = scaler.inverse_transform(dummy)[:, 0]  # 原始尺度真实速度

    dummy_pred = np.zeros_like(data_scaled[:len(y_test)])
    dummy_pred[:, 0] = y_pred.numpy().flatten()  # 预测速度（标准化后）
    y_pred_original = scaler.inverse_transform(dummy_pred)[:, 0]  # 原始尺度预测速度

    # 绘制预测 vs 真实值
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='真实速度', alpha=0.7)
    plt.plot(y_pred_original, label='预测速度', alpha=0.7)
    plt.xlabel('时间步')
    plt.ylabel('速度 (m/s)')
    plt.title('无人车速度预测结果')
    plt.legend()
    plt.show()

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.title('训练与测试损失曲线')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()