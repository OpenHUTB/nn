import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time

# -------------------------- 1. 配置参数 --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU/CPU
SEQ_LENGTH = 10  # 输入序列长度（用前10个时刻速度预测下1个）
PREDICT_STEP = 1  # 预测步长（预测未来1个时刻）
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
SPEED_THRESHOLD_MAX = 60.0  # 最大允许速度（km/h）
SPEED_THRESHOLD_MIN = 5.0  # 最小允许速度（km/h）
ERROR_THRESHOLD = 5.0  # 预测误差阈值（超过则认为异常）


# -------------------------- 2. 数据生成与预处理 --------------------------
def generate_speed_data(n_samples=1000, noise=0.5):
    """生成无人车速度时序数据（含正常行驶+随机异常）"""
    time_steps = np.arange(n_samples)
    # 正常速度：模拟加速-匀速-减速过程
    normal_speed = np.zeros(n_samples)
    normal_speed[:200] = np.linspace(0, 50, 200)  # 加速
    normal_speed[200:700] = 50 + np.random.normal(0, noise, 500)  # 匀速（含噪声）
    normal_speed[700:] = np.linspace(50, 10, 300)  # 减速

    # 添加异常数据（超速/低速）
    abnormal_indices = np.random.choice(n_samples, size=50, replace=False)  # 50个异常点
    for idx in abnormal_indices:
        if np.random.random() > 0.5:
            normal_speed[idx] = np.random.uniform(65, 80)  # 超速
        else:
            normal_speed[idx] = np.random.uniform(0, 3)  # 低速异常

    # 封装为DataFrame
    data = pd.DataFrame({
        "time": pd.date_range(start="2025-01-01", periods=n_samples, freq="1s"),
        "speed": normal_speed
    })
    return data


def create_sequences(data, seq_len, predict_step):
    """将时序数据转换为输入序列（X）和标签（y）"""
    X, y = [], []
    for i in range(len(data) - seq_len - predict_step + 1):
        seq = data[i:i + seq_len]  # 输入序列：前seq_len个数据
        label = data[i + seq_len:i + seq_len + predict_step]  # 标签：后predict_step个数据
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)


# 生成并预处理数据
raw_data = generate_speed_data(n_samples=1000)
scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化
scaled_speed = scaler.fit_transform(raw_data[["speed"]].values)

# 划分训练集（80%）和测试集（20%）
train_size = int(0.8 * len(scaled_speed))
train_data = scaled_speed[:train_size]
test_data = scaled_speed[train_size:]

# 创建序列数据
X_train, y_train = create_sequences(train_data, SEQ_LENGTH, PREDICT_STEP)
X_test, y_test = create_sequences(test_data, SEQ_LENGTH, PREDICT_STEP)

# 转换为PyTorch张量（[batch_size, seq_len, input_dim]）
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# 创建DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------- 3. LSTM预测模型 --------------------------
class SpeedPredictionLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(SpeedPredictionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2  # batch_first=True：输入格式[batch, seq, dim]
        )
        # 全连接层（输出预测值）
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_dim]
        # 取最后一个时间步的输出作为全连接层输入
        out = self.fc(lstm_out[:, -1, :])  # [batch_size, output_dim]
        return out


# 初始化模型
model = SpeedPredictionLSTM(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1).to(DEVICE)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam优化器


# -------------------------- 4. 模型训练 --------------------------
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()  # 梯度清零
            outputs = model(batch_x)  # 前向传播
            loss = criterion(outputs, batch_y.squeeze())  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.6f}")
    return train_losses


# 开始训练
print("开始训练模型...")
train_losses = train_model(model, train_loader, criterion, optimizer, EPOCHS)

# 绘制训练损失曲线
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Model Training Loss")
plt.legend()
plt.show()


# -------------------------- 5. 速度预测与异常检测 --------------------------
def predict_and_detect(model, data, seq_len, scaler):
    """预测速度并检测异常"""
    model.eval()
    predictions = []
    anomalies = []  # 存储异常信息：(时间索引, 实际速度, 预测速度, 异常类型)

    with torch.no_grad():
        for i in range(len(data) - seq_len):
            # 构建输入序列
            seq = data[i:i + seq_len].reshape(1, seq_len, 1)  # [1, seq_len, 1]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).to(DEVICE)

            # 预测
            pred_scaled = model(seq_tensor)
            pred = scaler.inverse_transform(pred_scaled.cpu().numpy())[0][0]  # 反归一化
            predictions.append(pred)

            # 获取实际速度（预测步长为1，实际速度是序列后一个时刻）
            actual = scaler.inverse_transform(data[i + seq_len].reshape(1, 1))[0][0]
            time_idx = i + seq_len  # 实际速度对应的时间索引

            # 计算预测误差
            error = abs(actual - pred)

            # 异常检测逻辑
            if actual > SPEED_THRESHOLD_MAX:
                anomalies.append((time_idx, actual, pred, "超速"))
            elif actual < SPEED_THRESHOLD_MIN:
                anomalies.append((time_idx, actual, pred, "低速"))
            elif error > ERROR_THRESHOLD:
                anomalies.append((time_idx, actual, pred, "预测异常（误差过大）"))

    return predictions, anomalies


# 对测试集进行预测和异常检测
test_predictions, test_anomalies = predict_and_detect(
    model, test_data, SEQ_LENGTH, scaler
)

# 处理测试集实际速度（对齐预测结果长度）
test_actual = scaler.inverse_transform(test_data[SEQ_LENGTH:SEQ_LENGTH + len(test_predictions)]).flatten()


# -------------------------- 6. 报警模块 --------------------------
def trigger_alarm(anomaly_info):
    """触发报警（控制台模拟+时间记录）"""
    time_idx, actual_speed, pred_speed, anomaly_type = anomaly_info
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alarm_msg = (
        f"\n【{current_time}】⚠️  速度异常报警！"
        f"\n- 时间索引：{time_idx}"
        f"\n- 实际速度：{actual_speed:.2f} km/h"
        f"\n- 预测速度：{pred_speed:.2f} km/h"
        f"\n- 异常类型：{anomaly_type}"
        f"\n----------------------------------------"
    )
    print(alarm_msg)

    # 可选：调用硬件报警（如蜂鸣器、LED灯），需根据硬件接口修改
    # import RPi.GPIO as GPIO
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(18, GPIO.OUT)
    # GPIO.output(18, GPIO.HIGH)  # 蜂鸣器响
    # time.sleep(0.5)
    # GPIO.output(18, GPIO.LOW)

# 输出所有异常报警
print("\n========== 异常检测结果 ==========")
if test_anomalies:
    for anomaly in test_anomalies:
        trigger_alarm(anomaly)
else:
    print("✅ 未检测到速度异常！")


# -------------------------- 7. 结果可视化 --------------------------
def plot_results(test_actual, test_predictions, test_anomalies, seq_len):
    """绘制实际速度、预测速度和异常点"""
    plt.figure(figsize=(15, 6))
    # 绘制实际速度和预测速度
    plt.plot(test_actual, label="实际速度", color="blue", linewidth=1.5)
    plt.plot(test_predictions, label="预测速度", color="orange", linewidth=1.5, linestyle="--")
    # 绘制速度阈值线
    plt.axhline(y=SPEED_THRESHOLD_MAX, color="red", linestyle=":", label=f"最大阈值 {SPEED_THRESHOLD_MAX} km/h")
    plt.axhline(y=SPEED_THRESHOLD_MIN, color="green", linestyle=":", label=f"最小阈值 {SPEED_THRESHOLD_MIN} km/h")
    # 标记异常点
    if test_anomalies:
        anomaly_indices = [a[0] - seq_len for a in test_anomalies]  # 对齐测试集索引
        anomaly_speeds = [a[1] for a in test_anomalies]
        plt.scatter(anomaly_indices, anomaly_speeds, color="red", s=50, label="异常点", zorder=5)
    # 图表配置
    plt.xlabel("时间步（1步=1秒）")
    plt.ylabel("速度（km/h）")
    plt.title("无人车速度预测与异常检测结果")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 绘制结果
plot_results(test_actual, test_predictions, test_anomalies, SEQ_LENGTH)