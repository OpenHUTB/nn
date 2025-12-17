import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 1. 生成模拟驾驶数据
def generate_driving_data(n_samples=10000):
    time = np.linspace(0, 100, n_samples)
    speed = np.zeros(n_samples)
    acceleration = np.zeros(n_samples)
    steering_angle = np.zeros(n_samples)

    # 模拟驾驶场景：起步→加速→匀速→减速→转弯→加速→停车
    segment1 = slice(0, 1500)
    segment2 = slice(1500, 4000)
    segment3 = slice(4000, 5000)
    segment4 = slice(5000, 6500)
    segment5 = slice(6500, 8500)
    segment6 = slice(8500, 10000)

    # 计算各段长度
    len1 = segment1.stop - segment1.start
    len2 = segment2.stop - segment2.start
    len3 = segment3.stop - segment3.start
    len4 = segment4.stop - segment4.start
    len5 = segment5.stop - segment5.start
    len6 = segment6.stop - segment6.start

    # 填充速度数据
    speed[segment1] = np.linspace(0, 15, len1) + np.random.normal(0, 0.3, len1)
    speed[segment2] = 15 + np.random.normal(0, 0.2, len2)
    speed[segment3] = np.linspace(15, 5, len3) + np.random.normal(0, 0.3, len3)
    speed[segment4] = 5 + np.random.normal(0, 0.2, len4)
    speed[segment5] = np.linspace(5, 12, len5) + np.random.normal(0, 0.3, len5)
    speed[segment6] = np.linspace(12, 0, len6) + np.random.normal(0, 0.3, len6)

    # 填充加速度数据
    acceleration[segment1] = 0.5 + np.random.normal(0, 0.1, len1)
    acceleration[segment2] = 0 + np.random.normal(0, 0.05, len2)
    acceleration[segment3] = -0.3 + np.random.normal(0, 0.1, len3)
    acceleration[segment4] = 0 + np.random.normal(0, 0.05, len4)
    acceleration[segment5] = 0.2 + np.random.normal(0, 0.1, len5)
    acceleration[segment6] = -0.4 + np.random.normal(0, 0.1, len6)

    # 填充转向角数据
    steering_angle[segment1] = 0 + np.random.normal(0, 0.5, len1)
    steering_angle[segment2] = 0 + np.random.normal(0, 0.5, len2)
    steering_angle[segment3] = 0 + np.random.normal(0, 0.5, len3)
    steering_angle[segment4] = np.linspace(0, 15, len4) + np.random.normal(0, 1, len4)
    steering_angle[segment5] = 0 + np.random.normal(0, 0.5, len5)
    steering_angle[segment6] = 0 + np.random.normal(0, 0.5, len6)

    # 组合成DataFrame
    data = pd.DataFrame({
        'time': time,
        'speed': speed,
        'acceleration': acceleration,
        'steering_angle': steering_angle
    })
    return data


# 2. 数据预处理：构建时间序列
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # 预测speed（第0列）
    return np.array(X), np.array(y)


# 主流程
if __name__ == "__main__":
    # 生成数据
    df = generate_driving_data()
    features = df[['speed', 'acceleration', 'steering_angle']].values

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # 构建时间序列
    seq_length = 50
    X, y = create_sequences(scaled_features, seq_length)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. 构建LSTM模型
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 早停机制
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_split=0.1,
        callbacks=[early_stopping]
    )

    # 4. 模型评估
    y_pred = model.predict(X_test)

    # 反归一化（仅针对speed）
    # 构造反归一化的临时数组
    y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]
    y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), 2)))))[:, 0]

    # 计算指标
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"测试集MSE: {mse:.4f}")
    print(f"测试集MAE: {mae:.4f}")
    print(f"测试集R2: {r2:.4f}")

    # 5. 结果可视化
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='真实速度')
    plt.plot(y_pred_inv, label='预测速度')
    plt.title('无人车速度预测结果')
    plt.xlabel('时间步')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.show()#预测