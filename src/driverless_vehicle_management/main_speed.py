# 导入必要的库
import torch  # PyTorch核心库，用于构建和训练神经网络
import torch.nn as nn  # PyTorch神经网络模块，包含层、损失函数等
import torch.optim as optim  # PyTorch优化器模块
import numpy as np  # 数值计算库，用于数据生成和处理
import pandas as pd  # 数据处理库（本示例未直接使用，保留用于扩展真实数据读取）
from sklearn.model_selection import train_test_split  # 划分训练集/测试集
from sklearn.preprocessing import StandardScaler  # 数据标准化预处理
import matplotlib.pyplot as plt  # 绘图库，用于结果可视化

# 设置随机种子，保证实验结果可复现（固定随机数生成器的初始状态）
torch.manual_seed(42)  # PyTorch随机种子
np.random.seed(42)  # NumPy随机种子

class SpeedPredictor(nn.Module):
    """
    无人车速度预测模型：基于LSTM（长短期记忆网络）的时序预测模型
    LSTM适合处理时序数据，能捕捉时间序列中的长期依赖关系，适合速度这类连续变化的时序预测任务
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        """
        初始化模型参数
        Args:
            input_size (int): 输入特征维度（每个时间步的特征数量，如速度、加速度等）
            hidden_size (int): LSTM隐藏层神经元数量（控制模型容量）
            num_layers (int): LSTM网络的层数（多层LSTM可捕捉更复杂的时序特征）
            output_size (int): 输出维度（默认1，预测未来1个时间步的速度）
        """
        super(SpeedPredictor, self).__init__()  # 调用父类nn.Module的初始化方法

        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,  # 输入特征数
            hidden_size=hidden_size,  # 隐藏层神经元数
            num_layers=num_layers,  # LSTM层数
            batch_first=True  # 输入数据格式为(batch_size, seq_len, input_size)，便于批量处理
        )

        # 定义全连接层：将LSTM的隐藏层输出映射到最终预测结果
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播过程（模型的核心计算逻辑）
        Args:
            x (torch.Tensor): 输入数据，形状为(batch_size, seq_len, input_size)
                batch_size: 批次大小（每次训练的样本数）
                seq_len: 输入序列长度（用过去多少个时间步的数据进行预测）
                input_size: 每个时间步的特征数
        Returns:
            torch.Tensor: 预测结果，形状为(batch_size, output_size)
        """
        # LSTM层前向传播：输出包含所有时间步的隐藏状态，以及最后一个时间步的隐藏态和细胞态
        # lstm_out形状：(batch_size, seq_len, hidden_size)，_表示忽略的隐藏态和细胞态
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的隐藏状态作为全连接层的输入（因为要预测下一个时间步的速度）
        last_out = lstm_out[:, -1, :]  # 切片操作，形状变为(batch_size, hidden_size)

        # 全连接层映射：将隐藏层特征转换为最终预测值
        pred = self.fc(last_out)  # 输出形状：(batch_size, output_size)

        return pred


def create_sequences(data, seq_len, pred_len=1):
    """
    将原始时序数据转换为LSTM模型所需的输入序列（X）和标签（y）
    核心逻辑：用过去seq_len个时间步的特征数据，预测未来pred_len个时间步的速度
    Args:
        data (np.ndarray): 原始特征矩阵，形状为(total_samples, input_size)
            total_samples: 总样本数（时间步数量）
            input_size: 特征数
        seq_len (int): 输入序列长度（过去多少个时间步）
        pred_len (int): 预测长度（未来多少个时间步，默认1，即单步预测）
    Returns:
        tuple: (X, y)，均为NumPy数组
            X: 输入序列，形状为(num_sequences, seq_len, input_size)
            y: 标签（待预测的速度），形状为(num_sequences, pred_len)
    """
    xs, ys = [], []  # 存储输入序列和对应的标签

    # 遍历原始数据，生成所有可能的序列（避免索引越界）
    # 循环终止条件：i + seq_len + pred_len - 1 < len(data)
    for i in range(len(data) - seq_len - pred_len + 1):
        # 输入序列：从i开始，取seq_len个连续时间步的特征数据
        x = data[i:(i + seq_len)]  # 形状：(seq_len, input_size)
        # 标签：从i+seq_len开始，取pred_len个时间步的速度（假设第0列是速度特征）
        y = data[i + seq_len:i + seq_len + pred_len, 0]  # 形状：(pred_len,)

        xs.append(x)
        ys.append(y)

    # 转换为NumPy数组返回（便于后续转换为PyTorch张量）
    return np.array(xs), np.array(ys)


def main():
    """主函数：完整的模型训练和评估流程"""
    # ===================== 1. 数据准备 =====================
    # 说明：本示例使用模拟数据验证模型，实际应用中需替换为真实传感器数据（如CSV文件读取）
    # 模拟的无人车特征包括：[速度(m/s), 加速度(m/s²), 方向盘角度(°), 油门开度(%), 刹车压力(bar)]
    num_samples = 10000  # 总时间步数量（模拟10000个连续的传感器采样点）
    time = np.linspace(0, 100, num_samples)  # 时间轴（0到100秒，均匀分布10000个点）

    # 模拟各特征数据（添加随机噪声模拟真实传感器误差）
    speed = 10 + 5 * np.sin(time) + np.random.normal(0, 0.5, num_samples)  # 基础速度10m/s，叠加正弦波动和噪声
    acceleration = np.gradient(speed, time)  # 加速度：速度对时间的导数
    steering = 10 * np.sin(time / 2) + np.random.normal(0, 1, num_samples)  # 方向盘角度：慢波动+噪声
    throttle = 30 + 10 * np.sin(time / 3) + np.random.normal(0, 2, num_samples)  # 油门开度：30%左右波动
    brake = np.where(
        speed < 8,  # 当速度低于8m/s时，刹车压力增大
        5 + np.random.normal(0, 1, num_samples),  # 低速时刹车压力（5bar左右）
        np.random.normal(0, 0.5, num_samples)  # 高速时刹车压力（接近0）
    )

    # 组合所有特征为特征矩阵（形状：(num_samples, 5)）
    data = np.column_stack([speed, acceleration, steering, throttle, brake])

    # ===================== 2. 数据预处理 =====================
    # 标准化：将所有特征转换为均值=0、方差=1的分布，避免不同量纲特征影响模型训练
    scaler = StandardScaler()  # 初始化标准化器
    data_scaled = scaler.fit_transform(data)  # 拟合训练数据并转换（注意：仅用训练数据拟合，避免数据泄露）

    # 生成LSTM输入序列和标签：用过去10个时间步预测未来1个时间步的速度
    seq_len = 10  # 输入序列长度（可根据数据特性调整，如5、15等）
    X, y = create_sequences(data_scaled, seq_len)  # X形状：(num_sequences, 10, 5)，y形状：(num_sequences, 1)

    # 划分训练集和测试集：时序数据禁止shuffle（保持时间顺序），测试集占比20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 转换为PyTorch张量（模型仅支持张量输入）
    X_train = torch.FloatTensor(X_train)  # 训练集输入：(train_samples, 10, 5)
    y_train = torch.FloatTensor(y_train)  # 训练集标签：(train_samples, 1)
    X_test = torch.FloatTensor(X_test)  # 测试集输入：(test_samples, 10, 5)
    y_test = torch.FloatTensor(y_test)  # 测试集标签：(test_samples, 1)

    # ===================== 3. 模型初始化 =====================
    input_size = X_train.shape[2]  # 输入特征数：5（速度、加速度、方向盘、油门、刹车）
    hidden_size = 64  # LSTM隐藏层神经元数（超参数，可调整：32、64、128等）
    num_layers = 2  # LSTM层数（超参数，1或2层较常用，过多易过拟合）
    model = SpeedPredictor(input_size, hidden_size, num_layers)  # 实例化模型

    # ===================== 4. 定义训练组件 =====================
    criterion = nn.MSELoss()  # 损失函数：均方误差（MSE），适用于回归任务（预测连续值）
    optimizer = optim.Adam(  # 优化器：Adam（自适应学习率，训练稳定性好）
        model.parameters(),  # 待优化的模型参数
        lr=0.001  # 学习率（超参数，可调整：0.0001、0.001、0.01等）
    )

    # ===================== 5. 模型训练与验证 =====================
    epochs = 50  # 训练轮数（整个训练集遍历50次）
    batch_size = 32  # 批次大小（每次训练用32个样本更新参数，平衡速度和稳定性）
    train_losses = []  # 记录每轮训练损失（用于后续可视化）
    test_losses = []  # 记录每轮测试损失（用于后续可视化）

    for epoch in range(epochs):
        # 训练模式：启用Dropout（本模型无Dropout，但保留规范）、BatchNorm更新
        model.train()
        epoch_loss = 0  # 累计当前轮次的总损失

        # 分批训练（避免一次性加载所有数据导致内存溢出）
        for i in range(0, len(X_train), batch_size):
            # 截取当前批次的训练数据
            batch_X = X_train[i:i + batch_size]  # 批次输入：(batch_size, 10, 5)
            batch_y = y_train[i:i + batch_size]  # 批次标签：(batch_size, 1)

            optimizer.zero_grad()  # 清零梯度（避免梯度累积）
            outputs = model(batch_X)  # 模型前向传播：计算预测值
            loss = criterion(outputs, batch_y)  # 计算当前批次的损失
            loss.backward()  # 反向传播：计算梯度
            optimizer.step()  # 梯度下降：更新模型参数

            # 累计损失（乘以批次大小，最后求平均）
            epoch_loss += loss.item() * batch_X.size(0)

        # 计算当前轮次的平均训练损失
        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)

        # 测试集验证（不更新模型参数，仅评估性能）
        model.eval()  # 评估模式：禁用Dropout、BatchNorm固定
        with torch.no_grad():  # 禁用梯度计算（节省内存，加速计算）
            y_pred = model(X_test)  # 测试集预测
            test_loss = criterion(y_pred, y_test).item()  # 测试集损失
            test_losses.append(test_loss)

        # 每5轮打印一次训练/测试损失（监控训练进度）
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

    # ===================== 6. 结果可视化 =====================
    # 反标准化：将标准化后的预测值和真实值转换为原始速度尺度（便于直观理解）
    # 说明：StandardScaler需要完整的特征向量才能反标准化，因此构造虚拟矩阵仅恢复速度列
    dummy = np.zeros_like(data_scaled[:len(y_test)])  # 虚拟矩阵：(test_samples, 5)，全0
    dummy[:, 0] = y_test.numpy().flatten()  # 仅在速度列（第0列）填入标准化后的真实速度
    y_test_original = scaler.inverse_transform(dummy)[:, 0]  # 反标准化得到原始尺度的真实速度

    # 对预测值执行同样的反标准化操作
    dummy_pred = np.zeros_like(data_scaled[:len(y_test)])
    dummy_pred[:, 0] = y_pred.numpy().flatten()
    y_pred_original = scaler.inverse_transform(dummy_pred)[:, 0]

    # 绘制预测值 vs 真实值曲线（直观对比预测效果）
    plt.figure(figsize=(12, 6))  # 设置图大小
    plt.plot(y_test_original, label='真实速度', alpha=0.7)  # 真实速度曲线（透明度0.7避免重叠）
    plt.plot(y_pred_original, label='预测速度', alpha=0.7)  # 预测速度曲线
    plt.xlabel('时间步')  # x轴标签
    plt.ylabel('速度 (m/s)')  # y轴标签（原始尺度）
    plt.title('无人车速度预测结果')  # 图标题
    plt.legend()  # 显示图例
    plt.show()  # 显示图像

    # 绘制训练/测试损失曲线（监控模型收敛情况，判断是否过拟合）
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='训练损失')  # 训练损失趋势
    plt.plot(test_losses, label='测试损失')  # 测试损失趋势
    plt.xlabel('Epoch')  # x轴：训练轮次
    plt.ylabel('MSE损失')  # y轴：均方误差
    plt.title('训练与测试损失曲线')
    plt.legend()
    plt.show()


# 程序入口：当脚本直接运行时，执行main函数
if __name__ == '__main__':
    main()