# 导入必要的库（按功能分类，便于查找）
import torch  # PyTorch核心库：提供张量计算和自动微分，是神经网络构建基础
import torch.nn as nn  # PyTorch神经网络模块：封装了常用层（LSTM、Linear）、损失函数等
import torch.optim as optim  # PyTorch优化器模块：提供Adam、SGD等参数更新算法
import numpy as np  # 数值计算库：用于数据生成、数组操作（适配PyTorch张量）
import pandas as pd  # 数据处理库：预留用于读取真实CSV格式传感器数据（本示例暂未使用）
from sklearn.model_selection import train_test_split  # 数据集划分工具：拆分训练集/测试集
from sklearn.preprocessing import StandardScaler  # 数据预处理工具：特征标准化（消除量纲影响）
import matplotlib.pyplot as plt  # 可视化库：绘制预测结果和损失曲线，直观评估模型

# 设置随机种子（固定所有随机数生成器的初始状态）
# 目的：保证每次运行代码的实验结果完全一致，便于调试和复现
torch.manual_seed(42)  # 固定PyTorch的随机数生成（张量初始化、dropout等）
np.random.seed(42)  # 固定NumPy的随机数生成（数据模拟、索引打乱等）


class SpeedPredictor(nn.Module):
    """
    无人车速度预测模型：基于LSTM（长短期记忆网络）的时序单步预测模型
    核心适配场景：速度是连续变化的时序数据，需捕捉历史状态与未来状态的依赖关系
    LSTM优势：解决传统RNN的梯度消失问题，能有效学习长时序数据的特征
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        """
        初始化模型结构与超参数
        Args:
            input_size (int): 每个时间步的输入特征维度
                例：本示例输入5个特征（速度、加速度、方向盘角度、油门开度、刹车压力）
            hidden_size (int): LSTM隐藏层神经元数量
                作用：控制模型的特征提取能力（值越大能力越强，但易过拟合）
            num_layers (int): LSTM网络的堆叠层数
                作用：多层LSTM可提取更复杂的时序特征（1-2层最常用，过多易过拟合）
            output_size (int): 模型输出维度（默认1，即预测未来1个时间步的速度）
        """
        super().__init__()  # 简化写法：等价于 super(SpeedPredictor, self).__init__()

        # 定义LSTM核心层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入数据格式：(batch_size, 序列长度, 特征维度)，符合日常数据组织习惯
        )

        # 定义全连接输出层：将LSTM的高维特征映射到最终预测值（速度）
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        模型前向传播逻辑（数据在模型中的计算路径）
        Args:
            x (torch.Tensor): 输入张量，形状=(batch_size, seq_len, input_size)
                batch_size：每次训练的样本数量（批次大小）
                seq_len：输入序列长度（用过去多少个时间步的数据做预测）
                input_size：每个时间步的特征维度
        Returns:
            torch.Tensor：模型预测输出，形状=(batch_size, output_size)（每个样本对应1个预测速度）
        """
        # LSTM层计算：输出包含两部分（所有时间步的隐藏态 + 最后一个时间步的隐藏态/细胞态）
        # lstm_out形状=(batch_size, seq_len, hidden_size)，_表示忽略不需要的隐藏态/细胞态
        lstm_out, _ = self.lstm(x)

        # 关键操作：仅取最后一个时间步的隐藏态作为全连接层输入
        # 原因：我们需要用整个序列的历史信息预测下一个时间步，最后一个时间步已整合所有历史特征
        last_time_step_out = lstm_out[:, -1, :]  # 切片后形状=(batch_size, hidden_size)

        # 全连接层映射：将LSTM提取的特征转换为最终速度预测值
        pred_speed = self.fc(last_time_step_out)

        return pred_speed


def create_sequences(data, seq_len, pred_len=1):
    """
    时序数据格式化：将原始1D时序数据转换为LSTM可处理的序列-标签对（X, y）
    核心逻辑：用「过去seq_len个时间步的所有特征」预测「未来pred_len个时间步的速度」
    Args:
        data (np.ndarray): 原始特征矩阵，形状=(总时间步数量, 特征维度)
        seq_len (int): 输入序列长度（即"过去多少个时间步"）
        pred_len (int): 预测长度（即"未来多少个时间步"，默认1为单步预测）
    Returns:
        tuple: (X, y) 均为NumPy数组
            X: 输入序列矩阵，形状=(有效序列数, seq_len, 特征维度)
            y: 标签矩阵（待预测的速度），形状=(有效序列数, pred_len)
    """
    input_sequences = []  # 存储所有输入序列
    target_speeds = []    # 存储对应每个序列的目标速度

    # 循环生成序列（避免索引越界：确保最后一个序列能取到完整的标签）
    # 有效序列数 = 总时间步 - 输入序列长度 - 预测长度 + 1
    for i in range(len(data) - seq_len - pred_len + 1):
        # 提取输入序列：从第i步开始，连续取seq_len个时间步的所有特征
        seq = data[i:i + seq_len]  # 形状=(seq_len, 特征维度)
        # 提取目标标签：从输入序列结束后开始，取pred_len个时间步的速度（速度是第0列特征）
        target = data[i + seq_len:i + seq_len + pred_len, 0]  # 形状=(pred_len,)

        input_sequences.append(seq)
        target_speeds.append(target)

    # 转换为NumPy数组（便于后续转换为PyTorch张量）
    return np.array(input_sequences), np.array(target_speeds)


def main():
    """主函数：整合数据准备、模型训练、评估、可视化的完整流程"""
    # ===================== 1. 模拟数据生成（替代真实传感器数据）=====================
    # 说明：实际应用时，可替换为 pd.read_csv("sensor_data.csv") 读取真实数据
    # 模拟特征维度：[速度(m/s), 加速度(m/s²), 方向盘角度(°), 油门开度(%), 刹车压力(bar)]
    total_time_steps = 10000  # 总时间步（模拟10000次传感器采样，约100秒）
    time_axis = np.linspace(0, 100, total_time_steps)  # 时间轴（0-100秒，均匀采样）

    # 模拟各特征（添加高斯噪声模拟真实传感器的测量误差）
    base_speed = 10  # 基础速度（10m/s ≈ 36km/h，城市道路常见速度）
    speed = base_speed + 5 * np.sin(time_axis) + np.random.normal(0, 0.5, total_time_steps)  # 速度含正弦波动
    acceleration = np.gradient(speed, time_axis)  # 加速度：速度对时间的一阶导数
    steering = 10 * np.sin(time_axis / 2) + np.random.normal(0, 1, total_time_steps)  # 方向盘角度（慢波动）
    throttle = 30 + 10 * np.sin(time_axis / 3) + np.random.normal(0, 2, total_time_steps)  # 油门开度（30%左右）
    brake = np.where(
        speed < 8,  # 逻辑：速度低于8m/s时（低速），刹车压力增大
        5 + np.random.normal(0, 1, total_time_steps),  # 低速刹车压力（5bar左右）
        np.random.normal(0, 0.5, total_time_steps)     # 高速刹车压力（接近0，很少刹车）
    )

    # 组合所有特征为矩阵：形状=(total_time_steps, 5)
    sensor_data = np.column_stack([speed, acceleration, steering, throttle, brake])

    # ===================== 2. 数据预处理（避免模型训练异常）=====================
    # 标准化：将所有特征转换为「均值=0，方差=1」的标准正态分布
    # 必要性：不同特征量纲不同（如速度m/s、角度°），标准化可避免特征权重失衡
    scaler = StandardScaler()
    # 注意：仅用训练数据拟合scaler（避免测试集信息泄露，保证评估客观性）
    scaled_data = scaler.fit_transform(sensor_data)

    # 生成LSTM输入序列和标签
    seq_len = 10  # 输入序列长度：用过去10个时间步（约0.1秒）的数据预测未来速度
    X, y = create_sequences(scaled_data, seq_len)  # X=(N,10,5), y=(N,1)，N为有效序列数

    # 划分训练集/测试集（时序数据必须禁用shuffle！否则破坏时间顺序）
    # 测试集占比20%：用于评估模型在未见过的未来数据上的泛化能力
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 转换为PyTorch张量（模型仅支持张量输入，不支持NumPy数组）
    X_train = torch.FloatTensor(X_train)  # 训练输入：(N_train, 10, 5)
    y_train = torch.FloatTensor(y_train)  # 训练标签：(N_train, 1)
    X_test = torch.FloatTensor(X_test)    # 测试输入：(N_test, 10, 5)
    y_test = torch.FloatTensor(y_test)    # 测试标签：(N_test, 1)

    # ===================== 3. 模型初始化（配置超参数）=====================
    input_size = X_train.shape[2]  # 输入特征维度：5（对应5个传感器特征）
    hidden_size = 64  # LSTM隐藏层神经元数（超参数：可调整为32/128，需配合正则化）
    num_layers = 2    # LSTM层数（超参数：1-2层最优，过多易过拟合）
    model = SpeedPredictor(input_size, hidden_size, num_layers)  # 实例化模型

    # ===================== 4. 配置训练组件（损失函数+优化器）=====================
    criterion = nn.MSELoss()  # 损失函数：均方误差（MSE）
    # 适配场景：回归任务（预测连续值如速度），对异常值敏感，惩罚较大误差
    optimizer = optim.Adam(
        model.parameters(),  # 待优化的参数：模型中所有可学习参数（LSTM权重、全连接层权重）
        lr=0.001  # 学习率（超参数：控制参数更新步长，0.001为Adam默认最优初始值）
    )

    # ===================== 5. 模型训练与实时验证 =====================
    epochs = 50  # 训练轮数：整个训练集将被遍历50次（足够模型收敛，避免欠拟合）
    batch_size = 32  # 批次大小：每次训练用32个样本更新参数（平衡训练速度和稳定性）
    train_loss_history = []  # 记录每轮训练损失（用于后续分析收敛情况）
    test_loss_history = []   # 记录每轮测试损失（用于判断是否过拟合）

    for epoch in range(epochs):
        # ---------------------- 训练阶段（更新模型参数）----------------------
        model.train()  # 开启训练模式（启用Dropout/BatchNorm的训练行为，本模型无但保留规范）
        total_train_loss = 0  # 累计当前轮次所有批次的损失

        # 分批迭代训练集（避免一次性加载所有数据导致内存溢出）
        for i in range(0, len(X_train), batch_size):
            # 截取当前批次数据（最后一批可能不足32个样本，自动适配）
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()  # 清零梯度（关键！避免前一轮梯度累积影响当前更新）
            pred = model(batch_X)  # 模型前向传播：计算批次预测值
            loss = criterion(pred, batch_y)  # 计算批次损失（预测值与真实值的差距）
            loss.backward()  # 反向传播：计算各参数的梯度（损失对参数的偏导数）
            optimizer.step()  # 梯度下降：根据梯度更新模型参数（最小化损失）

            # 累计损失（乘以批次样本数，确保最终平均损失计算准确）
            total_train_loss += loss.item() * batch_X.size(0)

        # 计算当前轮次的平均训练损失（总损失/训练集样本数）
        avg_train_loss = total_train_loss / len(X_train)
        train_loss_history.append(avg_train_loss)

        # ---------------------- 验证阶段（不更新模型参数）----------------------
        model.eval()  # 开启评估模式（禁用Dropout/BatchNorm更新，固定模型参数）
        with torch.no_grad():  # 禁用梯度计算（节省内存+加速计算，无需反向传播）
            test_pred = model(X_test)  # 测试集全量预测
            avg_test_loss = criterion(test_pred, y_test).item()  # 计算测试集平均损失
            test_loss_history.append(avg_test_loss)

        # 每5轮打印一次损失（监控训练进度，及时发现异常如损失不下降）
        if (epoch + 1) % 5 == 0:
            print(f'第 {epoch + 1:2d}/{epochs} 轮 | 训练损失：{avg_train_loss:.6f} | 测试损失：{avg_test_loss:.6f}')

    # ===================== 6. 结果可视化（直观评估模型性能）=====================
    # 反标准化：将标准化后的预测值/真实值转换为原始速度单位（m/s），便于理解
    # 原理：StandardScaler需要完整特征向量才能反归一化，故构造虚拟矩阵仅恢复速度列
    # 真实速度反标准化
    dummy_test = np.zeros_like(scaled_data[:len(y_test)])  # 虚拟矩阵：(N_test, 5)
    dummy_test[:, 0] = y_test.numpy().flatten()  # 仅速度列（第0列）填入标准化后的真实值
    true_speed = scaler.inverse_transform(dummy_test)[:, 0]  # 反标准化得到原始速度

    # 预测速度反标准化（与真实速度步骤一致）
    dummy_pred = np.zeros_like(scaled_data[:len(y_test)])
    dummy_pred[:, 0] = test_pred.numpy().flatten()
    pred_speed = scaler.inverse_transform(dummy_pred)[:, 0]

    # 图1：真实速度 vs 预测速度（直观对比拟合效果）
    plt.figure(figsize=(12, 6))
    plt.plot(true_speed, label='真实速度', alpha=0.8, linewidth=1.5)  # 真实值曲线（透明度0.8避免重叠）
    plt.plot(pred_speed, label='预测速度', alpha=0.8, linewidth=1.5, linestyle='--')  # 预测值曲线（虚线区分）
    plt.xlabel('测试集时间步', fontsize=12)
    plt.ylabel('速度 (m/s)', fontsize=12)
    plt.title('无人车速度预测结果对比', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)  # 添加网格线，便于读取数值
    plt.show()

    # 图2：训练/测试损失曲线（分析模型收敛与过拟合）
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_history, label='训练损失', linewidth=1.5)
    plt.plot(test_loss_history, label='测试损失', linewidth=1.5, linestyle='--')
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('均方误差 (MSE)', fontsize=12)
    plt.title('模型训练收敛曲线', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.show()

    # 补充说明：
    # 1. 若训练损失持续下降但测试损失上升 → 过拟合（可减小hidden_size/增加Dropout/增大批次）
    # 2. 若训练损失和测试损失均很高 → 欠拟合（可增大hidden_size/增加LSTM层数/增加训练轮数）
    # 3. 若损失波动大 → 减小学习率/增大批次大小


# 程序入口：当脚本被直接运行时，执行主函数（避免被导入时自动执行）
if __name__ == '__main__':
    main()