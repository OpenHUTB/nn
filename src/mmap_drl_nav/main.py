import torch
import torch.nn as nn
from torch.optim import Adam
from utils.dataloader import get_dataloader  # 导入自定义的数据加载工具
from models.perception_module import PerceptionModule  # 导入感知模块
from models.attention_module import CrossDomainAttention  # 导入跨域注意力模块
from models.decision_module import DecisionModule  # 导入决策模块
from models.sagm import SelfAssessmentGradientModel  # 导入自评估梯度模型


class IntegratedSystem:
    """
    集成决策系统类
    整合感知、注意力、决策和自评估模块，形成端到端的智能决策系统
    """

    def __init__(self, device='cpu', state_dim=128, action_dim=2):
        """
        初始化集成系统
        Args:
            device (str): 模型运行设备，默认CPU，可设为'cuda'使用GPU
            state_dim (int): 状态特征维度，默认128
            action_dim (int): 动作输出维度，默认2
        """
        self.device = device  # 保存设备信息
        # 初始化感知模块并移至指定设备
        self.perception = PerceptionModule().to(self.device)
        # 初始化跨域注意力模块（6个注意力块）并移至指定设备
        self.attention = CrossDomainAttention(num_blocks=6).to(self.device)
        # 初始化决策模块并移至指定设备
        self.decision = DecisionModule().to(self.device)
        # 初始化自评估梯度模型（输入维度为状态+动作维度）并移至指定设备
        self.sagm = SelfAssessmentGradientModel(input_dim=state_dim + action_dim).to(self.device)

    def forward(self, image, lidar_data, imu_data, action):
        """
        前向传播函数：整合各模块完成一次完整的前向计算
        Args:
            image: 图像输入数据
            lidar_data: 激光雷达输入数据
            imu_data: 惯性测量单元输入数据
            action: 动作输入数据
        Returns:
            policy: 策略输出
            value: 价值输出
            sagm_q_value: 自评估Q值输出
        """
        # 感知模块处理多源数据，输出场景信息、分割结果、里程计、障碍物和边界信息
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        # 跨域注意力模块融合感知模块输出的多维度特征
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)

        # 决策模块基于融合特征输出策略和价值估计
        policy, value = self.decision(fused_features)

        # 自评估梯度模型基于融合特征和动作计算Q值
        sagm_q_value = self.sagm(fused_features, action)
        return policy, value, sagm_q_value


def train_model(model, dataloader, optimizer, device, num_epochs=10):
    """
    模型训练函数
    Args:
        model: 待训练的集成系统模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        device: 训练设备
        num_epochs (int): 训练轮数，默认10
    """
    model.train()  # 将模型设为训练模式（启用dropout、batchnorm等训练特性）
    for epoch in range(num_epochs):  # 遍历每个训练轮次
        running_loss = 0.0  # 初始化累计损失
        # 遍历数据加载器中的每个批次
        for i, (image, lidar_data, imu_data, target_action) in enumerate(dataloader):
            # 将所有数据移至指定设备（CPU/GPU）
            image, lidar_data, imu_data, target_action = image.to(device), lidar_data.to(device), imu_data.to(
                device), target_action.to(device)

            optimizer.zero_grad()  # 清空梯度缓存

            # 前向传播：输入数据得到模型输出
            policy_output, value_output, sagm_q_value = model(image, lidar_data, imu_data, target_action)

            # 计算总损失：策略损失 + 价值损失 + 自评估Q值损失（均使用MSE损失）
            # 价值输出和Q值输出与目标动作的求和结果做回归
            loss = (nn.MSELoss()(policy_output, target_action) +
                    nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True)) +
                    nn.MSELoss()(sagm_q_value, target_action.sum(dim=1, keepdim=True)))

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新模型参数

            running_loss += loss.item()  # 累计损失值

            # 每10个批次打印一次训练信息
            if i % 10 == 9:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0  # 重置累计损失
    print('Training complete')  # 训练完成提示


def test_model(model, dataloader, device):
    """
    模型测试函数
    Args:
        model: 已训练的集成系统模型
        dataloader: 测试数据加载器
        device: 测试设备
    """
    model.eval()  # 将模型设为评估模式（禁用dropout、batchnorm等训练特性）
    total_loss = 0.0  # 初始化总损失

    # 禁用梯度计算（加速推理，节省内存）
    with torch.no_grad():
        # 遍历测试数据集
        for image, lidar_data, imu_data, target_action in dataloader:
            # 将所有数据移至指定设备
            image, lidar_data, imu_data, target_action = image.to(device), lidar_data.to(device), imu_data.to(
                device), target_action.to(device)

            # 前向传播得到模型输出
            policy_output, value_output, sagm_q_value = model(image, lidar_data, imu_data, target_action)

            # 计算批次损失（与训练损失计算方式一致）
            loss = (nn.MSELoss()(policy_output, target_action) +
                    nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True)) +
                    nn.MSELoss()(sagm_q_value, target_action.sum(dim=1, keepdim=True)))

            total_loss += loss.item()  # 累计测试损失

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    print(f'Test Average Loss: {avg_loss:.4f}')  # 打印测试平均损失