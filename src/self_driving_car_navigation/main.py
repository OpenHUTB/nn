# 导入PyTorch核心库，用于张量操作和神经网络构建
import torch
import torch.nn as nn
# 导入参数解析模块，用于命令行参数配置
import argparse 
# 导入Adam优化器，用于模型参数更新
from torch.optim import Adam
# 导入自定义数据加载器，用于加载训练/测试数据
from utils.dataloader import get_dataloader
# 导入自定义感知模块，负责处理多模态输入（图像、激光雷达、IMU）并提取基础特征
from models.perception_module import PerceptionModule
# 导入自定义跨域注意力模块，负责融合不同类型的感知特征
from models.attention_module import CrossDomainAttention
# 导入自定义决策模块，基于融合特征输出策略和价值估计
from models.decision_module import DecisionModule
# 导入自定义自评估梯度模型，用于计算动作相关的Q值估计
from models.sagm import SelfAssessmentGradientModel  

class IntegratedSystem(nn.Module):
    """
    多模态融合的智能决策集成系统
    功能：整合感知、注意力融合、决策和自评估梯度模块，处理多模态输入并输出决策结果
    输入：图像（视觉）、激光雷达（环境测距）、IMU（惯性测量）、动作（历史/目标动作）
    输出：策略分布、价值估计、自评估Q值
    """
    
    def __init__(self, device='cpu', state_dim=128, action_dim=2):
        """
        初始化集成系统的各个模块
        Args:
            device (str): 计算设备（'cpu'或'cuda'），默认使用CPU
            state_dim (int): 状态特征维度，默认128（感知模块输出特征维度）
            action_dim (int): 动作维度，默认2（例如：转向角、加速度）
        """
        super().__init__()
        # 记录计算设备，用于统一模块和数据的设备归属
        self.device = device
        
        # 初始化感知模块：处理IMU、图像、激光雷达多模态输入，提取场景关键信息
        self.perception = PerceptionModule().to(self.device)
        
        # 初始化跨域注意力模块：融合感知模块输出的多种特征（场景信息、分割图等）
        # 参数：2个注意力块，嵌入维度64（特征融合后的维度）
        self.attention = CrossDomainAttention(num_blocks=2, embed_dim=64).to(self.device)
        
        # 初始化决策模块：基于融合特征输出策略（动作分布）和价值（状态价值估计）
        self.decision = DecisionModule().to(self.device)
        
        # 初始化自评估梯度模型：输入融合特征和动作，输出动作对应的Q值（动作价值估计）
        # 参数：隐藏层维度64
        self.sagm = SelfAssessmentGradientModel(hidden_dim=64).to(self.device)  

    def forward(self, image, lidar_data, imu_data, action):
        """
        前向传播：定义数据在模型中的流动和计算过程
        Args:
            image (torch.Tensor): 视觉图像输入，形状通常为[batch_size, channels, height, width]
            lidar_data (torch.Tensor): 激光雷达数据输入，形状通常为[batch_size, lidar_dim, sequence_len]
            imu_data (torch.Tensor): IMU惯性测量数据，形状通常为[batch_size, imu_dim, sequence_len]
            action (torch.Tensor): 目标/历史动作输入，形状通常为[batch_size, action_dim]
        Returns:
            policy (torch.Tensor): 策略输出（动作预测），形状[batch_size, action_dim]
            value (torch.Tensor): 状态价值估计，形状[batch_size, 1]
            sagm_q_value (torch.Tensor): 自评估Q值（动作价值估计），形状[batch_size, 1]
        """
        # 1. 感知模块处理：将多模态原始输入转换为结构化特征
        # 输出：场景信息、语义分割结果、里程计数据、障碍物检测结果、边界信息
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        
        # 2. 跨域注意力融合：融合感知模块输出的多种异质特征，得到统一的融合特征
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        
        # 3. 决策模块推理：基于融合特征输出策略（动作预测）和状态价值估计
        policy, value = self.decision(fused_features)

        # 4. 维度调整：对策略和价值进行序列维度平均（适应时序数据的批量处理）
        # 假设fused_features形状为[batch_size, sequence_len, feature_dim]，需压缩sequence_len维度
        policy = torch.mean(policy, dim=1)  # [batch_size, action_dim]
        value = torch.mean(value, dim=1)    # [batch_size, 1]
        
        # 5. 动作维度适配：将动作扩展为3维（匹配融合特征的时序维度），用于SAGM模块输入
        action_3d = action.unsqueeze(1)  # [batch_size, 1, action_dim]
        seq_len = fused_features.shape[1]  # 获取时序长度（sequence_len）
        action_3d = action_3d.repeat(1, seq_len, 1)  # [batch_size, sequence_len, action_dim]

        # 6. 自评估梯度模型推理：输入融合特征和适配后的动作，输出动作价值估计（Q值）
        sagm_q_value = self.sagm(fused_features, action_3d) 
        # 维度调整：对Q值进行序列维度平均
        sagm_q_value = torch.mean(sagm_q_value, dim=1)  # [batch_size, 1]
        
        # 返回最终的决策输出（策略、价值、Q值）
        return policy, value, sagm_q_value

def train_model(model, dataloader, optimizer, device, num_epochs=10):
    """
    模型训练函数：实现批量数据训练、损失计算、梯度下降更新
    Args:
        model (nn.Module): 待训练的集成模型（IntegratedSystem实例）
        dataloader (DataLoader): 训练数据加载器，迭代输出批量数据
        optimizer (torch.optim.Optimizer): 优化器（此处为Adam），用于更新模型参数
        device (str): 计算设备（'cpu'或'cuda'）
        num_epochs (int): 训练轮数，默认10轮
    """
    # 设置模型为训练模式：启用Dropout、BatchNorm更新等训练特定行为
    model.train()
    
    # 迭代训练每一轮
    for epoch in range(num_epochs):
        # 初始化本轮累计损失
        running_loss = 0.0
        
        # 迭代处理每个批量数据
        for i, (image, lidar_data, imu_data, target_action) in enumerate(dataloader):
            # 将批量数据转移到指定计算设备（CPU/GPU），确保数据与模型设备一致
            image = image.to(device)
            lidar_data = lidar_data.to(device)
            imu_data = imu_data.to(device)
            target_action = target_action.to(device)
            
            # 清零优化器梯度：避免上一轮梯度累积影响当前更新
            optimizer.zero_grad()
            
            # 前向传播：模型输出预测结果（策略、价值、Q值）
            policy_output, value_output, sagm_q_value = model(image, lidar_data, imu_data, target_action)
            
            # 计算损失函数：多任务损失融合（策略回归损失 + 价值估计损失 + Q值估计损失）
            # 1. 策略损失：预测动作与目标动作的MSE损失（回归任务）
            policy_loss = nn.MSELoss()(policy_output, target_action)
            # 2. 价值损失：价值估计与目标动作总和的MSE损失（假设价值与动作累积效果相关）
            value_loss = nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True))
            # 3. Q值损失：SAGM输出Q值与目标动作总和的MSE损失（动作价值匹配）
            sagm_loss = nn.MSELoss()(sagm_q_value, target_action.sum(dim=1, keepdim=True))
            # 总损失：三个子损失直接相加（可根据任务重要性调整权重）
            total_loss = policy_loss + value_loss + sagm_loss
            
            # 反向传播：计算损失对模型参数的梯度
            total_loss.backward()
            
            # 梯度下降：优化器更新模型参数（基于计算出的梯度）
            optimizer.step()
            
            # 累积批量损失
            running_loss += total_loss.item()
            
            # 每10个批量打印一次训练状态（监控训练进度）
            if i % 10 == 9:
                # 计算10个批量的平均损失并打印
                avg_batch_loss = running_loss / 10
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {avg_batch_loss:.4f}')
                # 重置累计损失
                running_loss = 0.0
    
    # 训练完成提示
    print('Training complete')

def test_model(model, dataloader, device):
    """
    模型测试函数：评估模型在测试集上的性能（无梯度计算，仅前向传播）
    Args:
        model (nn.Module): 已训练的集成模型（IntegratedSystem实例）
        dataloader (DataLoader): 测试数据加载器，迭代输出批量测试数据
        device (str): 计算设备（'cpu'或'cuda'）
    """
    # 设置模型为评估模式：禁用Dropout、固定BatchNorm参数等测试特定行为
    model.eval()
    
    # 初始化测试集总损失
    total_loss = 0.0
    
    # 禁用梯度计算上下文：减少内存占用，加速推理（测试阶段无需反向传播）
    with torch.no_grad():
        # 迭代处理每个测试批量
        for image, lidar_data, imu_data, target_action in dataloader:
            # 数据转移到指定设备
            image = image.to(device)
            lidar_data = lidar_data.to(device)
            imu_data = imu_data.to(device)
            target_action = target_action.to(device)
            
            # 前向传播：获取模型预测结果
            policy_output, value_output, sagm_q_value = model(image, lidar_data, imu_data, target_action)
            
            # 计算测试损失（与训练损失计算逻辑一致，保证评估指标统一）
            policy_loss = nn.MSELoss()(policy_output, target_action)
            value_loss = nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True))
            sagm_loss = nn.MSELoss()(sagm_q_value, target_action.sum(dim=1, keepdim=True))
            batch_loss = policy_loss + value_loss + sagm_loss
            
            # 累积测试损失
            total_loss += batch_loss.item()
    
    # 计算测试集平均损失（总损失 / 测试批量数）
    avg_test_loss = total_loss / len(dataloader)
    # 打印测试结果
    print(f'Test Average Loss: {avg_test_loss:.4f}')

# 主函数入口：程序执行的起点
if __name__ == "__main__":
    # 1. 初始化命令行参数解析器：用于接收用户输入的运行模式（训练/测试）
    parser = argparse.ArgumentParser(description="多模态融合智能决策系统 - 训练/测试入口")
    
    # 添加模式参数：--mode，默认值为"train"，仅允许选择"train"或"test"
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "test"],
        help="运行模式：train（训练模型）/ test（测试模型）"
    )
    
    # 解析命令行参数（获取用户输入的模式）
    args = parser.parse_args()

    # 2. 自动选择计算设备：优先使用GPU（CUDA），无GPU时使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用计算设备: {device}")  # 打印当前使用的设备（便于调试）

    # 3. 初始化模型：创建集成系统实例，指定计算设备
    model = IntegratedSystem(device=device)

    # 4. 初始化优化器：使用Adam优化器，学习率0.001（可根据需求调整）
    optimizer = Adam(model.parameters(), lr=0.001)

    # 5. 加载数据：通过自定义数据加载器获取训练/测试数据（数据路径、批次大小等在get_dataloader中配置）
    dataloader = get_dataloader()
    print(f"数据加载完成，批量数: {len(dataloader)}")  # 打印批量数（便于确认数据量）

    # 6. 根据运行模式执行训练或测试
    if args.mode == "train":
        print("=== 开始模型训练 ===")
        train_model(model, dataloader, optimizer, device)
    elif args.mode == "test":
        print("=== 开始模型测试 ===")
        test_model(model, dataloader, device)