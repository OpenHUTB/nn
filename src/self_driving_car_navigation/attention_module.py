import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class CrossDomainAttention(nn.Module):
    def __init__(self, num_blocks=2, embed_dim=256):  # 确保embed_dim与感知模块输出匹配
        super().__init__()
        self.num_blocks = num_blocks
        # 多头注意力层的embed_dim必须与输入特征的最后一维一致
        self.attention_blocks = nn.ModuleList([
            MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # 层归一化维度与embed_dim一致

    def forward(self, scene_info, segmentation, odometry, obstacles, boundary):
        # 统一输入张量维度为3D [batch, seq_len, features]
        def adjust_dim(tensor):
            if tensor.dim() == 4:
                # 4D张量（如[batch, channel, h, w]）→ [batch, h*w, channel]
                batch, channel, h, w = tensor.shape
                return tensor.permute(0, 2, 3, 1).reshape(batch, h*w, channel)
            elif tensor.dim() == 2:
                # 2D张量（如[batch, features]）→ [batch, 1, features]
                return tensor.unsqueeze(1)
            else:
                return tensor

        # 调整所有输入特征的维度
        inputs = [
            adjust_dim(scene_info),
            adjust_dim(segmentation),
            adjust_dim(odometry),
            adjust_dim(obstacles),
            adjust_dim(boundary)
        ]

        # 统一所有特征的最后一维（特征维度）为 embed_dim
        target_feat_dim = self.attention_blocks[0].embed_dim
        adjusted_inputs = []
        for x in inputs:
            if x.shape[-1] != target_feat_dim:
                # 动态创建线性层并转移到相同设备
                linear = nn.Linear(x.shape[-1], target_feat_dim, device=x.device)
                x = linear(x)
            adjusted_inputs.append(x)

        # 拼接所有特征（在seq_len维度拼接）
        x = torch.cat(adjusted_inputs, dim=1)

        # 注意力计算
        for attn_block in self.attention_blocks:
            attn_output, _ = attn_block(x, x, x)  # 自注意力
            x = self.norm(x + attn_output)  # 残差连接 + 层归一化

        return x