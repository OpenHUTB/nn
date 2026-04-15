import torch
import torch.nn as nn

class CrossDomainAttention(nn.Module):
    def __init__(self, input_dims, num_blocks=6):
        super(CrossDomainAttention, self).__init__()
        # 多头注意力模块（embed_dim 需与投影后的维度一致，即256）
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=256, num_heads=8) for _ in range(num_blocks)
        ])
        # 投影层：将每个输入特征映射到256维
        self.proj_layers = nn.ModuleList([
            nn.Linear(dim, 256) for dim in input_dims
        ])
        # 可选：层归一化，稳定训练
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(256) for _ in range(num_blocks)
        ])
    
    def forward(self, *inputs):
        processed_inputs = []
        for i, x in enumerate(inputs):
            # 1. 处理高维特征（如 segmentation 的 (B, 256, H, W)）
            if x.dim() > 2:
                # 对空间维度求平均，降为 (B, C)，其中 C=256（与 input_dims 对应）
                x = x.mean(dim=tuple(range(2, x.dim())))  # 保留前2维 (B, C)
            
            # 2. 确保输入线性层的是2维张量 (B, D)
            assert x.dim() == 2, f"特征 {i} 经处理后必须是2维张量，当前维度：{x.dim()}"
            
            # 3. 投影到256维，并调整为 MultiheadAttention 要求的格式 (seq_len=1, B, 256)
            # （因为每个特征是全局描述，序列长度设为1）
            x = self.proj_layers[i](x)  # (B, 256)
            x = x.unsqueeze(0)  # (1, B, 256)，seq_len=1
            processed_inputs.append(x)
        
        # 4. 拼接所有跨域特征，形成序列 (seq_len=N, B, 256)，其中 N=5（5个特征）
        x = torch.cat(processed_inputs, dim=0)  # (5, B, 256)
        
        # 5. 经过多个注意力块
        for i, block in enumerate(self.attention_blocks):
            # 自注意力计算
            attn_output, _ = block(x, x, x)  # (5, B, 256)
            # 残差连接 + 层归一化
            x = self.norm_layers[i](x + attn_output)
        
        # 6. 对序列维度求平均，得到融合后的特征 (B, 256)
        return x.mean(dim=0)  # (B, 256)