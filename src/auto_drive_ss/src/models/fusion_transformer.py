import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """
    A single Transformer Encoder Block with Multi-Head Attention and Feed-Forward layers.
    """

    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-Head Self Attention
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)

        # Feed-Forward Network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass for the Transformer Block.
        :param x: Input tensor (B, N, dim)
        :return: Processed tensor (B, N, dim)
        """
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        x = self.norm1(x + attn_output)  # Add & Norm

        mlp_output = self.mlp(x)  # Feed-Forward Network
        x = self.norm2(x + mlp_output)  # Add & Norm

        return x


class FusionTransformer(nn.Module):
    """
    Cross-Modal Transformer for multi-sensor fusion.
    """

    def __init__(self, input_dim=256, num_modalities=3, depth=4, heads=8, dim_head=64, mlp_dim=512, output_dim=256):
        """
        input_dim: Feature dimension of each modality.
        num_modalities: Number of input modalities (RGB, Depth, LiDAR, Segmentation).
        depth: Number of transformer layers.
        heads: Number of attention heads.
        output_dim: Dimension of the final fused feature vector.
        """
        super(FusionTransformer, self).__init__()

        # Project all input modalities to the same embedding size
        self.projection = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_modalities)])

        # Transformer Encoder Blocks
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(dim=input_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
            for _ in range(depth)
        ])

        # Output projection to match required output dimension
        self.output_proj = nn.Linear(input_dim * num_modalities, output_dim)

    def forward(self, rgb_feat, depth_feat, seg_feat):
        """
        Forward pass for multi-sensor fusion.
        :param rgb_feat: Feature tensor from RGB Encoder (B, input_dim)
        :param depth_feat: Feature tensor from Depth Encoder (B, input_dim)
        :param seg_feat: Feature tensor from Segmentation Parser (B, input_dim)
        :return: Fused feature vector (B, output_dim)
        """

        # Project each modality
        rgb_feat = self.projection[0](rgb_feat)
        depth_feat = self.projection[1](depth_feat)
        seg_feat = self.projection[2](seg_feat)

        # Stack modalities: (B, N=3, input_dim)
        x = torch.stack([rgb_feat, depth_feat, seg_feat], dim=1)

        # Pass through Transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)

        # Flatten and project to output dimension
        x = x.view(x.shape[0], -1)  # (B, input_dim * num_modalities)
        x = self.output_proj(x)  # (B, output_dim)

        return x



