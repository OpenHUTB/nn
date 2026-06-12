import torch
import torch.nn as nn
from src.models.rgb_encoder import RGBEncoder
from src.models.depth_encoder import DepthEncoder
from src.models.segmentation_parser import SegmentationParser
from src.models.fusion_transformer import FusionTransformer
from src.models.temporal_aggregator import TemporalAggregator

class PerceptionSystem(nn.Module):
    """
    Multi-Sensor Perception System using RGB, Depth, and Segmentation.
    """

    def __init__(self, feature_dim=256, fusion_dim=256, output_dim=256, num_frames=5):
        super(PerceptionSystem, self).__init__()

        # Sensor Encoders (Only RGB, Depth, and Segmentation)
        self.rgb_encoder = RGBEncoder(output_dim=feature_dim)
        self.depth_encoder = DepthEncoder(output_dim=feature_dim)
        self.segmentation_parser = SegmentationParser(output_dim=feature_dim)

        # Fusion Transformer (Now only for 3 modalities)
        self.fusion_transformer = FusionTransformer(input_dim=feature_dim, num_modalities=3, output_dim=fusion_dim)

        # Temporal Aggregator
        self.temporal_aggregator = TemporalAggregator(input_dim=fusion_dim, output_dim=output_dim)

        # Buffer for storing past frames
        self.num_frames = num_frames
        self.memory = []

    def forward(self, sensor_data):
        """
        Forward pass with RGB, Depth, and Segmentation.
        :param sensor_data: Dictionary containing sensor tensors:
            - rgb: (B, 3, H, W)
            - depth: (B, 1, H, W)
            - segmentation: (B, 3, H, W)
        :return: RL-ready state representation (B, output_dim)
        """

        # Encode each modality
        rgb_feat = self.rgb_encoder(sensor_data['rgb'])  # (B, feature_dim)
        depth_feat = self.depth_encoder(sensor_data['depth'])  # (B, feature_dim)
        seg_feat = self.segmentation_parser(sensor_data['segmentation'])  # (B, feature_dim)

        # Fuse features with Transformer
        fused_feat = self.fusion_transformer(rgb_feat, depth_feat, seg_feat)  # (B, fusion_dim)

        # Maintain temporal memory
        self.memory.append(fused_feat)
        if len(self.memory) > self.num_frames:
            self.memory.pop(0)

        # Convert memory list to tensor
        temporal_input = torch.stack(self.memory, dim=1)

        # Apply Temporal Aggregation
        output, _ = self.temporal_aggregator(temporal_input)

        return output

