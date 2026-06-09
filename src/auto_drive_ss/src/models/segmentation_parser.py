import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation

class SemanticAttention(nn.Module):
    """
    Semantic Attention Module to enhance important segmented regions.
    """

    def __init__(self, num_classes=5):
        """
        num_classes: Number of semantic classes to attend to.
        (e.g., road, lane, vehicle, pedestrian, traffic light)
        """
        super(SemanticAttention, self).__init__()

        # Learnable attention weights for each class
        self.class_weights = nn.Parameter(torch.randn(num_classes))

        # Feature extractor after attention weighting
        self.conv = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, seg_map):
        """
        Applies attention weighting on the segmentation map.
        :param seg_map: Tensor (B, num_classes, H, W) - one-hot encoded segmentation map
        :return: Processed segmentation features (B, 32, H, W)
        """
        # Apply class attention weighting
        weighted_map = seg_map * self.class_weights.view(1, -1, 1, 1)  # Shape (B, num_classes, H, W)

        # Extract features
        features = self.conv(weighted_map)  # (B, 32, H, W)

        return features


class SegmentationParser(nn.Module):
    """
    Segmentation Parser that processes semantic segmentation maps
    using a pretrained DeepLabV3 model and a Semantic Attention module.
    """

    def __init__(self, pretrained=True, output_dim=256, num_classes=5):
        super(SegmentationParser, self).__init__()

        # Load DeepLabV3-ResNet101 for segmentation
        self.segmentation_model = segmentation.deeplabv3_resnet101(weights="DEFAULT" if pretrained else None)
        
        # Modify classifier to output the required number of classes
        in_channels = self.segmentation_model.classifier[4].in_channels
        self.segmentation_model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Semantic Attention
        self.semantic_attention = SemanticAttention(num_classes)

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Projection layer to output fixed feature size
        self.projection = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass for segmentation processing.
        :param x: Input RGB image tensor (B, 3, H, W)
        :return: Feature vector (B, output_dim)
        """
        # Get segmentation map from DeepLabV3
        seg_map = self.segmentation_model(x)["out"]  # (B, num_classes, H, W)
        seg_map = torch.sigmoid(seg_map)  # Normalize to (0,1)

        # Apply semantic attention
        processed_features = self.semantic_attention(seg_map)  # (B, 32, H, W)

        # Extract final feature representation
        features = self.feature_extractor(processed_features)  # (B, 64, 1, 1)
        features = features.view(features.shape[0], -1)  # Flatten (B, 64)
        features = self.projection(features)  # (B, output_dim)

        return features

