import torch
import torch.nn as nn
import torchvision.models as models

class RGBEncoder(nn.Module):
    """
    RGB Encoder using a pretrained ResNet-50 model.
    Extracts feature embeddings from RGB images.
    """
    
    def __init__(self, pretrained=True, output_dim=2048):
        super(RGBEncoder, self).__init__()

        # Load ResNet-50 with pretrained weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Remove the fully connected (FC) layer to get feature embeddings
        self.resnet.fc = nn.Identity()

        # Optional: Add a projection layer to reduce dimensionality
        if output_dim < 2048:
            self.projection = nn.Linear(2048, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        """
        Forward pass for RGB feature extraction.
        :param x: Input RGB image tensor (B, 3, H, W)
        :return: Feature vector (B, output_dim)
        """
        features = self.resnet(x)  # (B, 2048)
        features = self.projection(features)  # (B, output_dim)
        return features

