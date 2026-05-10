import torch
import torch.nn as nn
import timm

class DepthEncoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=1024):
        super(DepthEncoder, self).__init__()

        self.convnext = timm.create_model("convnext_tiny", pretrained=pretrained, in_chans=1, num_classes=0)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        if output_dim < 768:
            self.projection = nn.Linear(768, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        features = self.convnext(x)
        if len(features.shape) == 4:
            features = self.global_avg_pool(features)
            features = features.view(features.shape[0], -1)
        features = self.projection(features)
        return features