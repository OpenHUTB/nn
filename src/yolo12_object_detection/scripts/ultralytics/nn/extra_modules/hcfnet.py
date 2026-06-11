import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import Conv

__all__ = ['PPA', 'DASI']

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

# https://mp.weixin.qq.com/s/26H0PgN5sikD1MoSkIBJzg
class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = Conv(in_features, filters, act=False)
         self.c1 = Conv(filters, filters, 3)
         self.c2 = Conv(filters, filters, 3)
         self.c3 = Conv(filters, filters, 3)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.drop = nn.Dropout2d(0.1)
         self.bn1 = nn.BatchNorm2d(filters)
         self.silu = nn.SiLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x

class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

class DASI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Conv2d(out_features, out_features, 1)
         self.conv = nn.Conv2d(out_features // 2, out_features // 4, 1)
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = nn.Conv2d(in_features[1], out_features, 1)
         self.skips_2 = nn.Conv2d(in_features[0], out_features, 1)
         self.skips_3 = nn.Conv2d(in_features[2], out_features, kernel_size=3, stride=2, dilation=2, padding=2)
         self.silu = nn.SiLU()

    def forward(self, x_list):
        # x_high, x, x_low = x_list
        x_low, x, x_high = x_list
        if x_high != None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x = self.skips(x)
        x_skip = x
        x = torch.chunk(x, 4, dim=1)
        if x_high == None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low == None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.silu(x)

        return x