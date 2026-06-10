#------------------------------------------------------------------#
# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
# HS-FPN
# ├── HFP (High Frequency Perception Module)
# │   ├── DctSpatialInteraction (Spatial Path of HFP)
# │   └── DctChannelInteraction (Channel Path of HFP)
# └── SDP&SDP_Large (Spatial Dependency Perception Module
#-----------------------------------------------------------------#

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_dct as DCT
except Exception as e:
    pass
from einops import rearrange
from ..modules.conv import Conv

__all__ =['HFP', 'SDP']

#------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctSpatialInteraction(nn.Module):
    def __init__(self,
                in_channels,
                ratio,
                isdct = True):
        super(DctSpatialInteraction, self).__init__()
        self.ratio = ratio
        self.isdct = isdct # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)]
        )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho') 
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


#------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctChannelInteraction(nn.Module):
    def __init__(self,
                in_channels, 
                patch,
                ratio,
                isdct=True
                ):
        super(DctChannelInteraction, self).__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.channel2x1 = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, 1, groups=32)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct: # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x,  output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x,  output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp)) # 2025 03 15 szc 
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') 

        amaxp = F.adaptive_max_pool2d(dct_,  output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_,  output_size=(self.h, self.w))       
        amaxp = torch.sum(self.relu(amaxp), dim=[2,3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2,3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp) # 2025 03 15 szc 
        return x * torch.sigmoid(self.channel2x1(channel))
        
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight  


#------------------------------------------------------------------#
# High Frequency Perception Module HFP
#------------------------------------------------------------------#
class HFP(nn.Module):
    def __init__(self, 
                in_channels,
                ratio = (0.25, 0.25),
                patch = (8,8),
                isdct = True):
        super(HFP, self).__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct) 
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
        self.out =  nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels)]
            )
    def forward(self, x):
        spatial = self.spatial(x) # output of spatial path
        channel = self.channel(x) # output of channel path
        return self.out(spatial + channel)


#------------------------------------------------------------------#
# Spatial Dependency Perception Module SDP
#------------------------------------------------------------------#
class SDP(nn.Module):
    def __init__(self,
                in_dim,
                dim=256,
                patch_size=None,
                inter_dim=None
                ):
        super(SDP, self).__init__()
        self.conv1x1_0 = Conv(in_dim[0], dim) if in_dim[0] != dim else nn.Identity()
        self.conv1x1_1 = Conv(in_dim[1], dim) if in_dim[1] != dim else nn.Identity()

        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.softmax = nn.Softmax(dim=-1)
        self.patch_size = patch_size
    def forward(self, x):
        x_low, x_high = x
        x_low = self.conv1x1_0(x_low)
        x_high = self.conv1x1_1(x_high)
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size[0], p2=self.patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=self.patch_size[0], p2=self.patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=self.patch_size[0], p2=self.patch_size[1], h=h_//self.patch_size[0], w=w_//self.patch_size[1])
        return output + x_low

#------------------------------------------------------------------#
# Improved Version of Spatial Dependency Perception Module SDP
# 2025 03 15 szc 
#------------------------------------------------------------------#
class SDP_Improved(nn.Module):
    def __init__(self,
                dim=256,
                inter_dim=None):
        super(SDP_Improved, self).__init__()
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv = nn.Sequential(*[nn.Conv2d(self.inter_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim)])
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
        output = self.conv(output + x_low)
        return output