import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * res

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x_source

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttiton(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class HFFE(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(HFFE, self).__init__()
        feature_low_channel, feature_high_channel = in_channel
        self.conv_block_low = nn.Sequential(
            Conv(feature_low_channel, feature_low_channel // 16, kernel_size),
            nn.Conv2d(feature_low_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv_block_high = nn.Sequential(
            Conv(feature_high_channel, feature_high_channel // 16, kernel_size),
            nn.Conv2d(feature_high_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = Conv(feature_low_channel, out_channel, 1)
        self.conv2 = Conv(feature_high_channel, out_channel, 1)
        self.conv3 = Conv(feature_low_channel + feature_high_channel, out_channel, 1)

        self.Up_to_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.feature_low_sa = SpatialAttention()
        self.feature_high_sa = SpatialAttention()

        self.ca = CoordAttiton(out_channel,out_channel)

        self.conv_final = Conv(out_channel * 2, out_channel, 1)

    def forward(self, x):
        x_low, x_high = x
        b1, c1, w1, h1 = x_low.size()
        b2, c2, w2, h2 = x_high.size()
        if (w1, h1) != (w2, h2):
            x_high = F.interpolate(x_high, (w1, h1), mode='bilinear', align_corners=False)

        source_low = x_low
        source_high = x_high

        x_low = self.feature_low_sa(x_low)
        x_high = self.feature_high_sa(x_high)

        x_low_map = self.conv_block_low(x_low)
        x_high_map = self.conv_block_high(x_high)

        x_mix = torch.cat([source_low * x_high_map, source_high * x_low_map], 1)
        x_ca = torch.sigmoid(self.ca(self.conv3(x_mix)))

        x_low_att = x_ca * self.conv1((source_low + x_low))
        x_high_att = x_ca * self.conv2((source_high + x_high))

        out = self.conv_final(torch.cat([x_low_att, x_high_att], 1))

        return out