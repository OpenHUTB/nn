import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelPool(nn.Module):
    def forward(self, x):
        # 将maxpooling 与 global average pooling 结果拼接在一起
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))


        self.conv_1x1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()

        self.F_h = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_w = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        x_1, x_2 = x.split(C, 1)

        x_1 = self.F_h(x_1)
        x_2 = self.F_w(x_2)
        s_h = self.sigmoid(x_1)
        s_w = self.sigmoid(x_2)

        out = res * s_h.expand_as(res) * s_w.expand_as(res)

        return out

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class CSSC(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(CSSC, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(in_channel)
        )
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(in_channel, reduction)  ## Channel Attention

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        self.tail = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)  # 拼接
        x1 = self.conv1x1(x1)
        x2 = torch.sigmoid(
            torch.add(x, F.interpolate(self.SC(x), x.size()[2:])))
        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out

class CNCM(nn.Module):
    def __init__(self, channel_in, reduction=8):
        super(CNCM, self).__init__()

        # RCSSC
        self.unit_1 = CSSC(int(channel_in / 2.), reduction)
        self.unit_2 = CSSC(int(channel_in / 2.), reduction)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                      padding=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=1, padding=0,
                      stride=1),  # 做压缩
            nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        residual = x
        c1 = self.unit_1(self.conv1(x))
        x = torch.cat([residual, c1], 1)
        c2 = self.unit_2(self.conv2(x))
        x = torch.cat([c2, x], 1)
        x = self.conv3(x)
        x = torch.add(x, residual)
        return x