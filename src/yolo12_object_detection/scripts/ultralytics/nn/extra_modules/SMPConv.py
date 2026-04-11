import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from ..modules import Conv
from timm.layers import trunc_normal_, DropPath

try:
    from depthwise_conv2d_implicit_gemm import _DepthWiseConv2dImplicitGEMMFP16, _DepthWiseConv2dImplicitGEMMFP32
except ImportError as e:
    pass

__all__ = ['SMPBlock', 'SMPCNN_ConvFFN']

def rel_pos(kernel_size):
    tensors = [torch.linspace(-1, 1, steps=kernel_size) for _ in range(2)]
    kernel_coord = torch.stack(torch.meshgrid(*tensors), dim=-0)
    kernel_coord = kernel_coord.unsqueeze(0)
    return kernel_coord


class SMPConv(nn.Module):
    def __init__(self, planes, kernel_size, n_points, stride, padding, groups):
        super().__init__()

        self.planes = planes
        self.kernel_size = kernel_size
        self.n_points = n_points
        self.init_radius = 2 * (2/kernel_size)

        # kernel_coord
        kernel_coord = rel_pos(kernel_size)
        self.register_buffer('kernel_coord', kernel_coord)

        # weight_coord
        weight_coord = torch.empty(1, n_points, 2)
        nn.init.trunc_normal_(weight_coord, std=0.2, a=-1., b=1.)
        self.weight_coord = nn.Parameter(weight_coord)

        self.radius = nn.Parameter(torch.empty(1, n_points).unsqueeze(-1).unsqueeze(-1))
        self.radius.data.fill_(value=self.init_radius)

        # weight
        weights = torch.empty(1, planes, n_points)
        trunc_normal_(weights, std=.02)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        kernels = self.make_kernels().unsqueeze(1)
        x = x.contiguous()
        kernels = kernels.contiguous()

        if x.dtype == torch.float32:
            x = _DepthWiseConv2dImplicitGEMMFP32.apply(x, kernels)
        elif x.dtype == torch.float16:
            x = _DepthWiseConv2dImplicitGEMMFP16.apply(x, kernels)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
        return x        

    def make_kernels(self):
        diff = self.weight_coord.unsqueeze(-2) - self.kernel_coord.reshape(1,2,-1).transpose(1,2)  # [1, n_points, kernel_size^2, 2]
        diff = diff.transpose(2,3).reshape(1, self.n_points, 2, self.kernel_size, self.kernel_size)
        diff = F.relu(1 - torch.sum(torch.abs(diff), dim=2) / self.radius)  # [1, n_points, kernel_size, kernel_size]
        
        # Apply weighted diff for average weighted kernel
        # non_zero = (diff != 0) # [1, n_points, kernel_size, kernel_size]
        # count_weight = 1 / (torch.sum(non_zero, dim=1, keepdim=True) + 1e-6)  # [1, 1, kernel_size, kernel_size]
        # weighted_diff = count_weight * diff  # [1, n_points, kernel_size, kernel_size]

        kernels = torch.matmul(self.weights, diff.reshape(1, self.n_points, -1)) # [1, planes, kernel_size*kernel_size]
        kernels = kernels.reshape(1, self.planes, *self.kernel_coord.shape[2:]) # [1, planes, kernel_size, kernel_size]
        kernels = kernels.squeeze(0)
        kernels = torch.flip(kernels.permute(0,2,1), dims=(1,))
        return kernels
    
    def radius_clip(self, min_radius=1e-3, max_radius=1.):
        r = self.radius.data
        r = r.clamp(min_radius, max_radius)
        self.radius.data = r


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, n_points=None):
    if n_points != None and in_channels == out_channels and out_channels == groups and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        # print("SMPConv")
        return SMPConv(in_channels, kernel_size, n_points, stride, padding, groups)
    else:
        # print("Original convolution")
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)


use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True


def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, 
                                         n_points=n_points))
    result.add_module('bn', get_bn(out_channels))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation,
                                         n_points=n_points)
    result.add_module('nonlinear', nn.ReLU())
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class SMPCNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, n_points=None, n_points_divide=4):
        super().__init__()
        self.kernel_size = kernel_size
        if n_points == None:
            n_points = int((kernel_size**2) // n_points_divide)

        padding = kernel_size // 2
        self.smp = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=1, groups=groups, n_points=n_points)
        
        self.small_kernel = 5
        # self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=self.small_kernel,
        #                            stride=stride, padding=self.small_kernel//2, groups=groups)
        self.small_conv = Conv(in_channels, out_channels, self.small_kernel, stride, self.small_kernel // 2, groups, act=False)

    def forward(self, inputs):
        out = self.smp(inputs)
        out += self.small_conv(inputs)
        return out


class SMPCNN_ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = get_bn(in_channels)
        # self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        # self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw1 = Conv(in_channels, internal_channels, act=False)
        self.pw2 = Conv(internal_channels, out_channels, act=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class SMPBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, lk_size, drop_path, n_points=None, n_points_divide=4):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = SMPCNN(in_channels=dw_channels, out_channels=dw_channels, kernel_size=lk_size,
                                  stride=1, groups=dw_channels, n_points=n_points, n_points_divide=n_points_divide)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
