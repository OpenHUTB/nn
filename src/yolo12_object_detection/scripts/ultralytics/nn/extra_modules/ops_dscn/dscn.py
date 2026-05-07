# --------------------------------------------------------
# Modified from
# https://github.com/OpenGVLab/InternImage
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from .functions import DSCNFunction


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale

class DSCNX(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        remove_center=False,
    ):
        """
        DSCNX Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DSCN to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1,dw_kernel_size),
                stride=1,
                padding=(0,(dw_kernel_size - 1) // 2),
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)

    def forward(self, input, off_x):
        """
        :param query                       (N, C, H, W)
        :return output                     (N, C, H, W)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = self.dw_conv(off_x)
        offset = self.offset(x1)
        
        x = DSCNFunction.apply(
            x, offset,
            1, self.kernel_size,
            1, self.stride,
            0, self.pad,
            1, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center, True)
        return x
    
class DSCNY(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        remove_center=False,
    ):
        """
        DSCNY Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DSCN to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(dw_kernel_size,1),
                stride=1,
                padding=((dw_kernel_size - 1) // 2,0),
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))

        self.offset = nn.Linear(
            channels,
            group * (kernel_size - remove_center))
        self._reset_parameters()
        
    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)

    def forward(self, input, off_x):
        """
        :param query                       (N, C, H, W)
        :return output                     (N, C, H, W)
        """
        N, H, W, _ = input.shape

        x = input #self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = self.dw_conv(off_x)
        offset = self.offset(x1)
        
        x = DSCNFunction.apply(
            x, offset,
            self.kernel_size, 1,
            self.stride, 1,
            self.pad, 0,
            self.dilation, 1,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center, False)
        
        return x
