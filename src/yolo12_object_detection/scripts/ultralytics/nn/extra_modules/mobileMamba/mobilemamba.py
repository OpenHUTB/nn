import torch
import itertools
import torch.nn as nn
from timm.layers import SqueezeExcite, trunc_normal_, DropPath
from .vmambanew import SS2D
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data

__all__ = ['MobileMambaBlock', 'MobileMamba_T2', 'MobileMamba_T4', 'MobileMamba_S6', 'MobileMamba_B1', 'MobileMamba_B2', 'MobileMamba_B4']

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
        super(MBWTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.global_atten =SS2D(d_model=in_channels, d_state=1,
             ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.global_atten(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0,)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple

class MobileMambaModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(self.global_channels, self.global_channels, kernels, wt_levels=1, ssm_ratio=ssm_ratio, forward_type=forward_type,)
        else:
            self.global_op = nn.Identity()

        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0,))

    def forward(self, x):  # x (B,C,H,W)
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        x1 = self.global_op(x1)
        x2 = self.local_op(x2)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))
        return x


class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.attn = MobileMambaModule(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                                           kernels=kernels, ssm_ratio=ssm_ratio, forward_type=forward_type,)

    def forward(self, x):
        x = self.attn(x)
        return x


class MobileMambaBlock(torch.nn.Module):
    def __init__(self, ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5,  drop_path=0., has_skip=True, ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        self.mixer = Residual(MobileMambaBlockWindow(ed, global_ratio=global_ratio, local_ratio=local_ratio, kernels=kernels, ssm_ratio=ssm_ratio,forward_type=forward_type))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.,))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x

class MobileMamba(torch.nn.Module):
    def __init__(self, img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[192, 384, 448],
                 global_ratio=[0.8, 0.7, 0.6],
                 local_ratio=[0.2, 0.2, 0.3],
                 depth=[1, 2, 2],
                 kernels=[7, 5, 3],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False, drop_path=0., ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1,
                                                         ), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1,
                                                         ), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 1, 1,
                                                         ))

        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depth))]

        # Build MobileMamba blocks
        for i, (stg, ed, dpth, gr, lr, do) in enumerate(
                zip(stages, embed_dim, depth, global_ratio, local_ratio, down_ops)):
            dpr = dprs[sum(depth[:i]):sum(depth[:i + 1])]
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(MobileMambaBlock(ed, gr, lr, kernels[i], dpr[d], ssm_ratio=ssm_ratio, forward_type=forward_type))
            if do[0] == 'subsample':
                # Build MobileMamba downsample block
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                                               Residual(FFN(embed_dim[i], int(embed_dim[i] * 2))), ))
                blk.append(PatchMerging(*embed_dim[i:i + 2]))
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1],)),
                                               Residual(
                                                   FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2))), ))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        x1 = self.patch_embed(x)
        x2 = self.blocks1(x1)
        x3 = self.blocks2(x2)
        x4 = self.blocks3(x3)
        return [x1, x2, x3, x4]

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

CFG_MobileMamba_T2 = {
        'img_size': 192,
        'embed_dim': [144, 272, 368],
        'depth': [1, 2, 2],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0,
        'ssm_ratio': 2,
    }
CFG_MobileMamba_T4 = {
        'img_size': 192,
        'embed_dim': [176, 368, 448],
        'depth': [1, 2, 2],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0,
        'ssm_ratio': 2,
    }
CFG_MobileMamba_S6 = {
        'img_size': 224,
        'embed_dim': [192, 384, 448],
        'depth': [1, 2, 2],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0,
        'ssm_ratio': 2,
    }
CFG_MobileMamba_B1 = {
        'img_size': 256,
        'embed_dim': [200, 376, 448],
        'depth': [2, 3, 2],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0.03,
        'ssm_ratio': 2,
    }
CFG_MobileMamba_B2 = {
        'img_size': 384,
        'embed_dim': [200, 376, 448],
        'depth': [2, 3, 2],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0.03,
        'ssm_ratio': 2,
    }
CFG_MobileMamba_B4 = {
        'img_size': 512,
        'embed_dim': [200, 376, 448],
        'depth': [2, 3, 2],
        'global_ratio': [0.8, 0.7, 0.6],
        'local_ratio': [0.2, 0.2, 0.3],
        'kernels': [7, 5, 3],
        'drop_path': 0.03,
        'ssm_ratio': 2,
    }

def MobileMamba_T2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_T2):
    model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

def MobileMamba_T4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_T4):
    model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

def MobileMamba_S6(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_S6):
    model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

def MobileMamba_B1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_B1):
    model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

def MobileMamba_B2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_B2):
    model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model

def MobileMamba_B4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=CFG_MobileMamba_B4):
    model = MobileMamba(num_classes=num_classes, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model