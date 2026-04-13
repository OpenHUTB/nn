import math
import torch
import torch.nn as nn

class DWConv(nn.Module):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, dilation, bias=bias, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class PWConv(nn.Module):
    def __init__(self, in_channels=768, out_channels=None, bias=True):
        super(PWConv, self).__init__()
        out_channels = out_channels or in_channels
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.pwconv(x)
        return x

class CKS(nn.Module):
    def __init__(self, dim, dwconv=DWConv, pwconv=PWConv):
        super().__init__()
        hidden_channel = max(dim//16, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_reduce = nn.Sequential(
            pwconv(dim, hidden_channel),
            dwconv(hidden_channel),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)
        )
        self.fc_expand_small = nn.Sequential(
            pwconv(hidden_channel, dim),
            dwconv(dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.fc_expand_big_h = nn.Sequential(
            pwconv(hidden_channel, dim),
            dwconv(dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.fc_expand_big_v = nn.Sequential(
            pwconv(hidden_channel, dim),
            dwconv(dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv_squeeze = nn.Conv2d(2, 3, 7, padding=3)

    def norm_weight_channel(self, sig_small, sig_big_h, sig_big_v):
        sig_small = torch.sigmoid(sig_small)
        sig_big_h = torch.sigmoid(sig_big_h)
        sig_big_v = torch.sigmoid(sig_big_v)
        return sig_small, sig_big_h, sig_big_v

    def norm_weight_spatial(self, sig):
        sig = torch.sigmoid(sig)
        sig_small, sig_big_h, sig_big_v = torch.chunk(sig, chunks=3, dim=1)
        return sig_small, sig_big_h, sig_big_v
    
    def forward(self, attn_small, attn_big_h, attn_big_v):
        attn = attn_small + attn_big_h + attn_big_v
        feats_S = self.avg_pool(attn)
        feats_Z = self.fc_reduce(feats_S)
        sig_small_channel = self.fc_expand_small(feats_Z)
        sig_big_h_channel = self.fc_expand_big_h(feats_Z)
        sig_big_v_channel = self.fc_expand_big_v(feats_Z)
        sig_small_channel, sig_big_h_channel, sig_big_v_channel = self.norm_weight_channel(sig_small_channel, sig_big_h_channel, sig_big_v_channel)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg)
        sig_small, sig_big_h, sig_big_v = self.norm_weight_spatial(sig)
        attn = attn_small * sig_small_channel * sig_small + attn_big_h * sig_big_h_channel * sig_big_h + attn_big_v * sig_big_v_channel * sig_big_v
        return attn

class SDLSKA(nn.Module):
    def __init__(self, dim, kernel_size, dwconv=DWConv, pwconv=PWConv):
        super().__init__()
        if kernel_size == 7 or kernel_size == 11:
            self.DW_conv = dwconv(dim, kernel_size=(3, 3), stride=(1,1), padding=(1,1))
        else:
            self.DW_conv = dwconv(dim, kernel_size=(5, 5), stride=(1,1), padding=(2,2))
        if kernel_size == 7:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), dilation=2)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), dilation=2)
        elif kernel_size == 11:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), dilation=2)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), dilation=2)
        elif kernel_size == 23:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), dilation=3)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), dilation=3)
        elif kernel_size == 35:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), dilation=3)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), dilation=3)
        self.pwconv = pwconv(dim)
        self.FS_kernel = CKS(dim, dwconv=DWConv, pwconv=PWConv)

    def forward(self, x):
        u = x.clone()
        attn_small = self.DW_conv(x)
        attn_big_h = self.DW_D_conv_h(attn_small)
        attn_big_v = self.DW_D_conv_v(attn_small)
        attn = self.FS_kernel(attn_small, attn_big_h, attn_big_v)
        attn = self.pwconv(attn)
        return u * attn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dwconv=DWConv, pwconv=PWConv, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = pwconv(in_features, hidden_features)
        self.dwconv = dwconv(hidden_features)
        self.act = act_layer()
        self.fc2 = pwconv(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, lka, d_model, kernel_size, dwconv=DWConv, pwconv=PWConv):
        super().__init__()
        self.proj_1 = pwconv(d_model, d_model)
        self.activation = nn.GELU()
        self.spatial_gating_unit = lka(d_model, kernel_size, dwconv=dwconv, pwconv=pwconv)
        self.proj_2 = pwconv(d_model, d_model)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)
    

class EVA(nn.Module):
    """Implementation of Efficient Visual Attention Network Block."""
    def __init__(self, dim, kernel_size=35, lka=SDLSKA, dwconv=DWConv, pwconv=PWConv, mlp_expand=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(lka, dim, kernel_size, dwconv=dwconv, pwconv=pwconv)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_expand)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x