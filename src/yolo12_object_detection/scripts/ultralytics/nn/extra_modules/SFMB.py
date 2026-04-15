import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
from timm.layers import DropPath
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError as e:
    pass

class SSM2D_MB(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return y


class MB(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dropout=0.,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.out_norm1 = nn.LayerNorm(self.d_inner)
        self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_norm3 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=True,
                kernel_size=3,
                padding=(3 - 1) // 2, ),
            nn.SiLU())
        self.ca2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=True,
                kernel_size=3,
                padding=(3 - 1) // 2, ),
            nn.SiLU())
        self.ca3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=True,
                kernel_size=3,
                padding=(3 - 1) // 2, ),
            nn.SiLU())
        self.ssm1 = SSM2D_MB(d_model=self.d_model, d_state=self.d_state, expand=self.expand,
                             **kwargs)
        self.ssm2 = SSM2D_MB(d_model=self.d_model, d_state=self.d_state, expand=self.expand,
                             **kwargs)
        self.ssm3 = SSM2D_MB(d_model=self.d_model, d_state=self.d_state, expand=self.expand,
                             **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        skip = x
        x_dtype = x.dtype
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.ca1(x)
        x2 = self.pooling(x)
        y1 = self.ssm1(x)
        y1 = self.out_norm1(y1.to(x_dtype))

        x2 = self.ca2(x2)
        x3 = self.pooling(x2)
        y2 = self.ssm2(x2)
        y2 = self.out_norm2(y2.to(x_dtype))

        x3 = self.ca3(x3)
        y3 = self.ssm3(x3)
        y3 = self.out_norm3(y3.to(x_dtype))

        y3 = y3.permute(0, 3, 1, 2).contiguous()
        y3 = self.up(y3)
        y2 = y2.permute(0, 3, 1, 2).contiguous()
        y2 = y2 + y3
        y2 = self.up(y2)
        y2 = y2.permute(0, 2, 3, 1).contiguous()
        y = y1 + y2

        y = y * F.silu(z)
        out = self.out_proj(y)
        out = out + skip
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class AB(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_feat // 2, out_channels=num_feat // 2, kernel_size=1, padding=0)
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):  # 输入 BHWC 输出 BHWC
        x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        x = self.conv(x)
        skip = x
        x1, x2 = torch.split(x, C // 2, dim=1)
        avg_out = self.mlp1(self.avg_pool(x1))
        max_out = self.mlp2(self.max_pool(x2))
        y1 = self.sigmoid1(avg_out)
        y2 = self.sigmoid2(max_out)
        z = torch.cat((x1 * y1, x2 * y2), dim=1)
        perm = torch.randperm(C)
        z = z[:, perm, :, :]
        z = z + skip
        z = z.permute(0, 2, 3, 1).contiguous()
        return z


class FB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.select1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.select2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):  # 输入 BHWC 输出 BHWC
        B, H, W, C = x.shape
        skip = x
        x_dtype = x.dtype
        x = x.permute(0, 3, 1, 2).contiguous()
        y = torch.fft.rfft2(x.float()) + 1e-8
        a = torch.abs(y)
        p = torch.angle(y)
        a = self.select1(a.to(x_dtype))
        p = self.select2(p.to(x_dtype))
        real = a * torch.cos(p)
        imag = a * torch.sin(p)
        out = torch.complex(real.float(), imag.float()) + 1e-8
        out = torch.fft.irfft2(out, s=(H, W), norm='backward') + 1e-8
        out = torch.abs(out) + 1e-8

        out = out.permute(0, 2, 3, 1).contiguous().to(x_dtype)
        out = out + skip

        return out

class SFMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln = norm_layer(hidden_dim)
        self.mamba = MB(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate,
                        **kwargs)
        self.drop_path = DropPath(drop_path)
        self.attention = AB(hidden_dim)
        self.frequency = FB(hidden_dim)

    def forward(self, input):
        input = input.permute(0, 2, 3, 1)
        x = self.ln(input)
        x = self.drop_path(self.mamba(x)) + self.attention(x) + self.frequency(x)
        return x.permute(0, 3, 1, 2)