# Code Implementation of the MaIR Model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
from timm.layers import DropPath
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception as e:
    pass
from einops import rearrange, repeat

__all__ = ['RMB']

def test_crop_by_horz(inp, scan_len):
    # Flip the return way
    split_inp = rearrange(inp, "h (d1 w) -> d1 h w ", w=scan_len)
    for i in range(1, len(split_inp), 2):
        split_inp[i, :] = split_inp[i, :].flip(dims=[-2])
    inp = rearrange(split_inp, " d1 h w -> (d1 h) w ")
    # print(inp)

    inp_window = rearrange(inp, "(d1 h) (d2 w) -> (d2 d1) h w ", h=2, w=scan_len)

    inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(1, -1)
    print(inp_flatten)
    print(inp_flatten.shape)

def chw_2d(h, w):
    return torch.arange(1, (h*w+1)).reshape(h, w)

def chw_3d(c, h, w):
    return torch.arange(1, (c*h*w+1)).reshape(c, h, w)

def chw_4d(b, c, h, w, random=False):
    if random:
        return torch.randn(b*c*h*w).reshape(b, c, h, w)
    else:
        return torch.arange(1, (b*c*h*w+1)).reshape(b, c, h, w)

def create_idx(b, c, h, w):
    # return torch.arange(1, (b*c*h*w+1)).reshape(b, c, h, w)
    return torch.arange(b*c*h*w).reshape(b, c, h, w)

def test_2d_horz(inp_h, inp_w):
    scan_len = 2
    # inp_h, inp_w = 4, 4
    # inp  = torch.randn((4,4))
    inp = torch.tensor([[ 1,  2,  3,  4],
                        [ 5,  6,  7,  8],
                        [ 9,  10, 11, 12],
                        [ 13, 14, 15, 16]])
    inp = chw_2d(inp_h, inp_w)
    print(inp)
    test_crop_by_horz(inp, scan_len)

def sscan_einops(inp, scan_len):
    B, C, H, W = inp.shape
    # Flip the return way
    split_inp = rearrange(inp, "b c h (d1 w) -> d1 b c h w ", w=scan_len)
    for i in range(1, len(split_inp), 2):
        split_inp[i, :] = split_inp[i, :].flip(dims=[-2])
    reverse_inp = rearrange(split_inp, " d1 b c h w -> b c (d1 h) w ")
    # print(inp)

    inp_window = rearrange(reverse_inp, "b c (d1 h) (d2 w) -> (b c d2 d1) h w ", h=2, w=scan_len)

    inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(B, C, 1, -1)
    # print(inp_flatten)
    # print(inp_flatten.shape)

    return inp_flatten

def sscan(inp, scan_len, shift_len=0):
    B, C, H, W = inp.shape
    # Flip the return way
    # 将返回的时候的列，上下翻转
    if shift_len == 0:
        for i in range(1, (W // scan_len)+1, 2):
            # for j in range(scan_len):
            inp[:, :, :, i*scan_len:(i+1)*scan_len] = inp[:, :, :, i*scan_len:(i+1)*scan_len].flip(dims=[-2])
    else:
        for i in range(0, ((W-shift_len) // scan_len) +1, 2):
            inp[:, :, :,(shift_len+i*scan_len):(shift_len+(i+1)*scan_len)] = inp[:, :, :, (shift_len+i*scan_len):(shift_len+(i+1)*scan_len)].flip(dims=[-2])


    # 将当前return way的sub-line翻转
    # inp_window = rearrange(inp, "b c (d1 h) (d2 w) -> (b c d2 d1) h w ", h=2, w=scan_len)
    if shift_len == 0:
        for hi in range((H // 2)):
            for wi in range(W // scan_len):
                inp[:, :, 2*hi+1, wi*scan_len:(wi+1)*scan_len] = inp[:, :, 2*hi+1, wi*scan_len:(wi+1)*scan_len].flip(dims=[-1])
    else:
        for hi in range((H // 2)):
            inp[:, :, 2*hi+1, 0:shift_len] = inp[:, :, 2*hi+1, 0:shift_len].flip(dims=[-1])

            for wi in range((W-shift_len) // scan_len):
                start_ = shift_len + wi*scan_len
                end_ = shift_len + (wi+1)*scan_len
                inp[:, :, 2*hi+1, start_:end_] = inp[:, :, 2*hi+1, start_:end_].flip(dims=[-1])

        
    if (W-shift_len) % scan_len:
        # inp_last = inp[:,:,:,-(W % scan_len):].reshape(B, C, -1)
        inp_last = inp[:,:,:,-((W-shift_len) % scan_len):]
        inp_last[:,:, 1::2, :] =  inp_last[:,:, 1::2, :].flip(dims=[-1]) # 取偶数位，奇数位是::2
        inp_last = inp_last.reshape(B, C, -1)

        inp_rest = inp[:,:,:,:-((W-shift_len) % scan_len)]
    else:
        inp_rest = inp

    if shift_len==0:
        inp_window = rearrange(inp_rest, "b c h (d2 w) -> (b c d2) h w ", w=scan_len)
    else:
        inp_first = inp_rest[:,:,:,:shift_len].reshape(B, C, -1)

        inp_middle = inp_rest[:,:,:, shift_len:]
        inp_window = rearrange(inp_middle, "b c h (d2 w) -> (b c d2) h w ", w=scan_len)

    # inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(B, C, -1)
    # inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    # inp_flatten = inp.reshape(B, C, 1, -1)
    # print(inp_flatten)
    # print(inp_flatten.shape)
    if shift_len != 0:
        inp_flatten = torch.concat((inp_first, inp_flatten), dim=-1)

    if (W-shift_len) % scan_len:
        inp_flatten = torch.concat((inp_flatten, inp_last), dim=-1)
        # print(inp_last.shape)
    return inp_flatten


# def sscan_4d(inp, scan_len, ues_einops=True, fix_ending=True):
def sscan_4d(inp, scan_len, shift_len=0, fix_ending=True, use_einops=False):
    B, C, H, W = inp.shape
    L = H * W
    if fix_ending:
        inp_reverse = torch.flip(inp, dims=[-1,-2])
        inp_cat = torch.concat((inp, inp_reverse), dim=1)
        inp_cat_t = inp_cat.transpose(-1, -2).contiguous()

        if use_einops:
            line1 = sscan_einops(inp_cat, scan_len)
            line2 = sscan_einops(inp_cat_t, scan_len)
        else:
            line1 = sscan(inp_cat, scan_len, shift_len)
            line2 = sscan(inp_cat_t, scan_len, shift_len)

        xs = torch.stack([line1.reshape(B, 2, -1, L), line2.reshape(B, 2, -1, L)], dim=1).reshape(B, 4, -1, L)
    else:
        inp_t = inp.transpose(-1, -2).contiguous()
        if use_einops:
            line1 = sscan_einops(inp, scan_len)
            line2 = sscan_einops(inp_t, scan_len)
        else:
            line1 = sscan(inp, scan_len, shift_len)
            line2 = sscan(inp_t, scan_len, shift_len)

        x_hwwh = torch.stack([line1.reshape(B, -1, L), line2.reshape(B, -1, L)], dim=1).reshape(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) 
    # print(xs)
    return xs

def inverse_ids_generate(origin_ids, K=4):
    '''
        Input: origin_ids: (B, K, C, L)
        Output: (B, K, C, L)
        Note: C is set to 1 for speeding up.
    '''
    inverse_ids = torch.argsort(origin_ids, dim=-1)
    return inverse_ids


def mair_ids_generate(inp_shape, scan_len=4, K=4):
    inp_b, inp_c, inp_h, inp_w = inp_shape

    # inp_idx = create_idx(1, inp_c, inp_h, inp_w)
    inp_idx = create_idx(1, 1, inp_h, inp_w)

    xs_scan_ids = sscan_4d(inp_idx, scan_len)[0]

    xs_inverse_ids = inverse_ids_generate(xs_scan_ids, K=K)

    return xs_scan_ids, xs_inverse_ids


def mair_shift_ids_generate(inp_shape, scan_len=4, shift_len=0, K=4):
    inp_b, inp_c, inp_h, inp_w = inp_shape

    # create_idx函数运行时间：0.0050699710845947266 秒
    # start_time = time.time()
    inp_idx = create_idx(1, 1, inp_h, inp_w)
    # print(f"create_idx函数运行时间：{time.time() - start_time} 秒")

    # start_time = time.time()
    xs_scan_ids = sscan_4d(inp_idx, scan_len, shift_len=shift_len)[0]
    # print(f"sscan_4d函数运行时间：{time.time() - start_time} 秒")

    # xs_scan_ids函数运行时间：0.05201005935668945 秒
    # start_time = time.time()
    xs_scan_ids = xs_scan_ids.repeat(inp_b, 1, 1, 1)
    # print(f"xs_scan_ids函数运行时间：{time.time() - start_time} 秒")

    # start_time = time.time()
    xs_inverse_ids = inverse_ids_generate(xs_scan_ids, K=K)
    # print(f"inverse_ids_generate函数运行时间：{time.time() - start_time} 秒")

    return xs_scan_ids, xs_inverse_ids


def mair_ids_scan(inp, xs_scan_ids, bkdl=False, K=4):
    '''
        inp: B, C, H, W
        xs_scan_ids: K, 1, L
    '''
    B, C, H, W = inp.shape
    L = H * W
    xs_scan_ids = xs_scan_ids.reshape(K, L)

    y1 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[0])
    y2 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[1])
    y3 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[2])
    y4 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[3])

    if bkdl:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1)
    else:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1).reshape(B, 4, -1)
    return inp_flatten

def mair_ids_inverse(inp, xs_scan_ids, shape=None):
    '''
        inp: (B, K, -1, L)
        xs_scan_ids: (1, K, 1, L)
    '''
    B, K, _, L = inp.shape
    xs_scan_ids = xs_scan_ids.reshape(K, L)
    if not shape:
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, L)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, L)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, L)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, L)
    else:
        B, C, H, W = shape
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, H, W)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, H, W)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, H, W)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, H, W)
    return torch.cat((y1,y2,y3,y4), dim=1)

class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, group=4):
        super().__init__()
        self.group = group
        self.in_features = in_features
        self.out_features = out_features
        
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x
    
    def channel_rearrange(self,x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)

        return x

    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., input_resolution=(64,64)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.input_resolution = input_resolution
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += 2 * H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features

        return flops


class VMM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            input_resolution=(64, 64),
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.input_resolution = input_resolution

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
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
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.gating = ShuffleAttn(in_features=self.d_inner*4, out_features=self.d_inner*4, group=self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, 
                     mair_ids,
                     x_proj_bias: torch.Tensor=None,
                     ):
        # print(x.shape) C=360
        B, C, H, W = x.shape
        L = H * W
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        K=4
        # print("hello")
        xs = mair_ids_scan(x, mair_ids[0])

        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight.reshape(-1, D, 1), bias=(x_proj_bias.reshape(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        out_y = self.selective_scan(
            xs, dts,
            -torch.exp(self.A_logs.float()).view(-1, self.d_state), Bs, Cs, self.Ds.float().view(-1), z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return mair_ids_inverse(out_y, mair_ids[1], shape=(B, -1, H, W)) #B, C, L

    def forward(self, x: torch.Tensor, mair_ids, **kwargs):
        B, H, W, C = x.shape
        x_dtype = x.dtype

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, mair_ids)
        # assert y.dtype == torch.float32
        y = y.to(x_dtype)
        y = y * self.gating(y)
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y = y1 + y2 + y3 + y4
        y = y.permute(0, 2, 3, 1).contiguous()
        
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout()
        return y


class RMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            inp_shape: tuple = None,
            shift_size=0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            ssm_ratio: float = 2.,
            mlp_ratio=1.5,
            scan_len=8,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VMM(d_model=hidden_dim, d_state=d_state,expand=ssm_ratio,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.conv_blk = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim)
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim

        self.shift_size = shift_size
        self.scan_len = scan_len

        self.inp_shape = inp_shape
        self._generate_ids()

    def _generate_ids(self):
        H,W = self.inp_shape

        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)# [B,H,W,C]
        self.xs_scan_ids = xs_scan_ids
        self.xs_inverse_ids = xs_inverse_ids

        xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        self.xs_shift_scan_ids = xs_shift_scan_ids
        self.xs_shift_inverse_ids = xs_shift_inverse_ids

        del xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids

    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()

        self.xs_shift_scan_ids = self.xs_shift_scan_ids.to(input.device)
        self.xs_shift_inverse_ids = self.xs_shift_inverse_ids.to(input.device)
        self.xs_scan_ids = self.xs_scan_ids.to(input.device)
        self.xs_inverse_ids = self.xs_inverse_ids.to(input.device)

        x = self.ln_1(input)
        if self.shift_size > 0:
            x = input*self.skip_scale + self.drop_path(self.self_attention(x, (self.xs_shift_scan_ids, self.xs_shift_inverse_ids)))
        else:
            x = input*self.skip_scale + self.drop_path(self.self_attention(x, (self.xs_scan_ids, self.xs_inverse_ids)))
        
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x))

        x = x.permute(0, 3, 1, 2).contiguous()

        return x