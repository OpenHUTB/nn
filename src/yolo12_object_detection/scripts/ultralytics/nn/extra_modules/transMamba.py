import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import repeat, rearrange

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from causal_conv1d import causal_conv1d_fn
except ImportError as e:
    pass

__all__ = ['TransMambaBlock', 'SpectralEnhancedFFN']

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # b,c,hw -> b,c,1
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1   = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.silu1 = nn.SiLU()
        self.fc2   = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # b,c,hw -> b,c,1
        avg_out = self.fc2(self.silu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.silu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # b,c,hw -> b,1,hw
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1) # b,1,hw -> b,2,hw
        x = self.conv1(x) # b,1,hw
        return self.sigmoid(x)

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

#        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, 1, bias=bias, )

        self.dwconv = nn.Conv2d(self.d_inner*2, self.d_inner*2, kernel_size=(5,5), stride=1, padding=(2,2), groups=self.d_inner*2, bias=bias)


        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.sa = SpatialAttention(7)
        self.ca = ChannelAttention(self.d_inner, )
        
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, dim, height, width = hidden_states.shape
        seqlen = height * width

        conv_state = None

        # We do matmul and transpose BLH -> HBL at the same time
        '''
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        '''
        xz = self.dwconv(self.in_proj(hidden_states))
        xz = rearrange(xz, "b d h w -> b d (h w)") 

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            '''
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            else:
            '''
            x, z = xz.chunk(2, dim=1)
            x = self.ca(x) * x
            z = self.sa(z) * z
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]

                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        out = rearrange(out, "b (h w) d -> b d h w", h=height, w=width)


        return out

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
# Spectral Enhanced Feed-Forward
class SpectralEnhancedFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SpectralEnhancedFFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=2, groups=hidden_features*2, bias=bias, dilation=2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]

    def forward(self, x):
        x_dtype = x.dtype
        x = self.project_in(x)
        x = self.dwconv(x)
        x, pad_w = self.pad(x,2)
        x = torch.fft.rfft2(x.float())
        x = self.fft_channel_weight * x + self.fft_channel_bias
#        x = torch.nn.functional.normalize(x, 1)
        x = torch.fft.irfft2(x)
        x = self.unpad(x, pad_w)
        x1, x2 = x.chunk(2, dim=1)
        
        x = F.silu(x1) * x2
        x = self.project_out(x.to(x_dtype))
        return x

##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.factor = 2
        self.idx_dict = {}
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]
        
    def comp2real(self, x):
        b, _, h, w = x.shape
        return torch.cat([x.real, x.imag], 1)
#        return torch.stack([x.real, x.imag], 2).view(b,-1,h,w)
    def real2comp(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.complex(xr, xi)

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def get_idx_map(self, h, w):
        l1_u = torch.arange(h//2).view(1,1,-1,1)
        l2_u = torch.arange(w).view(1,1,1,-1)
        half_map_u = l1_u @ l2_u
        l1_d = torch.arange(h - h//2).flip(0).view(1,1,-1,1)
        l2_d = torch.arange(w).view(1,1,1,-1)
        half_map_d = l1_d @ l2_d
        return torch.cat([half_map_u, half_map_d], 2).view(1,1,-1).argsort(-1)
    def get_idx(self, x):
        h, w = x.shape[-2:]
        if (h, w) in self.idx_dict:
            return self.idx_dict[(h, w)]
        idx_map = self.get_idx_map(h, w).to(x.device).detach()
        self.idx_dict[(h, w)] = idx_map
        return idx_map
    def attn(self, qkv):
        h = qkv.shape[2]
        q,k,v = qkv.chunk(3, dim=1)
        
        q, pad_w, idx = self.fft(q)
        q, pad = self.pad(q, self.factor)
        k, pad_w, _ = self.fft(k)
        k, pad = self.pad(k, self.factor)
        v, pad_w, _ = self.fft(v)
        v, pad = self.pad(v, self.factor)
        
        q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor)
        out = self.unpad(out, pad)
        out = self.ifft(out, pad_w, idx, h)
        return out
    def fft(self, x):
        x, pad = self.pad(x, 2)
        x = torch.fft.rfft2(x.float(), norm="ortho")
        x = self.comp2real(x)
        idx = self.get_idx(x).to(x.device)
        b, c = x.shape[:2]
        x = x.contiguous().view(b, c, -1)
        x = torch.gather(x, 2, index=idx.repeat(b,c,1)) # b, 6c, h*(w//2+1)
        return x, pad, idx
    def ifft(self, x, pad, idx, h):
        b, c = x.shape[:2]
        x = torch.scatter(x, 2, idx.repeat(b,c,1), x)
        x = x.view(b, c, h, -1)
        x = self.real2comp(x)
        x = torch.fft.irfft2( x, norm='ortho' )#.abs()
        x = self.unpad(x, pad)
        return x
    def forward(self, x):
        b,c,h,w = x.shape

        attn_map = x

        qkv = self.qkv_dwconv(self.qkv(x))

#        qkv, pad_w, idx = self.fft(qkv)
#        qkv, pad = self.pad(qkv, self.factor)

        attn_map = qkv  
        out = self.attn(qkv) 
        attn_map = out


#        out = self.unpad(out, pad)
#        out = self.ifft(out, pad_w, idx, h)

        out = self.project_out(out)
        attn_map = out
        return out

    '''
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    '''

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = SpectralEnhancedFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, LayerNorm_type,):
        super(MambaBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mamba1 = DRMamba(dim, reverse=False)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mamba2 = DRMamba(dim, reverse=True)# FeedForward(dim, ffn_expansion_factor, bias, True)

    def forward(self, x):
        x = x + self.mamba1(self.norm1(x))

        x = x + self.mamba2(self.norm2(x))

        return x

class DRMamba(nn.Module):
    def __init__(self, dim, reverse):
        super(DRMamba, self).__init__()
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.reverse = reverse

    def forward(self, x):
        b,c,h,w = x.shape
        if self.reverse:
            x = x.flip(1)
        x = self.mamba(x)
        if self.reverse:
            x = x.flip(1)
        return x

class TransMambaBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=1.5, bias=False, LayerNorm_type='BiasFree'):
        super(TransMambaBlock, self).__init__()

        self.trans_block = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.mamba_block = MambaBlock(dim, LayerNorm_type)
        self.conv = nn.Conv2d(int(dim*2), dim, kernel_size=1, bias=bias) 

    def forward(self, x):
        x1 = self.trans_block(x)
        x2 = self.mamba_block(x)
        out = torch.cat((x1, x2), 1)
        out = self.conv(out)
        return out
