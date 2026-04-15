import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from ..modules.conv import Conv

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP,self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)  #TODO,1e-6?
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class ConvolutionalAttention(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        torch.nn.init.trunc_normal_(self.kv, std=0.001)
        torch.nn.init.trunc_normal_(self.kv3, std=0.001)

    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)   
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)  
        x = x.reshape([x_shape[0], self.inter_channels, h, w]) 
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w)
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        x1 = F.conv2d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=(3,0))  
        x1 = self._act_dn(x1)  
        x1 = F.conv2d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=(3,0))  
        x3 = F.conv2d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=(0,3)) 
        x3 = self._act_dn(x3)
        x3 = F.conv2d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,3)) 
        x=x1+x3
        return x

class CFBlock(nn.Module):
    """
    The CFBlock implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.):
        super(CFBlock,self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.conv1x1 = Conv(in_channels_l, out_channels_l, 1) if in_channels_l != out_channels_l else nn.Identity()
        self.attn_l = ConvolutionalAttention(
            out_channels_l,
            out_channels_l,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv1x1(x)
        x_res = x
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x)) 
        return x