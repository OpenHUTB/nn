import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Converse2D', 'ConverseBlock']

"""
# --------------------------------------------
# LayerNorm for Vision Normalization
# --------------------------------------------
"""
class LayerNorm(nn.Module):
    '''
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    '''

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

"""
# --------------------------------------------
# implementation of Converse2d operator ^_^
# --------------------------------------------
"""
class Converse2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding_mode='circular', eps=1e-5, act=False):
        super(Converse2D, self).__init__()
        """
        Converse2D Operator for Image Restoration Tasks.

        Args:
            x (Tensor): Input tensor of shape (N, in_channels, H, W), where
                        N is the batch size, H and W are spatial dimensions.
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the operation.
            kernel_size (int): Size of the kernel.
            scale (int): Upsampling factor. For example, `scale=2` doubles the resolution.
            padding_mode (str, optional): Padding method. One of {'reflect', 'replicate', 'circular', 'constant'}.
                                        Default is `circular`.
            eps (float, optional): Small value added to denominators for numerical stability.
                                Default is a small value like 1e-5.

        Returns:
            Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale), where spatial dimensions
                    are upsampled by the given scale factor.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  kernel_size
        self.scale = scale
        self.padding = kernel_size - 1
        self.padding_mode = padding_mode
        self.eps = eps


        # ensure depthwise
        assert self.out_channels == self.in_channels
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1,self.in_channels,-1), dim=-1).view(1, self.in_channels, self.kernel_size, self.kernel_size)

        self.act = nn.Identity()
        if act:
            self.act = act()
        
    def forward(self, x):

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)

        biaseps = torch.sigmoid(self.bias-9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
            # x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear',align_corners=False)
        # x = torch.zeros_like(x)

        FB = self.p2o(self.weight, (h*self.scale, w*self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
        
        FR = FBFy + torch.fft.fftn(biaseps * x, dim=(-2,-1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + biaseps)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR-FCBinvWBR) / biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding*self.scale:-self.padding*self.scale, self.padding*self.scale:-self.padding*self.scale]

        return self.act(out)

    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.
        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor
        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b


    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.
        Args:
            psf: NxCxhxw
            shape: [H, W]
        Returns:
            otf: NxCxHxWx2
        '''
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[...,:psf.shape[-2],:psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2]/2), -int(psf.shape[-1]/2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2,-1))

        return otf

    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros
        x: tensor image, NxCxWxH
        '''
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*scale, x.shape[3]*scale)).type_as(x)
        z[..., st::scale, st::scale].copy_(x)
        return z


"""
# --------------------------------------------
# implementation of Converse Block
# --------------------------------------------
"""
class ConverseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=1, padding_mode='replicate', eps=1e-5):
        super(ConverseBlock, self).__init__()
        """
        ConverseBlock: A Convolutional Block for Image Restoration using Converse2D Operations.

        This block consists of two main sub-blocks, each incorporating normalization, pointwise convolution,
        non-linearity, and (optionally) a custom reverse convolution (`Converse2D`) for learnable upsampling.
        It also includes residual connections to preserve information and improve gradient flow.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels to be produced by the block.
            kernel_size (int, optional): Kernel size used in the `Converse2D` operation. Default: 3.
            scale (int, optional): Upsampling scale factor. Default: 1 (no upsampling).
            padding_mode (str, optional): Padding mode to use in `Converse2D`. One of {'reflect', 'replicate', 'circular', 'constant'}. Default: 'circular'.
            eps (float, optional): A small epsilon value for numerical stability in normalization layers. Default: 1e-6.

        Forward:
            x (Tensor): Input tensor of shape (N, in_channels, H, W)
            Returns:
                Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale)
        """

        self.conv1 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(in_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   Converse2D(2*out_channels, 2*out_channels, kernel_size, scale=scale, padding_mode=padding_mode, eps=eps), 
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
                                  
        self.conv2 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(out_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
                                  
    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        return x