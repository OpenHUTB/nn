
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Optional, Tuple, Union, List
from ..modules.conv import Conv

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Block1x1(nn.Module):
    """The 1x1_1x1 path of the GCBlock.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            stride (int or tuple): Stride of the convolution. Default: 1
            padding (int, tuple): Padding added to all four sides of
                the input. Default: 1
            bias (bool) : Whether to use bias.
                Default: True
            norm_cfg (dict): Config dict to build norm layer.
                Default: dict(type='BN', requires_grad=True)
            deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 deploy: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = False
        self.deploy = deploy

        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=True)
        else:
            self.conv1 = Conv(
                in_channels,
                out_channels,
                k=1,
                s=stride,
                p=padding,
                act=False)
            self.conv2 = Conv(
                out_channels,
                out_channels,
                k=1,
                s=1,
                p=padding,
                act=False)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        return x

    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std if self.bias else beta - running_mean * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class Block3x3(nn.Module):
    """The 3x3_1x1 path of the GCBlock.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            stride (int or tuple): Stride of the convolution. Default: 1
            padding (int, tuple): Padding added to all four sides of
                the input. Default: 1
            bias (bool) : Whether to use bias.
                Default: True
            norm_cfg (dict): Config dict to build norm layer.
                Default: dict(type='BN', requires_grad=True)
            deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 deploy: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.bias = False
        self.deploy = deploy

        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=True)
        else:
            self.conv1 = Conv(
                in_channels,
                out_channels,
                k=3,
                s=stride,
                p=padding,
                act=False)
            self.conv2 = Conv(
                out_channels,
                out_channels,
                k=1,
                s=1,
                p=0,
                act=False)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        return x

    def _fuse_bn_tensor(self, conv: nn.Module):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std if self.bias else beta - running_mean * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride,
                              padding=self.padding, bias=True)

        self.conv.weight.data = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)

        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


class GCConv(nn.Module):
    """GCConv.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True)
        act (bool) : Whether to use activation function.
            Default: False
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 deploy: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.act = nn.SiLU()

        if deploy:
            self.reparam_3x3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                padding_mode=padding_mode)

        else:
            if (out_channels == in_channels) and stride == 1:
                self.path_residual = nn.BatchNorm2d(in_channels)
            else:
                self.path_residual = None

            self.path_3x3_1 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding
            )
            self.path_3x3_2 = Block3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding
            )
            self.path_1x1 = Block1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding_11
            )

    def forward(self, inputs: Tensor) -> Tensor:

        if hasattr(self, 'reparam_3x3'):
            return self.act(self.reparam_3x3(inputs))

        if self.path_residual is None:
            id_out = 0
        else:
            id_out = self.path_residual(inputs)

        return self.act(self.path_3x3_1(inputs) + self.path_3x3_2(inputs) + self.path_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        self.path_3x3_1.switch_to_deploy()
        kernel3x3_1, bias3x3_1 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data
        self.path_3x3_2.switch_to_deploy()
        kernel3x3_2, bias3x3_2 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data
        self.path_1x1.switch_to_deploy()
        kernel1x1, bias1x1 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kernelid, biasid = self._fuse_bn_tensor(self.path_residual)

        return kernel3x3_1 + kernel3x3_2 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3_1 + bias3x3_2 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific conv layer.

        Args:
            conv (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if conv is None:
            return 0, 0
        if isinstance(conv, Conv):
            kernel = conv.conv.weight
            running_mean = conv.bn.running_mean
            running_var = conv.bn.running_var
            gamma = conv.bn.weight
            beta = conv.bn.bias
            eps = conv.bn.eps
        else:
            assert isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_in_channels = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_in_channels, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_in_channels, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    conv.weight.device)
            kernel = self.id_tensor
            running_mean = conv.running_mean
            running_var = conv.running_var
            gamma = conv.weight
            beta = conv.bias
            eps = conv.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'reparam_3x3'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
        # for para in self.parameters():
        #     para.detach_()
        self.__delattr__('path_3x3_1')
        self.__delattr__('path_3x3_2')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual'):
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True