# --------------------------------------------------------
# Modified from
# https://github.com/OpenGVLab/InternImage
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
import DSCN

import pkg_resources
dscn_version = float(pkg_resources.get_distribution('DSCN').version)


class DSCNFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
            ctx, input, offset,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            group, group_channels, offset_scale, im2col_step, remove_center, on_x):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.on_x = on_x

        args = [
            input, offset, kernel_h,
            kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group,
            group_channels, offset_scale, ctx.im2col_step,
            ctx.remove_center, ctx.on_x
        ]

        output = DSCN.dscn_forward(*args)
        ctx.save_for_backward(input, offset)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset = ctx.saved_tensors

        args = [
            input, offset, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, grad_output.contiguous(), ctx.im2col_step,
            ctx.remove_center, ctx.on_x
        ]

        grad_input, grad_offset = \
            DSCN.dscn_backward(*args)

        return grad_input, grad_offset, \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, offset, kernel_h, kernel_w, stride_h,
                 stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                 group_channels, offset_scale, im2col_step, remove_center):
        """Symbolic function for mmdeploy::DSCN.

        Returns:
            DSCN op for onnx.
        """
        return g.op(
            'mmdeploy::TRTDSCN',
            input,
            offset,
            kernel_h_i=int(kernel_h),
            kernel_w_i=int(kernel_w),
            stride_h_i=int(stride_h),
            stride_w_i=int(stride_w),
            pad_h_i=int(pad_h),
            pad_w_i=int(pad_w),
            dilation_h_i=int(dilation_h),
            dilation_w_i=int(dilation_w),
            group_i=int(group),
            group_channels_i=int(group_channels),
            offset_scale_f=float(offset_scale),
            im2col_step_i=int(im2col_step),
            remove_center=int(remove_center),
        )