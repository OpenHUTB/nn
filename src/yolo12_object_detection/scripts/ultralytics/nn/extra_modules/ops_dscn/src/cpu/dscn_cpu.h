/*!
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
* Modified from
*https://github.com/OpenGVLab/InternImage
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

at::Tensor dscn_cpu_forward(const at::Tensor &input, const at::Tensor &offset,
                             const int kernel_h,
                             const int kernel_w, const int stride_h,
                             const int stride_w, const int pad_h,
                             const int pad_w, const int dilation_h,
                             const int dilation_w, const int group,
                             const int group_channels, const float offset_scale,
                             const int im2col_step);

std::vector<at::Tensor>
dscn_cpu_backward(const at::Tensor &input, const at::Tensor &offset,
                   const int kernel_h,
                   const int kernel_w, const int stride_h, const int stride_w,
                   const int pad_h, const int pad_w, const int dilation_h,
                   const int dilation_w, const int group,
                   const int group_channels, const float offset_scale,
                   const at::Tensor &grad_output, const int im2col_step);
