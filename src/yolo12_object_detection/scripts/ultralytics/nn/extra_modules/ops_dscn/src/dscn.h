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

#include "cpu/dscn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/dscn_cuda.h"
#endif

at::Tensor dscn_forward(const at::Tensor &input, const at::Tensor &offset,
                         const int kernel_h,
                         const int kernel_w, const int stride_h,
                         const int stride_w, const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int group, const int group_channels,
                         const float offset_scale, const int im2col_step, const int remove_center, const bool on_x) {
    if (input.type().is_cuda()) {
#ifdef WITH_CUDA
        return dscn_cuda_forward(input, offset, kernel_h, kernel_w,
                                  stride_h, stride_w, pad_h, pad_w, dilation_h,
                                  dilation_w, group, group_channels,
                                  offset_scale, im2col_step, remove_center, on_x);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
dscn_backward(const at::Tensor &input, const at::Tensor &offset,
               const int kernel_h, const int kernel_w,
               const int stride_h, const int stride_w, const int pad_h,
               const int pad_w, const int dilation_h, const int dilation_w,
               const int group, const int group_channels,
               const float offset_scale, const at::Tensor &grad_output,
               const int im2col_step, const int remove_center, const bool on_x) {
    if (input.type().is_cuda()) {
#ifdef WITH_CUDA
        return dscn_cuda_backward(input, offset, kernel_h, kernel_w,
                                   stride_h, stride_w, pad_h, pad_w, dilation_h,
                                   dilation_w, group, group_channels,
                                   offset_scale, grad_output, im2col_step, remove_center, on_x);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}
