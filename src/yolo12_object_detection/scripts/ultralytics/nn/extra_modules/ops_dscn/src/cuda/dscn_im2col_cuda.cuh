/*!
***************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
* Modified from
*https://github.com/OpenGVLab/InternImage
**************************************************************************************************
*/

/*
更改目标：1. 删除与 modulation mask 相关的操作代码  -> 完成
         2. 增加判断 单个维度的条件 -> 完成 
         3. 更改代码使其仅在单个维度上变形  -> 完成
         4. 双线性插值 改为 线性插值 -> 完成
*/

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
inline int GET_BLOCKS(const int N, const int num_threads) {
    return (N + num_threads - 1) / num_threads;
}

#define opmath_t at::opmath_type<scalar_t>

template <typename scalar_t>
__device__ opmath_t dscn_im2col_linear(const scalar_t *&bottom_data,
                                          const int &height, const int &width,
                                          const int &ngroup,
                                          const int &group_channels,
                                          const opmath_t &h, const opmath_t &w,
                                          const int &g, const int &c, const bool on_x) {

    const int w_stride = ngroup * group_channels;
    const int h_stride = width * w_stride;
    const int base_ptr = g * group_channels + c;
    opmath_t val = 0.0;

    if(on_x){
        const int w_low = floor(w); //on_x=false, w_low = w
        const int w_high = w_low + 1;
        const opmath_t lw = w - w_low; // on_x=false, lw = 0
        const opmath_t hw = 1 - lw;
        const int h_low_ptr_offset = h * h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

        opmath_t v1 = 0;
        if (h >= 0 && w_low >= 0) {
            const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
            v1 = bottom_data[ptr1];
        }
        opmath_t v2 = 0;
        if (h >= 0 && w_high <= width - 1) {
            const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
            v2 = bottom_data[ptr2];
        }
        val = hw * v1 + lw * v2;
    }else{
        const int h_low = floor(h); // on_x=true, h_low = h
        const int h_high = h_low + 1;
        const opmath_t lh = h - h_low; // on_x=true, lh = 0
        const opmath_t hh = 1 - lh;

        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w * w_stride;

        opmath_t v1 = 0;
        if (h_low >= 0 && w >= 0) {
            const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
            v1 = bottom_data[ptr1];
        }
        opmath_t v3 = 0;
        if (h_high <= height - 1 && w >= 0) {
            const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
            v3 = bottom_data[ptr3];
        }
        val = hh * v1 + lh * v3;
    }

    return val;
}

template <typename scalar_t>
__device__ void dscn_col2im_linear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &ngroup, const int &group_channels, const opmath_t &h,
    const opmath_t &w, const int &m, const int &c, const opmath_t offset_scale,
    const opmath_t &top_grad, opmath_t *&grad_im,
    opmath_t *grad_offset, const bool on_x) {

    const int w_stride = ngroup * group_channels;
    const int h_stride = width * w_stride;
    const int base_ptr = m * group_channels + c;
    const opmath_t top_grad_im = top_grad;

    if(on_x){
        const int w_low = floor(w); // on_x = flase, w_low = w
        const int w_high = w_low + 1;
        const opmath_t lw = w - w_low; // on_x = flase, lw = 0
        const opmath_t hw = 1 - lw;

        const int h_low_ptr_offset = h * h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

        opmath_t grad_w_weight = 0.0;
        opmath_t v1 = 0;
        if (h >= 0 && w_low >= 0) {
            const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
            v1 = bottom_data[ptr1];
            grad_w_weight -= v1;
            atomicAdd(grad_im + ptr1, hw * top_grad_im);
        }
        opmath_t v2 = 0;
        if (h >= 0 && w_high <= width - 1) {
            const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
            v2 = bottom_data[ptr2];
            grad_w_weight += v2;
            atomicAdd(grad_im + ptr2, lw * top_grad_im);
        }

        *grad_offset = offset_scale * grad_w_weight * top_grad_im;
    }else{
        const int h_low = floor(h); // on_x = true, h_low = h
        const int h_high = h_low + 1;
        const opmath_t lh = h - h_low; // on_x = true, lh = 0
        const opmath_t hh = 1 - lh;

        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w * w_stride;

        opmath_t grad_h_weight = 0;
        opmath_t v1 = 0;
        if (h_low >= 0 && w >= 0) {
            const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
            v1 = bottom_data[ptr1];
            grad_h_weight -= v1;
            atomicAdd(grad_im + ptr1, hh * top_grad_im);
        }
        opmath_t v3 = 0;
        if (h_high <= height - 1 && w >= 0) {
            const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
            v3 = bottom_data[ptr3];
            grad_h_weight += v3;
            atomicAdd(grad_im + ptr3, lh * top_grad_im);
        }

        *grad_offset = offset_scale * grad_h_weight * top_grad_im;
    }
}


template <typename scalar_t>
__device__ void dscn_col2im_linear_gm(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &ngroup, const int &group_channels, const opmath_t &h,
    const opmath_t &w, const int &m, const int &c, const opmath_t offset_scale,
    const opmath_t &top_grad, opmath_t *&grad_im,
    opmath_t *grad_offset, const bool on_x) {

    const int w_stride = ngroup * group_channels;
    const int h_stride = width * w_stride;
    const int base_ptr = m * group_channels + c;
    const opmath_t top_grad_im = top_grad;

    if(on_x){
        const int w_low = floor(w); // on_x = false, w_low = w
        const int w_high = w_low + 1;

        const opmath_t lw = w - w_low; // on_x = false, lw = 0.
        const opmath_t hw = 1 - lw;

        const int h_low_ptr_offset = h * h_stride;
        const int w_low_ptr_offset = w_low * w_stride;
        const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

        opmath_t grad_w_weight = 0.0;
        opmath_t v1 = 0;
        if (h >= 0 && w_low >= 0) {
            const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
            v1 = bottom_data[ptr1];
            grad_w_weight -= v1;
            atomicAdd(grad_im + ptr1, hw * top_grad_im);
        }
        opmath_t v2 = 0;
        if (h >= 0 && w_high <= width - 1) {
            const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
            v2 = bottom_data[ptr2];
            grad_w_weight += v2;
            atomicAdd(grad_im + ptr2, lw * top_grad_im);
        }

        atomicAdd(grad_offset, offset_scale * grad_w_weight * top_grad_im);
    }else{
        const int h_low = floor(h); // on_x = true, h_low = h
        const int h_high = h_low + 1;

        const opmath_t lh = h - h_low; // on_x = true, lh = 0.
        const opmath_t hh = 1 - lh;

        const int h_low_ptr_offset = h_low * h_stride;
        const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
        const int w_low_ptr_offset = w * w_stride;

        opmath_t grad_h_weight = 0;

        opmath_t v1 = 0;
        if (h_low >= 0 && w >= 0) {
            const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
            v1 = bottom_data[ptr1];
            grad_h_weight -= v1;
            atomicAdd(grad_im + ptr1, hh * top_grad_im);
        }
        opmath_t v3 = 0;
        if (h_high <= height - 1 && w >= 0) {
            const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
            v3 = bottom_data[ptr3];
            grad_h_weight += v3;
            atomicAdd(grad_im + ptr3, lh * top_grad_im);
        }

        atomicAdd(grad_offset, offset_scale * grad_h_weight * top_grad_im);
    }
}

template <typename scalar_t>
__global__ void dscn_im2col_gpu_kernel(
    const int num_kernels, const scalar_t *data_im, const scalar_t *data_offset,
    scalar_t *data_col, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, const bool on_x) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        int _temp = index;
        const int c_col = _temp % group_channels;
        _temp /= group_channels;
        const int sampling_index = _temp; // batch_n * H * W * group
        const int g_col = _temp % group;
        _temp /= group;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                         (_temp % width_out) * stride_w;
        _temp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                         (_temp % height_out) * stride_h;
        _temp /= height_out;
        const int b_col = _temp;

        const int input_size = height_in * width_in;
        scalar_t *data_col_ptr = data_col + index;
        const int kernel_size = kernel_h * kernel_w - remove_center;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr;// << 1;
        const int qid_stride = group * group_channels;
        opmath_t col = 0;
        const scalar_t *data_im_ptr = data_im + b_col * input_size * qid_stride;
        // top-left
        const opmath_t p0_w_ =
            p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ =
            p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        const int center_h = kernel_h / 2;
        const int center_w = kernel_w / 2;

        for (int i = 0; i < kernel_w; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                // if not remove center, or remove center and not the center
                // data_offset shape (B,H,W,groups, (Kh*Kw - center_remove))
                if (i!=center_w || j!=center_h || !remove_center) {
                    opmath_t offset_w = 0.0, offset_h = 0.0;
                    if(on_x){
                        offset_w = data_offset[data_loc_w_ptr];
                    }else{
                        offset_h = data_offset[data_loc_w_ptr];
                    }
                    const opmath_t loc_w =
                        p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                    const opmath_t loc_h =
                        p0_h_ + (j * dilation_h + offset_h) * offset_scale;
                    if (loc_h > -1 && loc_w > -1 && loc_h < height_in &&
                        loc_w < width_in) {
                        col += dscn_im2col_linear(
                                data_im_ptr, height_in, width_in, group,
                                group_channels, loc_h, loc_w, g_col, c_col, on_x);
                    }
                    data_weight_ptr += 1;
                    data_loc_w_ptr += 1;
                }
            }
        }
        *data_col_ptr = col;
    }
}

// debug
template <typename scalar_t, unsigned int blockSize>
__global__ void dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int num_kernels, const scalar_t *grad_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, opmath_t *grad_im, opmath_t *grad_offset, const bool on_x) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        __shared__ opmath_t cache_grad_offset[blockSize];
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % group_channels;
        _temp /= group_channels;
        const int sampling_index = _temp;
        const int g_col = _temp % group;
        _temp /= group;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                         (_temp % width_out) * stride_w;
        _temp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                         (_temp % height_out) * stride_h;
        _temp /= height_out;
        const int b_col = _temp;

        const opmath_t top_grad = grad_col[index];
        const int input_size = height_in * width_in;
        const int kernel_size = kernel_h * kernel_w - remove_center;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr;// << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr;//<< 1;
        const int qid_stride = group * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;
        const opmath_t p0_w_ =
            p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ =
            p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        const int center_h = kernel_h / 2;
        const int center_w = kernel_w / 2;

        for (int i = 0; i < kernel_w; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                // if not remove center, or remove center and not the center
                if (i!=center_w || j!=center_h || !remove_center) {
                    opmath_t offset_w = 0.0, offset_h = 0.0;
                    if(on_x){
                        offset_w = data_offset[data_loc_w_ptr];
                    }else{
                        offset_h = data_offset[data_loc_w_ptr];
                    }
                    const opmath_t loc_w =
                        p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                    const opmath_t loc_h =
                        p0_h_ + (j * dilation_h + offset_h) * offset_scale;
                    *(cache_grad_offset + threadIdx.x) = 0;
                    if (loc_h > -1 && loc_w > -1 && loc_h < height_in &&
                        loc_w < width_in) {
                        dscn_col2im_linear(
                            data_im_ptr, height_in, width_in, group, group_channels,
                            loc_h, loc_w, g_col, c_col, offset_scale, top_grad,
                            grad_im_ptr,
                            cache_grad_offset + threadIdx.x,
                            on_x);
                    }

                    __syncthreads();
                    if (tid == 0) {
                        opmath_t _grad_w = 0.0, _grad_h = 0.0;
                        if(on_x){
                            _grad_w = cache_grad_offset[0];
                        }else{
                            _grad_h = cache_grad_offset[0];
                        }
                        for (unsigned int tid = 1; tid < blockSize; ++tid) {
                            if(on_x){
                                _grad_w += cache_grad_offset[tid];
                            }else{
                                _grad_h += cache_grad_offset[tid];
                            }
                        }
                        if(on_x){
                            *grad_offset = _grad_w;
                        }else{
                            *grad_offset = _grad_h;
                        }
                    }
                    __syncthreads();

                    data_weight_ptr += 1;
                    data_loc_w_ptr += 1;
                    grad_offset += 1;
                }
            }
        }
    }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int num_kernels, const scalar_t *grad_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, opmath_t *grad_im, opmath_t *grad_offset, const bool on_x) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        __shared__ opmath_t cache_grad_offset[blockSize];
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % group_channels;
        _temp /= group_channels;
        const int sampling_index = _temp;
        const int g_col = _temp % group;
        _temp /= group;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                         (_temp % width_out) * stride_w;
        _temp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                         (_temp % height_out) * stride_h;
        _temp /= height_out;
        const int b_col = _temp;

        const opmath_t top_grad = grad_col[index];
        const int input_size = height_in * width_in;
        const int kernel_size = kernel_h * kernel_w - remove_center;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr;// << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr;// << 1;
        const int qid_stride = group * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;
        const opmath_t p0_w_ =
            p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ =
            p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        const int center_h = kernel_h / 2;
        const int center_w = kernel_w / 2;

        for (int i = 0; i < kernel_w; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                // if not remove center, or remove center and not the center
                if (i!=center_w || j!=center_h || !remove_center) {
                    opmath_t offset_w = 0.0, offset_h = 0.0;
                    if(on_x){
                        offset_w = data_offset[data_loc_w_ptr];
                    }else{
                        offset_h = data_offset[data_loc_w_ptr];
                    }
                    const opmath_t loc_w =
                        p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                    const opmath_t loc_h =
                        p0_h_ + (j * dilation_h + offset_h) * offset_scale;
                    *(cache_grad_offset + threadIdx.x) = 0;
                    if (loc_h > -1 && loc_w > -1 && loc_h < height_in &&
                        loc_w < width_in) {
                        dscn_col2im_linear(
                            data_im_ptr, height_in, width_in, group, group_channels,
                            loc_h, loc_w, g_col, c_col, offset_scale, top_grad,
                            grad_im_ptr,
                            cache_grad_offset + threadIdx.x,
                            on_x);
                    }

                    __syncthreads();

                    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
                        if (tid < s) {
                            cache_grad_offset[tid] += cache_grad_offset[tid + s];
                        }
                        __syncthreads();
                    }

                    if (tid == 0) {
                        *grad_offset = cache_grad_offset[0];
                    }
                    __syncthreads();

                    data_weight_ptr += 1;
                    data_loc_w_ptr += 1;
                    grad_offset += 1;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void dscn_col2im_gpu_kernel_shm_reduce_v1(
    const int num_kernels, const scalar_t *grad_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, opmath_t *grad_im, opmath_t *grad_offset, const bool on_x) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        extern __shared__ int _s[];
        opmath_t *cache_grad_offset = (opmath_t *)_s;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % group_channels;
        _temp /= group_channels;
        const int sampling_index = _temp;
        const int g_col = _temp % group;
        _temp /= group;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                         (_temp % width_out) * stride_w;
        _temp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                         (_temp % height_out) * stride_h;
        _temp /= height_out;
        const int b_col = _temp;

        const opmath_t top_grad = grad_col[index];
        const int input_size = height_in * width_in;
        const int kernel_size = kernel_h * kernel_w - remove_center;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr;// << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr;// << 1;
        const int qid_stride = group * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;
        const opmath_t p0_w_ =
            p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ =
            p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        const int center_h = kernel_h / 2;
        const int center_w = kernel_w / 2;

        for (int i = 0; i < kernel_w; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                // if not remove center, or remove center and not the center
                if (i!=center_w || j!=center_h || !remove_center) {
                    opmath_t offset_w = 0.0, offset_h = 0.0;
                    if(on_x){
                        offset_w = data_offset[data_loc_w_ptr];
                    }else{
                        offset_h = data_offset[data_loc_w_ptr];
                    }
                    const opmath_t loc_w =
                        p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                    const opmath_t loc_h =
                        p0_h_ + (j * dilation_h + offset_h) * offset_scale;
                    *(cache_grad_offset + threadIdx.x) = 0;
                    if (loc_h > -1 && loc_w > -1 && loc_h < height_in &&
                        loc_w < width_in) {
                        dscn_col2im_linear(
                            data_im_ptr, height_in, width_in, group, group_channels,
                            loc_h, loc_w, g_col, c_col, offset_scale, top_grad,
                            grad_im_ptr,
                            cache_grad_offset + threadIdx.x,
                            on_x);
                    }

                    __syncthreads();
                    if (tid == 0) {
                        opmath_t _grad_w = 0.0, _grad_h = 0.0;
                        if(on_x){
                            opmath_t _grad_w = cache_grad_offset[0];
                        }else{
                            opmath_t _grad_h = cache_grad_offset[0];
                        }
                        for (unsigned int tid = 1; tid < blockDim.x; ++tid) {
                            if(on_x){
                                _grad_w += cache_grad_offset[tid];
                            }else{
                                _grad_h += cache_grad_offset[tid];
                            }
                        }
                        if(on_x){
                            *grad_offset = _grad_w;
                        }else{
                            *grad_offset = _grad_h;
                        }
                        
                    }
                    __syncthreads();

                    data_weight_ptr += 1;
                    data_loc_w_ptr += 1;
                    grad_offset += 1;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void dscn_col2im_gpu_kernel_shm_reduce_v2(
    const int num_kernels, const scalar_t *grad_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, opmath_t *grad_im, opmath_t *grad_offset, const bool on_x) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        extern __shared__ int _s[];
        opmath_t *cache_grad_offset = (opmath_t *)_s;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % group_channels;
        _temp /= group_channels;
        const int sampling_index = _temp;
        const int g_col = _temp % group;
        _temp /= group;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                         (_temp % width_out) * stride_w;
        _temp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                         (_temp % height_out) * stride_h;
        _temp /= height_out;
        const int b_col = _temp;

        const opmath_t top_grad = grad_col[index];
        const int input_size = height_in * width_in;
        const int kernel_size = kernel_h * kernel_w - remove_center;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr;// << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr;// << 1;
        const int qid_stride = group * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;
        const opmath_t p0_w_ =
            p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ =
            p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        const int center_h = kernel_h / 2;
        const int center_w = kernel_w / 2;

        for (int i = 0; i < kernel_w; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                // if not remove center, or remove center and not the center
                if (i!=center_w || j!=center_h || !remove_center) {
                    opmath_t offset_w = 0.0, offset_h = 0.0;
                    if(on_x){
                        offset_w = data_offset[data_loc_w_ptr];
                    }else{
                        offset_h = data_offset[data_loc_w_ptr];
                    }
                    const opmath_t loc_w =
                        p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                    const opmath_t loc_h =
                        p0_h_ + (j * dilation_h + offset_h) * offset_scale;
                    *(cache_grad_offset + threadIdx.x) = 0;
                    if (loc_h > -1 && loc_w > -1 && loc_h < height_in &&
                        loc_w < width_in) {
                        dscn_col2im_linear(
                            data_im_ptr, height_in, width_in, group, group_channels,
                            loc_h, loc_w, g_col, c_col, offset_scale, top_grad,
                            grad_im_ptr,
                            cache_grad_offset + threadIdx.x,
                            on_x);
                    }

                    __syncthreads();

                    for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
                         s >>= 1, spre >>= 1) {
                        if (tid < s) {
                            cache_grad_offset[tid] += cache_grad_offset[tid + s];
                            if (tid + (s << 1) < spre) {
                                cache_grad_offset[tid] +=
                                    cache_grad_offset[tid + (s << 1)];
                            }
                        }
                        __syncthreads();
                    }

                    if (tid == 0) {
                        *grad_offset = cache_grad_offset[0];
                    }
                    __syncthreads();

                    data_weight_ptr += 1;
                    data_loc_w_ptr += 1;
                    grad_offset += 1;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void dscn_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int num_kernels, const scalar_t *grad_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, opmath_t *grad_im, opmath_t *grad_offset, const bool on_x) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        extern __shared__ int _s[];
        opmath_t *cache_grad_offset = (opmath_t *)_s;
        unsigned int tid = threadIdx.x;
        int _temp = index;
        const int c_col = _temp % group_channels;
        _temp /= group_channels;
        const int sampling_index = _temp;
        const int g_col = _temp % group;
        _temp /= group;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                         (_temp % width_out) * stride_w;
        _temp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                         (_temp % height_out) * stride_h;
        _temp /= height_out;
        const int b_col = _temp;

        const opmath_t top_grad = grad_col[index];
        const int input_size = height_in * width_in;
        const int kernel_size = kernel_h * kernel_w - remove_center;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr;// << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr;// << 1;
        const int qid_stride = group * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;
        const opmath_t p0_w_ =
            p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ =
            p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        const int center_h = kernel_h / 2;
        const int center_w = kernel_w / 2;

        for (int i = 0; i < kernel_w; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                // if not remove center, or remove center and not the center
                if (i!=center_w || j!=center_h || !remove_center) {
                    opmath_t offset_w = 0.0, offset_h = 0.0;
                    if(on_x){
                        offset_w = data_offset[data_loc_w_ptr];
                    }else{
                        offset_h = data_offset[data_loc_w_ptr];
                    }
                    const opmath_t loc_w =
                        p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                    const opmath_t loc_h =
                        p0_h_ + (j * dilation_h + offset_h) * offset_scale;
                    *(cache_grad_offset + threadIdx.x) = 0;
                    if (loc_h > -1 && loc_w > -1 && loc_h < height_in &&
                        loc_w < width_in) {
                        dscn_col2im_linear(
                            data_im_ptr, height_in, width_in, group, group_channels,
                            loc_h, loc_w, g_col, c_col, offset_scale, top_grad,
                            grad_im_ptr,
                            cache_grad_offset + threadIdx.x,
                            on_x);
                    }

                    __syncthreads();

                    for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
                         s >>= 1, spre >>= 1) {
                        if (tid < s) {
                            cache_grad_offset[tid] += cache_grad_offset[tid + s];
                            if (tid + (s << 1) < spre) {
                                cache_grad_offset[tid] +=
                                    cache_grad_offset[tid + (s << 1)];
                            }
                        }
                        __syncthreads();
                    }

                    if (tid == 0) {
                        atomicAdd(grad_offset, cache_grad_offset[0]);
                    }
                    __syncthreads();

                    data_weight_ptr += 1;
                    data_loc_w_ptr += 1;
                    grad_offset += 1;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void dscn_col2im_gpu_kernel_gm(
    const int num_kernels, const scalar_t *grad_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int height_in,
    const int width_in, const int height_out, const int width_out,
    const opmath_t offset_scale, const int remove_center, opmath_t *grad_im, opmath_t *grad_offset, const bool on_x) {
    CUDA_KERNEL_LOOP(index, num_kernels) {
        int _temp = index;
        const int c_col = _temp % group_channels;
        _temp /= group_channels;
        const int sampling_index = _temp;
        const int g_col = _temp % group;
        _temp /= group;
        const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                         (_temp % width_out) * stride_w;
        _temp /= width_out;
        const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                         (_temp % height_out) * stride_h;
        _temp /= height_out;
        const int b_col = _temp;

        const opmath_t top_grad = grad_col[index];
        const int input_size = height_in * width_in;
        const int kernel_size = kernel_h * kernel_w - remove_center;
        int data_weight_ptr = sampling_index * kernel_size;
        int data_loc_w_ptr = data_weight_ptr;// << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        grad_offset += grad_sampling_ptr;// << 1;
        const int qid_stride = group * group_channels;
        const int im_ptr_offset = b_col * input_size * qid_stride;
        const scalar_t *data_im_ptr = data_im + im_ptr_offset;
        opmath_t *grad_im_ptr = grad_im + im_ptr_offset;
        const opmath_t p0_w_ =
            p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
        const opmath_t p0_h_ =
            p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;

        const int center_h = kernel_h / 2;
        const int center_w = kernel_w / 2;

        for (int i = 0; i < kernel_w; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                // if not remove center, or remove center and not the center
                if (i!=center_w || j!=center_h || !remove_center) {
                    opmath_t offset_w = 0.0, offset_h = 0.0;
                    if(on_x){
                        offset_w = data_offset[data_loc_w_ptr];
                    }else{
                        offset_h = data_offset[data_loc_w_ptr];
                    }
                    const opmath_t loc_w =
                        p0_w_ + (i * dilation_w + offset_w) * offset_scale;
                    const opmath_t loc_h =
                        p0_h_ + (j * dilation_h + offset_h) * offset_scale;
                    if (loc_h > -1 && loc_w > -1 && loc_h < height_in &&
                        loc_w < width_in) {
                        dscn_col2im_linear_gm(
                            data_im_ptr, height_in, width_in, group, group_channels,
                            loc_h, loc_w, g_col, c_col, offset_scale, top_grad,
                            grad_im_ptr, grad_offset, on_x);
                    }
                    data_weight_ptr += 1;
                    data_loc_w_ptr += 1;
                    grad_offset += 1;
                }
            }
        }
    }
}

template <typename scalar_t>
void dscn_im2col_cuda(cudaStream_t stream, const scalar_t *data_im,
                       const scalar_t *data_offset,
                       scalar_t *data_col, const int kernel_h,
                       const int kernel_w, const int stride_h,
                       const int stride_w, const int pad_h, const int pad_w,
                       const int dilation_h, const int dilation_w,
                       const int group, const int group_channels,
                       const int batch_n, const int height_in,
                       const int width_in, const int height_out,
                       const int width_out, const opmath_t offset_scale, const int remove_center, const bool on_x) {
    const int num_kernels =
        batch_n * height_out * width_out * group * group_channels;
    const int num_actual_kernels =
        batch_n * height_out * width_out * group * group_channels;
    const int num_threads = CUDA_NUM_THREADS;
    dscn_im2col_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
           stream>>>(num_kernels, data_im, data_offset, data_col,
                     kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                     dilation_h, dilation_w, group, group_channels, height_in,
                     width_in, height_out, width_out, offset_scale, remove_center, on_x);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in dscn_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
void dscn_col2im_cuda(
    cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_im,
    const scalar_t *data_offset, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int group, const int group_channels, const int batch_n,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    opmath_t *grad_im, opmath_t *grad_offset, const bool on_x) {
    const int num_threads =
        (group_channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : group_channels;
    const int num_kernels =
        batch_n * height_out * width_out * group * group_channels;
    const int num_actual_kernels =
        batch_n * height_out * width_out * group * group_channels;
    if (group_channels > 1024) {
        if ((group_channels & 1023) == 0) {
            dscn_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                   num_threads * 3 * sizeof(opmath_t), stream>>>(
                    num_kernels, grad_col, data_im, data_offset,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, group, group_channels, height_in,
                    width_in, height_out, width_out, offset_scale, remove_center, grad_im,
                    grad_offset, on_x);
        } else {
            dscn_col2im_gpu_kernel_gm<scalar_t>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
        }
    } else {
        switch (group_channels) {
        case 1:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 2:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 4:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 8:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 16:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 32:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 64:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 128:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 256:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 512:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        case 1024:
            dscn_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                  1024>
                <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                   stream>>>(num_kernels, grad_col, data_im, data_offset,
                             kernel_h, kernel_w, stride_h, stride_w,
                             pad_h, pad_w, dilation_h, dilation_w, group,
                             group_channels, height_in, width_in, height_out,
                             width_out, offset_scale, remove_center, grad_im, grad_offset, on_x);
            break;
        default:
            if (group_channels < 64) {
                dscn_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                       num_threads * 3 * sizeof(opmath_t), stream>>>(
                        num_kernels, grad_col, data_im, data_offset,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                        dilation_h, dilation_w, group, group_channels,
                        height_in, width_in, height_out, width_out,
                        offset_scale, remove_center, grad_im, grad_offset, on_x);
            } else {
                dscn_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
                    <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                       num_threads * 3 * sizeof(opmath_t), stream>>>(
                        num_kernels, grad_col, data_im, data_offset,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                        dilation_h, dilation_w, group, group_channels,
                        height_in, width_in, height_out, width_out,
                        offset_scale, remove_center, grad_im, grad_offset, on_x);
            }
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in dscn_col2im_cuda: %s\n", cudaGetErrorString(err));
    }
}