/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GROUP_NORM_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GROUP_NORM_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "include/cuda_fp16.h"

template <typename T>
struct DynamicSharedMem;
template <>
struct DynamicSharedMem<float> {
  __device__ float *addr() {
    extern __shared__ float addr_float[];
    return addr_float;
  }
};
template <>
struct DynamicSharedMem<half> {
  __device__ half *addr() {
    extern __shared__ half addr_half[];
    return addr_half;
  }
};
template <>
struct DynamicSharedMem<double> {
  __device__ double *addr() {
    extern __shared__ double addr_ptr[];
    return addr_ptr;
  }
};
template <typename T>
CUDA_LIB_EXPORT cudaError_t GroupNorm(const int outer, const int inner, const int num_channel, const int HxW,
                                      const float epsilon, const T *x, const T *gamma, const T *beta, T *y,
                                      T *mean, T *rstd, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GROUP_NORM_IMPL_CUH_
