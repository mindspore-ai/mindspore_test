/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BIAS_DROPOUT_ADD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BIAS_DROPOUT_ADD_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t BiasDropoutAdd(const T *x, const T *bias, const T *residual, T *y, T *mask,
                                           size_t num_count, size_t n_strides, size_t channel_strides, float keep_prob,
                                           uint64_t seed, uint64_t seed_offset, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BIAS_DROPOUT_ADD_IMPL_CUH_
