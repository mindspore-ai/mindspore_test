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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_IMPL_CUH_

#include <vector>
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T, typename S, typename G>
CUDA_LIB_EXPORT cudaError_t CalFractionalmaxpoolwithfixedksize(const T *input, const S *random_samples, T *output,
                                                               G *argmax, int64_t outputH, int64_t outputW,
                                                               int64_t inputN, int64_t inputC, int64_t inputH,
                                                               int64_t inputW, int64_t kernelsizeH, int64_t kernelsizeW,
                                                               const int64_t outer_size, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_IMPL_CUH_
