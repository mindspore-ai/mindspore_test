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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MASKED_FILL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MASKED_FILL_CUH_

#include <vector>
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT cudaError_t ElewiseMaskedFill(size_t inner_size, size_t output_size, const T *input, const bool *mask,
                                              T *value, T *output, const uint32_t device_id, cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t BroadcastMaskedFill(size_t inner_size, const std::vector<size_t> &input_shape,
                                                const std::vector<size_t> &mask_shape,
                                                const std::vector<size_t> &output_shape, const T *input,
                                                const bool *mask, T *value, T *output, const uint32_t device_id,
                                                cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MASKED_FILL_CUH_
