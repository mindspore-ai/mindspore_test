/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAPTIVE_AVGPOOL3D_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAPTIVE_AVGPOOL3D_GRAD_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool3DGrad(const uint in_size, const uint out_size,
                                                       const uint input_channel, const uint input_height,
                                                       const uint input_width, const uint input_depth,
                                                       const uint output_channel, const uint output_height,
                                                       const uint output_width, const uint output_depth, T *input_data,
                                                       T *output_data, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAPTIVE_AVGPOOL3D_GRAD_IMPL_CUH_
