/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MINMAX_UPDATE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MINMAX_UPDATE_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"

CUDA_LIB_EXPORT cudaError_t CalMinMaxPerChannel(float *input, float *input_min, float *input_max, float *output_min,
                                                float *output_max, const int total_num, const int channel_num,
                                                const float ema_decay, const bool ema, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalMinMaxPerLayer(float *input, float *input_min, float *input_max, float *output_min,
                                              float *output_max, const int size, const float ema_decay, const bool ema,
                                              cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MINMAX_UPDATE_IMPL_CUH_
