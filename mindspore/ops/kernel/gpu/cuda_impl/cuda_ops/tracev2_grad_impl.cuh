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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CORRELATE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CORRELATE_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_device_info.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(T *din_addr, const T *dout_addr, const size_t row_st, const size_t col_st,
                                            const size_t diag_count, const size_t row_size, const size_t mat_size,
                                            const size_t batch_size, const uint32_t &device_id, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CORRELATE_IMPL_CUH_
