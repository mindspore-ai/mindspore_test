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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MOMENTUM_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MOMENTUM_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
template <typename T, typename S, typename G>
CUDA_LIB_EXPORT cudaError_t MomentumUpdateVariable(const size_t size, T *variable, T *accumulation,
                                                   const S *learning_rate, const G *gradient, const S *momentum,
                                                   bool use_nesterov, cudaStream_t cuda_stream);
template <typename T, typename S, typename G>
CUDA_LIB_EXPORT cudaError_t FusedWeightDecayScaleMomentum(const size_t element_num, S *weight_decay, S *scale,
                                                          T *variable, T *accumulation, const S *learning_rate,
                                                          const G *gradient, const S *momentum,
                                                          cudaStream_t cuda_stream);
template <typename T, typename S, typename G>
CUDA_LIB_EXPORT cudaError_t FusedWeightDecayMomentum(const size_t element_num, S *weight_decay, T *variable,
                                                     T *accumulation, const S *learning_rate, const G *gradient,
                                                     const S *momentum, cudaStream_t cuda_stream);
template <typename T, typename S, typename G>
CUDA_LIB_EXPORT cudaError_t FusedScaleMomentum(const size_t element_num, S *scale, T *variable, T *accumulation,
                                               const S *learning_rate, const G *gradient, const S *momentum,
                                               cudaStream_t cuda_stream);
template <typename T, typename S, typename G>
CUDA_LIB_EXPORT cudaError_t CombineFusedWeightDecayScaleMomentum(const size_t max, const size_t num,
                                                                 const size_t *element, S **weight_decay, S **scale,
                                                                 T **variable, T **accumulation, S **learning_rate,
                                                                 G **gradient, S **momentum, cudaStream_t cuda_stream);
template <typename T, typename S, typename G>
CUDA_LIB_EXPORT cudaError_t CombineFusedScaleMomentum(const size_t max, const size_t num, const size_t *element,
                                                      S **scale, T **variable, T **accumulation, S **learning_rate,
                                                      G **gradient, S **momentum, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_MOMENTUM_IMPL_CUH_
