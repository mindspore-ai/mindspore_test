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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_APPLY_PROXIMAL_GRADIENT_DESCENT_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_APPLY_PROXIMAL_GRADIENT_DESCENT_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent(
  const size_t size, const size_t indices_size, T *var, const T *alpha, const T *l1, const T *l2, const T *grad,
  const S *indices, T *var_out, S *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink,
  int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2(
  const size_t size, const size_t indices_size, T *var, const T *alpha, const T *l1, const T *l2, const T *grad,
  const S *indices, T *var_out, S *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink,
  int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SPARSE_APPLY_PROXIMAL_GRADIENT_DESCENT_IMPL_CUH_
