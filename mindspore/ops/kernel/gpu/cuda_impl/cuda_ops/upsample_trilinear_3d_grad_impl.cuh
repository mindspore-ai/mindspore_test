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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UPSAMPLE_TRILINEAR_3D_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UPSAMPLE_TRILINEAR_3D_GRAD_IMPL_CUH_
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_device_info.h"
template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t CalUpsampleTrilinear3DGrad(const T *grad, const int n, const int c, const int grad_d,
                                                       const int grad_h, const int grad_w, const int dinput_d,
                                                       const int dinput_h, const int dinput_w, const S d_scale,
                                                       const S h_scale, const S w_scale, const bool align_corner,
                                                       T *dinput, const uint32_t device_id, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_UPSAMPLE_TRILINEAR_3D_GRAD_IMPL_CUH_
