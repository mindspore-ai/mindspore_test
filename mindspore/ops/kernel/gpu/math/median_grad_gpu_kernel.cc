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

#include "kernel/gpu/math/median_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_THREE(MedianGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeInt16)
                          .AddInputAttr(kNumberTypeInt16)
                          .AddInputAttr(kNumberTypeInt16)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeFloat32),
                        MedianGradGpuKernelMod, int16_t, int64_t, float)
MS_REG_GPU_KERNEL_THREE(MedianGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt32)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeFloat32),
                        MedianGradGpuKernelMod, int32_t, int64_t, float)
MS_REG_GPU_KERNEL_THREE(MedianGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeFloat32),
                        MedianGradGpuKernelMod, int64_t, int64_t, float)
MS_REG_GPU_KERNEL_THREE(MedianGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeFloat32)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeFloat32),
                        MedianGradGpuKernelMod, float, int64_t, float)
MS_REG_GPU_KERNEL_THREE(MedianGrad,
                        KernelAttr()
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeFloat64)
                          .AddInputAttr(kNumberTypeInt64)
                          .AddOutputAttr(kNumberTypeFloat64),
                        MedianGradGpuKernelMod, double, int64_t, double)
}  // namespace kernel
}  // namespace mindspore
