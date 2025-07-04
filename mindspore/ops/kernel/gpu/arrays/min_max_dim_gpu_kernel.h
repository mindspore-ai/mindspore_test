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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MINMAXDIM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MINMAXDIM_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/arrays/argmaxandminwithvalue_gpu_kernel.h"
namespace mindspore {
namespace kernel {
class MinMaxDimGpuKernelMod : public ArgMaxAndMinWithValueGpuKernelMod {
 public:
  MinMaxDimGpuKernelMod() : ArgMaxAndMinWithValueGpuKernelMod(1, 0) {}

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using MinMaxDimFunc = std::function<bool(MinMaxDimGpuKernelMod *, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, MinMaxDimFunc>> func_list_;
  MinMaxDimFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MINMAXDIM_GPU_KERNEL_H_
