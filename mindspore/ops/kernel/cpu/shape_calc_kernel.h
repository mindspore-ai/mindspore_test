/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SHAPE_CALC_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SHAPE_CALC_KERNEL_H_

#include <map>
#include <set>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"
#include "ir/functor.h"

namespace mindspore {
namespace kernel {
class ShapeCalcCpuKernelMod : public NativeCpuKernelMod {
 public:
  ShapeCalcCpuKernelMod() = default;
  ~ShapeCalcCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override;

 private:
  ShapeArray outs_shape_;
  bool is_dynamic_len_out_{false};
  size_t inputs_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SHAPE_CALC_KERNEL_H_
