/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_PLACE_OP_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_PLACE_OP_CPU_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace in_place_op_cpu {
class InPlaceOpCpuKernelMod : public NativeCpuKernelMod {
 public:
  InPlaceOpCpuKernelMod() = default;
  explicit InPlaceOpCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~InPlaceOpCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    constexpr size_t inputs_num = 2;
    constexpr size_t outputs_num = 1;
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), inputs_num, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
  std::string kernel_type_{"Unknown"};
};
}  // namespace in_place_op_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_IN_PLACE_OP_CPU_KERNEL_H_
