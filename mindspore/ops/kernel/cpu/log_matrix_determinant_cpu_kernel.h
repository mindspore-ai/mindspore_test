/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LOG_MATRIX_DETERMINANT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LOG_MATRIX_DETERMINANT_CPU_KERNEL_H_
#include <map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace log_matrix_determinant_cpu {
class LogMatrixDeterminantCpuKernelMod : public NativeCpuKernelMod {
 public:
  LogMatrixDeterminantCpuKernelMod() = default;
  ~LogMatrixDeterminantCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddAllSameAttr(true),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddAllSameAttr(true),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex64)
        .AddOutputAttr(kNumberTypeComplex64)
        .AddOutputAttr(kNumberTypeComplex64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex128)
        .AddOutputAttr(kNumberTypeComplex128)
        .AddOutputAttr(kNumberTypeComplex128)};
    return support_list;
  }

 private:
  CNodeWeakPtr node_wpt_;
  ShapeVector shape_x_;
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void LaunchLogMatrixDeterminant(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs);
};
}  // namespace log_matrix_determinant_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LOG_MATRIX_DETERMINANT_CPU_KERNEL_H_
