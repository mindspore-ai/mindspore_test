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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_DIAG_PART_V3_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_DIAG_PART_V3_CPU_KERNEL_H_

#include <map>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace matrix_diag_part_v3_cpu {
class MatrixDiagPartV3CpuKernelMod : public NativeCpuKernelMod {
 public:
  MatrixDiagPartV3CpuKernelMod() = default;
  ~MatrixDiagPartV3CpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> & /* inputs */,
            const std::vector<KernelTensor *> & /* outputs */) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using MatrixDiagPartV3Func =
    std::function<bool(MatrixDiagPartV3CpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, MatrixDiagPartV3Func>> func_list_;
  MatrixDiagPartV3Func kernel_func_;

  template <typename T>
  bool DoLaunch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  std::vector<int64_t> x_shape_;
  std::vector<int64_t> k_shape_;
  TypeId input_dtype_{kTypeUnknown};
  std::string align_{"RIGHT_LEFT"};
  int64_t num_diags_ = 1;
  int64_t max_diag_len_ = 0;
  int64_t output_elements_in_batch_ = 0;
  bool align_superdiag_ = true;
  bool align_subdiag_ = true;
  int64_t num_cols_ = 1;
  int64_t num_rows_ = 1;
  int64_t upper_diag_index_ = 0;
  int64_t data_num_ = 0;
  int64_t num_array_ = 0;
};
}  // namespace matrix_diag_part_v3_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_DIAG_PART_V3_CPU_KERNEL_H_
