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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SEGMENT_SQRT_N_WITH_NUM_SGEMENTS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SEGMENT_SQRT_N_WITH_NUM_SGEMENTS_CPU_KERNEL_H_

#include <functional>
#include <vector>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace sparse_segment_sqrt_n_with_num_segments_cpu {
class SparseSegmentSqrtNWithNumSegmentsCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseSegmentSqrtNWithNumSegmentsCpuKernelMod() = default;
  ~SparseSegmentSqrtNWithNumSegmentsCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  template <typename T1, typename T2>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  ShapeVector x_shape_;
  ShapeVector indices_shape_;
  ShapeVector segment_ids_shape_;
  ShapeVector num_segments_shape_;
  ShapeVector y_shape_;
  TypeId xdtype_{kTypeUnknown};
  TypeId dtype1_{kTypeUnknown};
};
}  // namespace sparse_segment_sqrt_n_with_num_segments_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SEGMENT_SQRT_N_WITH_NUM_SGEMENTS_CPU_KERNEL_H_
