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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXIMUM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXIMUM_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <memory>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace maximum_cpu {
class MaximumCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<MaximumCpuKernelMod> {
 public:
  MaximumCpuKernelMod() = default;
  ~MaximumCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return MatchKernelHelper::OpSupport(); }

 private:
  bool IsBroadcast() const;
  size_t Index(const size_t &index, const size_t &dim) const;
  void InitTensorBroadcastShape();
  void InitInputTensorAndScalar(size_t max_input_shape_size);
  void InitInputTensors();

  // Broadcast Arithmetic
  template <typename T>
  void BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                            const size_t l5, const size_t l6, const size_t l7, const size_t r0, const size_t r1,
                            const size_t r2, const size_t r3, const size_t r4, const size_t r5, const size_t r6,
                            const size_t r7, const size_t d0, const size_t d1, const size_t d2, const size_t d3,
                            const size_t d4, const size_t d5, const size_t d6, const size_t d7, const T *input_x,
                            const T *input_y, T *output);
  template <typename T>
  T MaximumFunc(const T &lhs, const T &rhs) const {
    return lhs > rhs ? lhs : rhs;
  }
  template <typename T>
  void BroadcastArithOneScalarOneTensor(const T *input_x, const T *input_y, T *output);
  template <typename T>
  void BroadcastArithTensors(const T *input_x, const T *input_y, T *output);
  template <typename T>
  void BroadcastArith(const T *input_x, const T *input_y, T *output);
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);

  bool need_broadcast_{false};
  size_t input_x_num_{1};
  size_t input_y_num_{1};
  size_t output_num_{1};
  std::vector<int64_t> input_x_shape_;
  std::vector<int64_t> input_y_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<size_t> broadcast_input_x_shape_;
  std::vector<size_t> broadcast_input_y_shape_;
  std::vector<size_t> broadcast_output_shape_;
  const size_t max_dims_{8};
};
}  // namespace maximum_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXIMUM_CPU_KERNEL_H_
